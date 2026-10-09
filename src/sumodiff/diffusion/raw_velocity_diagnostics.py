"""Bounded joint-gradient calibration and isolated full-strength peak stress."""
from copy import deepcopy
import math
import time
import numpy as np
import torch
from sumodiff.models import ConditionalDenoiser,resolve_model_config
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.data.dataset import write_json
from .controlled_training import inherited_state,tensor_hash
from .data import select_batch
from .epoch_training import epoch_batch,probe_subset
from .training import draw_noise,synchronize,probe
from .process import NoiseSchedule,normalize
from .config import resolve_diffusion_config
from .unified_losses import RoadGeometryCache
from .unified_training import new_optimizer
from .raw_velocity_training import inherit_optimizer
from .raw_velocity_losses import resolve_raw_velocity_loss,raw_velocity_objective
from .velocity_validation import gradient_vector

FIELDS=('noise_mse','weighted_imitation','weighted_temporal','weighted_road',
        'weighted_raw_interior','weighted_raw_boundary','inherited_auxiliary_total',
        'raw_auxiliary_total','auxiliary_total','total_loss')

def vector_statistics(vector,reference,clip):
    n=float(vector.norm());r=float(reference.norm())
    cos=float(torch.dot(vector,reference)/n/r) if n and r else None
    return dict(parameter_gradient_norm=n,norm_ratio_to_epsilon=n/r if r else None,
        cosine_with_epsilon=cos,angle_degrees=math.degrees(math.acos(max(-1.,min(1.,cos)))) if cos is not None else None,
        would_clip=n>clip,clip_scale=min(1.,clip/(n+1e-6)))

def summarize_rows(rows):
    eligible=[r for r in rows if r['eligible_scenes']]
    def distribution(values):
        values=[x for x in values if x is not None]
        return dict(median=float(np.median(values)) if values else None,maximum=max(values) if values else None,
                    minimum=min(values) if values else None)
    return dict(batches=len(rows),eligible_batches=len(eligible),
        components={key:{name:distribution([r['components'][key][name] for r in eligible])
                    for name in ('parameter_gradient_norm','norm_ratio_to_epsilon','cosine_with_epsilon','angle_degrees')}
                    for key in FIELDS},
        total_would_clip_batches=sum(r['components']['total_loss']['would_clip'] for r in rows),
        total_clip_scale=distribution([r['components']['total_loss']['clip_scale'] for r in rows]),
        half_imitation_joint_ratio=distribution([r['half_imitation_sensitivity']['norm_ratio_to_epsilon'] for r in eligible]),
        interpretation='Conditional eligible ratios; clipping counted over all batches. Vector sums, never sums of norm ratios.')

def joint_gradient_audit(dataset,parent,c,cache,output,batches=16):
    if type(batches) is not int or not 1<=batches<=16:raise ValueError('Bounded at 16 B4 batches')
    configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
    model=ConditionalDenoiser(resolve_model_config(parent['metadata']['model_config'])).to(device)
    optimizer=new_optimizer(model,c);generator=torch.Generator()
    inherit_optimizer(parent,model,optimizer,generator,device);before=inherited_state(model,optimizer,generator)
    config=resolve_raw_velocity_loss(c['loss']);geometry=RoadGeometryCache(dataset,cache[0][2],config)
    schedule=NoiseSchedule(resolve_diffusion_config(c['diffusion']),device);parameters=list(model.parameters())
    rows=[];started=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for i in range(batches):
        indices,epoch,index=epoch_batch(45001+i,len(cache[0][2]),c['batch_size'],c['seed'])
        cond,target,ids=select_batch(cache[0],indices,device)
        natural,noise,ts=draw_noise(target,cond,schedule,generator)
        for mode in ('natural','fixed_t200'):
            times=ts if mode=='natural' else torch.full_like(ts,200)
            noisy=natural if mode=='natural' else torch.where(cond['agent_mask'][...,None,None],
                schedule.add_noise(normalize(target['future'],schedule.config),noise,times),0.)
            model.train()
            terms=raw_velocity_objective(model(noisy,times,cond),noisy,noise,times,cond,target,
                schedule,config,roads=geometry.batch(ids),ramp=1.)
            vectors={key:gradient_vector(terms[key],parameters) for key in FIELDS}
            reference=vectors['noise_mse']
            stats={key:dict(loss=float(terms[key].detach()),
                **vector_statistics(v,reference,c['gradient_clip_norm'])) for key,v in vectors.items()}
            parts=sum((vectors[key] for key in ('weighted_imitation','weighted_temporal','weighted_road',
                      'weighted_raw_interior','weighted_raw_boundary')),torch.zeros_like(reference))
            residual=float((parts-vectors['auxiliary_total']).norm())
            relative=residual/max(float(vectors['auxiliary_total'].norm()),1e-12)
            if relative>1e-4:raise RuntimeError('Joint gradient does not match component vector sum')
            sensitivity=vectors['auxiliary_total']-.5*vectors['weighted_imitation']
            rows.append(dict(batch=i+1,mode=mode,global_next_step=45001+i,epoch=epoch,epoch_batch=index,
                window_ids=ids,timesteps=times.cpu().tolist(),noise_sha256=tensor_hash(noise),
                eligible_scenes=terms['auxiliary_eligible_scenes'],raw_ramp=1.,inherited_ramp=1.,
                components=stats,component_vector_relative_residual=relative,
                half_imitation_sensitivity=vector_statistics(sensitivity,reference,c['gradient_clip_norm'])))
            del vectors,parts,reference,terms
    synchronize(device)
    after=inherited_state(model,optimizer,generator)
    result=dict(schema_version='sumodiff.raw.velocity.gradient.audit.v1',objective=config.to_dict(),
        summaries={mode:summarize_rows([r for r in rows if r['mode']==mode]) for mode in ('natural','fixed_t200')},
        rows=rows,optimizer_updates=0,model_unchanged=before['model']==after['model'],
        optimizer_unchanged=before['optimizer']==after['optimizer'],
        parameter_grad_buffers_unset=all(p.grad is None for p in parameters),
        wall_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
        imitation_review='Current 0.003 remains. Half-weight is a frozen vector sensitivity, not a trained candidate or automatic selection.',
        quality_or_convergence_claim=False)
    if not all(result[k] for k in ('model_unchanged','optimizer_unchanged','parameter_grad_buffers_unset')):
        raise RuntimeError('Frozen joint audit changed parameters or optimizer')
    write_json(output,result)
    return result

def peak_stress(dataset,parent,c,cache,output,updates=24):
    """Separate copies at peak lr and rho_raw=1; no production checkpoint.

    This bypasses warmup deliberately and is not a continuation schedule prefix.
    It can catch numerical/gradient instability, not determine an optimal LR.
    """
    if type(updates) is not int or not 1<=updates<=24:raise ValueError('Peak stress bounded at 24 updates per LR')
    device=torch.device(c['device']);objective=resolve_raw_velocity_loss(c['loss'])
    geometry=RoadGeometryCache(dataset,cache[0][2],objective)
    schedule=NoiseSchedule(resolve_diffusion_config(c['diffusion']),device)
    trials=[];trace_reference=None;source=deepcopy(parent['optimizer'])
    for peak in (2e-5,5e-5):
        configure(c['seed'],c['torch_threads'])
        model=ConditionalDenoiser(resolve_model_config(parent['metadata']['model_config'])).to(device)
        opt=new_optimizer(model,c);g=torch.Generator();inherit_optimizer(parent,model,opt,g,device)
        start_parameter=parameter_hash(model)
        start_moments=inherited_state(model,opt,g)
        rows=[];started=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        for i in range(updates):
            for group in opt.param_groups:group['lr']=peak
            indices,epoch,index=epoch_batch(45001+i,len(cache[0][2]),c['batch_size'],c['seed'])
            cond,target,ids=select_batch(cache[0],indices,device)
            model.train();opt.zero_grad(set_to_none=True)
            noisy,noise,ts=draw_noise(target,cond,schedule,g)
            terms=raw_velocity_objective(model(noisy,ts,cond),noisy,noise,ts,cond,target,schedule,
                                         objective,roads=geometry.batch(ids),ramp=1.)
            if not torch.isfinite(terms['total_loss']):raise FloatingPointError('Nonfinite peak stress loss')
            terms['total_loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True)
            opt.step()
            rows.append(dict(probe_update=i+1,source_batch_step=45001+i,epoch=epoch,epoch_batch=index,
                window_ids=ids,timesteps=ts.cpu().tolist(),noise_sha256=tensor_hash(noise),learning_rate=peak,
                total_loss=float(terms['total_loss'].detach()),noise_mse=float(terms['noise_mse'].detach()),
                auxiliary_total=float(terms['auxiliary_total'].detach()),raw_ramp=1.,inherited_ramp=1.,
                gradient_norm_before_clip=float(norm),clipped=bool(norm>c['gradient_clip_norm'])))
        synchronize(device)
        pairing=[{k:r[k] for k in ('window_ids','timesteps','noise_sha256')} for r in rows]
        if trace_reference is None:trace_reference=pairing
        elif pairing!=trace_reference:raise RuntimeError('Peak stress pairing differs')
        trials.append(dict(peak_lr=peak,updates=updates,initial_state=start_moments,rows=rows,
            initial_parameter_sha256=start_parameter,final_parameter_sha256=parameter_hash(model),
            clipped_updates=sum(r['clipped'] for r in rows),
            gradient_norm_max=max(r['gradient_norm_before_clip'] for r in rows),
            validation_epsilon=probe(model,cache[1],schedule,device,c['batch_size'],c['validation_seed']),
            training_probe_epsilon=probe(model,probe_subset(cache[0],c['train_probe_limit']),schedule,
                                        device,c['batch_size'],c['validation_seed']),
            wall_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))
    from .audit import exact_tree
    if not exact_tree(parent['optimizer'],source):raise RuntimeError('Peak stress mutated the loaded parent')
    result=dict(schema_version='sumodiff.raw.velocity.peak.stress.v1',trials=trials,noise_and_batches_paired=True,
        source_checkpoint_state_unchanged=True,production_training_updates=0,
        isolated_optimizer_updates=2*updates,warmup_bypassed=True,production_checkpoints_saved=False,
        interpretation='24 paired peak/full-aux updates per private copy; finite stress check only, not LR optimum or generation quality.',
        default_peak_lr=c['schedule']['peak_lr'],automatic_lr_or_loss_selection=False)
    write_json(output,result)
    return result
