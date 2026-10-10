"""Training-only frozen joint calibration and bounded full-strength stress."""
from collections import defaultdict, Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from sumodiff.data.dataset import WindowDataset, write_json
from sumodiff.models import ConditionalDenoiser, resolve_model_config
from sumodiff.models.cli import configure
from .boundary_validation import descriptors
from .controlled_training import inherited_state, tensor_hash
from .data import select_batch
from .process import NoiseSchedule, normalize
from .config import resolve_diffusion_config
from .training import draw_noise, synchronize, probe
from .epoch_training import epoch_batch
from .unified_training import new_optimizer
from .unified_losses import RoadGeometryCache
from .decoded_temporal_training import initialize
from .decoded_temporal_losses import resolve_decoded_temporal_loss, decoded_temporal_objective
from .raw_velocity_diagnostics import vector_statistics
from .velocity_validation import gradient_vector

FIELDS=('noise_mse','weighted_imitation','weighted_road','weighted_raw_boundary','weighted_raw_interior',
        'weighted_decoded_boundary','weighted_decoded_interior','decoded_temporal_boundary',
        'decoded_temporal_interior','auxiliary_total','total_loss')
PARTS=('weighted_imitation','weighted_road','weighted_raw_boundary','weighted_raw_interior',
       'weighted_decoded_boundary','weighted_decoded_interior')
MODES=('natural','fixed_t0','fixed_t50','fixed_t200')


def balanced_indices(meta,count,seed,dense):
    """Geometry/episode round robin using present metadata, no future error."""
    if type(count) is not int or not 4<=count<=16 or count%4:
        raise ValueError('Audit cohort is bounded at 4 through 16 complete B4 scenes')
    buckets=defaultdict(list)
    for i,r in enumerate(meta):
        is_dense=r['family']=='unsignalized_intersection' and r['agents']>=8
        if is_dense==dense:
            buckets[(r['family'],r['geometry_id'],r['episode_key'])].append(i)
    for v in buckets.values():
        v.sort(key=lambda i:hashlib.sha256((str(seed)+'|'+meta[i]['window_id']).encode()).hexdigest())
    chosen=[]
    while len(chosen)<count and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(chosen)<count:chosen.append(buckets[key].pop(0))
    if len(chosen)!=count:raise ValueError('Insufficient cohort; no silent undersampling')
    return chosen


def distribution(values):
    a=[x for x in values if x is not None]
    return dict(median=float(np.median(a)) if a else None,minimum=min(a) if a else None,maximum=max(a) if a else None)


def summarize(rows):
    eligible=[r for r in rows if r['eligible_scenes']]
    return dict(batches=len(rows),eligible_batches=len(eligible),
        components={key:{stat:distribution([r['components'][key][stat] for r in eligible])
            for stat in ('parameter_gradient_norm','norm_ratio_to_epsilon','cosine_with_epsilon','angle_degrees')}
            for key in FIELDS},total_would_clip_batches=sum(r['components']['total_loss']['would_clip'] for r in rows),
        candidates={str(w):dict(auxiliary_ratio=distribution([r['candidate_weight_sensitivity'][str(w)]['auxiliary']['norm_ratio_to_epsilon'] for r in eligible]),
            cosine=distribution([r['candidate_weight_sensitivity'][str(w)]['auxiliary']['cosine_with_epsilon'] for r in eligible]),
            total_norm=distribution([r['candidate_weight_sensitivity'][str(w)]['total']['parameter_gradient_norm'] for r in rows]),
            would_clip_batches=sum(r['candidate_weight_sensitivity'][str(w)]['total']['would_clip'] for r in rows))
            for w in (.0001,.0004,.001,.003)})


def joint_gradient_audit(dataset,parent,c,cache,output,batches_per_group=4):
    if type(batches_per_group) is not int or not 1<=batches_per_group<=4:
        raise ValueError('At most four B4 batches per group')
    configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
    ds=WindowDataset(dataset,'train');meta=descriptors(ds)
    byid={r['window_id']:r for r in meta}
    if len(ds)!=1200 or set(byid)!=set(cache[0][2]):raise ValueError('Certified training cohort mismatch')
    ids_to_index={wid:i for i,wid in enumerate(cache[0][2])}
    selected={name:[ids_to_index[meta[i]['window_id']] for i in balanced_indices(meta,4*batches_per_group,c['seed'],dense)]
        for name,dense in [('ordinary',False),('dense_intersection',True)]}
    obj=resolve_decoded_temporal_loss(c['loss']);roads=RoadGeometryCache(dataset,cache[0][2],obj)
    model=ConditionalDenoiser(resolve_model_config(parent['metadata']['model_config'])).to(device)
    optimizer=new_optimizer(model,c);generator=torch.Generator();initialize(parent,model,optimizer,generator,device)
    before=inherited_state(model,optimizer,generator)
    noise_schedule=NoiseSchedule(resolve_diffusion_config(c['diffusion']),device)
    params=list(model.parameters());rows=[];calls=0;started=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for group,indices in selected.items():
        for j in range(batches_per_group):
            cond,target,ids=select_batch(cache[0],np.asarray(indices[j*4:(j+1)*4],dtype=np.int64),device)
            natural,noise,ts=draw_noise(target,cond,noise_schedule,generator)
            for mode in MODES:
                times=ts if mode=='natural' else torch.full_like(ts,int(mode[7:]))
                noisy=natural if mode=='natural' else torch.where(cond['agent_mask'][...,None,None],
                    noise_schedule.add_noise(normalize(target['future'],noise_schedule.config),noise,times),0.)
                model.train();calls+=1
                terms=decoded_temporal_objective(model(noisy,times,cond),noisy,noise,times,cond,target,
                    noise_schedule,obj,roads=roads.batch(ids),ramp=1.)
                vectors={k:gradient_vector(terms[k],params) for k in FIELDS};ref=vectors['noise_mse']
                combined=sum((vectors[k] for k in PARTS),torch.zeros_like(ref))
                relative=float((combined-vectors['auxiliary_total']).norm())/max(float(combined.norm()),1e-12)
                if relative>1e-4:raise RuntimeError('Joint auxiliary vector differs from actual component sum')
                other=sum((vectors[k] for k in PARTS[:4]),torch.zeros_like(ref))
                candidates={}
                for w in (.0001,.0004,.001,.003):
                    aux=other+w*vectors['decoded_temporal_boundary']+obj.interior_weight*vectors['decoded_temporal_interior']
                    candidates[str(w)]=dict(auxiliary=vector_statistics(aux,ref,c['gradient_clip_norm']),
                        total=vector_statistics(ref+aux,ref,c['gradient_clip_norm']))
                rows.append(dict(group=group,batch=j+1,mode=mode,window_ids=ids,
                    metadata=[byid[wid] for wid in ids],timesteps=times.cpu().tolist(),noise_sha256=tensor_hash(noise),
                    noised_input_sha256=tensor_hash(noisy),eligible_scenes=terms['auxiliary_eligible_scenes'],
                    temporal_reweight_ramp=1.,raw_auxiliary_ramp=1.,inherited_auxiliary_ramp=1.,
                    components={k:dict(loss=float(terms[k].detach()),**vector_statistics(v,ref,c['gradient_clip_norm']))
                        for k,v in vectors.items()},component_vector_relative_residual=relative,
                    candidate_weight_sensitivity=candidates))
                del vectors,combined,other,terms
    synchronize(device);after=inherited_state(model,optimizer,generator)
    result=dict(schema_version='sumodiff.decoded.temporal.gradient.audit.v1',objective=obj.to_dict(),rows=rows,
        summaries={group:{mode:summarize([r for r in rows if r['group']==group and r['mode']==mode])
            for mode in MODES} for group in selected},
        cohorts={group:[byid[cache[0][2][i]] for i in v] for group,v in selected.items()},
        optimizer_updates=0,model_unchanged=before['model']==after['model'],
        optimizer_unchanged=before['optimizer']==after['optimizer'],parameter_grad_buffers_unset=all(p.grad is None for p in params),
        actual_model_forward_calls=calls,wall_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
        interpretation='Training-only metadata controls; natural ratios conditional on low-t eligibility. Frozen vector sensitivities do not prove training benefit; norms are never added as ratios.',
        validation_or_test_used_for_selection=False,automatic_loss_selection=False)
    if not all(result[k] for k in ('model_unchanged','optimizer_unchanged','parameter_grad_buffers_unset')):
        raise RuntimeError('Frozen audit mutated training state')
    write_json(output,result);return result


def full_strength_stress(dataset,parent,c,cache,output,updates=12):
    if type(updates) is not int or not 1<=updates<=24:raise ValueError('Stress limited to 24 private updates')
    configure(c['seed'],c['torch_threads']);device=torch.device(c['device']);obj=resolve_decoded_temporal_loss(c['loss'])
    model=ConditionalDenoiser(resolve_model_config(parent['metadata']['model_config'])).to(device)
    opt=new_optimizer(model,c);g=torch.Generator();initialize(parent,model,opt,g,device)
    source=deepcopy(parent['optimizer']);initial=inherited_state(model,opt,g)
    road=RoadGeometryCache(dataset,cache[0][2],obj);schedule=NoiseSchedule(resolve_diffusion_config(c['diffusion']),device)
    rows=[];started=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for i in range(updates):
        idx,epoch,batch=epoch_batch(c['source_step']+i+1,len(cache[0][2]),c['batch_size'],c['seed'])
        cond,target,ids=select_batch(cache[0],idx,device);model.train();opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:group['lr']=c['schedule']['peak_lr']
        noisy,n,t=draw_noise(target,cond,schedule,g)
        terms=decoded_temporal_objective(model(noisy,t,cond),noisy,n,t,cond,target,schedule,obj,roads=road.batch(ids),ramp=1.)
        if not torch.isfinite(terms['total_loss']):raise FloatingPointError('Nonfinite full-strength stress')
        terms['total_loss'].backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True);opt.step()
        rows.append(dict(update=i+1,window_ids=ids,timesteps=t.cpu().tolist(),noise_sha256=tensor_hash(n),
            epoch=epoch,epoch_batch=batch,learning_rate=c['schedule']['peak_lr'],temporal_reweight_ramp=1.,
            gradient_norm_before_clip=float(norm),clipped=bool(norm>c['gradient_clip_norm'])))
    synchronize(device)
    from .audit import exact_tree
    if not exact_tree(parent['optimizer'],source):raise RuntimeError('Stress mutated source AdamW')
    result=dict(schema_version='sumodiff.decoded.temporal.stress.v1',isolated_optimizer_updates=updates,
        production_training_updates=0,production_checkpoints_saved=False,warmup_bypassed=True,
        initial_state=initial,rows=rows,clipped_updates=sum(r['clipped'] for r in rows),
        gradient_norm_max=max(r['gradient_norm_before_clip'] for r in rows),
        validation_epsilon=probe(model,cache[1],schedule,device,c['batch_size'],c['validation_seed']),
        wall_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        interpretation='Peak/full-temporal private stress only; does not determine optimum lr or generation quality.')
    write_json(output,result);return result


def timestep_coverage(path,meta,max_auxiliary_timestep=300):
    """Observed parent training gates, not gradient shares or future-based selection."""
    by_id={r['window_id']:r for r in meta};counts={};steps=0;scene_visits=0
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if len(r['window_ids'])!=len(r['timesteps']) or not r['window_ids']:
            raise ValueError('Invalid parent timestep trace')
        steps+=1
        for wid,t in zip(r['window_ids'],r['timesteps']):
            if wid not in by_id or type(t) is not int or not 0<=t<1000:raise ValueError('Unknown training window or timestep')
            meta_row=by_id[wid];dense=meta_row['family']=='unsignalized_intersection' and meta_row['agents']>=8
            for group in ('all','dense_intersection' if dense else 'ordinary',meta_row['family']):
                x=counts.setdefault(group,dict(scene_visits=0,eligible_visits=0,t0=0,t1_50=0,t51_200=0,t201_300=0,t301_999=0))
                x['scene_visits']+=1;x['eligible_visits']+=t<=max_auxiliary_timestep
                key='t0' if t==0 else 't1_50' if t<=50 else 't51_200' if t<=200 else 't201_300' if t<=300 else 't301_999'
                x[key]+=1
            scene_visits+=1
    for value in counts.values():value['eligible_fraction']=value['eligible_visits']/value['scene_visits']
    return dict(source=str(Path(path).resolve()),steps=steps,scene_visits=scene_visits,groups=counts,
        interpretation='Parent natural uniform timestep exposure; scene counts do not measure gradient influence.',
        training_distribution_changed=False,validation_failure_used=False)
