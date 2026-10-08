"""Bounded real-training-batch parameter-gradient audit, without optimizer updates."""
import time
import numpy as np
import torch
from sumodiff.models import ConditionalDenoiser,resolve_model_config
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.data.dataset import write_json
from .config import resolve_diffusion_config
from .data import select_batch
from .epoch_training import epoch_batch
from .training import draw_noise,synchronize
from .process import NoiseSchedule
from .unified_losses import resolve_unified_loss,unified_objective,RoadGeometryCache
from .unified_training import warm_start,new_optimizer

FIELDS=('noise_mse','weighted_imitation','weighted_temporal','weighted_road','auxiliary_total','total_loss')


def gradient_statistics(loss,parameters,reference=None):
    gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True) if loss.requires_grad else (None,)*len(parameters)
    square=sum(float(g.detach().double().square().sum()) for g in gradients if g is not None)
    norm=float(np.sqrt(square))
    dot=None;cosine=None
    if reference is not None:
        dot=sum(float((x.detach().double()*y.detach().double()).sum())
                for x,y in zip(gradients,reference) if x is not None and y is not None)
        other=float(np.sqrt(sum(float(g.detach().double().square().sum()) for g in reference if g is not None)))
        if norm and other:cosine=dot/norm/other
    return dict(parameter_gradient_norm=norm,cosine_with_noise=cosine),gradients


def audit_gradients(dataset,parent,c,cache,output,updates=16):
    if type(updates) is not int or not 1<=updates<=24:
        raise ValueError('Gradient audit is limited to 1..24 batches')
    configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
    model=ConditionalDenoiser(resolve_model_config(parent['metadata']['model_config'])).to(device)
    optimizer=new_optimizer(model,c);generator=torch.Generator()
    warm_start(parent,model,optimizer,generator,device)
    before=parameter_hash(model)
    config=resolve_unified_loss(c['loss'])
    geometry=RoadGeometryCache(dataset,cache[0][2],config) if config.road_weight else None
    schedule=NoiseSchedule(resolve_diffusion_config(c['diffusion']),device)
    parameters=list(model.parameters());rows=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for additional in range(1,updates+1):
        synchronize(device);tick=time.perf_counter()
        indices,_,_=epoch_batch(parent['step']+additional,len(cache[0][2]),c['batch_size'],c['seed'])
        conditions,targets,ids=select_batch(cache[0],indices,device)
        noisy,noise,timestep=draw_noise(targets,conditions,schedule,generator)
        model.train()
        terms=unified_objective(model(noisy,timestep,conditions),noisy,noise,timestep,conditions,targets,schedule,
                                config,roads=geometry.batch(ids) if geometry else None,ramp=1.)
        stats={};reference=None
        for field in FIELDS:
            stat,gradient=gradient_statistics(terms[field],parameters,reference)
            stats[field]=dict(loss=float(terms[field]),**stat)
            if field=='noise_mse':reference=gradient
        base=stats['noise_mse']['parameter_gradient_norm']
        for value in stats.values():
            value['norm_ratio_to_noise']=value['parameter_gradient_norm']/base if base else None
        synchronize(device)
        rows.append(dict(additional_step=additional,window_ids=ids,timesteps=timestep.cpu().tolist(),
            auxiliary_eligible_scenes=terms['auxiliary_eligible_scenes'],components=stats,
            seconds=time.perf_counter()-tick))
    eligible=[r for r in rows if r['auxiliary_eligible_scenes']]
    ratios=[r['components']['auxiliary_total']['norm_ratio_to_noise'] for r in eligible]
    result=dict(schema_version='sumodiff.unified.gradient.audit.v1',updates=updates,optimizer_updates=0,
        model_unchanged=parameter_hash(model)==before,optimizer_state_count=len(optimizer.state),
        auxiliary_ramp_for_audit=1.,interpretation='full-strength candidate weights on paired reconstruction; actual training ramps over 5 epochs',
        objective=config.to_dict(),eligible_batches=len(eligible),
        eligible_auxiliary_to_noise_norm_ratio=dict(median=float(np.median(ratios)) if ratios else None,
            maximum=max(ratios) if ratios else None),
        component_norm_ratio_median={field:float(np.median([r['components'][field]['norm_ratio_to_noise'] for r in eligible]))
                                    if eligible else None for field in FIELDS},
        rows=rows,wall_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
        quality_or_convergence_claim=False)
    write_json(output,result)
    if not result['model_unchanged'] or result['optimizer_state_count']:
        raise RuntimeError('Gradient audit accidentally changed model/optimizer')
    return result
