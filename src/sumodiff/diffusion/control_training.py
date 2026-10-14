"""Bounded control-target recipe, certified scales, and rollout-state supervision."""
from copy import deepcopy
import json
import math
from pathlib import Path
import torch
from sumodiff.data.dataset import WindowDataset, collate_numpy
from sumodiff.models import prepare_conditioning
from sumodiff.experiments.recorder import file_identity
from sumodiff.decoding.control_replay import validate_plan, verify_task_metadata
from .config import ControlDiffusionConfig, resolve_diffusion_config, is_control
from .data import dataset_identity, input_files
from .schedule import ScheduleConfig, UpdateSchedule, resolve_schedule
from .unified_losses import resolve_unified_loss, unified_objective
from .jerk_tail import JerkTailConfig, jerk_tail_scenes
from .process import denormalize
from .representation import decode_prediction

SCHEMA='sumodiff.control.train.config.v1'
EXTRA=dict(task_plan='configs/experiments/action_state_replay_tasks.json',unified_loss={},
           jerk_tail={},tail_weight=.0003,schedule={})


def control_config(value):
    from .training import TRAIN_DEFAULTS, train_config
    if not isinstance(value,dict) or value.get('schema_version')!=SCHEMA or set(value)-(set(TRAIN_DEFAULTS)|set(EXTRA)):
        raise ValueError('Unknown control training options')
    base={k:v for k,v in value.items() if k in TRAIN_DEFAULTS}
    diffusion=resolve_diffusion_config(base.pop('diffusion',ControlDiffusionConfig(additional_losses_enabled=True).to_dict()))
    if not is_control(diffusion) or not diffusion.additional_losses_enabled:
        raise ValueError('Control training requires its explicit representation and auxiliary capability')
    base.update(schema_version=TRAIN_DEFAULTS['schema_version'],diffusion={})
    base.setdefault('max_steps',16);base.setdefault('train_limit',28);base.setdefault('validation_limit',28)
    base.setdefault('require_probe_improvement',False)
    c=train_config(base);c.update(schema_version=SCHEMA,diffusion=diffusion.to_dict())
    extra=deepcopy(EXTRA);extra.update({k:v for k,v in value.items() if k in EXTRA})
    if not 3<=c['max_steps']<=64:
        raise ValueError('Control integration is limited to 3..64 updates; formal long-training protocol is not implemented')
    if any(type(c[k]) is not int or not 1<=c[k]<=60 for k in ('train_limit','validation_limit')):
        raise ValueError('Control integration requires explicit bounded training/validation cohorts')
    if not isinstance(extra['task_plan'],str) or not extra['task_plan'].strip():
        raise ValueError('Explicit diagnostic task plan is required')
    extra['unified_loss']=resolve_unified_loss(extra['unified_loss']).to_dict()
    extra['jerk_tail']=JerkTailConfig(**extra['jerk_tail']).to_dict()
    if type(extra['tail_weight']) not in (int,float) or not math.isfinite(extra['tail_weight']) or extra['tail_weight']<0:
        raise ValueError('Invalid control tail weight')
    schedule=dict(peak_lr=1e-4,minimum_lr=1e-6,warmup_updates=2,
                  total_updates=c['max_steps'],auxiliary_warmup_updates=min(4,c['max_steps']))
    schedule.update(extra['schedule'])
    extra['schedule']=resolve_schedule(schedule).to_dict()
    if extra['schedule']['total_updates']!=c['max_steps']:
        raise ValueError('Control schedule must match the immutable declared budget')
    c.update(extra);c['learning_rate']=extra['schedule']['minimum_lr']
    return c


def verified_control_scales(path,directory,config):
    from .config import resolve_diffusion_config
    path=Path(path).resolve(strict=True)
    candidate=json.loads(path.read_text(encoding='utf-8'))
    source=path.parent
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    results=json.loads((source/'results.json').read_text(encoding='utf-8'))
    status=json.loads((source/'status.json').read_text(encoding='utf-8'))
    data=dataset_identity(directory)
    if (path.name!='candidate_normalization.json' or not candidate.get('certified_training_statistics')
        or not candidate.get('available') or not candidate.get('source_run_formal') or candidate.get('mean_centering') is not False
        or candidate.get('clipping') is not False or results.get('candidate_normalization')!=candidate
        or results.get('failed_windows')!=0 or results.get('validation_arrays_loaded')!=0 or results.get('test_arrays_loaded')!=0
        or not manifest['formal'] or manifest['git']['dirty'] or status['state']!='completed'
        or candidate['source_run_sha']!=manifest['git']['commit_sha'] or candidate['data_id']!=manifest['data']['id']):
        raise ValueError('Control scales require intact completed training-only certified statistics')
    if candidate['dataset_manifest_sha256']!=data['manifest']['sha256'] or candidate['window_index_sha256']!=data['index']['sha256']:
        raise ValueError('Control scale dataset mismatch')
    from .representation import CONTROL
    from dataclasses import asdict
    if candidate['control_definition']!=asdict(CONTROL):
        raise ValueError('Control scale inversion definition mismatch')
    expected=[config.acceleration_scale_mps2,config.yaw_rate_scale_radps,config.lateral_velocity_scale_mps]
    actual=[candidate['scales'][k]['scale'] for k in ('acceleration','yaw_rate','lateral_velocity')]
    if expected!=actual:
        raise ValueError('Control normalization differs from certified scales')
    identity=file_identity(path)
    identity['source_files']=[file_identity(source/name) for name in ('manifest.json','results.json','status.json')]
    identity['certified_data_id']=candidate['data_id']
    return identity


def control_cache(directory,split,c,*,model_config=None):
    if split not in ('train','validation'):
        raise ValueError('Control diagnostics cannot access test labels')
    tasks=validate_plan(json.loads(Path(c['task_plan']).read_text(encoding='utf-8')))
    selected=[t for t in tasks if t['split']==split]
    limit=c['train_limit' if split=='train' else 'validation_limit']
    if len(selected)!=limit:
        raise ValueError('Diagnostic plan population differs from the declared bound')
    ds=WindowDataset(directory,split,core_only=True)
    by_id={e['window_id']:i for i,e in enumerate(ds.entries)}
    samples=[];indices=[]
    for task in selected:
        if task['window_id'] not in by_id:raise ValueError('Diagnostic task is outside its split/core cohort')
        i=by_id[task['window_id']];meta=ds._json(ds.entries[i],'input.json')
        verify_task_metadata(task,ds.entries[i],meta)
        samples.append(ds[i]);indices.append(i)
    batch=collate_numpy(samples)
    conditions=prepare_conditioning(batch,'cpu',config=model_config)
    targets={k:torch.as_tensor(v) for k,v in batch['targets'].items()}
    if not targets['future_mask'][conditions['agent_mask']].all():
        raise ValueError('Control targets require complete active labels')
    return conditions,targets,[t['window_id'] for t in selected],input_files(ds,indices,True)


def control_objective(epsilon,noisy,noise,timestep,conditioning,targets,schedule,c,*,roads=None,ramp=1.):
    loss_config=resolve_unified_loss(c['unified_loss'])
    out=unified_objective(epsilon,noisy,noise,timestep,conditioning,targets,schedule,
                          loss_config,roads=roads,ramp=ramp)
    zero=out['noise_mse'].new_zeros(())
    out.update(tail=zero,tail_mean=zero,tail_top=zero,weighted_tail=zero,tail_top_counts=[])
    eligible=torch.nonzero(timestep<=loss_config.max_auxiliary_timestep).flatten()
    if c['tail_weight'] and len(eligible) and ramp:
        subset={k:v[eligible] for k,v in conditioning.items()}
        clean=schedule.clean_from_noise(noisy[eligible],epsilon[eligible],timestep[eligible])
        mask=subset['agent_mask'][...,None,None]
        physical=denormalize(torch.where(mask,clean,0.),schedule.config)
        decoded=decode_prediction(physical,subset,schedule.config)
        loss,mean,top,counts=jerk_tail_scenes(decoded.states,subset,targets['future_mask'][eligible],JerkTailConfig(**c['jerk_tail']))
        alpha=schedule.alpha_bar[timestep[eligible]].to(loss.dtype).detach()
        for key,term in [('tail',loss),('tail_mean',mean),('tail_top',top)]:
            out[key]=(term*alpha).sum()/len(epsilon)
        out['tail_top_counts']=counts
        out['weighted_tail']=ramp*c['tail_weight']*out['tail']
    out['auxiliary_total']=out['auxiliary_total']+out['weighted_tail']
    out['total_loss']=out['noise_mse']+out['auxiliary_total']
    return out


def validate_runtime(payload):
    m=payload['metadata'];c=m['training_config']
    if c!=control_config(c) or m['training_signature']['training']!=c or c['diffusion']!=m['diffusion_config']:
        raise ValueError('Control checkpoint recipe/signature mismatch')
    scheduler=UpdateSchedule(ScheduleConfig(**c['schedule']))
    scheduler.load_state_dict(m['runtime']['update_schedule'],expected_updates=payload['step'])
    if any(g['lr']!=scheduler.last_lr for g in payload['optimizer']['param_groups']):
        raise ValueError('Control optimizer/schedule mismatch')
    if not payload['optimizer']['state'] or any(int(s['step'])!=payload['step'] for s in payload['optimizer']['state'].values()):
        raise ValueError('Control optimizer update counts mismatch')
