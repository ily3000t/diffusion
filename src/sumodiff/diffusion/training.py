"""Bounded base training, fixed validation probes, and reproducible continuation."""
from copy import deepcopy
from pathlib import Path
import json
import math
import time
import numpy as np
import torch
from sumodiff.models import ConditionalDenoiser,resolve_model_config
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from sumodiff.data.dataset import write_json
from .config import resolve_diffusion_config, is_control, validate_architecture
from .representation import training_target
from .process import NoiseSchedule,normalize,masked_noise_mse
from .data import dataset_identity,labeled_cache,select_batch,verified_scale_audit
from .checkpoint import save_checkpoint,load_checkpoint,restore_training

TRAIN_DEFAULTS=dict(schema_version='sumodiff.base.train.config.v1',seed=20261005,noise_seed=20261006,
    validation_seed=20261007,device='cuda:0',torch_threads=1,batch_size=4,max_steps=240,
    learning_rate=0.0003,weight_decay=0.0001,gradient_clip_norm=1.0,
    validation_interval=40,checkpoint_interval=120,train_limit=12,validation_limit=6,
    require_probe_improvement=True,minimum_probe_relative_improvement=0.05,
    formal_estimate_steps=10000,diffusion={})


def train_config(value):
    if not isinstance(value,dict) or set(value)-set(TRAIN_DEFAULTS): raise ValueError('Unknown training options')
    c=deepcopy(TRAIN_DEFAULTS);c.update(value)
    c['diffusion']=resolve_diffusion_config(c['diffusion']).to_dict()
    if is_control(resolve_diffusion_config(c['diffusion'])):
        raise ValueError('Control targets require the explicit control training schema')
    if c['diffusion']['additional_losses_enabled']:
        raise ValueError('Physical objectives require the controlled continuation entry; legacy training is epsilon-only')
    if c['schema_version']!=TRAIN_DEFAULTS['schema_version'] or c['device'] not in ('cpu','cuda:0'): raise ValueError('Unsupported train schema/device')
    for key in ('seed','noise_seed','validation_seed'):
        if type(c[key]) is not int or not 0<=c[key]<2**32: raise ValueError(f'Invalid {key}')
    for key in ('torch_threads','batch_size','max_steps','validation_interval','checkpoint_interval','formal_estimate_steps'):
        if type(c[key]) is not int or c[key]<1: raise ValueError(f'Invalid {key}')
    for key in ('train_limit','validation_limit'):
        if c[key] is not None and (type(c[key]) is not int or c[key]<1): raise ValueError(f'Invalid {key}')
    for key in ('learning_rate','gradient_clip_norm'):
        if type(c[key]) not in (float,int) or not math.isfinite(c[key]) or c[key]<=0: raise ValueError(f'Invalid {key}')
    if type(c['weight_decay']) not in (float,int) or not math.isfinite(c['weight_decay']) or c['weight_decay']<0: raise ValueError('Invalid weight decay')
    if type(c['require_probe_improvement']) is not bool or type(c['minimum_probe_relative_improvement']) not in (float,int) or not 0<=c['minimum_probe_relative_improvement']<1: raise ValueError('Invalid probe criterion')
    return c


def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


def draw_noise(targets,conditioning,schedule,generator):
    clean=training_target(targets,conditioning,schedule.config)
    noise=torch.randn(clean.shape,generator=generator).to(clean.device)
    timestep=torch.randint(schedule.config.training_steps,(len(clean),),generator=generator).to(clean.device)
    noisy=schedule.add_noise(clean,noise,timestep)
    # Only padded AGENTS are zeroed; future_mask is never a denoiser input.
    noisy=torch.where(conditioning['agent_mask'][:,:,None,None],noisy,0.)
    return noisy,noise,timestep


@torch.no_grad()
def probe(model,cache,schedule,device,batch_size,seed):
    was_training=model.training;model.eval();generator=torch.Generator().manual_seed(seed)
    total=0.;count=0
    try:
        for start in range(0,len(cache[2]),batch_size):
            indices=torch.arange(start,min(start+batch_size,len(cache[2])))
            c,t,_=select_batch(cache,indices,device)
            noisy,noise,timestep=draw_noise(t,c,schedule,generator)
            loss=masked_noise_mse(model(noisy,timestep,c),noise,c['agent_mask'],t['future_mask'])
            total+=float(loss)*len(indices);count+=len(indices)
    finally: model.train(was_training)
    return total/count


def training_signature(model_config,c,data,scale,train_ids,val_ids):
    mutable={'max_steps','validation_interval','checkpoint_interval','require_probe_improvement','minimum_probe_relative_improvement','formal_estimate_steps','epochs','validation_every_epochs','checkpoint_every_epochs','monitor_every_epochs','train_probe_limit','monitor'}
    if c['schema_version'] in ('sumodiff.base.epoch.config.v2','sumodiff.control.train.config.v1'):mutable=set()
    result=dict(model=model_config.to_dict(),training={k:v for k,v in c.items() if k not in mutable},
        dataset_hashes={k:v['sha256'] for k,v in data.items()},scale_audit_sha256=scale['sha256'],
        train_window_ids=train_ids,validation_window_ids=val_ids)
    if c['schema_version']=='sumodiff.control.train.config.v1':result['task_plan_sha256']=file_identity(c['task_plan'])['sha256']
    return result


def train(dataset_path,model_config_path,config_path,scale_audit_path,output,repository,command,formal=False,resume=None,allow_long_run=False,prefix_updates=None):
    source_config=load_config(config_path)
    epoch_mode=source_config.get('schema_version') in ('sumodiff.base.epoch.config.v1','sumodiff.base.epoch.config.v2')
    fresh=source_config.get('schema_version')=='sumodiff.base.epoch.config.v2'
    control=source_config.get('schema_version')=='sumodiff.control.train.config.v1'
    objective_mode=fresh or control
    if epoch_mode:
        from .epoch_training import epoch_config,verified_curated_dataset,epoch_batch,probe_subset,periodic_sample
        c=epoch_config(source_config);selection=verified_curated_dataset(dataset_path,c)
        c['formal_estimate_steps']=c['max_steps']
    elif control:
        from .control_training import control_config
        c=control_config(source_config);selection=None
    else: c=train_config(source_config);selection=None
    architecture=resolve_model_config(load_config(model_config_path));diffusion=resolve_diffusion_config(c['diffusion'])
    validate_architecture(architecture,diffusion)
    if c['max_steps']>1000 and not allow_long_run: raise ValueError('More than 1000 total steps requires explicit --allow-long-run')
    data=dataset_identity(dataset_path)
    if control:
        from .control_training import verified_control_scales
        scale=verified_control_scales(scale_audit_path,dataset_path,diffusion)
    else:scale=verified_scale_audit(scale_audit_path,dataset_path,diffusion)
    if fresh:
        from .fresh_protocol import verify_calibration
        c['calibration_identity']=verify_calibration(c,architecture,data)
    configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    cache_started=time.perf_counter()
    if control:
        from .control_training import control_cache
        train_cache=control_cache(dataset_path,'train',c);val_cache=control_cache(dataset_path,'validation',c)
    else:
        train_cache=labeled_cache(dataset_path,'train',c['train_limit']);val_cache=labeled_cache(dataset_path,'validation',c['validation_limit'])
    cache_seconds=time.perf_counter()-cache_started
    train_probe_cache=probe_subset(train_cache,c['train_probe_limit']) if epoch_mode else train_cache
    if epoch_mode and (len(train_cache[2])!=c['expected_train_windows'] or len(val_cache[2])!=c['expected_validation_windows']):
        raise ValueError('Actual epoch cache differs from the certified population')
    signature=training_signature(architecture,c,data,scale,train_cache[2],val_cache[2])
    parent=load_checkpoint(resume,signature=signature) if resume else None
    start_step=parent['step'] if parent else 0
    if start_step>=c['max_steps']: raise ValueError('Resume checkpoint already reaches the requested total steps')
    if prefix_updates is not None and (not objective_mode or type(prefix_updates) is not int or not 1<=prefix_updates<=64):
        raise ValueError('Prefix diagnostic supports at most 64 optimizer updates')
    stop_step=min(c['max_steps'],start_step+prefix_updates) if prefix_updates else c['max_steps']
    effective=dict(prefix_updates=prefix_updates,schema_version='sumodiff.base.train.run.v1',model=architecture.to_dict(),training=c,data=data,scale_audit=scale,
        task_selection=dict(train_window_ids=train_cache[2],validation_window_ids=val_cache[2],core_only=True,selection='fixed train/validation diagnostic task plan; no test' if control else ('certified quality cohort; complete H+F; epochs traverse each train window exactly once' if epoch_mode else 'input-family round robin; complete H+F for training/probe only')),
        resume=file_identity(resume) if resume else None,allow_long_run=allow_long_run,
        precision='float32 network/optimizer; no AMP or TF32; strict determinism',sampling_not_performed=not epoch_mode or c['monitor_every_epochs']==0,quality_selection=selection)
    files=[model_config_path,config_path,scale_audit_path,*[v['location'] for v in data.values()],*train_cache[3],*val_cache[3]]
    if control:
        files.append(c['task_plan']);files.extend(v['location'] for v in scale['source_files'])
    if selection:files.append(selection['location'])
    if fresh:files.append(c['calibration_identity']['location'])
    with RunRecorder(output,repository,effective,command,{'model':c['seed'],'training_noise':c['noise_seed'],'validation_noise':c['validation_seed']},
        purpose='stage5a_epoch_base_training' if epoch_mode else ('stage5_short_base_training' if c['max_steps']<=1000 else 'base_training'),data_files=sorted(set(map(str,files))),
        data_id=data['manifest']['id'],checkpoint=resume,formal=formal) as run:
        model=ConditionalDenoiser(architecture).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        generator=torch.Generator().manual_seed(c['noise_seed']);schedule=NoiseSchedule(diffusion,device)
        if parent: restore_training(parent,model,optimizer,generator,device)
        metadata=dict(model_config=architecture.to_dict(),diffusion_config=diffusion.to_dict(),training_config=c,
            training_signature=signature,data=data,scale_audit=scale,run_sha=run.manifest['git']['commit_sha'],run_branch=run.manifest['git']['branch'],
            run_dirty=run.manifest['git']['dirty'],parent_checkpoint=file_identity(resume) if resume else None,
            runtime=dict(torch=str(torch.__version__),device=str(device),strict_determinism=True,network_dtype='float32'))
        update_schedule=None
        if fresh:
            from .schedule import UpdateSchedule,ScheduleConfig
            from .raw_velocity_losses import resolve_raw_velocity_loss
            from .jerk_tail import JerkTailConfig,fresh_objective
            from .unified_losses import RoadGeometryCache
            raw_loss=resolve_raw_velocity_loss(c['raw_loss']);tail=JerkTailConfig(**c['jerk_tail'])
            roads=RoadGeometryCache(dataset_path,train_cache[2],raw_loss)
            update_schedule=UpdateSchedule(ScheduleConfig(**c['schedule']))
            if parent:update_schedule.load_state_dict(parent['metadata']['runtime']['update_schedule'],optimizer,start_step)
            metadata['runtime']['update_schedule']=update_schedule.state_dict()
        if control:
            from .schedule import UpdateSchedule,ScheduleConfig
            from .control_training import control_objective
            from .unified_losses import RoadGeometryCache,resolve_unified_loss
            roads=RoadGeometryCache(dataset_path,train_cache[2],resolve_unified_loss(c['unified_loss']))
            update_schedule=UpdateSchedule(ScheduleConfig(**c['schedule']))
            if parent:update_schedule.load_state_dict(parent['metadata']['runtime']['update_schedule'],optimizer,start_step)
            metadata['runtime']['update_schedule']=update_schedule.state_dict()
        initial_hash=parameter_hash(model)
        validation=[dict(step=start_step,train_probe_mse=probe(model,train_probe_cache,schedule,device,c['batch_size'],c['validation_seed']),
            validation_probe_mse=probe(model,val_cache,schedule,device,c['batch_size'],c['validation_seed']))]
        steps=[];checkpoints=[];times=[];monitoring=[];actual_batch_sizes=[];peak_allocated=0;peak_reserved=0
        started=time.perf_counter()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        with (run.output/'steps.jsonl').open('w',encoding='utf-8') as log:
            for step in range(start_step+1,stop_step+1):
                synchronize(device);tick=time.perf_counter()
                if epoch_mode: indices,epoch,epoch_batch_index=epoch_batch(step,len(train_cache[2]),c['batch_size'],c['seed'])
                else: indices=torch.randint(len(train_cache[2]),(c['batch_size'],),generator=generator)
                actual_batch_sizes.append(len(indices))
                conditions,targets,ids=select_batch(train_cache,indices,device)
                model.train();optimizer.zero_grad(set_to_none=True)
                noisy,noise,timestep=draw_noise(targets,conditions,schedule,generator)
                epsilon=model(noisy,timestep,conditions)
                components=None
                if fresh:
                    lr,ramp=update_schedule.apply(optimizer,step)
                    components=fresh_objective(epsilon,noisy,noise,timestep,conditions,targets,schedule,
                        raw_loss,tail,c['tail_weight'],roads=roads.batch(ids),ramp=ramp)
                    loss=components['total_loss']
                elif control:
                    lr,ramp=update_schedule.apply(optimizer,step)
                    components=control_objective(epsilon,noisy,noise,timestep,conditions,targets,schedule,c,
                                                 roads=roads.batch(ids),ramp=ramp)
                    loss=components['total_loss']
                else:loss=masked_noise_mse(epsilon,noise,conditions['agent_mask'],targets['future_mask'])
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                optimizer.step();synchronize(device);elapsed=time.perf_counter()-tick;times.append(elapsed)
                if objective_mode:
                    update_schedule.complete(step);metadata['runtime']['update_schedule']=update_schedule.state_dict()
                row=dict(step=step,noise_mse=float(components['noise_mse']) if objective_mode else float(loss),gradient_norm_before_clip=float(norm),window_ids=ids,timesteps=timestep.cpu().tolist(),seconds=elapsed)
                if objective_mode:
                    row.update(learning_rate=lr,auxiliary_ramp=ramp,total_loss=float(loss),
                        clipping_applied=float(norm)>c['gradient_clip_norm'],
                        loss_components={k:float(v.detach()) for k,v in components.items() if isinstance(v,torch.Tensor) and v.ndim==0})
                if epoch_mode: row.update(epoch=epoch,epoch_batch=epoch_batch_index,epoch_complete=epoch_batch_index==c['steps_per_epoch'])
                log.write(json.dumps(row,allow_nan=False)+chr(10));log.flush()
                steps.append(row)
                if device.type=='cuda':
                    peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated(device));peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved(device))
                if step%c['validation_interval']==0 or step==stop_step:
                    validation.append(dict(step=step,train_probe_mse=probe(model,train_probe_cache,schedule,device,c['batch_size'],c['validation_seed']),
                        validation_probe_mse=probe(model,val_cache,schedule,device,c['batch_size'],c['validation_seed'])))
                    write_json(run.output/'validation.json',validation)
                    print(f"step={step} train_probe={validation[-1]['train_probe_mse']:.6f} validation_probe={validation[-1]['validation_probe_mse']:.6f}",flush=True)
                    write_json(run.output/'progress.json',dict(step=step,total_steps=c['max_steps'],
                        epoch=epoch if epoch_mode else None,total_epochs=c.get('epochs'),validation=validation[-1]))
                monitor_due=bool(epoch_mode and c['monitor_every_epochs'] and epoch_batch_index==c['steps_per_epoch'] and
                    (epoch%c['monitor_every_epochs']==0 or step==c['max_steps']))
                if step%c['checkpoint_interval']==0 or step==stop_step or monitor_due:
                    path=run.output/f'checkpoint_step_{step:06d}.pt'
                    save_checkpoint(path,model,optimizer,generator,step,metadata);checkpoints.append(file_identity(path))
                    write_json(run.output/'latest_checkpoint.json',dict(step=step,checkpoint=checkpoints[-1]))
                if monitor_due:
                    monitoring.append(periodic_sample(dataset_path,path,run,repository,c,epoch,formal))
                    write_json(run.output/'generation_monitor.json',monitoring)
        last_hash=parameter_hash(model)
        improvement=1-validation[-1]['train_probe_mse']/validation[0]['train_probe_mse']
        timed=times[1:] or times;median=float(np.median(timed));mean=float(np.mean(timed))
        result=dict(schema_version='sumodiff.base.train.metrics.v1',run_sha=metadata['run_sha'],start_step=start_step,end_step=stop_step,nominal_total_updates=c['max_steps'],complete_budget=stop_step==c['max_steps'],updates=len(times),
            train_windows=len(train_cache[2]),validation_windows=len(val_cache[2]),batch_size=c['batch_size'],validation=validation,
            train_probe_relative_improvement=improvement,weight_update_verified=last_hash!=initial_hash,initial_parameter_sha256=initial_hash,final_parameter_sha256=last_hash,
            cache_preparation_seconds=cache_seconds,training_validation_checkpoint_wall_seconds=time.perf_counter()-started,
            optimizer_step_timing=dict(warmup_updates_excluded=min(1,len(times)),mean_seconds=mean,median_seconds=median,windows_per_second=float(np.mean(actual_batch_sizes[1:] or actual_batch_sizes))/mean,
                includes='CPU batch indexing, device transfer, forward/backward, gradient clipping and AdamW; excludes cache construction/validation/checkpoint IO'),
            peak_allocated_bytes=peak_allocated if device.type=='cuda' else None,peak_reserved_bytes=peak_reserved if device.type=='cuda' else None,
            optimizer_state_tensor_bytes=sum(v.numel()*v.element_size() for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)),
            estimate=dict(hypothetical_updates=c['formal_estimate_steps'],optimizer_only_seconds=mean*c['formal_estimate_steps'],
                basis='same cached batch/map/agent workload and precision; excludes data preparation/IO, validation and checkpoints; not convergence prediction'),
            checkpoints=checkpoints,limitations=['small-data learning probe, not adequate training or generated-quality evidence','validation uses fixed noise/timesteps','normal-data quality defects remain'])
        if epoch_mode:
            result.update(epoch_schedule=dict(total_epochs=c['epochs'],steps_per_epoch=c['steps_per_epoch'],
                start_epoch=(start_step//c['steps_per_epoch'])+1,each_epoch_without_replacement=True,
                last_partial_batch_retained=True,permutation_seed=c['seed']),
                train_probe_windows=len(train_probe_cache[2]),generation_monitor=monitoring,
                quality_selection=selection,cached_tensor_bytes=sum(v.numel()*v.element_size() for cache in (train_cache,val_cache)
                    for part in cache[:2] for v in part.values()),
                convergence_status='not_assessed',limitations=[f"{c['epochs']} epochs is an initial budget, not proof of convergence",
                    'curated SUMO labels do not establish real driving realism','held-out geometry/episode validation; overlapping within-episode windows remain',
                    'quality thresholds are declared engineering candidates'])
        if control:
            result.update(representation='control_a_omega_r',convergence_status='not_assessed',
                engineering_budget_only=True,
                cached_tensor_bytes=sum(v.numel()*v.element_size() for cache in (train_cache,val_cache) for part in cache[:2] for v in part.values()))
        run.write_metrics(result);write_json(run.output/'training_summary.json',result)
        if not result['weight_update_verified']: raise RuntimeError('Training did not change weights')
        if c['require_probe_improvement'] and improvement<c['minimum_probe_relative_improvement']: raise RuntimeError('Short learning probe did not meet its predeclared training-loss improvement')
    return result
