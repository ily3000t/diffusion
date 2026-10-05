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
from .config import resolve_diffusion_config
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
    clean=normalize(targets['future'],schedule.config)
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
    mutable={'max_steps','validation_interval','checkpoint_interval','require_probe_improvement','minimum_probe_relative_improvement','formal_estimate_steps'}
    return dict(model=model_config.to_dict(),training={k:v for k,v in c.items() if k not in mutable},
        dataset_hashes={k:v['sha256'] for k,v in data.items()},scale_audit_sha256=scale['sha256'],
        train_window_ids=train_ids,validation_window_ids=val_ids)


def train(dataset_path,model_config_path,config_path,scale_audit_path,output,repository,command,formal=False,resume=None,allow_long_run=False):
    c=train_config(load_config(config_path));architecture=resolve_model_config(load_config(model_config_path));diffusion=resolve_diffusion_config(c['diffusion'])
    if architecture.diffusion_embedding_steps!=diffusion.training_steps: raise ValueError('Model embedding/schedule mismatch')
    if c['max_steps']>1000 and not allow_long_run: raise ValueError('More than 1000 total steps requires explicit --allow-long-run')
    data=dataset_identity(dataset_path);scale=verified_scale_audit(scale_audit_path,dataset_path,diffusion)
    configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    cache_started=time.perf_counter()
    train_cache=labeled_cache(dataset_path,'train',c['train_limit']);val_cache=labeled_cache(dataset_path,'validation',c['validation_limit'])
    cache_seconds=time.perf_counter()-cache_started
    signature=training_signature(architecture,c,data,scale,train_cache[2],val_cache[2])
    parent=load_checkpoint(resume,signature=signature) if resume else None
    start_step=parent['step'] if parent else 0
    if start_step>=c['max_steps']: raise ValueError('Resume checkpoint already reaches the requested total steps')
    effective=dict(schema_version='sumodiff.base.train.run.v1',model=architecture.to_dict(),training=c,data=data,scale_audit=scale,
        task_selection=dict(train_window_ids=train_cache[2],validation_window_ids=val_cache[2],core_only=True,selection='input-family round robin; complete H+F for training/probe only'),
        resume=file_identity(resume) if resume else None,allow_long_run=allow_long_run,
        precision='float32 network/optimizer; no AMP or TF32; strict determinism',sampling_not_performed=True)
    files=[model_config_path,config_path,scale_audit_path,*[v['location'] for v in data.values()],*train_cache[3],*val_cache[3]]
    with RunRecorder(output,repository,effective,command,{'model':c['seed'],'training_noise':c['noise_seed'],'validation_noise':c['validation_seed']},
        purpose='stage5_short_base_training' if c['max_steps']<=1000 else 'base_training',data_files=sorted(set(map(str,files))),
        data_id=data['manifest']['id'],checkpoint=resume,formal=formal) as run:
        model=ConditionalDenoiser(architecture).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        generator=torch.Generator().manual_seed(c['noise_seed']);schedule=NoiseSchedule(diffusion,device)
        if parent: restore_training(parent,model,optimizer,generator,device)
        metadata=dict(model_config=architecture.to_dict(),diffusion_config=diffusion.to_dict(),training_config=c,
            training_signature=signature,data=data,scale_audit=scale,run_sha=run.manifest['git']['commit_sha'],run_branch=run.manifest['git']['branch'],
            run_dirty=run.manifest['git']['dirty'],parent_checkpoint=file_identity(resume) if resume else None,
            runtime=dict(torch=str(torch.__version__),device=str(device),strict_determinism=True,network_dtype='float32'))
        initial_hash=parameter_hash(model)
        validation=[dict(step=start_step,train_probe_mse=probe(model,train_cache,schedule,device,c['batch_size'],c['validation_seed']),
            validation_probe_mse=probe(model,val_cache,schedule,device,c['batch_size'],c['validation_seed']))]
        steps=[];checkpoints=[];times=[];peak_allocated=0;peak_reserved=0
        started=time.perf_counter()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        with (run.output/'steps.jsonl').open('w',encoding='utf-8') as log:
            for step in range(start_step+1,c['max_steps']+1):
                synchronize(device);tick=time.perf_counter()
                indices=torch.randint(len(train_cache[2]),(c['batch_size'],),generator=generator)
                conditions,targets,ids=select_batch(train_cache,indices,device)
                model.train();optimizer.zero_grad(set_to_none=True)
                noisy,noise,timestep=draw_noise(targets,conditions,schedule,generator)
                loss=masked_noise_mse(model(noisy,timestep,conditions),noise,conditions['agent_mask'],targets['future_mask'])
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                optimizer.step();synchronize(device);elapsed=time.perf_counter()-tick;times.append(elapsed)
                row=dict(step=step,noise_mse=float(loss),gradient_norm_before_clip=float(norm),window_ids=ids,timesteps=timestep.cpu().tolist(),seconds=elapsed)
                log.write(json.dumps(row,allow_nan=False)+chr(10));log.flush()
                steps.append(row)
                if device.type=='cuda':
                    peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated(device));peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved(device))
                if step%c['validation_interval']==0 or step==c['max_steps']:
                    validation.append(dict(step=step,train_probe_mse=probe(model,train_cache,schedule,device,c['batch_size'],c['validation_seed']),
                        validation_probe_mse=probe(model,val_cache,schedule,device,c['batch_size'],c['validation_seed'])))
                    write_json(run.output/'validation.json',validation)
                    print(f"step={step} train_probe={validation[-1]['train_probe_mse']:.6f} validation_probe={validation[-1]['validation_probe_mse']:.6f}",flush=True)
                if step%c['checkpoint_interval']==0 or step==c['max_steps']:
                    path=run.output/f'checkpoint_step_{step:06d}.pt'
                    save_checkpoint(path,model,optimizer,generator,step,metadata);checkpoints.append(file_identity(path))
        last_hash=parameter_hash(model)
        improvement=1-validation[-1]['train_probe_mse']/validation[0]['train_probe_mse']
        timed=times[1:] or times;median=float(np.median(timed));mean=float(np.mean(timed))
        result=dict(schema_version='sumodiff.base.train.metrics.v1',run_sha=metadata['run_sha'],start_step=start_step,end_step=c['max_steps'],updates=len(times),
            train_windows=len(train_cache[2]),validation_windows=len(val_cache[2]),batch_size=c['batch_size'],validation=validation,
            train_probe_relative_improvement=improvement,weight_update_verified=last_hash!=initial_hash,initial_parameter_sha256=initial_hash,final_parameter_sha256=last_hash,
            cache_preparation_seconds=cache_seconds,training_validation_checkpoint_wall_seconds=time.perf_counter()-started,
            optimizer_step_timing=dict(warmup_updates_excluded=min(1,len(times)),mean_seconds=mean,median_seconds=median,windows_per_second=c['batch_size']/mean,
                includes='CPU batch indexing, device transfer, forward/backward, gradient clipping and AdamW; excludes cache construction/validation/checkpoint IO'),
            peak_allocated_bytes=peak_allocated if device.type=='cuda' else None,peak_reserved_bytes=peak_reserved if device.type=='cuda' else None,
            optimizer_state_tensor_bytes=sum(v.numel()*v.element_size() for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)),
            estimate=dict(hypothetical_updates=c['formal_estimate_steps'],optimizer_only_seconds=mean*c['formal_estimate_steps'],
                basis='same cached batch/map/agent workload and precision; excludes data preparation/IO, validation and checkpoints; not convergence prediction'),
            checkpoints=checkpoints,limitations=['small-data learning probe, not adequate training or generated-quality evidence','validation uses fixed noise/timesteps','normal-data quality defects remain'])
        run.write_metrics(result);write_json(run.output/'training_summary.json',result)
        if not result['weight_update_verified']: raise RuntimeError('Training did not change weights')
        if c['require_probe_improvement'] and improvement<c['minimum_probe_relative_improvement']: raise RuntimeError('Short learning probe did not meet its predeclared training-loss improvement')
    return result
