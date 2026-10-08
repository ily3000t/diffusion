"""Explicit same-parent continuations for the four stage5B objective arms."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from sumodiff.models import ConditionalDenoiser, resolve_model_config
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.data.dataset import write_json
from sumodiff.experiments.recorder import RunRecorder, load_config, file_identity
from .checkpoint import load_checkpoint, restore_training, save_checkpoint
from .config import resolve_diffusion_config
from .data import dataset_identity, labeled_cache, select_batch, verified_scale_audit
from .epoch_training import verified_curated_dataset, epoch_batch, probe_subset, periodic_sample
from .training import draw_noise, probe, synchronize, training_signature
from .process import NoiseSchedule
from .physical_losses import resolve_physical_config, physical_objective

SCHEMA = 'sumodiff.controlled.train.config.v1'
DEFAULTS = dict(schema_version=SCHEMA, arm='A', additional_epochs=10,
    expected_parent_step=30000, expected_parent_sha256='',
    expected_train_windows=1200, expected_validation_windows=200,
    batch_size=4, learning_rate=.0003, device='cuda:0', torch_threads=1,
    validation_every_epochs=1, checkpoint_every_epochs=1, monitor_every_epochs=2,
    physical_loss={})
PATTERNS = {'A':(False,False,False), 'B':(True,False,False),
            'C':(True,True,False), 'D':(True,True,True)}


def tensor_hash(value):
    array=value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(str((str(array.dtype),array.shape)).encode()+array.tobytes()).hexdigest()


def tree_hash(value):
    digest=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            digest.update(b'tensor');digest.update(tensor_hash(x).encode())
        elif isinstance(x,dict):
            digest.update(b'dict')
            for key in sorted(x,key=str):
                digest.update(repr(key).encode());visit(x[key])
        elif isinstance(x,(list,tuple)):
            digest.update(type(x).__name__.encode())
            for item in x:visit(item)
        else:digest.update(repr(x).encode())
    visit(value)
    return digest.hexdigest()


def controlled_config(value, parent, *, prefix_updates=None):
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown controlled continuation options')
    requested=deepcopy(DEFAULTS);requested.update(value)
    if requested['schema_version']!=SCHEMA or requested['arm'] not in PATTERNS:
        raise ValueError('Unsupported continuation schema/arm')
    if requested['additional_epochs']!=10 or type(requested['additional_epochs']) is not int:
        raise ValueError('First ablation budget is exactly ten additional epochs')
    for key in ('expected_parent_step','expected_train_windows','expected_validation_windows',
                'batch_size','torch_threads','validation_every_epochs','checkpoint_every_epochs'):
        if type(requested[key]) is not int or requested[key]<1:
            raise ValueError(f'Invalid {key}')
    if type(requested['monitor_every_epochs']) is not int or requested['monitor_every_epochs']<0:
        raise ValueError('Invalid monitor schedule')
    if requested['device'] not in ('cpu','cuda:0'):
        raise ValueError('Unsupported continuation device')
    sha=requested['expected_parent_sha256']
    if not isinstance(sha,str) or len(sha)!=64 or any(x not in '0123456789abcdef' for x in sha):
        raise ValueError('A complete common parent SHA-256 is required')
    if type(requested['learning_rate']) not in (int,float) or not math.isfinite(requested['learning_rate']) or requested['learning_rate']<=0:
        raise ValueError('Invalid continuation learning rate')
    previous=parent['metadata']['training_config']
    if previous['schema_version']!='sumodiff.base.epoch.config.v1' or previous['diffusion']['additional_losses_enabled']:
        raise ValueError('Fork must start from the original epsilon-only epoch checkpoint')
    if parent['step']!=requested['expected_parent_step'] or previous['epochs']!=100:
        raise ValueError('Expected the completed 100-epoch common parent')
    if parent['step']!=previous['epochs']*previous['steps_per_epoch']:
        raise ValueError('Parent is not at its certified epoch boundary')
    for key in ('batch_size','learning_rate','device','torch_threads','expected_train_windows','expected_validation_windows'):
        if requested[key]!=previous[key]:
            raise ValueError(f'{key} must match the common parent; no optimizer/sampling confound')
    cfg=resolve_physical_config(requested['physical_loss'])
    weights=(cfg.consistency_weight,cfg.velocity_difference_weight,cfg.boundary_weight)
    if tuple(x>0 for x in weights)!=PATTERNS[requested['arm']]:
        raise ValueError('Objective weights do not match the declared A/B/C/D arm')
    for group in parent['optimizer']['param_groups']:
        if group['lr']!=requested['learning_rate'] or group['weight_decay']!=previous['weight_decay']:
            raise ValueError('Actual parent AdamW settings differ from the continuation contract')
    c=deepcopy(previous)
    c.update(schema_version=SCHEMA, arm=requested['arm'], source_step=parent['step'],
        source_epochs=previous['epochs'], fork_source_sha256=sha,
        additional_epochs=requested['additional_epochs'], physical_loss=cfg.to_dict(),
        epochs=previous['epochs']+requested['additional_epochs'],
        validation_every_epochs=requested['validation_every_epochs'],
        checkpoint_every_epochs=requested['checkpoint_every_epochs'],
        monitor_every_epochs=requested['monitor_every_epochs'],
        initialization='inherit_model_AdamW_noise_generator_and_torch_RNG')
    c['diffusion']['additional_losses_enabled']=cfg.enabled
    c['nominal_additional_updates']=requested['additional_epochs']*previous['steps_per_epoch']
    if prefix_updates is not None and (type(prefix_updates) is not int or not 1<=prefix_updates<=24):
        raise ValueError('Preflight prefix must contain 1 to 24 updates')
    c['execution_budget_kind']='preflight_prefix' if prefix_updates is not None else 'ten_epoch_ablation'
    c['planned_additional_updates']=prefix_updates if prefix_updates is not None else c['nominal_additional_updates']
    c['max_steps']=parent['step']+c['planned_additional_updates']
    c['validation_interval']=c['validation_every_epochs']*c['steps_per_epoch']
    c['checkpoint_interval']=c['checkpoint_every_epochs']*c['steps_per_epoch']
    c['formal_estimate_steps']=c['nominal_additional_updates']
    return c


def shared_cache(dataset,c):
    begun=time.perf_counter()
    selection=verified_curated_dataset(dataset,c)
    train=labeled_cache(dataset,'train',None)
    validation=labeled_cache(dataset,'validation',None)
    if len(train[2])!=c['expected_train_windows'] or len(validation[2])!=c['expected_validation_windows']:
        raise ValueError('Cached population differs from the certified cohort')
    return train,validation,selection,time.perf_counter()-begun


def inherited_state(model,optimizer,generator):
    return dict(model=parameter_hash(model),optimizer=tree_hash(optimizer.state_dict()),
        noise_generator=tensor_hash(generator.get_state()),cpu_rng=tensor_hash(torch.get_rng_state()),
        cuda_rng=[tensor_hash(x) for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def optimize_batch(model,optimizer,conditions,targets,schedule,generator,clip_norm,objective):
    model.train();optimizer.zero_grad(set_to_none=True)
    noisy,noise,timestep=draw_noise(targets,conditions,schedule,generator)
    epsilon=model(noisy,timestep,conditions)
    terms=physical_objective(epsilon,noisy,noise,timestep,conditions,targets,schedule,objective)
    loss=terms['total_loss'];loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm,error_if_nonfinite=True)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite controlled training loss')
    optimizer.step()
    return terms,norm,noise,timestep


def train_controlled(dataset,parent_checkpoint,config_path,scale_path,output,repository,command,
                     formal=False,resume=None,prefix_updates=None,allow_long_run=False,cache=None):
    source=file_identity(parent_checkpoint)
    parent=load_checkpoint(parent_checkpoint)
    c=controlled_config(load_config(config_path),parent,prefix_updates=prefix_updates)
    if source['sha256']!=c['fork_source_sha256']:
        raise ValueError('Common parent checksum mismatch')
    if c['planned_additional_updates']>1000 and not allow_long_run:
        raise ValueError('Ten-epoch continuation needs explicit --allow-long-run')
    data=dataset_identity(dataset)
    architecture=resolve_model_config(parent['metadata']['model_config'])
    # The auxiliary flag changes training only; physical scales and schedule
    # remain exactly the previously audited parent values.
    original_diffusion=resolve_diffusion_config(parent['metadata']['diffusion_config'])
    scale=verified_scale_audit(scale_path,dataset,original_diffusion)
    for key in data:
        if data[key]['sha256']!=parent['metadata']['data'][key]['sha256']:
            raise ValueError('Continuation dataset differs from common parent')
    if scale['sha256']!=parent['metadata']['scale_audit']['sha256']:
        raise ValueError('Continuation scale-audit identity changed')
    configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    train_cache,val_cache,selection,cache_seconds=cache if cache is not None else shared_cache(dataset,c)
    source_signature=parent['metadata']['training_signature']
    if train_cache[2]!=source_signature['train_window_ids'] or val_cache[2]!=source_signature['validation_window_ids']:
        raise ValueError('Continuation cache/task order differs from the common parent')
    signature=training_signature(architecture,c,data,scale,train_cache[2],val_cache[2])
    previous=load_checkpoint(resume,signature=signature) if resume else parent
    if previous['step']<c['source_step'] or previous['step']>=c['max_steps']:
        raise ValueError('Resume step is outside the declared continuation budget')
    start=previous['step']
    files=[config_path,scale_path,parent_checkpoint,*[x['location'] for x in data.values()],*train_cache[3],*val_cache[3]]
    if resume:files.append(resume)
    effective=dict(schema_version='sumodiff.controlled.train.run.v1',training=c,model=architecture.to_dict(),
        data=data,scale_audit=scale,common_parent=source,resume=file_identity(resume) if resume else None,
        train_window_ids=train_cache[2],validation_window_ids=val_cache[2],selection=selection,
        evaluation_unchanged=True,precision='float32 network/AdamW; float64 boundary decoder; no AMP/TF32',
        physical_loss_reduction='scene mean; t<=max_auxiliary_timestep; alpha_bar weight; full-batch denominator')
    with RunRecorder(output,repository,effective,command,
        {'model':c['seed'],'training_noise':c['noise_seed'],'validation_noise':c['validation_seed']},
        purpose='stage5b_controlled_continuation',data_files=sorted(set(map(str,files))),
        data_id=data['manifest']['id'],checkpoint=resume or parent_checkpoint,formal=formal) as run:
        model=ConditionalDenoiser(architecture).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        generator=torch.Generator().manual_seed(c['noise_seed'])
        restore_training(previous,model,optimizer,generator,device)
        initial=inherited_state(model,optimizer,generator)
        diffusion=resolve_diffusion_config(c['diffusion']);schedule=NoiseSchedule(diffusion,device)
        objective=resolve_physical_config(c['physical_loss'])
        metadata=dict(model_config=architecture.to_dict(),diffusion_config=diffusion.to_dict(),training_config=c,
            training_signature=signature,data=data,scale_audit=scale,run_sha=run.manifest['git']['commit_sha'],
            run_branch=run.manifest['git']['branch'],run_dirty=run.manifest['git']['dirty'],
            parent_checkpoint=file_identity(resume or parent_checkpoint),
            runtime=dict(torch=str(torch.__version__),device=str(device),strict_determinism=True,network_dtype='float32'))
        train_probe=probe_subset(train_cache,c['train_probe_limit'])
        validations=[dict(step=start,train_probe_mse=probe(model,train_probe,schedule,device,c['batch_size'],c['validation_seed']),
            validation_probe_mse=probe(model,val_cache,schedule,device,c['batch_size'],c['validation_seed']))]
        write_json(run.output/'initial_state.json',initial)
        write_json(run.output/'validation.json',validations)
        times=[];checkpoints=[];monitoring=[];batches=[];counts=0;component_sums={}
        peak_allocated=0;peak_reserved=0
        begun=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        with (run.output/'steps.jsonl').open('w',encoding='utf-8') as log:
            for step in range(start+1,c['max_steps']+1):
                synchronize(device);tick=time.perf_counter()
                indices,epoch,index=epoch_batch(step,len(train_cache[2]),c['batch_size'],c['seed'])
                conditions,targets,ids=select_batch(train_cache,indices,device)
                terms,norm,noise,timestep=optimize_batch(model,optimizer,conditions,targets,schedule,generator,
                    c['gradient_clip_norm'],objective)
                synchronize(device);seconds=time.perf_counter()-tick
                times.append(seconds);batches.append(len(indices))
                row=dict(step=step,additional_step=step-c['source_step'],epoch=epoch,epoch_batch=index,
                    epoch_complete=index==c['steps_per_epoch'],window_ids=ids,timesteps=timestep.cpu().tolist(),
                    noise_sha256=tensor_hash(noise),gradient_norm_before_clip=float(norm),seconds=seconds)
                for name,value in terms.items():
                    row[name]=float(value) if isinstance(value,torch.Tensor) else value
                    component_sums[name]=component_sums.get(name,0.)+row[name]
                counts+=terms['auxiliary_eligible_scenes']
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if device.type=='cuda':
                    peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated(device))
                    peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved(device))
                if step%c['validation_interval']==0 or step==c['max_steps']:
                    validations.append(dict(step=step,
                        train_probe_mse=probe(model,train_probe,schedule,device,c['batch_size'],c['validation_seed']),
                        validation_probe_mse=probe(model,val_cache,schedule,device,c['batch_size'],c['validation_seed'])))
                    write_json(run.output/'validation.json',validations)
                    write_json(run.output/'progress.json',dict(arm=c['arm'],step=step,end_step=c['max_steps'],
                        epoch=epoch,epoch_complete=index==c['steps_per_epoch'],execution_budget_kind=c['execution_budget_kind'],
                        nominal_additional_epochs=10,validation=validations[-1]))
                    print(f"arm={c['arm']} step={step} epoch={epoch} validation={validations[-1]['validation_probe_mse']:.6f}",flush=True)
                save_due=step%c['checkpoint_interval']==0 or step==c['max_steps']
                if save_due:
                    path=run.output/f'checkpoint_step_{step:06d}.pt'
                    save_checkpoint(path,model,optimizer,generator,step,metadata)
                    checkpoints.append(file_identity(path))
                    write_json(run.output/'latest_checkpoint.json',dict(step=step,checkpoint=checkpoints[-1]))
                monitor_due=c['execution_budget_kind']=='ten_epoch_ablation' and c['monitor_every_epochs'] and index==c['steps_per_epoch'] and (
                    (epoch-c['source_epochs'])%c['monitor_every_epochs']==0 or step==c['max_steps'])
                if monitor_due:
                    if not save_due:
                        path=run.output/f'checkpoint_step_{step:06d}.pt'
                        save_checkpoint(path,model,optimizer,generator,step,metadata)
                        checkpoints.append(file_identity(path))
                    monitoring.append(periodic_sample(dataset,path,run,repository,c,epoch,formal))
                    write_json(run.output/'generation_monitor.json',monitoring)
        duration=time.perf_counter()-begun
        timed=times[1:] or times;mean=float(np.mean(timed))
        result=dict(schema_version='sumodiff.controlled.train.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            arm=c['arm'],common_parent=source,start_step=start,end_step=c['max_steps'],updates=len(times),
            execution_budget_kind=c['execution_budget_kind'],nominal_additional_epochs=10,
            newly_completed_epochs=c['max_steps']//c['steps_per_epoch']-start//c['steps_per_epoch'],
            epoch_schedule=dict(first_global_epoch=start//c['steps_per_epoch']+1,last_global_epoch=(c['max_steps']-1)//c['steps_per_epoch']+1,
                steps_per_epoch=c['steps_per_epoch'],each_epoch_without_replacement=True),
            train_windows=len(train_cache[2]),validation_windows=len(val_cache[2]),batch_size=c['batch_size'],
            initial_state=initial,final_parameter_sha256=parameter_hash(model),
            cache_preparation_seconds=cache_seconds,training_validation_checkpoint_wall_seconds=duration,
            optimizer_step_timing=dict(mean_seconds=mean,median_seconds=float(np.median(timed)),
                warmup_updates_excluded=min(1,len(times)),windows_per_second=float(np.mean(batches[1:] or batches))/mean),
            peak_allocated_bytes=peak_allocated if device.type=='cuda' else None,
            peak_reserved_bytes=peak_reserved if device.type=='cuda' else None,
            validation=validations,generation_monitor=monitoring,checkpoints=checkpoints,
            physical_loss=c['physical_loss'],auxiliary_eligible_scenes=counts,
            mean_components={k:v/len(times) for k,v in component_sums.items()},
            estimate=dict(optimizer_only_seconds_per_arm=mean*c['nominal_additional_updates'],
                basis='same measured workload; validation/checkpoint/IO/cache excluded; not a convergence claim'),
            cached_tensor_bytes=sum(v.numel()*v.element_size() for item in (train_cache,val_cache)
                for part in item[:2] for v in part.values()),
            convergence_status='not_assessed')
        if result['final_parameter_sha256']==initial['model']:
            raise RuntimeError('Continuation did not update model weights')
        if objective.enabled and (not counts or result['mean_components']['physical_total']<=0):
            raise RuntimeError('Enabled physical objectives were not exercised')
        write_json(run.output/'training_summary.json',result);run.write_metrics(result)
    return result
