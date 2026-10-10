"""Stage5C: parameter warm start, new AdamW, and strict same-experiment recovery."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from sumodiff.models import ConditionalDenoiser,resolve_model_config
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.data.dataset import write_json
from sumodiff.experiments.recorder import RunRecorder,load_config,file_identity
from .checkpoint import load_checkpoint,restore_training,save_checkpoint
from .config import resolve_diffusion_config
from .controlled_training import shared_cache,inherited_state,tensor_hash
from .data import dataset_identity,select_batch,verified_scale_audit
from .epoch_training import epoch_batch,probe_subset,periodic_sample
from .training import draw_noise,probe,synchronize
from .process import NoiseSchedule
from .unified_losses import resolve_unified_loss,unified_objective,RoadGeometryCache
from .schedule import ScheduleConfig,UpdateSchedule,resolve_schedule

SCHEMA = 'sumodiff.unified.train.config.v1'
DEFAULTS = dict(schema_version=SCHEMA,objective_name='unified',additional_epochs=50,
    expected_parent_step=30000,expected_parent_sha256='',
    expected_train_windows=1200,expected_validation_windows=200,batch_size=4,
    device='cuda:0',torch_threads=1,peak_learning_rate=1e-4,minimum_learning_rate=1e-6,
    warmup_epochs=2,auxiliary_warmup_epochs=5,validation_every_epochs=1,
    checkpoint_every_epochs=5,monitor_every_epochs=5,monitor_tasks=20,loss={})


def unified_config(value,parent,*,prefix_updates=None):
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown unified training options')
    requested=deepcopy(DEFAULTS);requested.update(value)
    if requested['schema_version']!=SCHEMA or requested['objective_name'] not in ('unified','epsilon_cosine'):
        raise ValueError('Unsupported unified training schema/objective')
    for key in ('additional_epochs','expected_parent_step','expected_train_windows','expected_validation_windows',
                'batch_size','torch_threads','warmup_epochs','auxiliary_warmup_epochs',
                'validation_every_epochs','checkpoint_every_epochs','monitor_every_epochs','monitor_tasks'):
        if type(requested[key]) is not int or requested[key]<1:
            raise ValueError(f'Invalid {key}')
    if requested['additional_epochs']!=50 or requested['warmup_epochs']!=2 or requested['auxiliary_warmup_epochs']!=5:
        raise ValueError('Stage5C budget is 50 additional epochs, lr warmup 2, auxiliary warmup 5')
    if requested['monitor_tasks']!=20:
        raise ValueError('Stage5C uses 20 fixed validation monitoring tasks')
    sha=requested['expected_parent_sha256']
    if not isinstance(sha,str) or len(sha)!=64 or any(x not in '0123456789abcdef' for x in sha):
        raise ValueError('Complete original parent SHA-256 required')
    previous=parent['metadata']['training_config']
    if previous.get('schema_version')!='sumodiff.base.epoch.config.v1' or previous['diffusion']['additional_losses_enabled']:
        raise ValueError('Stage5C warm start requires the original epsilon-only 100-epoch parent')
    if parent['step']!=requested['expected_parent_step'] or previous['epochs']!=100 or parent['step']!=100*previous['steps_per_epoch']:
        raise ValueError('Expected completed original 100-epoch parent')
    for key in ('batch_size','device','torch_threads','expected_train_windows','expected_validation_windows'):
        if requested[key]!=previous[key]:
            raise ValueError(f'{key} must retain the audited parent architecture/cohort')
    loss=resolve_unified_loss(requested['loss'])
    if loss.enabled!=(requested['objective_name']=='unified'):
        raise ValueError('Objective name and auxiliary-loss weights disagree')
    steps=math.ceil(requested['expected_train_windows']/requested['batch_size'])
    schedule=ScheduleConfig(peak_lr=requested['peak_learning_rate'],minimum_lr=requested['minimum_learning_rate'],
        warmup_updates=steps*requested['warmup_epochs'],total_updates=steps*requested['additional_epochs'],
        auxiliary_warmup_updates=steps*requested['auxiliary_warmup_epochs'])
    if prefix_updates is not None and (type(prefix_updates) is not int or not 1<=prefix_updates<=24):
        raise ValueError('Short preflight prefix must contain 1 to 24 updates')
    c=deepcopy(previous)
    c.update(schema_version=SCHEMA,objective_name=requested['objective_name'],
        source_step=parent['step'],source_epochs=100,fork_source_sha256=sha,
        additional_epochs=50,epochs=150,steps_per_epoch=steps,loss=loss.to_dict(),schedule=schedule.to_dict(),
        initialization='inherit_model_and_noise_Torch_RNG_reset_AdamW_and_scheduler',
        optimizer=dict(type='AdamW',weight_decay=previous['weight_decay'],betas=[.9,.999],eps=1e-8),
        learning_rate=schedule.minimum_lr,nominal_additional_updates=schedule.total_updates,
        planned_additional_updates=prefix_updates if prefix_updates is not None else schedule.total_updates,
        execution_budget_kind='preflight_prefix' if prefix_updates is not None else 'fifty_epoch_unified',
        validation_every_epochs=requested['validation_every_epochs'],
        checkpoint_every_epochs=requested['checkpoint_every_epochs'],monitor_every_epochs=requested['monitor_every_epochs'])
    c['monitor']=deepcopy(previous['monitor']);c['monitor']['limit']=20
    c['diffusion']['additional_losses_enabled']=loss.enabled
    c.update(max_steps=parent['step']+c['planned_additional_updates'],
        validation_interval=c['validation_every_epochs']*steps,checkpoint_interval=c['checkpoint_every_epochs']*steps,
        formal_estimate_steps=schedule.total_updates)
    return c


def unified_signature(architecture,c,data,scale,train_ids,val_ids):
    # Unlike the legacy signature, every schedule/objective/budget/monitor option
    # is immutable during recovery. A new objective is a new experiment.
    return dict(model=architecture.to_dict(),training=deepcopy(c),
        dataset_hashes={k:v['sha256'] for k,v in data.items()},scale_audit_sha256=scale['sha256'],
        train_window_ids=list(train_ids),validation_window_ids=list(val_ids))


def warm_start(parent,model,optimizer,generator,device):
    if optimizer.state:
        raise ValueError('Warm start requires a fresh empty AdamW state')
    if parent['metadata']['runtime']['torch']!=str(torch.__version__):
        raise ValueError('Warm start RNG inheritance requires the recorded torch version')
    model.load_state_dict(parent['model'],strict=True)
    generator.set_state(parent['generator_state']);torch.set_rng_state(parent['cpu_rng'])
    if torch.device(device).type=='cuda':
        if len(parent['cuda_rng'])!=torch.cuda.device_count():
            raise ValueError('Warm start CUDA RNG/device mismatch')
        torch.cuda.set_rng_state_all(parent['cuda_rng'])
    return inherited_state(model,optimizer,generator)


def new_optimizer(model,c):
    settings=c['optimizer']
    return torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=settings['weight_decay'],
                            betas=tuple(settings['betas']),eps=settings['eps'])


def validate_runtime(value):
    """Called on every new-protocol checkpoint load, including inference."""
    m=value['metadata'];c=m['training_config']
    if c.get('schema_version')!=SCHEMA or m['training_signature']['training']!=c:
        raise ValueError('Unified checkpoint requires complete immutable training provenance')
    objective=resolve_unified_loss(c['loss'])
    if objective.enabled!=m['diffusion_config']['additional_losses_enabled'] or c['diffusion']!=m['diffusion_config']:
        raise ValueError('Unified loss/checkpoint flag mismatch')
    schedule=UpdateSchedule(resolve_schedule(c['schedule']))
    runtime=m['runtime']
    schedule.load_state_dict(runtime['scheduler_state'],expected_updates=value['step']-c['source_step'])
    if value['step']>c['max_steps'] or runtime['snapshot_kind'] not in ('training_recovery','monitor_assessment'):
        raise ValueError('Unified checkpoint step/snapshot is invalid')
    if value['step']>c['source_step'] and any(g['lr']!=schedule.last_lr for g in value['optimizer']['param_groups']):
        raise ValueError('Saved optimizer lr differs from unified scheduler')


def optimize_batch(model,optimizer,conditions,targets,schedule,generator,clip_norm,objective,roads,lr_schedule,update,
                   *, objective_fn=unified_objective):
    lr,ramp=lr_schedule.apply(optimizer,update)
    model.train();optimizer.zero_grad(set_to_none=True)
    noisy,noise,timestep=draw_noise(targets,conditions,schedule,generator)
    terms=objective_fn(model(noisy,timestep,conditions),noisy,noise,timestep,conditions,targets,schedule,
                            objective,roads=roads,ramp=ramp)
    loss=terms['total_loss'];loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm,error_if_nonfinite=True)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite unified loss')
    optimizer.step();lr_schedule.complete(update)
    return terms,norm,noise,timestep,lr


def monitor_quality(directory,validation_cache):
    from .quality_audit import physical_derivatives
    root=Path(directory)
    catalog=json.loads((root/'trajectory_index.json').read_text(encoding='utf-8'))
    cached={wid:i for i,wid in enumerate(validation_cache[2])}
    counts=dict(planned=len(catalog),quality_passes=0,road_bad=0,route_bad=0,motion_bad=0,unknown=0,failures=0)
    jerks=[]
    for item in catalog:
        metrics=json.loads((root/'trajectories'/item['window_id']/'metrics.json').read_text(encoding='utf-8'))
        d=metrics['decoded']
        counts['quality_passes']+=d.get('quality_pass') is True
        counts['failures']+=item['generation_status']!='completed' or d['status']!='completed'
        known=d.get('quality_pass') is not None and d['status']=='completed'
        counts['unknown']+=not known
        for name in ('road','route'):
            fraction=d.get('roads',{}).get(name,{}).get('violation_fraction')
            counts[name+'_bad']+=fraction is not None and fraction>0
        counts['motion_bad']+=d.get('motion',{}).get('hard_quality_pass') is False
        path=root/'trajectories'/item['window_id']/'stages.npz'
        if item['generation_status']=='completed' and path.exists():
            i=cached[item['window_id']]
            c={k:v[i].numpy() for k,v in validation_cache[0].items()}
            with np.load(path,allow_pickle=False) as a:
                _,_,j=physical_derivatives(c,a['decoded_absolute_states'])
            mask=c['agent_mask']
            jerks.extend(np.linalg.norm(j[mask,4:],axis=-1).ravel().tolist())
    return dict(**counts,decoded_interior_jerk_p99=float(np.quantile(jerks,.99)) if jerks else None)


def quality_frontier(records):
    """Separate multiobjective candidates; no weighted scene score."""
    eligible=[r for r in records if r['quality']['unknown']==0 and r['quality']['failures']==0
              and r['quality']['decoded_interior_jerk_p99'] is not None]
    def vector(r):
        q=r['quality'];n=q['planned']
        return (-q['quality_passes']/n,q['road_bad']/n,q['route_bad']/n,
                q['motion_bad']/n,q['decoded_interior_jerk_p99'])
    return [r['checkpoint'] for r in eligible if not any(
        all(x<=y for x,y in zip(vector(other),vector(r))) and
        any(x<y for x,y in zip(vector(other),vector(r))) for other in eligible if other is not r)]


class UnifiedProtocol:
    """Stage5C semantics; shared loop with an explicitly separate protocol."""
    config = staticmethod(unified_config)
    scale_diffusion = staticmethod(lambda parent: resolve_diffusion_config(parent['metadata']['diffusion_config']))
    resolve_loss = staticmethod(resolve_unified_loss)
    objective = staticmethod(unified_objective)
    initialize = staticmethod(warm_start)
    monitor_quality = staticmethod(monitor_quality)
    frontier = staticmethod(quality_frontier)
    max_prefix_updates = 24
    run_schema = 'sumodiff.unified.train.run.v1'
    metrics_schema = 'sumodiff.unified.train.metrics.v1'
    purpose = 'stage5c_unified_continuation'
    full_budget_kind = 'fifty_epoch_unified'
    optimizer_reset = True
    loss_reduction = 'per-scene valid means; t<=300 alpha_bar; whole-minibatch denominator; linear auxiliary ramp'


def train_unified(dataset,parent_checkpoint,config_path,scale_path,output,repository,command,
                  formal=False,resume=None,prefix_updates=None,allow_long_run=False,cache=None,stop_after_updates=None,
                  *, _protocol=None):
    protocol = _protocol or UnifiedProtocol()
    source=file_identity(parent_checkpoint);parent=load_checkpoint(parent_checkpoint)
    c=protocol.config(load_config(config_path),parent,prefix_updates=prefix_updates)
    if source['sha256']!=c['fork_source_sha256']:
        raise ValueError('Original parent checksum mismatch')
    if c['planned_additional_updates']>1000 and not allow_long_run:
        raise ValueError('Continuation requires explicit --allow-long-run')
    if stop_after_updates is not None and (
        type(stop_after_updates) is not int or not 1<=stop_after_updates<=protocol.max_prefix_updates or prefix_updates is None):
        raise ValueError('Diagnostic stop must be bounded within a short prefix')
    data=dataset_identity(dataset)
    scale_config=protocol.scale_diffusion(parent)
    scale=verified_scale_audit(scale_path,dataset,scale_config)
    if any(data[k]['sha256']!=parent['metadata']['data'][k]['sha256'] for k in data):
        raise ValueError('Warm-start dataset differs from the original parent')
    if scale['sha256']!=parent['metadata']['scale_audit']['sha256']:
        raise ValueError('Scale audit differs from the original parent')
    configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    train_cache,val_cache,selection,cache_seconds=cache if cache is not None else shared_cache(dataset,c)
    signature=unified_signature(resolve_model_config(parent['metadata']['model_config']),c,data,scale,train_cache[2],val_cache[2])
    old_signature=parent['metadata']['training_signature']
    if train_cache[2]!=old_signature['train_window_ids'] or val_cache[2]!=old_signature['validation_window_ids']:
        raise ValueError('Warm-start task order differs from original parent')
    previous=load_checkpoint(resume,signature=signature) if resume else None
    if previous and previous['metadata']['runtime']['snapshot_kind']!='training_recovery':
        raise ValueError('Monitor assessment weights are not recovery checkpoints')
    start=previous['step'] if previous else c['source_step']
    if not c['source_step']<=start<c['max_steps']:
        raise ValueError('Resume step outside this experiment budget')
    end=min(c['max_steps'],start+stop_after_updates) if stop_after_updates else c['max_steps']
    objective=protocol.resolve_loss(c['loss'])
    geometry_start=time.perf_counter()
    geometry=RoadGeometryCache(dataset,train_cache[2],objective) if objective.road_weight else None
    geometry_seconds=time.perf_counter()-geometry_start
    files=[config_path,scale_path,parent_checkpoint,*[v['location'] for v in data.values()],*train_cache[3],*val_cache[3]]
    if resume:files.append(resume)
    effective=dict(schema_version=protocol.run_schema,training=c,data=data,scale_audit=scale,
        scale_audit_diffusion_config=scale_config.to_dict(),
        common_parent=source,resume=file_identity(resume) if resume else None,
        train_window_ids=train_cache[2],validation_window_ids=val_cache[2],selection=selection,
        stop_after_updates=stop_after_updates,evaluation_unchanged=True,
        precision='float32 network/AdamW; float64 decoder; no AMP/TF32',
        loss_reduction=protocol.loss_reduction,
        road_proxy='5x5 oriented body points; exact vector outside distance; detached nearest projection; no raster clipping')
    with RunRecorder(output,repository,effective,command,
        {'model':c['seed'],'training_noise':c['noise_seed'],'validation_noise':c['validation_seed']},
        purpose=protocol.purpose,data_files=sorted(set(map(str,files))),
        data_id=data['manifest']['id'],checkpoint=resume or parent_checkpoint,formal=formal) as run:
        model=ConditionalDenoiser(resolve_model_config(parent['metadata']['model_config'])).to(device)
        optimizer=new_optimizer(model,c);generator=torch.Generator().manual_seed(c['noise_seed'])
        learning=UpdateSchedule(resolve_schedule(c['schedule']))
        if previous:
            restore_training(previous,model,optimizer,generator,device)
            learning.load_state_dict(previous['metadata']['runtime']['scheduler_state'],optimizer,
                                     expected_updates=start-c['source_step'])
            initial=inherited_state(model,optimizer,generator)
            best=deepcopy(previous['metadata']['runtime']['best_epsilon'])
            if best['checkpoint'] is None:
                if best['step']!=start:raise ValueError('Missing previous best epsilon identity')
                best['checkpoint']=file_identity(resume)
            elif file_identity(best['checkpoint']['location'])['sha256']!=best['checkpoint']['sha256']:
                raise ValueError('Previous best epsilon checkpoint changed')
            monitoring=deepcopy(previous['metadata']['runtime']['quality_history'])
        else:
            initial=protocol.initialize(parent,model,optimizer,generator,device)
            best=None;monitoring=[]
        schedule=NoiseSchedule(resolve_diffusion_config(c['diffusion']),device)
        probe_cache=probe_subset(train_cache,c['train_probe_limit'])
        validations=[dict(step=start,train_probe_mse=probe(model,probe_cache,schedule,device,c['batch_size'],c['validation_seed']),
            validation_probe_mse=probe(model,val_cache,schedule,device,c['batch_size'],c['validation_seed']))]
        if best is None:
            best=dict(step=start,mse=validations[0]['validation_probe_mse'],checkpoint=source)
        metadata=dict(model_config=parent['metadata']['model_config'],diffusion_config=c['diffusion'],training_config=c,
            training_signature=signature,data=data,scale_audit=scale,run_sha=run.manifest['git']['commit_sha'],
            run_branch=run.manifest['git']['branch'],run_dirty=run.manifest['git']['dirty'],
            parent_checkpoint=file_identity(resume or parent_checkpoint),
            runtime=dict(torch=str(torch.__version__),device=str(device),strict_determinism=True,network_dtype='float32'))
        def save(path,step,kind):
            m=deepcopy(metadata);m['runtime'].update(scheduler_state=learning.state_dict(),
                best_epsilon=deepcopy(best),quality_history=deepcopy(monitoring),snapshot_kind=kind)
            save_checkpoint(path,model,optimizer,generator,step,m)
            return file_identity(path)
        write_json(run.output/'initial_state.json',dict(**initial,optimizer_parameter_states=len(optimizer.state),
            scheduler_state=learning.state_dict(),initialization='strict_resume' if resume else c['initialization']))
        write_json(run.output/'validation.json',validations)
        write_json(run.output/'geometry_repairs.json',geometry.repairs if geometry else [])
        times=[];component_sums={};checkpoints=[];peak_allocated=0;peak_reserved=0;eligible=0;clips=0
        started=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        with (run.output/'steps.jsonl').open('w',encoding='utf-8') as log:
            for step in range(start+1,end+1):
                synchronize(device);tick=time.perf_counter()
                indices,epoch,index=epoch_batch(step,len(train_cache[2]),c['batch_size'],c['seed'])
                conditions,targets,ids=select_batch(train_cache,indices,device)
                terms,norm,noise,timestep,lr=optimize_batch(model,optimizer,conditions,targets,schedule,generator,
                    c['gradient_clip_norm'],objective,geometry.batch(ids) if geometry else None,learning,step-c['source_step'],
                    objective_fn=protocol.objective)
                synchronize(device);elapsed=time.perf_counter()-tick;times.append(elapsed)
                row=dict(step=step,additional_step=step-c['source_step'],epoch=epoch,epoch_batch=index,
                    epoch_complete=index==c['steps_per_epoch'],window_ids=ids,timesteps=timestep.cpu().tolist(),
                    noise_sha256=tensor_hash(noise),gradient_norm_before_clip=float(norm),learning_rate=lr,seconds=elapsed)
                for key,value in terms.items():
                    row[key]=float(value) if isinstance(value,torch.Tensor) else value
                    component_sums[key]=component_sums.get(key,0.)+row[key]
                eligible+=terms['auxiliary_eligible_scenes'];clips+=float(norm)>c['gradient_clip_norm']
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if device.type=='cuda':
                    peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated(device))
                    peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved(device))
                best_changed=False
                if step%c['validation_interval']==0 or step==end:
                    v=dict(step=step,train_probe_mse=probe(model,probe_cache,schedule,device,c['batch_size'],c['validation_seed']),
                           validation_probe_mse=probe(model,val_cache,schedule,device,c['batch_size'],c['validation_seed']))
                    validations.append(v);write_json(run.output/'validation.json',validations)
                    best_changed=v['validation_probe_mse']<best['mse']
                    if best_changed:best=dict(step=step,mse=v['validation_probe_mse'],checkpoint=None)
                    print(f"objective={c['objective_name']} step={step} epoch={epoch} lr={lr:.3g} validation={v['validation_probe_mse']:.6f}",flush=True)
                monitor_due=c['execution_budget_kind']==protocol.full_budget_kind and index==c['steps_per_epoch'] and (
                    (epoch-c['source_epochs'])%c['monitor_every_epochs']==0 or step==end)
                if monitor_due:
                    assessment=run.output/f'monitor_checkpoint_step_{step:06d}.pt'
                    identity=save(assessment,step,'monitor_assessment')
                    record=periodic_sample(dataset,assessment,run,repository,c,epoch,formal)
                    record.update(step=step,quality=protocol.monitor_quality(record['run'],val_cache),checkpoint=identity)
                    monitoring.append(record)
                    write_json(run.output/'quality_candidates.json',
                        dict(history=monitoring,pareto_checkpoints=protocol.frontier(monitoring),
                             selection='separate quality/road/route/motion/jerk axes; no aggregate score'))
                prefix_midpoint = prefix_updates is not None and (step-c['source_step'])==max(1,prefix_updates//2)
                if step%c['checkpoint_interval']==0 or step==end or best_changed or monitor_due or prefix_midpoint:
                    identity=save(run.output/f'checkpoint_step_{step:06d}.pt',step,'training_recovery')
                    checkpoints.append(identity)
                    if best['checkpoint'] is None:
                        best['checkpoint']=identity
                    write_json(run.output/'latest_checkpoint.json',dict(step=step,checkpoint=identity))
                    write_json(run.output/'best_epsilon.json',best)
                write_json(run.output/'progress.json',dict(step=step,end_step=c['max_steps'],executed_segment_end=end,
                    epoch=epoch,epoch_complete=index==c['steps_per_epoch'],learning_rate=lr,
                    auxiliary_ramp=learning.last_auxiliary_ramp,scheduler_state=learning.state_dict(),
                    best_epsilon=best,full_budget_complete=step==c['max_steps'] and c['execution_budget_kind']==protocol.full_budget_kind))
        duration=time.perf_counter()-started;timed=times[1:] or times;mean=float(np.mean(timed))
        result=dict(schema_version=protocol.metrics_schema,run_sha=run.manifest['git']['commit_sha'],
            objective=c['objective_name'],common_parent=source,start_step=start,end_step=end,updates=len(times),
            planned_end_step=c['max_steps'],nominal_additional_epochs=c['additional_epochs'],execution_budget_kind=c['execution_budget_kind'],
            full_budget_complete=end==c['max_steps'] and c['execution_budget_kind']==protocol.full_budget_kind,
            newly_completed_epochs=end//c['steps_per_epoch']-start//c['steps_per_epoch'],
            optimizer_reset_at_fork=protocol.optimizer_reset,initial_state=initial,final_parameter_sha256=parameter_hash(model),
            optimizer_step_timing=dict(mean_seconds=mean,median_seconds=float(np.median(timed)),warmup_updates_excluded=1),
            cache_preparation_seconds=cache_seconds,geometry_preparation_seconds=geometry_seconds,
            training_validation_checkpoint_wall_seconds=duration,
            peak_allocated_bytes=peak_allocated if device.type=='cuda' else None,
            peak_reserved_bytes=peak_reserved if device.type=='cuda' else None,
            validation=validations,checkpoints=checkpoints,best_epsilon=best,generation_monitor=monitoring,
            quality_pareto_checkpoints=protocol.frontier(monitoring),gradient_clipped_updates=clips,
            auxiliary_eligible_scenes=eligible,mean_components={k:v/len(times) for k,v in component_sums.items()},
            scheduler_state=learning.state_dict(),
            cached_tensor_bytes=sum(v.numel()*v.element_size() for item in (train_cache,val_cache)
                for part in item[:2] for v in part.values()),
            estimate=dict(optimizer_only_seconds=mean*c['nominal_additional_updates'],
                basis='measured same batch/cohort workload; excludes validation/monitoring/checkpoint/cache; no convergence claim'),
            convergence_status='not_assessed')
        write_json(run.output/'training_summary.json',result);run.write_metrics(result)
    return result
