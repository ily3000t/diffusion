"""Full-cohort control training; immutable epoch schedule and exact continuation."""
from collections import Counter, OrderedDict
from copy import deepcopy
from itertools import combinations
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import psutil
from sumodiff.data.dataset import WindowDataset, collate_numpy, write_json
from sumodiff.data.control_tasks import FAMILIES, POLICIES, motion_metadata
from sumodiff.models import ConditionalDenoiser, resolve_model_config, prepare_conditioning
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.experiments.recorder import RunRecorder, file_identity, load_config
from .config import resolve_diffusion_config, is_control, validate_architecture
from .control_training import verified_control_scales, control_objective
from .checkpoint import save_checkpoint, load_checkpoint, restore_training
from .data import dataset_identity
from .epoch_training import epoch_batch, periodic_sample
from .process import NoiseSchedule, masked_noise_mse
from .schedule import ScheduleConfig, UpdateSchedule
from .unified_losses import resolve_unified_loss, prepare_roads
from .jerk_tail import JerkTailConfig

SCHEMA = 'sumodiff.control.epoch.config.v1'
DEFAULTS = dict(schema_version=SCHEMA, seed=20261017, noise_seed=20261018,
    validation_seed=20261019, device='cuda:0', torch_threads=1, batch_size=16,
    epochs=200, expected_train_windows=6000, expected_validation_windows=600,
    expected_test_windows=600, learning_rate=1e-6, peak_learning_rate=1e-4,
    lr_warmup_epochs=10, auxiliary_warmup_epochs=10, weight_decay=.0001,
    gradient_clip_norm=1., validation_every_epochs=1, checkpoint_every_epochs=5,
    monitor_every_epochs=10, train_probe_limit=60, stream_cache_windows=64,
    calibration_config=None, expected_data_id=None, unified_loss={}, jerk_tail={},
    tail_weight=.0003, diffusion={}, monitor={})
DERIVED = {'steps_per_epoch', 'max_steps', 'schedule', 'initialization'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def epoch_config(value):
    if not isinstance(value, dict) or set(value)-(set(DEFAULTS)|DERIVED) or value.get('schema_version')!=SCHEMA:
        raise ValueError('Unknown control epoch options')
    c=deepcopy(DEFAULTS); c.update({k:v for k,v in value.items() if k in DEFAULTS})
    for key in ('seed','noise_seed','validation_seed'):
        if type(c[key]) is not int or not 0<=c[key]<2**32: raise ValueError(f'Invalid {key}')
    for key in ('torch_threads','batch_size','epochs','expected_train_windows','expected_validation_windows',
                'expected_test_windows','lr_warmup_epochs','auxiliary_warmup_epochs',
                'validation_every_epochs','checkpoint_every_epochs','monitor_every_epochs','train_probe_limit','stream_cache_windows'):
        if type(c[key]) is not int or c[key]<1: raise ValueError(f'Invalid {key}')
    if c['device'] not in ('cpu','cuda:0'): raise ValueError('Unsupported control training device')
    for key in ('learning_rate','peak_learning_rate','gradient_clip_norm','tail_weight'):
        if type(c[key]) not in (int,float) or not math.isfinite(c[key]) or c[key]<=0: raise ValueError(f'Invalid {key}')
    if type(c['weight_decay']) not in (int,float) or not math.isfinite(c['weight_decay']) or c['weight_decay']<0:
        raise ValueError('Invalid weight decay')
    if any(c[k]%6 for k in ('expected_train_windows','expected_validation_windows','expected_test_windows')):
        raise ValueError('Motion quotas must allow equal family and reference partitions')
    if c['train_probe_limit']>c['expected_train_windows'] or c['stream_cache_windows']>1024:
        raise ValueError('Invalid bounded probe/cache budget')
    for key in ('calibration_config','expected_data_id'):
        if not isinstance(c[key],str) or not c[key].strip(): raise ValueError(f'Explicit {key} required')
    d=resolve_diffusion_config(c['diffusion'])
    if not is_control(d) or not d.additional_losses_enabled: raise ValueError('Control auxiliary representation required')
    c['diffusion']=d.to_dict()
    c['unified_loss']=resolve_unified_loss(c['unified_loss']).to_dict()
    c['jerk_tail']=JerkTailConfig(**c['jerk_tail']).to_dict()
    from .sampling import sample_config
    c['monitor']=sample_config({**c['monitor'],'device':c['device'],'torch_threads':c['torch_threads']})
    if (c['monitor']['split']!='validation' or c['monitor']['limit']!=60 or not c['monitor']['task_plan']
        or c['monitor']['steps']!=20 or c['monitor']['eta']!=0):
        raise ValueError('Control epoch monitoring requires the fixed 60 validation tasks and DDIM20 eta0')
    steps=math.ceil(c['expected_train_windows']/c['batch_size'])
    derived=dict(steps_per_epoch=steps,max_steps=steps*c['epochs'],initialization='random_new_model_and_adamw',
        schedule=ScheduleConfig(peak_lr=c['peak_learning_rate'],minimum_lr=c['learning_rate'],
            warmup_updates=c['lr_warmup_epochs']*steps,total_updates=steps*c['epochs'],
            auxiliary_warmup_updates=c['auxiliary_warmup_epochs']*steps).to_dict())
    if any(value[k]!=derived[k] for k in DERIVED if k in value):
        raise ValueError('Derived epoch budget/schedule cannot be overridden')
    c.update(derived)
    return c


def verified_dataset(directory,c):
    """Authenticate the frozen selection and source ledger; never open test arrays."""
    directory=Path(directory).resolve(strict=True)
    data=dataset_identity(directory); m=read(directory/'dataset_manifest.json')
    cur=m.get('curation',{}); identity=file_identity(directory/'selection.json')
    if (m['data_id']!=c['expected_data_id'] or identity['sha256']!=cur.get('selection',{}).get('sha256')
        or m['window_index']['sha256']!=data['index']['sha256']
        or cur.get('required_model_schema')!='sumodiff.model.control.config.v2'
        or not cur.get('label_quality_filtered') or not cur.get('inference_conditions_unchanged')):
        raise ValueError('Motion dataset/model/selection identity mismatch')
    s=read(directory/'selection.json'); rows=s['selected']; index=read(directory/'windows.json')
    if s.get('schema_version')!='sumodiff.stage5a.selection.v3' or s.get('profile')!='full' or not s.get('no_smoothing_or_clipping'):
        raise ValueError('Full Motion selection v3 is required')
    by_id={r['window_id']:r for r in rows}
    if len(by_id)!=len(rows) or len(index)!=len(rows) or {r['window_id'] for r in index}!=set(by_id):
        raise ValueError('Motion selected window population mismatch')
    for row in index:
        selected=by_id[row['window_id']]
        if (not row['core_training_eligible'] or not selected['eligible'] or not selected['core_complete']
            or selected['reasons'] or selected['quality_status']!='assessed'
            or any(row[k]!=selected[k] for k in ('split','episode_key','geometry_id','family'))):
            raise ValueError('Ineligible or mismatched selected window')
        motion_metadata(selected)
    if len({(r['episode_key'],r['reference_tick']) for r in rows})!=len(rows):
        raise ValueError('Duplicate Motion reference time views')
    for split,n in [('train',c['expected_train_windows']),('validation',c['expected_validation_windows']),('test',c['expected_test_windows'])]:
        selected=[r for r in rows if r['split']==split]
        if len(selected)!=n or Counter((r['family'],r['reference_policy']) for r in selected)!=Counter(
            {(f,p):n//6 for f in FAMILIES for p in POLICIES}):
            raise ValueError('Motion split/family/reference quotas differ')
        for family in FAMILIES:
            summary=s['selection'][split][family]
            if not summary['passed'] or not all(summary['checks'].values()) or any(not v['passed'] for v in summary['categories'].values()):
                raise ValueError('Motion coverage certification failed')
            cohort=[r for r in selected if r['family']==family]; req=summary['requirement']
            event_counts=Counter(e for r in cohort for e in {v for a in r['motion_event_ids'].values() for v in a})
            if (Counter(r['geometry_id'] for r in cohort)!=summary['geometry_counts']
                or Counter(r['episode_key'] for r in cohort)!=summary['episode_counts']
                or max(Counter(r['episode_key'] for r in cohort).values())>req['max_episode_windows']
                or max(event_counts.values(),default=0)>req['max_windows_per_event']):
                raise ValueError('Motion coverage counts/caps differ')
            for kind,report in summary['categories'].items():
                motion=[r for r in cohort if r['motion_event_ids'][kind]]
                actual=[len(motion),len({e for r in motion for e in r['motion_event_ids'][kind]}),
                    len({r['geometry_id'] for r in motion}),len({r['episode_key'] for r in motion})]
                if (actual!=[report[k] for k in ('selected_windows','selected_unique','geometry_count','episode_count')]
                    or any(a<b for a,b in zip(actual,[req['minimum_windows'][kind],req['minimum_unique'][kind],
                        req['minimum_motion_geometries'],req['minimum_motion_episodes']]))):
                    raise ValueError('Motion event coverage differs')
    for left,right in combinations(('train','validation','test'),2):
        for field in ('episode_key','geometry_id'):
            if {r[field] for r in rows if r['split']==left}&{r[field] for r in rows if r['split']==right}:
                raise ValueError('Motion geometry/episode leakage')
    source=file_identity(m['source_registry']['location'])
    if source['sha256']!=m['source_registry']['sha256']: raise ValueError('Source registry integrity failure')
    registry=read(source['location']); population=cur['source_population']; summary=population['summary']
    if (summary['pending_jobs'] or summary['processed_jobs']!=summary['planned_jobs']
        or summary['accepted_jobs']+summary['rejected_jobs']!=summary['planned_jobs']
        or registry['schema_version']!='sumodiff.sources.v2'):
        raise ValueError('Incomplete Motion source ledger')
    ledger=registry['job_ledger']
    accepted=[r for r in ledger if r['outcome']=='accepted']
    if (len(ledger)!=summary['planned_jobs'] or len({r['job_id'] for r in ledger})!=len(ledger)
        or len(accepted)!=summary['accepted_jobs'] or sum(r['outcome']=='rejected' for r in ledger)!=summary['rejected_jobs']
        or {r['job_id'] for r in accepted}!=set(population['accepted_job_ids'])
        or registry['quality_policy']!=population['quality_policy']
        or any(r['source_manifest'] not in {v['episode_manifest']['location'] for v in accepted} for r in rows)):
        raise ValueError('Motion accepted/rejected ledger population differs')
    return dict(selection=identity,source_registry=source,source_population=population)


def calibration_evidence(c,architecture,directory,scale):
    """Use the measured objective, not the diagnostic 16-update learning-rate schedule."""
    path=Path(c['calibration_config']).resolve(strict=True); root=path.parent
    selected=read(root/'results.json'); manifest=read(root/'manifest.json')
    status=read(root/'status.json'); effective=load_config(root/'resolved_config.yaml')
    identity=file_identity(path)
    if (manifest['purpose']!='motion_control_calibration' or not manifest['formal'] or manifest['git']['dirty']
        or status['state']!='completed' or not selected['passed'] or not selected['numerically_finite']
        or selected['training_config']!=identity or selected['scale_audit']['sha256']!=scale['sha256']
        or selected['data_id']!=c['expected_data_id'] or selected!=read(root/'metrics.json')
        or file_identity(root/'resolved_config.yaml')['sha256']!=manifest['resolved_config_sha256']):
        raise ValueError('Completed selected Motion calibration required')
    from .control_training import control_config
    old=control_config(load_config(path)); suite=effective['suite']
    for key in ('diffusion','unified_loss','jerk_tail','tail_weight','batch_size','gradient_clip_norm','weight_decay','seed','noise_seed','validation_seed'):
        if old[key]!=c[key]: raise ValueError(f'Formal objective differs from calibrated {key}')
    if resolve_model_config(load_config(suite['model_config'])).to_dict()!=architecture.to_dict():
        raise ValueError('Calibrated model architecture differs')
    if suite['peak_lr']!=c['peak_learning_rate'] or suite['minimum_lr']!=c['learning_rate']:
        raise ValueError('Formal learning-rate bounds differ from preflight')
    from sumodiff.data.control_tasks import verify_motion_task_plan
    task=read(c['monitor']['task_plan']); verify_motion_task_plan(task,directory)
    if file_identity(c['monitor']['task_plan'])!=selected['task_plan']:
        raise ValueError('Monitoring tasks differ from certified calibration tasks')
    gradient=file_identity(selected['selected_gradient']['location'])
    if gradient!=selected['selected_gradient']: raise ValueError('Calibrated gradient report changed')
    report=read(gradient['location'])
    if (not report['numerically_finite'] or report['optimizer_updates'] or report['test_arrays_loaded']
        or selected['selected_max_combined_ratio']>suite['auxiliary_ratio_budget']
        or any(r['combination_relative_residual']>1e-4 for r in report['rows'])):
        raise ValueError('Selected gradient budget or numerical check failed')
    return dict(config=identity,result=file_identity(root/'results.json'),
        manifest=file_identity(root/'manifest.json'),status=file_identity(root/'status.json'),
        resolved=file_identity(root/'resolved_config.yaml'),gradient=gradient,task_plan=selected['task_plan'])


class WindowStream:
    """Bounded CPU LRU, per-batch lane padding; maps/targets hashed on every cache miss."""
    def __init__(self,directory,split,architecture,capacity):
        if split not in ('train','validation'): raise ValueError('Training stream cannot read test labels')
        self.ds=WindowDataset(directory,split,core_only=True)
        self.ids=[r['window_id'] for r in self.ds.entries]
        self.architecture=architecture;self.capacity=capacity;self.cache=OrderedDict()
        self.roads=OrderedDict();self.loaded=set();self.repairs=[]
    def batch(self,indices,device):
        samples=[]
        for index in list(map(int,indices)):
            if index not in self.cache:
                self.cache[index]=self.ds[index]; self.loaded.add(index)
                if len(self.cache)>self.capacity:self.cache.popitem(last=False)
            self.cache.move_to_end(index); samples.append(self.cache[index])
        batch=collate_numpy(samples)
        conditions=prepare_conditioning(batch,device,config=self.architecture)
        targets={k:torch.as_tensor(v,device=device) for k,v in batch['targets'].items()}
        if not targets['future_mask'][conditions['agent_mask']].all(): raise ValueError('Incomplete core labels')
        return conditions,targets,[s['input_metadata']['window_id'] for s in samples],samples
    def road_batch(self,ids,samples,config):
        result=[]
        for wid,sample in zip(ids,samples):
            if wid not in self.roads:
                geometry,repairs=prepare_roads(sample['exact_map'],config)
                self.roads[wid]=geometry
                # Repair details are written per update by the caller, rather than growing unbounded.
                self.repairs.extend(dict(window_id=wid,**r) for r in repairs)
                if len(self.roads)>self.capacity:self.roads.popitem(last=False)
            self.roads.move_to_end(wid); result.append(self.roads[wid])
        return result


@torch.no_grad()
def stream_probe(model,stream,schedule,device,batch_size,seed,indices=None):
    from .training import draw_noise
    was_training=model.training;model.eval(); generator=torch.Generator().manual_seed(seed)
    indices=list(range(len(stream.ids))) if indices is None else list(indices)
    total=0.
    try:
        for start in range(0,len(indices),batch_size):
            chosen=indices[start:start+batch_size]
            conditions,targets,_,_=stream.batch(chosen,device)
            noisy,noise,timestep=draw_noise(targets,conditions,schedule,generator)
            loss=masked_noise_mse(model(noisy,timestep,conditions),noise,conditions['agent_mask'],targets['future_mask'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite validation')
            total+=float(loss)*len(chosen)
    finally:model.train(was_training)
    return total/len(indices)


def validate_runtime(payload):
    m=payload['metadata'];c=m['training_config'];runtime=m['runtime']
    if c!=epoch_config(c) or m['training_signature']['training']!=c or m['diffusion_config']!=c['diffusion']:
        raise ValueError('Control epoch checkpoint recipe mismatch')
    if not 1<=payload['step']<=c['max_steps']:raise ValueError('Invalid control epoch checkpoint step')
    schedule=UpdateSchedule(ScheduleConfig(**c['schedule']))
    schedule.load_state_dict(runtime['update_schedule'],expected_updates=payload['step'])
    if (not payload['optimizer']['state'] or any(int(s['step'])!=payload['step'] for s in payload['optimizer']['state'].values())
        or any(g['lr']!=schedule.last_lr for g in payload['optimizer']['param_groups'])):
        raise ValueError('Control epoch AdamW/schedule mismatch')
    best=runtime['best_validation']
    if (set(best)!={'step','mse','checkpoint'} or not 0<best['step']<=payload['step']
        or not math.isfinite(best['mse']) or best['mse']<0 or not isinstance(best['checkpoint'],str)):
        raise ValueError('Invalid best-validation continuation state')
    pending=runtime['pending_monitor_epoch'];done=runtime['monitored_through_epoch']
    epoch=payload['step']//c['steps_per_epoch']
    if type(done) is not int or not 0<=done<=epoch:
        raise ValueError('Invalid completed monitor state')
    owed=(epoch//c['monitor_every_epochs'])*c['monitor_every_epochs']
    if epoch==c['epochs']:owed=epoch
    if (pending is None and done!=owed) or (pending is not None and (pending!=owed or done>=owed)):
        raise ValueError('Checkpoint skipped an owed periodic monitor')
    if pending is not None and (type(pending) is not int or pending!=epoch or done>=pending
        or payload['step']%c['steps_per_epoch'] or (pending%c['monitor_every_epochs'] and pending!=c['epochs'])):
        raise ValueError('Invalid pending monitor state')


def train(dataset_path,model_config_path,config_path,scale_audit_path,output,repository,command,
          formal=False,resume=None,allow_long_run=False,prefix_updates=None):
    from .training import draw_noise,synchronize
    from sumodiff.experiments.recorder import _write_json
    c=epoch_config(load_config(config_path))
    if not allow_long_run: raise ValueError('Control epoch training requires explicit --allow-long-run')
    if prefix_updates is not None and (type(prefix_updates) is not int or not 1<=prefix_updates<=64):
        raise ValueError('Prefix checks permit 1..64 updates without changing the 200-epoch schedule')
    architecture=resolve_model_config(load_config(model_config_path)); diffusion=resolve_diffusion_config(c['diffusion'])
    validate_architecture(architecture,diffusion)
    if architecture.schema_version!='sumodiff.model.control.config.v2': raise ValueError('Directional control model v2 required')
    data=dataset_identity(dataset_path); selection=verified_dataset(dataset_path,c)
    scale=verified_control_scales(scale_audit_path,dataset_path,diffusion)
    evidence=calibration_evidence(c,architecture,dataset_path,scale)
    train_stream=WindowStream(dataset_path,'train',architecture,c['stream_cache_windows'])
    val_stream=WindowStream(dataset_path,'validation',architecture,c['stream_cache_windows'])
    if len(train_stream.ids)!=c['expected_train_windows'] or len(val_stream.ids)!=c['expected_validation_windows']:
        raise ValueError('Full training/validation stream counts differ')
    signature=dict(model=architecture.to_dict(),training=c,dataset_hashes={k:v['sha256'] for k,v in data.items()},
        scale_audit_sha256=scale['sha256'],calibration_hashes={k:v['sha256'] for k,v in evidence.items()},
        selection_sha256=selection['selection']['sha256'],train_window_ids=train_stream.ids,validation_window_ids=val_stream.ids)
    parent=load_checkpoint(resume,signature=signature) if resume else None
    start_step=parent['step'] if parent else 0
    if start_step>=c['max_steps']: raise ValueError('Training budget already complete')
    stop_step=min(c['max_steps'],start_step+prefix_updates) if prefix_updates else c['max_steps']
    files=[model_config_path,config_path,scale_audit_path,*[v['location'] for v in data.values()],
        *[v['location'] for v in evidence.values()],selection['selection']['location'],selection['source_registry']['location'],
        *[v['location'] for v in scale['source_files']]]
    for stream in (train_stream,val_stream):
        files.extend(stream.ds._path(entry,name) for entry in stream.ds.entries for name in entry['files'])
    effective=dict(schema_version='sumodiff.control.epoch.run.v1',training=c,model=architecture.to_dict(),data=data,
        scale_audit=scale,calibration=evidence,selection=selection,resume=file_identity(resume) if resume else None,
        prefix_updates=prefix_updates,train_window_ids=train_stream.ids,validation_window_ids=val_stream.ids,
        precision='float32; strict determinism; no AMP/TF32',input_pipeline='bounded CPU LRU; batch-local lane padding; synchronous')
    configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    with RunRecorder(output,repository,effective,command,dict(model=c['seed'],noise=c['noise_seed'],validation=c['validation_seed']),
        purpose='motion_control_epoch_training',data_files=sorted(set(map(str,files))),data_id=c['expected_data_id'],checkpoint=resume,formal=formal) as run:
        model=ConditionalDenoiser(architecture).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        generator=torch.Generator().manual_seed(c['noise_seed']);schedule=NoiseSchedule(diffusion,device)
        updates=UpdateSchedule(ScheduleConfig(**c['schedule']))
        runtime=dict(torch=str(torch.__version__),device=str(device),strict_determinism=True,network_dtype='float32',
            update_schedule=updates.state_dict(),best_validation=None,pending_monitor_epoch=None,monitored_through_epoch=0)
        if parent:
            restore_training(parent,model,optimizer,generator,device)
            runtime=deepcopy(parent['metadata']['runtime'])
            updates.load_state_dict(runtime['update_schedule'],optimizer,start_step)
            best_path=runtime['best_validation']['checkpoint']
            best_payload=load_checkpoint(best_path,signature=signature)
            if best_payload['step']!=runtime['best_validation']['step']:raise ValueError('Best checkpoint changed/missing')
            del best_payload
        metadata=dict(model_config=architecture.to_dict(),diffusion_config=diffusion.to_dict(),training_config=c,
            training_signature=signature,data=data,scale_audit=scale,run_sha=run.manifest['git']['commit_sha'],
            run_branch=run.manifest['git']['branch'],run_dirty=run.manifest['git']['dirty'],
            parent_checkpoint=file_identity(resume) if resume else None,runtime=runtime)
        initial_hash=parameter_hash(model);records=[];validation=[];monitors=[];timings=[]
        peak_allocated=0;peak_reserved=0;clip_count=0;rss_peak=psutil.Process().memory_info().rss
        probe_ids=sorted(range(len(train_stream.ids)),key=lambda i:__import__('hashlib').sha256(
            f"probe:{c['validation_seed']}:{train_stream.ids[i]}".encode()).hexdigest())[:c['train_probe_limit']]
        def validate(step):
            record=dict(step=step,epoch=step/c['steps_per_epoch'],
                train_probe_windows=len(probe_ids),validation_windows=len(val_stream.ids),
                train_probe_mse=stream_probe(model,train_stream,schedule,device,c['batch_size'],c['validation_seed'],probe_ids),
                validation_probe_mse=stream_probe(model,val_stream,schedule,device,c['batch_size'],c['validation_seed']))
            validation.append(record);_write_json(run.output/'validation.json',validation)
            print(f"validation step={step}/{c['max_steps']} mse={record['validation_probe_mse']:.7f}",flush=True)
            return record
        def save(step,suffix=''):
            path=run.output/f'checkpoint_step_{step:06d}{suffix}.pt'
            save_checkpoint(path,model,optimizer,generator,step,metadata)
            identity=file_identity(path);records.append(identity)
            _write_json(run.output/'latest_checkpoint.json',dict(step=step,checkpoint=identity))
            best=runtime['best_validation']
            _write_json(run.output/'best_validation_checkpoint.json',dict(**best,identity=file_identity(best['checkpoint']),
                selection='minimum fixed-noise validation MSE; candidate only, confirm generated quality on validation'))
            return path
        def monitor_pending(checkpoint):
            epoch=runtime['pending_monitor_epoch']
            if epoch is None:return checkpoint
            monitors.append(periodic_sample(dataset_path,checkpoint,run,repository,c,epoch,formal))
            _write_json(run.output/'generation_monitor.json',monitors)
            runtime['pending_monitor_epoch']=None;runtime['monitored_through_epoch']=epoch
            return save(int(epoch*c['steps_per_epoch']),'_monitored')
        if runtime['pending_monitor_epoch'] is not None:monitor_pending(resume)
        initial_probe_started=time.perf_counter();validate(start_step)
        initial_probe_seconds=time.perf_counter()-initial_probe_started
        started=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        loss_config=resolve_unified_loss(c['unified_loss'])
        with (run.output/'steps.jsonl').open('w',encoding='utf-8') as log:
            for step in range(start_step+1,stop_step+1):
                synchronize(device);tick=time.perf_counter()
                indices,epoch,batch_index=epoch_batch(step,len(train_stream.ids),c['batch_size'],c['seed'])
                conditions,targets,ids,samples=train_stream.batch(indices.tolist(),device)
                model.train();optimizer.zero_grad(set_to_none=True)
                noisy,noise,timestep=draw_noise(targets,conditions,schedule,generator)
                epsilon=model(noisy,timestep,conditions);lr,ramp=updates.apply(optimizer,step)
                # Static road queries are needed only when a scene has an eligible auxiliary timestep.
                roads=train_stream.road_batch(ids,samples,loss_config) if (timestep<=loss_config.max_auxiliary_timestep).any() else None
                components=control_objective(epsilon,noisy,noise,timestep,conditions,targets,schedule,c,roads=roads,ramp=ramp)
                loss=components['total_loss']
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step();updates.complete(step);runtime['update_schedule']=updates.state_dict()
                synchronize(device);elapsed=time.perf_counter()-tick;timings.append(elapsed)
                rss_peak=max(rss_peak,psutil.Process().memory_info().rss)
                clipped=float(norm)>c['gradient_clip_norm'];clip_count+=int(clipped)
                row=dict(step=step,epoch=epoch,epoch_batch=batch_index,epoch_complete=batch_index==c['steps_per_epoch'],
                    window_ids=ids,timesteps=timestep.cpu().tolist(),batch_size=len(ids),seconds=elapsed,
                    learning_rate=lr,auxiliary_ramp=ramp,noise_mse=float(components['noise_mse']),
                    total_loss=float(loss),gradient_norm_before_clip=float(norm),clipping_applied=clipped,
                    loss_components={k:float(v.detach()) for k,v in components.items() if isinstance(v,torch.Tensor) and v.ndim==0},
                    road_repairs=train_stream.repairs)
                train_stream.repairs=[]
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if device.type=='cuda':
                    peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated(device))
                    peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved(device))
                epoch_end=batch_index==c['steps_per_epoch']
                validate_due=(epoch_end and epoch%c['validation_every_epochs']==0) or step==stop_step
                best_changed=False
                if validate_due:
                    latest=validate(step)
                    if runtime['best_validation'] is None or latest['validation_probe_mse']<runtime['best_validation']['mse']:
                        runtime['best_validation']=dict(step=step,mse=latest['validation_probe_mse'],
                            checkpoint=str(run.output/f'checkpoint_step_{step:06d}.pt'))
                        best_changed=True
                monitor_due=epoch_end and (epoch%c['monitor_every_epochs']==0 or step==c['max_steps'])
                if monitor_due:runtime['pending_monitor_epoch']=epoch
                if best_changed or (epoch_end and epoch%c['checkpoint_every_epochs']==0) or step==stop_step or monitor_due:
                    # A checkpoint must carry an evaluated candidate even for unusual checkpoint/validation schedules.
                    if runtime['best_validation'] is None:
                        latest=validate(step);runtime['best_validation']=dict(step=step,mse=latest['validation_probe_mse'],
                            checkpoint=str(run.output/f'checkpoint_step_{step:06d}.pt'))
                    path=save(step)
                    if monitor_due:monitor_pending(path)
                if step%25==0 or validate_due:
                    _write_json(run.output/'progress.json',dict(step=step,total_steps=c['max_steps'],epoch=epoch,total_epochs=c['epochs'],
                        learning_rate=lr,auxiliary_ramp=ramp,last_validation=validation[-1],best_validation=runtime['best_validation']))
                    print(f"epoch={epoch}/{c['epochs']} batch={batch_index}/{c['steps_per_epoch']} step={step} loss={float(loss):.6f} lr={lr:.3g}",flush=True)
        timed=timings[1:] or timings;mean=float(np.mean(timed));final_hash=parameter_hash(model)
        result=dict(schema_version='sumodiff.control.epoch.metrics.v1',run_sha=metadata['run_sha'],
            data_id=c['expected_data_id'],representation='control_a_omega_r',start_step=start_step,end_step=stop_step,
            updates=len(timings),nominal_total_updates=c['max_steps'],complete_budget=stop_step==c['max_steps'],
            train_windows=len(train_stream.ids),validation_windows=len(val_stream.ids),batch_size=c['batch_size'],
            epoch_schedule=dict(total_epochs=c['epochs'],steps_per_epoch=c['steps_per_epoch'],each_epoch_without_replacement=True,last_partial_batch_retained=True),
            unique_training_windows_loaded=len(train_stream.loaded),unique_validation_windows_loaded=len(val_stream.loaded),
            training_validation_checkpoint_wall_seconds=time.perf_counter()-started,validation=validation,
            initial_parameter_sha256=initial_hash,final_parameter_sha256=final_hash,weight_update_verified=final_hash!=initial_hash,
            initial_full_validation_and_train_probe_seconds=initial_probe_seconds,
            sampled_process_peak_rss_bytes=rss_peak,process_memory_scope='RSS sampled after initialization and optimizer updates; not continuous peak',
            clipped_updates=clip_count,optimizer_step_timing=dict(mean_seconds=mean,median_seconds=float(np.median(timed)),
                warmup_updates_excluded=min(1,len(timings)),includes='synchronous file reads/hash/LRU, CPU batch padding/conditioning, device transfer, road geometry, forward/backward/clip/AdamW; excludes validation/checkpoint/monitor'),
            peak_allocated_bytes=peak_allocated if device.type=='cuda' else None,peak_reserved_bytes=peak_reserved if device.type=='cuda' else None,
            checkpoints=records,best_validation=runtime['best_validation'],generation_monitor=monitors,
            convergence_status='not_assessed',test_arrays_loaded=0,
            limitations=['initial budget does not establish convergence','MSE-best is a candidate, not generated-quality selection',
                'SUMO labels do not certify real driving realism','short prefixes do not certify long-run stability or quality'])
        write_json(run.output/'training_summary.json',result);run.write_metrics(result)
    return result
