"""Label-free fixed-horizon base generation with separate raw/decoded metrics."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time
import numpy as np
import torch
from sumodiff.data.dataset import WindowDataset,collate_numpy,write_json
from sumodiff.models import ConditionalDenoiser,resolve_model_config,prepare_conditioning
from sumodiff.models.cli import configure
from sumodiff.decoding import DecoderConfig,decode_future
from sumodiff.evaluation.config import resolve_config as evaluation_config
from sumodiff.evaluation.motion import MotionConfig
from sumodiff.evaluation.roads import RoadConfig
from sumodiff.evaluation.collisions import CollisionConfig
from sumodiff.evaluation.metrics import _heading,applicability,evaluate_trajectory,aggregate_scenes,correction_summary
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from .config import resolve_diffusion_config
from .checkpoint import load_checkpoint
from .data import choose_indices,input_files,dataset_identity
from .process import NoiseSchedule,ddim_sample,denormalize
from .training import synchronize

SAMPLE_DEFAULTS=dict(schema_version='sumodiff.base.sample.config.v1',seed=20261008,device='cuda:0',torch_threads=1,
    split='validation',limit=6,minimum_agents=2,steps=20,eta=0.0,mode='base',cfg=False,partial_diffusion=False,rolling_generation=False,evaluation={})


def sample_config(value):
    if not isinstance(value,dict) or set(value)-set(SAMPLE_DEFAULTS): raise ValueError('Unknown sample options')
    c=deepcopy(SAMPLE_DEFAULTS);c.update(value)
    if c['schema_version']!=SAMPLE_DEFAULTS['schema_version'] or c['device'] not in ('cpu','cuda:0'): raise ValueError('Unsupported sample schema/device')
    if c['split'] not in ('train','validation','test'): raise ValueError('Invalid sampling split')
    if c['mode']!='base': raise NotImplementedError('Guided/diffscene_style sampling is stage6, not implemented')
    for key in ('cfg','partial_diffusion','rolling_generation'):
        if type(c[key]) is not bool: raise ValueError(f'Invalid {key}')
        if c[key]: raise NotImplementedError(f'{key} is not supported')
    for key in ('torch_threads','minimum_agents','steps'):
        if type(c[key]) is not int or c[key]<1: raise ValueError(f'Invalid {key}')
    if c['limit'] is not None and (type(c['limit']) is not int or c['limit']<1): raise ValueError('Invalid task limit')
    if type(c['seed']) is not int or not 0<=c['seed']<2**32: raise ValueError('Invalid sampling seed')
    if type(c['eta']) not in (float,int) or not math.isfinite(c['eta']) or not 0<=c['eta']<=1: raise ValueError('Invalid eta')
    c['evaluation']=evaluation_config(c['evaluation'])
    return c


def task_seed(seed,window_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{window_id}'.encode()).digest()[:8],'little')%(2**63-1)


def failed_report(meta,error):
    agents=np.asarray([v is not None for v in meta['agent_ids']],bool)
    return dict(schema_version='sumodiff.scene.metrics.v1',status='failed',window_id=meta['window_id'],
        planned_active_agents=int(agents.sum()),excluded_vehicle_count=meta['excluded_vehicle_count'],task_applicability=applicability(agents,None),
        quality_pass=None,effective_target_event=None,error=dict(type=type(error).__name__,message=str(error)))


def sample(dataset_path,checkpoint,config_path,output,repository,command,formal=False):
    c=sample_config(load_config(config_path));payload=load_checkpoint(checkpoint,requested={k:c[k] for k in ('cfg','partial_diffusion','rolling_generation')})
    architecture=resolve_model_config(payload['metadata']['model_config']);diffusion=resolve_diffusion_config(payload['metadata']['diffusion_config'])
    ds=WindowDataset(dataset_path,c['split']);indices=choose_indices(ds,c['limit'],minimum_agents=c['minimum_agents'])
    if not indices: raise ValueError('No eligible current-state sampling tasks')
    tasks=[ds._json(ds.entries[i],'input.json') for i in indices]
    data=dataset_identity(dataset_path)
    if data['manifest']['sha256']!=payload['metadata']['data']['manifest']['sha256']: raise ValueError('Sampling dataset/checkpoint mismatch; cross-dataset inference needs an explicit later protocol')
    effective=dict(schema_version='sumodiff.base.sample.run.v1',sampling=c,model=architecture.to_dict(),diffusion=diffusion.to_dict(),data=data,
        checkpoint=file_identity(checkpoint),selection='input-family round robin after t0 minimum-agent filter; never future completeness',
        task_window_ids=[t['window_id'] for t in tasks],checkpoint_training_sha=payload['metadata']['run_sha'],target_pair=None,
        precision='float32 model, float64 decoder solve; strict determinism, no AMP/TF32')
    files=[config_path,*[v['location'] for v in data.values()],*input_files(ds,indices,False)]
    with RunRecorder(output,repository,effective,command,{'initial_noise':c['seed']},purpose='stage5_offline_base_generation',
        data_files=sorted(set(map(str,files))),data_id=data['manifest']['id'],checkpoint=checkpoint,formal=formal) as run:
        configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
        if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
        model=ConditionalDenoiser(architecture).to(device);model.load_state_dict(payload['model'],strict=True);model.eval();model.requires_grad_(False)
        del payload
        schedule=NoiseSchedule(diffusion,device);schedule.timesteps(c['steps'])
        write_json(run.output/'tasks.json',[dict(window_id=t['window_id'],agent_ids=t['agent_ids'],target_pair=None,noise_seed=task_seed(c['seed'],t['window_id'])) for t in tasks])
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        raw_records=[];decoded_records=[];rows=[];durations=[];corrections=[];evaluation_seconds=0.;generation_failures=0;started=time.perf_counter()
        kwargs=dict(motion_config=MotionConfig(**c['evaluation']['motion']),road_config=RoadConfig(**c['evaluation']['road']),collision_config=CollisionConfig(**c['evaluation']['collision']))
        for i,meta in zip(indices,tasks):
            row_dir=run.output/'trajectories'/meta['window_id'];row_dir.mkdir(parents=True,exist_ok=False)
            row=dict(window_id=meta['window_id'],family=meta['family'],generation_status='failed',guided_stage=dict(status='not_applicable',reason='base mode has no guidance'))
            arrays={}
            try:
                source=ds.inference(i);conditions=source['conditioning'];heading=_heading(conditions)
                tensors=prepare_conditioning(collate_numpy([source]),device)
                generator=torch.Generator().manual_seed(task_seed(c['seed'],meta['window_id']))
                initial=torch.randn((1,len(conditions['agent_mask']),40,6),generator=generator).to(device)
                arrays['initial_noise']=initial.cpu().numpy()[0]
                synchronize(device);tick=time.perf_counter()
                normalized=ddim_sample(model,tensors,initial,schedule,c['steps'],c['eta'],generator=torch.Generator(device=device).manual_seed(task_seed(c['seed'],meta['window_id'])))
                physical=denormalize(normalized,diffusion)[0]
                decoded=decode_future(physical,tensors['initial_positions'][0],torch.as_tensor(heading,device=device),tensors['agent_mask'][0],DecoderConfig(**c['evaluation']['decoder']))
                synchronize(device);duration=time.perf_counter()-tick;durations.append(duration)
                row.update(generation_status='completed',sampling_decoder_seconds=duration,denoiser_calls=c['steps'],actual_gradient_calls=0)
                arrays.update(raw_normalized_future=normalized[0].cpu().numpy(),raw_future_delta=physical.cpu().numpy(),
                    raw_absolute_states=decoded.raw_states.cpu().numpy(),decoded_absolute_states=decoded.states.cpu().numpy(),
                    initial_positions=conditions['initial_positions'],initial_heading=heading,agent_mask=conditions['agent_mask'],
                    position_correction=decoded.position_correction.cpu().numpy(),velocity_correction=decoded.velocity_correction.cpu().numpy(),
                    heading_correction=decoded.heading_correction.cpu().numpy(),heading_fallback_mask=decoded.heading_fallback_mask.cpu().numpy(),
                    raw_velocity_residual=decoded.raw_velocity_residual.cpu().numpy())
                row['corrections']=correction_summary(decoded);corrections.append(row['corrections'])
                tick=time.perf_counter()
                for stage in ('raw','decoded'):
                    try:
                        report=evaluate_trajectory(conditions,meta,source['exact_map'],arrays[f'{stage}_absolute_states'],initial_heading=heading,**kwargs)
                    except Exception as exc: report=failed_report(meta,exc)
                    row[stage]=report
                evaluation_seconds+=time.perf_counter()-tick
            except Exception as exc:
                generation_failures+=1;row['raw']=failed_report(meta,exc);row['decoded']=failed_report(meta,exc)
                row['generation_error']=dict(type=type(exc).__name__,message=str(exc))
            raw_records.append(row['raw']);decoded_records.append(row['decoded'])
            if arrays: np.savez_compressed(row_dir/'stages.npz',**arrays)
            write_json(row_dir/'metrics.json',row)
            rows.append(dict(window_id=meta['window_id'],family=meta['family'],generation_status=row['generation_status'],
                files={p.name:file_identity(p) for p in row_dir.iterdir() if p.is_file()}))
            print(f"sample={len(rows)}/{len(tasks)} generation={row['generation_status']} quality={row['decoded'].get('quality_pass')}",flush=True)
        result=dict(schema_version='sumodiff.base.sample.metrics.v1',run_sha=run.manifest['git']['commit_sha'],checkpoint_training_sha=effective['checkpoint_training_sha'],
            planned_tasks=len(tasks),generation_failures=generation_failures,raw=aggregate_scenes(raw_records),decoded=aggregate_scenes(decoded_records),
            selected_agent_windows=sum(r['planned_active_agents'] for r in decoded_records),excluded_current_vehicle_windows=sum(t['excluded_vehicle_count'] for t in tasks),
            denoiser_calls_per_completed_scene=c['steps'],actual_gradient_calls=0,steps=c['steps'],eta=c['eta'],
            sampling_decoder_seconds=dict(count=len(durations),mean=float(np.mean(durations)) if durations else None,p99=float(np.quantile(durations,.99)) if durations else None,
                includes='conditioning encoder, DDIM and unified decoder; excludes input IO, independent metrics and output IO'),
            independent_evaluation_seconds=evaluation_seconds,total_loop_wall_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
            max_position_correction_m=max((r['position_correction_m']['max'] for r in corrections),default=None),
            max_velocity_correction_mps=max((r['velocity_correction_mps']['max'] for r in corrections),default=None),
            heading_fallback_count=sum(r['heading_fallback_count'] for r in corrections),
            limitations=['offline selected vehicles only','checkpoint training budget does not establish convergence or quality','no attack roles, target event N/A','failed and unqualified scenes retained'])
        write_json(run.output/'trajectory_index.json',rows);write_json(run.output/'sampling_summary.json',result);run.write_metrics(result)
        if generation_failures or result['raw']['failed_scenes'] or result['decoded']['failed_scenes']: raise RuntimeError('Base sampling/evaluation failures retained in metrics')
    return result
