"""Bounded, fixed-data diagnostics; reconstruction and free generation stay separate."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from sumodiff.data.dataset import WindowDataset, write_json
from sumodiff.models import ConditionalDenoiser, resolve_model_config
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.decoding import DecoderConfig, decode_future
from sumodiff.experiments.recorder import RunRecorder, file_identity, load_config
from .checkpoint import load_checkpoint
from .config import resolve_diffusion_config
from .data import dataset_identity, labeled_cache, select_batch, choose_indices, input_files
from .process import NoiseSchedule, normalize, denormalize
from .sampling import sample, sample_config, task_seed

DEFAULTS=dict(schema_version='sumodiff.training.diagnostic.config.v1',noise_seed=20261009,
    timesteps=[0,50,100,250,500,750,900,950,999],noise_repeats=3,
    sampling_steps=[20],free_sampling={})


def diagnostic_config(value):
    if not isinstance(value,dict) or set(value)-set(DEFAULTS): raise ValueError('Unknown diagnostic options')
    c=deepcopy(DEFAULTS);c.update(value)
    if c['schema_version']!=DEFAULTS['schema_version']: raise ValueError('Unsupported diagnostic schema')
    if type(c['noise_seed']) is not int or not 0<=c['noise_seed']<2**32: raise ValueError('Invalid diagnostic noise seed')
    if type(c['noise_repeats']) is not int or not 1<=c['noise_repeats']<=5: raise ValueError('Diagnostic repeats must be 1..5')
    for key in ('timesteps','sampling_steps'):
        if not isinstance(c[key],list) or not c[key] or len(set(c[key]))!=len(c[key]) or any(type(t) is not int for t in c[key]): raise ValueError(f'Invalid {key}')
    if any(not 0<=t<1000 for t in c['timesteps']): raise ValueError('Diagnostic timesteps must be 0..999')
    if any(t not in (20,50,100) for t in c['sampling_steps']): raise ValueError('Only bounded 20/50/100 sampling comparison is supported')
    c['free_sampling']=sample_config(c['free_sampling'])
    if c['free_sampling']['limit'] is None or c['free_sampling']['limit']>12: raise ValueError('Free diagnostic tasks require a limit <=12')
    return c


def channel_mse(prediction,target,mask):
    """Equal scene weight, physical/state channels separately, no padded NaNs."""
    if prediction.shape!=target.shape or prediction.ndim!=4 or mask.shape!=prediction.shape[:3] or mask.dtype!=torch.bool: raise ValueError('Invalid diagnostic shapes')
    counts=mask.sum((1,2))
    if (counts==0).any(): raise ValueError('No valid diagnostic elements')
    p=torch.where(mask[...,None],prediction,0.).double();q=torch.where(mask[...,None],target,0.).double()
    if not torch.isfinite(p).all() or not torch.isfinite(q).all(): raise FloatingPointError('Nonfinite diagnostic output')
    return (p-q).square().sum((1,2))/counts[:,None]


def noise_for_ids(ids,shape,seed,repeat):
    return torch.stack([torch.randn(shape,generator=torch.Generator().manual_seed(task_seed(seed,f'{window_id}:repeat{repeat}'))) for window_id in ids])


@torch.no_grad()
def labeled_probe(model,cache,schedule,c,device,batch_size,output,split):
    rows=[];channels=['delta_x','delta_y','vx','vy','sin_heading','cos_heading']
    for timestep in c['timesteps']:
        epsilon_errors=[];state_errors=[];saved=[]
        for start in range(0,len(cache[2]),batch_size):
            indices=torch.arange(start,min(start+batch_size,len(cache[2])))
            conditions,targets,ids=select_batch(cache,indices,device)
            clean=normalize(targets['future'],schedule.config)
            valid=conditions['agent_mask'][...,None]&targets['future_mask']
            condition=model.condition_encoder(conditions)
            t=torch.full((len(ids),),timestep,dtype=torch.long,device=device)
            for repeat in range(c['noise_repeats']):
                noise=noise_for_ids(ids,clean.shape[1:],c['noise_seed'],repeat).to(device)
                noisy=schedule.add_noise(clean,noise,t)
                noisy=torch.where(conditions['agent_mask'][:,:,None,None],noisy,0.)
                prediction=model.unet(noisy,t,condition,conditions['agent_mask'])
                recovered=schedule.clean_from_noise(noisy,prediction,t)
                physical=denormalize(recovered,schedule.config)
                epsilon_errors.append(channel_mse(prediction,noise,valid).cpu())
                state_errors.append(channel_mse(physical,targets['future'],valid).cpu())
                if repeat==0:
                    saved.append((indices.numpy(),noisy.cpu().numpy(),prediction.cpu().numpy(),physical.cpu().numpy()))
        e=torch.cat(epsilon_errors).mean(0);s=torch.cat(state_errors).mean(0)
        a=float(schedule.alpha_bar[timestep]);factor=((1-a)/a)**.5
        row=dict(timestep=timestep,scenes=len(cache[2]),noise_repeats=c['noise_repeats'],epsilon_mse=float(e.mean()),
            epsilon_channel_mse=dict(zip(channels,e.tolist())),physical_channel_mse=dict(zip(channels,s.tolist())),
            position_vector_rmse_m=float((s[0]+s[1]).sqrt()),velocity_vector_rmse_mps=float((s[2]+s[3]).sqrt()),
            heading_vector_rmse=float((s[4]+s[5]).sqrt()),epsilon_error_amplification=factor)
        rows.append(row)
        # Arrays are labeled diagnostic reconstructions, never generated scenes.
        np.savez_compressed(output/f'{split}_reconstruction_t{timestep:04d}.npz',
            window_ids=np.asarray(cache[2]),future_label=cache[1]['future'].numpy(),future_label_mask=cache[1]['future_mask'].numpy(),
            noisy_normalized=np.concatenate([x[1] for x in saved]),pred_noise=np.concatenate([x[2] for x in saved]),
            reconstructed_physical_delta=np.concatenate([x[3] for x in saved]))
    return rows


def oracle_probe(cache,schedule,decoder):
    future=cache[1]['future'].double();mask=cache[0]['agent_mask'];clean=normalize(future,schedule.config)
    noise=torch.randn(clean.shape,dtype=torch.float64,generator=torch.Generator().manual_seed(1729))
    cpu_schedule=NoiseSchedule(schedule.config);errors=[]
    for time in (0,500,999):
        t=torch.full((len(clean),),time,dtype=torch.long)
        restored=denormalize(cpu_schedule.clean_from_noise(cpu_schedule.add_noise(clean,noise,t),noise,t),schedule.config)
        errors.append(float((restored-future)[mask].abs().max()))
    initial=cache[0]['initial_positions'].double();heading=cache[0]['history'][:,:,-1,4:6].double()
    decoded=decode_future(future,initial,heading,mask,decoder)
    p=float(decoded.position_correction[mask].norm(dim=-1).max());v=float(decoded.velocity_correction[mask].norm(dim=-1).max())
    return dict(exact_noise_reconstruction_max_physical_error=max(errors),label_decode_max_position_correction_m=p,
        label_decode_max_velocity_correction_mps=v,passed=max(errors)<1e-8 and p<3e-4 and v<3e-3,
        kind='CPU float64 exact-noise arithmetic and observed-label decoder check; not learned or generated performance')


def sample_statistics(directory,training_ids):
    directory=Path(directory);catalog=load_json(directory/'trajectory_index.json');values={k:[] for k in ('raw_displacement_m','decoded_displacement_m','position_correction_m','raw_speed_mps')};rows=[]
    for entry in catalog:
        meta=load_json(directory/'trajectories'/entry['window_id']/'metrics.json')
        rows.append(dict(window_id=entry['window_id'],family=entry['family'],seen_in_training=entry['window_id'] in training_ids,
            generation_status=entry['generation_status'],raw_quality_pass=meta['raw']['quality_pass'],decoded_quality_pass=meta['decoded']['quality_pass']))
        path=directory/'trajectories'/entry['window_id']/'stages.npz'
        if entry['generation_status']!='completed': continue
        with np.load(path,allow_pickle=False) as d:
            a=d['agent_mask'];values['raw_displacement_m'].extend(np.linalg.norm(d['raw_future_delta'][a,:,:2],axis=-1).ravel().tolist())
            values['decoded_displacement_m'].extend(np.linalg.norm(d['decoded_absolute_states'][a,:,:2]-d['initial_positions'][a,None,:],axis=-1).ravel().tolist())
            values['position_correction_m'].extend(np.linalg.norm(d['position_correction'][a],axis=-1).ravel().tolist())
            values['raw_speed_mps'].extend(np.linalg.norm(d['raw_future_delta'][a,:,2:4],axis=-1).ravel().tolist())
    return dict(rows=rows,statistics={k:dict(count=len(v),median=float(np.median(v)),p99=float(np.quantile(v,.99)),max=float(np.max(v))) if v else None for k,v in values.items()})


def load_json(path):
    import json
    return json.loads(Path(path).read_text(encoding='utf-8'))


def diagnose(dataset_path,checkpoint,config_path,output,repository,command,formal=False):
    c=diagnostic_config(load_config(config_path));payload=load_checkpoint(checkpoint);meta=payload['metadata'];data=dataset_identity(dataset_path)
    if any(data[k]['sha256']!=meta['data'][k]['sha256'] for k in data): raise ValueError('Diagnostic dataset/checkpoint mismatch')
    caches={s:labeled_cache(dataset_path,s,meta['training_config'][f'{s if s=="train" else "validation"}_limit']) for s in ('train','validation')}
    signature=meta['training_signature']
    if caches['train'][2]!=signature['train_window_ids'] or caches['validation'][2]!=signature['validation_window_ids']: raise ValueError('Diagnostic labeled task IDs changed')
    files=[config_path,*[v['location'] for v in data.values()],*[p for cache in caches.values() for p in cache[3]]]
    tasks={}
    for split in caches:
        ds=WindowDataset(dataset_path,split);ids=choose_indices(ds,c['free_sampling']['limit'],c['free_sampling']['minimum_agents'])
        tasks[split]=[ds.entries[i]['window_id'] for i in ids];files+=input_files(ds,ids,False)
    effective=dict(schema_version='sumodiff.training.diagnostic.run.v1',diagnostic=c,model=meta['model_config'],diffusion=meta['diffusion_config'],data=data,
        checkpoint=file_identity(checkpoint),checkpoint_step=payload['step'],labeled_task_ids={s:cache[2] for s,cache in caches.items()},
        free_task_ids=tasks,interpretation='labeled noisy-input reconstruction versus label-free pure-noise generation; no convergence gate',convergence_status='not_assessed')
    with RunRecorder(output,repository,effective,command,{'reconstruction_noise':c['noise_seed'],'free_noise':c['free_sampling']['seed']},
        purpose='stage5_bounded_training_diagnosis',data_files=sorted(set(map(str,files))),checkpoint=checkpoint,data_id=data['manifest']['id'],formal=formal) as run:
        started=time.perf_counter();configure(meta['training_config']['seed'],c['free_sampling']['torch_threads']);device=torch.device(c['free_sampling']['device'])
        if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        model=ConditionalDenoiser(resolve_model_config(meta['model_config'])).to(device);model.load_state_dict(payload['model']);model.eval();model.requires_grad_(False)
        schedule=NoiseSchedule(resolve_diffusion_config(meta['diffusion_config']),device);initial_hash=parameter_hash(model)
        probes={};oracles={};(run.output/'reconstructions').mkdir()
        for split,cache in caches.items():
            oracles[split]=oracle_probe(cache,schedule,DecoderConfig(**c['free_sampling']['evaluation']['decoder']))
            if not oracles[split]['passed']: raise RuntimeError('Exact-noise/label decoder numerical check failed')
            probes[split]=labeled_probe(model,cache,schedule,c,device,meta['training_config']['batch_size'],run.output/'reconstructions',split)
        if parameter_hash(model)!=initial_hash: raise RuntimeError('Diagnostic changed model weights')
        del model,schedule,payload
        if device.type=='cuda': torch.cuda.empty_cache()
        free={}
        for steps in c['sampling_steps']:
            for split in caches:
                sc=deepcopy(c['free_sampling']);sc.update(split=split,steps=steps)
                path=run.output/f'{split}_{steps}_config.yaml';path.write_text(yaml.safe_dump(sc,sort_keys=True),encoding='utf-8')
                subdir=run.output/f'free_{split}_{steps}'
                cmd=[command[0],'-m','sumodiff.diffusion','sample','--dataset',str(Path(dataset_path).resolve()),'--checkpoint',str(Path(checkpoint).resolve()),'--config',str(path.resolve()),'--output',str(subdir.resolve())]+(['--formal'] if formal else [])
                sampled=sample(dataset_path,checkpoint,path,subdir,repository,cmd,formal)
                free[f'{split}_{steps}']=dict(metrics=sampled,details=sample_statistics(subdir,signature['train_window_ids']),manifest=file_identity(subdir/'manifest.json'))
        result=dict(schema_version='sumodiff.training.diagnostic.metrics.v1',run_sha=run.manifest['git']['commit_sha'],checkpoint_step=effective['checkpoint_step'],
            parameter_sha256=initial_hash,oracle=oracles,reconstruction=probes,free_generation=free,wall_seconds=time.perf_counter()-started,
            convergence_status='not_assessed',scope='fixed small train/validation diagnosis; no test, retraining, architecture changes or guidance')
        write_json(run.output/'diagnosis.json',result);run.write_metrics(result)
    return result
