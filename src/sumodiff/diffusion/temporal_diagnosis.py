"""Frozen paired objective/reconstruction probes for the completed 200/225 models."""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import torch
from sumodiff.data.dataset import WindowDataset, write_json
from sumodiff.models import ConditionalDenoiser, resolve_model_config
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.decoding import decode_future, observed_heading, DecoderConfig
from sumodiff.evaluation.motion import evaluate_motion, MotionConfig, summary
from sumodiff.experiments.recorder import RunRecorder, file_identity, load_config
from .boundary_validation import descriptors, select_training, make_cache, groups
from .checkpoint import load_checkpoint
from .config import resolve_diffusion_config
from .data import dataset_identity, select_batch
from .process import NoiseSchedule, normalize, denormalize
from .sampling import task_seed
from .training import synchronize
from .controlled_training import tensor_hash
from .quality_audit import physical_derivatives
from .raw_velocity_losses import velocity_residual
from .decoded_temporal_losses import resolve_decoded_temporal_loss, decoded_temporal_objective
from .unified_losses import prepare_roads

CHECKPOINTS = {
    'epoch200': ('b7dcb40fcd9032b76d56ad8871e986067cd29873c8d84ec461feb55b8c96db54', 60000),
    'epoch225': ('3625b5c3028e93fde90e527161f988bcba5f97cf4dd918c70daf809140eac937', 67500)}
SPANS = dict(boundary=slice(0,4), middle=slice(4,36), endpoint=slice(36,40), interior=slice(4,40))
LOSS_FIELDS = ('noise_mse', 'longitudinal', 'lateral', 'velocity', 'heading', 'imitation',
    'first_difference', 'second_difference', 'decoded_temporal_boundary', 'decoded_temporal_interior',
    'first_difference_boundary', 'first_difference_interior', 'second_difference_boundary',
    'second_difference_interior', 'raw_second_boundary', 'raw_second_interior', 'road',
    'weighted_imitation', 'weighted_road', 'weighted_raw_boundary', 'weighted_raw_interior',
    'weighted_temporal', 'inherited_weighted_temporal', 'total_loss', 'auxiliary_total')

def resolve_probe(value):
    defaults = dict(schema_version='sumodiff.temporal.diagnosis.config.v1', seed=20261020,
                    device='cuda:0', torch_threads=1, timesteps=[0,50,200,300])
    if not isinstance(value,dict) or set(value)-set(defaults):
        raise ValueError('Unknown temporal diagnostic options')
    c = dict(defaults,**value)
    if c['schema_version']!=defaults['schema_version'] or c['device'] not in ('cpu','cuda:0'):
        raise ValueError('Unsupported probe schema/device')
    if type(c['seed']) is not int or not 0<=c['seed']<2**32:
        raise ValueError('Invalid seed')
    if type(c['torch_threads']) is not int or not 1<=c['torch_threads']<=4:
        raise ValueError('Invalid thread budget')
    ts=c['timesteps']
    if not isinstance(ts,list) or not 1<=len(ts)<=4 or len(set(ts))!=len(ts) or any(
        type(t) is not int or not 0<=t<=300 for t in ts):
        raise ValueError('At most four unique auxiliary timesteps in [0,300]')
    return c

def residual_curves(raw, decoded, label, conditioning, targets, dt=.1):
    """Physical norms, indexed by the newest future point; mask before differences."""
    active=conditioning['agent_mask']
    safe=lambda x: torch.where(active[...,None,None],x,0.)
    observed=safe(conditioning['history'][...,-2:,2:4]).to(decoded.dtype)
    p=torch.cat((observed,safe(decoded[...,2:4])),dim=-2)
    q=torch.cat((observed,safe(label[...,2:4])),dim=-2)
    first=(torch.diff(p,dim=-2)-torch.diff(q,dim=-2))[...,1:,:]
    second=torch.diff(p,n=2,dim=-2)-torch.diff(q,n=2,dim=-2)
    raw_second,_=velocity_residual(raw,conditioning,targets)
    return dict(decoded_acceleration_error=torch.linalg.vector_norm(first,dim=-1)/dt,
        decoded_velocity_jerk_error=torch.linalg.vector_norm(second,dim=-1)/dt**2,
        raw_velocity_jerk_error=torch.linalg.vector_norm(raw_second,dim=-1)/dt**2)

def summarize_rows(rows,pools,times):
    return dict(planned_scenes=len(rows),selected_agents=sum(r['agents'] for r in rows),
        motion_pass_scenes=sum(int(r['motion_pass']) for r in rows),
        comfort_pass_agents=sum(r['comfort_pass_agents'] for r in rows),
        loss_mean={k:float(np.mean([r['loss'][k] for r in rows])) for k in rows[0]['loss']},
        pooled={k:summary(v) for k,v in pools.items()},
        by_time={k:[summary(v) for v in frames] for k,frames in times.items()})

def probe_model(model,cache,schedule,config,timestep,device,metadata,roads,loss_config):
    rows=[];pools=defaultdict(lambda:defaultdict(list))
    timepools=defaultdict(lambda:defaultdict(lambda:[[] for _ in range(40)]))
    grouped=defaultdict(list);noisy_hashes={}
    for start in range(0,len(cache[2]),4):
        cond,target,ids=select_batch(cache,torch.arange(start,min(start+4,len(cache[2]))),device)
        active=cond['agent_mask']
        noise=torch.stack([torch.randn(target['future'][i].shape,generator=
            torch.Generator().manual_seed(task_seed(config['seed'],wid))) for i,wid in enumerate(ids)]).to(device)
        ts=torch.full((len(ids),),timestep,dtype=torch.long,device=device)
        noisy=torch.where(active[...,None,None],schedule.add_noise(normalize(target['future'],schedule.config),noise,ts),0.)
        with torch.no_grad():
            epsilon=model(noisy,ts,cond)
            raw=torch.where(active[...,None,None],denormalize(schedule.clean_from_noise(noisy,epsilon,ts),schedule.config),0.)
            heading=observed_heading(cond['history'],cond['history_mask'],active)
            decoded=decode_future(raw,cond['initial_positions'],heading,active)
            truth=decode_future(target['future'].to(raw.dtype),cond['initial_positions'],heading,active)
            curves=residual_curves(raw,decoded.states,truth.states,cond,target)
            for i,wid in enumerate(ids):
                ci={k:v[i:i+1] for k,v in cond.items()}
                ti={k:v[i:i+1] for k,v in target.items()}
                # Use the exact current training objective for both frozen models.
                objective=decoded_temporal_objective(epsilon[i:i+1],noisy[i:i+1],noise[i:i+1],
                    ts[i:i+1],ci,ti,schedule,loss_config,roads=[roads[wid]],ramp=1.)
                loss={k:float(objective[k]) for k in LOSS_FIELDS}
                loss['total_under_parent_weights']=loss['total_loss']-loss['weighted_temporal']+loss['inherited_weighted_temporal']
                cc={k:v[i].cpu().numpy() for k,v in cond.items()}
                states=decoded.states[i].cpu().numpy();a=cc['agent_mask']
                _,_,jerk=physical_derivatives(cc,states)
                values={k:v[i][active[i]].cpu().numpy() for k,v in curves.items()}
                values['decoded_jerk']=np.linalg.norm(jerk[a],axis=-1)
                values['position_correction']=np.linalg.norm((decoded.states[i][...,:2]-
                    raw[i][...,:2]-cond['initial_positions'][i,:,None,:])[active[i]].cpu().numpy(),axis=-1)
                motion=evaluate_motion(cc['history'],cc['history_mask'],states,target['future_mask'][i].cpu().numpy(),
                    cc['initial_positions'],decoded.initial_heading[i].cpu().numpy(),a)
                row=dict(**metadata[wid],timestep=timestep,loss=loss,motion_pass=motion['hard_quality_pass'],
                    comfort_pass_agents=motion['comfort_pass_agents'],violation_counts=motion['violation_counts'],
                    metrics={k+'_'+span:summary(v[:,sl]) for k,v in values.items() for span,sl in SPANS.items()})
                noisy_hashes[wid]=tensor_hash(noisy[i]);rows.append(row)
                for group in groups(row):
                    grouped[group].append(row)
                    for k,v in values.items():
                        for span,sl in SPANS.items():pools[group][k+'_'+span].extend(v[:,sl].ravel().tolist())
                        for frame in range(40):timepools[group][k][frame].extend(v[:,frame].tolist())
    return {k:summarize_rows(v,pools[k],timepools[k]) for k,v in grouped.items()},rows,noisy_hashes

def diagnose(config_path,output,formal=False):
    suite=load_config(config_path)
    if not isinstance(suite,dict) or set(suite)!={'dataset','checkpoints','probe'} or set(suite['checkpoints'])!=set(CHECKPOINTS):
        raise ValueError('Fixed epoch200/225 suite required')
    c=resolve_probe(suite['probe']);data=dataset_identity(suite['dataset'])
    payloads={};identities={}
    for name,path in suite['checkpoints'].items():
        identity=file_identity(path);expected,step=CHECKPOINTS[name]
        if identity['sha256']!=expected:raise ValueError('Fixed checkpoint identity changed')
        p=load_checkpoint(path)
        if p['step']!=step or p['metadata']['data']!=data:raise ValueError('Candidate step/data mismatch')
        payloads[name]=p;identities[name]=identity
    old,new=(payloads[n]['metadata'] for n in ('epoch200','epoch225'))
    for field in ('model_config','diffusion_config','data'):
        if old[field]!=new[field]:raise ValueError('Architecture/noising/data mismatch')
    loss_config=resolve_decoded_temporal_loss(new['training_config']['loss'])
    if loss_config.inherited.to_dict()!=old['training_config']['loss']:
        raise ValueError('Inherited supervision differs')
    datasets={s:WindowDataset(suite['dataset'],s,core_only=True) for s in ('train','validation')}
    desc={s:descriptors(d) for s,d in datasets.items()}
    if len(desc['train'])!=1200 or len(desc['validation'])!=200:
        raise ValueError('Certified 1200/200 dataset required')
    for field in ('episode_key','geometry_id'):
        if {d[field] for d in desc['train']}&{d[field] for d in desc['validation']}:
            raise ValueError('Train/validation leakage')
    selected={'train':select_training(desc['train'],24,c['seed']), 'validation':list(range(200))}
    caches={s:make_cache(d,selected[s]) for s,d in datasets.items()}
    if len(caches['train'][2])!=119:raise ValueError('Certified dense/control cohort changed')
    metadata={s:{d['window_id']:d for d in desc[s]} for s in desc}
    maps={};repairs=[];label_control={}
    for split,ds in datasets.items():
        maps[split]={};cache=caches[split];passed=0;comfort=0;agents=0
        for i,index in enumerate(selected[split]):
            wid=ds.entries[index]['window_id']
            maps[split][wid],rr=prepare_roads(ds._json(ds.entries[index],'map.json'),loss_config)
            repairs.extend(dict(split=split,window_id=wid,**r) for r in rr)
            cc={k:v[i].numpy() for k,v in cache[0].items()}
            label=cache[1]['future'][i].numpy().astype(float).copy()
            label[...,:2]+=cc['initial_positions'][:,None,:]
            m=evaluate_motion(cc['history'],cc['history_mask'],label,cache[1]['future_mask'][i].numpy(),
                cc['initial_positions'],cc['history'][:,-1,4:6],cc['agent_mask'])
            passed+=int(m['hard_quality_pass']);comfort+=m['comfort_pass_agents'];agents+=int(cc['agent_mask'].sum())
        label_control[split]=dict(motion_pass_scenes=passed,comfort_pass_agents=comfort,selected_agents=agents)
    files=[config_path,Path(__file__),*[v['location'] for v in data.values()],*[v['location'] for v in identities.values()]]
    for cache in caches.values():files.extend(cache[3])
    effective=dict(schema_version='sumodiff.temporal.diagnosis.run.v1',suite=suite,probe=c,data=data,
        checkpoints=identities,selected_window_ids={s:cache[2] for s,cache in caches.items()},
        common_objective=loss_config.to_dict(),common_objective_ramp=1.,maps_repairs=repairs,
        selection='All dense train cases plus 24 metadata/episode/geometry-balanced ordinary controls per family; all validation.',
        batch_size=4,uses_test=False,future_labels_used=True,optimizer_updates=0,new_free_generation=0,
        evaluation=dict(decoder=vars(DecoderConfig()),motion=vars(MotionConfig())))
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'diagnostic_noise':c['seed']},
        purpose='stage5e_frozen_temporal_objective_diagnosis',data_files=sorted(set(map(str,files))),
        checkpoint=identities['epoch225']['location'],formal=formal) as run:
        configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        started=time.perf_counter();results={};pair_inputs={};model_state={};calls=0;peak=0;reserved=0
        for name,p in payloads.items():
            model=ConditionalDenoiser(resolve_model_config(p['metadata']['model_config'])).to(device)
            model.load_state_dict(p['model'],strict=True);model.eval();model.requires_grad_(False)
            before=parameter_hash(model);results[name]={}
            schedule=NoiseSchedule(resolve_diffusion_config(p['metadata']['diffusion_config']),device)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            for split,cache in caches.items():
                results[name][split]={}
                for t in c['timesteps']:
                    aggregate,rows,hashes=probe_model(model,cache,schedule,c,t,device,metadata[split],maps[split],loss_config)
                    if name=='epoch200':pair_inputs[(split,t)]=hashes
                    elif pair_inputs[(split,t)]!=hashes:raise RuntimeError('Paired noised inputs differ')
                    results[name][split][str(t)]=aggregate
                    write_json(run.output/f'{name}_{split}_t{t:03}.json',rows)
                    calls+=(len(rows)+3)//4
                    print(f'{name} {split} t={t}: motion={aggregate["all"]["motion_pass_scenes"]}/{len(rows)}',flush=True)
            synchronize(device)
            model_state[name]=dict(parameter_hash=before,model_unchanged=before==parameter_hash(model),
                grad_buffers_unset=all(v.grad is None for v in model.parameters()))
            if not model_state[name]['model_unchanged'] or not model_state[name]['grad_buffers_unset']:
                raise RuntimeError('Frozen model mutated')
            if device.type=='cuda':
                peak=max(peak,torch.cuda.max_memory_allocated(device));reserved=max(reserved,torch.cuda.max_memory_reserved(device))
            del model,schedule
        for identity in identities.values():
            if file_identity(identity['location'])!=identity:raise RuntimeError('Source checkpoint mutated')
        result=dict(schema_version='sumodiff.temporal.diagnosis.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            checkpoints=identities,reconstruction=results,label_control=label_control,model_state=model_state,
            cohorts={s:[metadata[s][wid] for wid in cache[2]] for s,cache in caches.items()},
            paired_noisy_inputs=True,optimizer_updates=0,new_free_generation=0,actual_model_forward_calls=calls,
            peak_allocated_bytes=peak if device.type=='cuda' else None,
            peak_reserved_bytes=reserved if device.type=='cuda' else None,wall_seconds=time.perf_counter()-started,
            limits=['Known-future reconstruction is not free generation; no generated quality benefit implied.',
                't=0 still has Gaussian noise (alpha_bar=0.9999); not noiseless reconstruction.',
                'One Gaussian per window, reused over timesteps/models; four fixed timesteps are not the natural training distribution.',
                'Train controls overrepresent density; split differences are not an unbiased generalization-gap estimate.',
                'Both model predictions use the same full-ramp 225 loss; component values include alpha_bar, but not lambda.',
                'Parent/current weighted totals are alternative fixed objectives, not historical step logs.',
                'No causal separation of learning-rate restart and temporal reweighting.'])
        write_json(run.output/'diagnosis.json',result);run.write_metrics(result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--formal',action='store_true')
    a=p.parse_args();r=diagnose(a.config,a.output,a.formal)
    print(json.dumps({k:r[k] for k in ('run_sha','optimizer_updates','new_free_generation',
        'actual_model_forward_calls','wall_seconds','paired_noisy_inputs')},indent=2))
if __name__=='__main__':main()
