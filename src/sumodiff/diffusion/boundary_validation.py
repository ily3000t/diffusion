"""Frozen boundary and dense-intersection probes; no optimizer or free sampling."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch
from sumodiff.data.dataset import WindowDataset, collate_numpy, write_json
from sumodiff.models import ConditionalDenoiser, resolve_model_config, prepare_conditioning
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.decoding import decode_future, observed_heading, DecoderConfig
from sumodiff.evaluation.motion import evaluate_motion, MotionConfig, summary
from sumodiff.experiments.recorder import RunRecorder, file_identity, load_config
from .checkpoint import load_checkpoint
from .config import resolve_diffusion_config
from .data import dataset_identity, input_files, select_batch
from .process import NoiseSchedule, normalize, denormalize
from .sampling import task_seed
from .training import synchronize
from .controlled_training import tensor_hash
from .quality_audit import physical_derivatives

SCHEMA='sumodiff.boundary.density.config.v1'
CHECKPOINTS={
    'parent150':('2a25293744e0f9ff8bec9a03b3a1c3e077275bc7c9cece4119cdda13f070900e',45000),
    'candidate200':('b7dcb40fcd9032b76d56ad8871e986067cd29873c8d84ec461feb55b8c96db54',60000)}
DEFAULTS=dict(schema_version=SCHEMA,seed=20261020,device='cuda:0',torch_threads=1,batch_size=4,
    dense_min_agents=8,normal_train_per_family=24,timesteps=[0,50,100,200,300])

def config(value):
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown boundary-density probe options')
    c=dict(DEFAULTS,**value)
    if c['schema_version']!=SCHEMA or c['device'] not in ('cpu','cuda:0'):
        raise ValueError('Unsupported diagnostic schema/device')
    if type(c['seed']) is not int or not 0<=c['seed']<2**32:
        raise ValueError('Invalid diagnostic seed')
    if c['batch_size']!=4 or type(c['batch_size']) is not int or c['dense_min_agents']!=8:
        raise ValueError('This bounded protocol fixes B4 and density >=8')
    if type(c['torch_threads']) is not int or not 1<=c['torch_threads']<=4:
        raise ValueError('Invalid CPU thread budget')
    if type(c['normal_train_per_family']) is not int or not 1<=c['normal_train_per_family']<=24:
        raise ValueError('At most 24 normal train controls per family')
    ts=c['timesteps']
    if not isinstance(ts,list) or not 1<=len(ts)<=5 or len(set(ts))!=len(ts) or any(
        type(t) is not int or not 0<=t<=300 for t in ts):
        raise ValueError('At most five unique low-noise timesteps in [0,300]')
    return c

def select_training(descriptors,count,seed,dense_min=8):
    """All dense cases plus metadata-only geometry/episode-balanced controls."""
    chosen=[i for i,d in enumerate(descriptors) if d['agents']>=dense_min]
    for family in sorted({d['family'] for d in descriptors}):
        groups=defaultdict(list)
        for i,d in enumerate(descriptors):
            if d['family']==family and d['agents']<dense_min:
                groups[(d['geometry_id'],d['episode_key'])].append(i)
        for values in groups.values():
            values.sort(key=lambda i:hashlib.sha256((str(seed)+'|'+descriptors[i]['window_id']).encode()).hexdigest())
        picked=[]
        while len(picked)<count and any(groups.values()):
            for key in sorted(groups):
                if groups[key] and len(picked)<count:picked.append(groups[key].pop(0))
        if len(picked)!=count:raise ValueError('Insufficient normal train controls; no silent undersampling')
        chosen.extend(picked)
    if len(set(chosen))!=len(chosen):raise ValueError('Duplicate diagnostic selection')
    return chosen

def descriptors(dataset):
    result=[]
    for e in dataset.entries:
        m=dataset._json(e,'input.json')
        result.append(dict(window_id=e['window_id'],family=m['family'],geometry_id=e['geometry_id'],
            episode_key=e['episode_key'],agents=sum(v is not None for v in m['agent_ids'])))
    return result

def make_cache(dataset,indices):
    samples=[dataset[i] for i in indices]
    packed=collate_numpy(samples)
    c=prepare_conditioning(packed,'cpu')
    t={k:torch.as_tensor(v) for k,v in packed['targets'].items()}
    if not t['future_mask'][c['agent_mask']].all() or not c['history_mask'][c['agent_mask']].all():
        raise ValueError('Diagnostic requires complete historical and future labels')
    return c,t,[s['input_metadata']['window_id'] for s in samples],input_files(dataset,indices,True)

def error_jet(error,dt=.1):
    """Derivative errors share exactly the same observed history: three zeros."""
    e=np.asarray(error,np.float64)
    if e.ndim!=3 or e.shape[-1]!=2 or not np.isfinite(e).all() or not dt>0:
        raise ValueError('Invalid finite position error or dt')
    p=np.concatenate((np.zeros((len(e),3,2)),e),axis=1)
    v=np.diff(p,axis=1)/dt
    a=np.diff(v,axis=1)/dt
    j=np.diff(a,axis=1)/dt
    return v[:,2:],a[:,1:],j

def prefix_position(raw,label,frames):
    if type(frames) is not int or not 0<=frames<=raw.shape[-2] or raw.shape!=label.shape:
        raise ValueError('Invalid oracle prefix')
    out=raw.copy()
    out[...,:frames,:2]=label[...,:frames,:2]
    return out

def groups(meta):
    return ('all',meta['family'],
        'intersection_dense' if meta['family']=='unsignalized_intersection' and meta['agents']>=8 else
        'intersection_normal' if meta['family']=='unsignalized_intersection' else 'other_roads')

def summarize_group(rows,pools):
    counts=Counter()
    for r in rows:
        counts['motion_pass_scenes']+=r['motion_pass'] is True
        counts['comfort_pass_agents']+=r['comfort_pass_agents']
        counts['selected_agents']+=r['agents']
        for k,v in r['violation_counts'].items():counts[k+'_bad_scenes']+=v>0
    return dict(planned_scenes=len(rows),counts=dict(counts),
        mean_scene_epsilon_mse=float(np.mean([r['epsilon_mse'] for r in rows])),
        channel_epsilon_mse=np.mean([r['channel_epsilon_mse'] for r in rows],axis=0).tolist(),
        pooled={k:summary(v) for k,v in pools.items()},
        mean_scene_metrics={k:summary([r['metrics'][k]['mean'] for r in rows]) for k in pools},
        label_conditioned_reconstruction=True,free_generation=False)

def quantities(c,raw,decoded,label):
    agents=np.asarray(c['agent_mask'],bool)
    # Mask padding before finite differences. Valid histories are immutable.
    r=np.where(agents[:,None,None],raw,0.).astype(np.float64)
    q=np.where(agents[:,None,None],label,0.).astype(np.float64)
    initial=np.asarray(c['initial_positions'],float)
    rabs=r.copy();rabs[...,:2]+=initial[:,None,:]
    qabs=q.copy();qabs[...,:2]+=initial[:,None,:]
    rv,ra,rj=physical_derivatives(c,rabs)
    dv,da,dj=physical_derivatives(c,decoded)
    lv,la,lj=physical_derivatives(c,qabs)
    er=r[...,:2]-q[...,:2];ed=decoded[...,:2]-qabs[...,:2]
    _,_,jet=error_jet(np.where(agents[:,None,None],ed,0.))
    linear_error=float(np.abs((dj-lj-jet)[agents]).max())
    if linear_error>1e-6:raise RuntimeError('Boundary derivative errors violate fixed-history identity')
    norm=lambda x:np.linalg.norm(x,axis=-1)[agents]
    return dict(
        raw_position_first=norm(er[:,0]),decoded_position_first=norm(ed[:,0]),
        raw_position_first4=norm(er[:,:4]),decoded_position_first4=norm(ed[:,:4]),
        raw_velocity_first4=norm(r[:,:4,2:4]-q[:,:4,2:4]),
        raw_position_velocity_first4=norm((rv-lv)[:,:4]),
        decoded_velocity_first4=norm((dv-lv)[:,:4]),
        decoded_accel_error_first4=norm((da-la)[:,:4]),
        raw_velocity_consistency=norm(rv-r[...,2:4]),
        decoded_jerk_boundary=norm(dj[:,:4]),decoded_jerk_interior=norm(dj[:,4:]),
        decoded_jerk_error_boundary=norm((dj-lj)[:,:4]),
        decoded_jerk_error_interior=norm((dj-lj)[:,4:]),
        decoded_jerk_first=norm(dj[:,0]),
        raw_jerk_boundary=norm(rj[:,:4]),
        position_correction=norm(decoded[...,:2]-rabs[...,:2]),
        first4_position_linear_identity_error=np.asarray([linear_error]))

def reconstruct(model,cache,schedule,c,timestep,device,metadata):
    rows=[];pools=defaultdict(lambda:defaultdict(list));bygroup=defaultdict(list)
    all_noise={}
    for start in range(0,len(cache[2]),4):
        ix=torch.arange(start,min(start+4,len(cache[2])))
        cond,target,ids=select_batch(cache,ix,device);mask=cond['agent_mask']
        noise=torch.stack([torch.randn(target['future'][i].shape,
            generator=torch.Generator().manual_seed(task_seed(c['seed'],wid))) for i,wid in enumerate(ids)]).to(device)
        ts=torch.full((len(ids),),timestep,dtype=torch.long,device=device)
        noisy=torch.where(mask[...,None,None],schedule.add_noise(normalize(target['future'],schedule.config),noise,ts),0.)
        with torch.no_grad():
            epsilon=model(noisy,ts,cond)
            raw=torch.where(mask[...,None,None],denormalize(schedule.clean_from_noise(noisy,epsilon,ts),schedule.config),0.)
            decoded=decode_future(raw,cond['initial_positions'],observed_heading(cond['history'],cond['history_mask'],mask),mask)
        for i,wid in enumerate(ids):
            all_noise[wid]=tensor_hash(noisy[i])
            cc={k:v[i].cpu().numpy() for k,v in cond.items()}
            value=quantities(cc,raw[i].cpu().numpy(),decoded.states[i].cpu().numpy(),target['future'][i].cpu().numpy())
            a=cc['agent_mask'];perfield=(epsilon[i][mask[i]]-noise[i][mask[i]]).square().mean((0,1)).cpu().tolist()
            motion=evaluate_motion(cc['history'],cc['history_mask'],decoded.states[i].cpu().numpy(),
                target['future_mask'][i].cpu().numpy(),cc['initial_positions'],decoded.initial_heading[i].cpu().numpy(),a)
            row=dict(**metadata[wid],timestep=timestep,epsilon_mse=float(np.mean(perfield)),channel_epsilon_mse=perfield,
                motion_pass=motion['hard_quality_pass'],violation_counts=motion['violation_counts'],
                comfort_pass_agents=motion['comfort_pass_agents'],metrics={k:summary(v) for k,v in value.items()})
            rows.append(row)
            for group in groups(row):
                bygroup[group].append(row)
                for k,v in value.items():pools[group][k].extend(v.ravel().tolist())
    return {k:summarize_group(v,pools[k]) for k,v in bygroup.items()},rows,all_noise

def saved_prefix_probe(directory,dataset):
    """Oracle labels localize existing output error; never a generated baseline."""
    root=Path(directory);catalog=json.loads((root/'trajectory_index.json').read_text(encoding='utf-8'))
    if len(catalog)!=200 or len({r['window_id'] for r in catalog})!=200:
        raise ValueError('Saved diagnostic requires the complete 200-task catalog')
    lookup={e['window_id']:i for i,e in enumerate(dataset.entries)}
    modes=['unchanged','position_1','position_2','position_4','position_8','position_40','velocity_40','position_velocity_4']
    pooled=defaultdict(lambda:defaultdict(list));counts=defaultdict(Counter);rows=[];maxerr=0.
    for item in catalog:
        wid=item['window_id'];sample=dataset[lookup[wid]];c=sample['conditioning'];a=c['agent_mask']
        label=sample['targets']['future'].astype(np.float64)
        source=root/'trajectories'/wid
        for name,ident in item['files'].items():
            if file_identity(source/name)['sha256']!=ident['sha256']:
                raise ValueError('Saved source artifact identity mismatch')
        with np.load(source/'stages.npz',allow_pickle=False) as f:
            raw=f['raw_future_delta'].astype(np.float64);original=f['decoded_absolute_states'].copy()
            heading=f['initial_heading'].copy()
        for mode in modes:
            value=raw.copy()
            if mode.startswith('position_') and mode!='position_velocity_4':
                value=prefix_position(raw,label,int(mode.split('_')[1]))
            if mode=='velocity_40':value[...,2:4]=label[...,2:4]
            if mode=='position_velocity_4':value[...,:4,:4]=label[...,:4,:4]
            decoded=decode_future(torch.from_numpy(value),torch.from_numpy(c['initial_positions']),
                torch.from_numpy(heading),torch.from_numpy(a)).states.numpy()
            if mode=='unchanged':
                err=float(np.abs(decoded[a]-original[a]).max());maxerr=max(maxerr,err)
                if err>1e-9:raise RuntimeError('Saved decoder replay changed')
            _,_,j=physical_derivatives(c,decoded);jn=np.linalg.norm(j[a],axis=-1)
            for span,slice_ in [('boundary',slice(0,4)),('interior',slice(4,None)),('all',slice(None))]:
                pooled[mode]['jerk_'+span].extend(jn[:,slice_].ravel().tolist())
            counts[mode]['jerk_pass_scenes']+=bool((jn<=15).all())
            counts[mode]['boundary_jerk_pass_scenes']+=bool((jn[:,:4]<=15).all())
            counts[mode]['interior_jerk_pass_scenes']+=bool((jn[:,4:]<=15).all())
            rows.append(dict(window_id=wid,mode=mode,boundary_jerk_max=float(jn[:,:4].max()),
                interior_jerk_max=float(jn[:,4:].max()),jerk_pass=bool((jn<=15).all())))
    return dict(planned_scenes=200,decoder_replay_max_error=maxerr,
        modes={k:dict(counts=dict(counts[k]),pooled={q:summary(v) for q,v in pool.items()}) for k,pool in pooled.items()},
        future_labels_used=True,optimizer_updates=0,free_generation=False,
        interpretation='Oracle swaps in saved trajectories; prefix edits may create a junction at the edit endpoint. Not achievable generated quality.'),rows

def diagnose(config_path,output,formal=False):
    suite=load_config(config_path)
    if not isinstance(suite,dict) or set(suite)!= {'probe','dataset','checkpoints','saved_generation'}:
        raise ValueError('Unexpected suite fields')
    c=config(suite['probe']);data=dataset_identity(suite['dataset'])
    if set(suite['checkpoints'])!=set(CHECKPOINTS):raise ValueError('Both fixed 150/200 candidates required')
    payloads={};identities={}
    for name,path in suite['checkpoints'].items():
        ident=file_identity(path);expected,step=CHECKPOINTS[name]
        if ident['sha256']!=expected:raise ValueError('Wrong fixed candidate identity')
        p=load_checkpoint(path)
        if p['step']!=step or p['metadata']['data']['manifest']['sha256']!=data['manifest']['sha256']:
            raise ValueError('Candidate step/dataset mismatch')
        payloads[name]=p;identities[name]=ident
    parent=payloads['parent150']['metadata'];new=payloads['candidate200']['metadata']
    for field in ('model_config','diffusion_config','data'):
        if parent[field]!=new[field]:raise ValueError('Candidate architecture, noising or data differ')
    datasets={s:WindowDataset(suite['dataset'],s,core_only=True) for s in ('train','validation')}
    if len(datasets['train'])!=1200 or len(datasets['validation'])!=200:
        raise ValueError('Requires the certified 1200/200 cohort')
    desc={s:descriptors(d) for s,d in datasets.items()}
    for k in ('episode_key','geometry_id'):
        if {d[k] for d in desc['train']}&{d[k] for d in desc['validation']}:
            raise ValueError('Train/validation leakage')
    selected={'train':select_training(desc['train'],c['normal_train_per_family'],c['seed']),
              'validation':list(range(200))}
    if sum(d['agents']>=8 for d in desc['train'])!=47 or sum(d['agents']>=8 for d in desc['validation'])!=16:
        raise ValueError('Audited density composition changed')
    if any(d['agents']>=8 and d['family']!='unsignalized_intersection' for s in desc for d in desc[s]):
        raise ValueError('Density diagnostic cohort semantics changed')
    metadata={s:{d['window_id']:d for d in desc[s]} for s in desc}
    caches={s:make_cache(d,selected[s]) for s,d in datasets.items()}
    generation=Path(suite['saved_generation'])
    gm=json.loads((generation/'manifest.json').read_text(encoding='utf-8'))
    if gm['checkpoint']['sha256']!=identities['candidate200']['sha256']:
        raise ValueError('Saved generation candidate mismatch')
    if json.loads((generation/'status.json').read_text(encoding='utf-8'))['state']!='completed':
        raise ValueError('Saved generation is incomplete')
    files=[config_path,*[v['location'] for v in data.values()],identities['parent150']['location']]
    for cache in caches.values():files.extend(cache[3])
    files += [generation/n for n in ('manifest.json','status.json','resolved_config.yaml','trajectory_index.json','sampling_summary.json')]
    for item in json.loads((generation/'trajectory_index.json').read_text(encoding='utf-8')):
        files.extend(generation/'trajectories'/item['window_id']/n for n in item['files'])
    effective=dict(schema_version='sumodiff.boundary.density.run.v1',suite=suite,probe=c,data=data,
        checkpoints=identities,selected_window_ids={s:v[2] for s,v in caches.items()},
        selection='all dense train cases; geometry/episode round-robin normal controls ordered by seed/window hash; all validation',
        uses_test=False,labels='Future labels only for noised-label reconstruction, error references and oracle diagnostics.',
        evaluation=dict(decoder=vars(DecoderConfig()),motion=vars(MotionConfig())),
        optimizer_updates=0,new_free_generation=0)
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],
        {'diagnostic_noise':c['seed']},purpose='stage5d_frozen_boundary_density_diagnosis',
        data_files=sorted(set(map(str,files))),checkpoint=identities['candidate200']['location'],formal=formal) as run:
        configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
        started=time.perf_counter();prefix,prefixrows=saved_prefix_probe(generation,datasets['validation'])
        write_json(run.output/'prefix_oracle.json',prefix);write_json(run.output/'prefix_rows.json',prefixrows)
        results={};paired_inputs={};unchanged={};calls=0;peak=0;reserved=0;labelcontrol={}
        for split,cache in caches.items():
            cc,tt,ids,_=cache
            labelcontrol[split]=Counter()
            for i,wid in enumerate(ids):
                cond={k:v[i].numpy() for k,v in cc.items()};target=tt['future'][i].numpy().astype(float)
                target[...,:2]+=cond['initial_positions'][:,None,:]
                motion=evaluate_motion(cond['history'],cond['history_mask'],target,tt['future_mask'][i].numpy(),
                    cond['initial_positions'],cond['history'][:,-1,4:6],cond['agent_mask'])
                labelcontrol[split]['motion_pass_scenes']+=motion['hard_quality_pass'] is True
                labelcontrol[split]['comfort_pass_agents']+=motion['comfort_pass_agents']
                labelcontrol[split]['agents']+=int(cond['agent_mask'].sum())
        for name,p in payloads.items():
            model=ConditionalDenoiser(resolve_model_config(p['metadata']['model_config'])).to(device)
            model.load_state_dict(p['model'],strict=True);model.eval();model.requires_grad_(False)
            before=parameter_hash(model);schedule=NoiseSchedule(resolve_diffusion_config(p['metadata']['diffusion_config']),device)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            results[name]={}
            for split,cache in caches.items():
                results[name][split]={}
                for t in c['timesteps']:
                    aggregate,rows,noises=reconstruct(model,cache,schedule,c,t,device,metadata[split])
                    key=(split,t)
                    if name=='parent150':paired_inputs[key]=noises
                    elif paired_inputs[key]!=noises:raise RuntimeError('Candidate noised inputs are not identical')
                    results[name][split][str(t)]=aggregate
                    write_json(run.output/f'{name}_{split}_t{t:03}.json',rows)
                    calls+=(len(cache[2])+3)//4
                    print(f'{name} {split} t={t}: motion={aggregate["all"]["counts"]["motion_pass_scenes"]}/{len(rows)}',flush=True)
            synchronize(device)
            unchanged[name]=dict(parameter_hash=before,model_unchanged=before==parameter_hash(model),
                parameter_grad_buffers_unset=all(x.grad is None for x in model.parameters()))
            if not all(unchanged[name][k] for k in ('model_unchanged','parameter_grad_buffers_unset')):
                raise RuntimeError('Frozen model state changed')
            if device.type=='cuda':
                peak=max(peak,torch.cuda.max_memory_allocated(device));reserved=max(reserved,torch.cuda.max_memory_reserved(device))
            del model,schedule
        for name,ident in identities.items():
            if file_identity(ident['location'])!=ident:raise RuntimeError('Source checkpoint file changed')
        result=dict(schema_version='sumodiff.boundary.density.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            checkpoints=identities,reconstruction=results,prefix_oracle=prefix,
            cohorts={s:[metadata[s][wid] for wid in cache[2]] for s,cache in caches.items()},
            label_control={k:dict(v) for k,v in labelcontrol.items()},model_state=unchanged,
            paired_noisy_inputs=True,optimizer_updates=0,new_free_generation=0,actual_model_forward_calls=calls,
            peak_allocated_bytes=peak if device.type=='cuda' else None,
            peak_reserved_bytes=reserved if device.type=='cuda' else None,wall_seconds=time.perf_counter()-started,
            limits=['Single fixed Gaussian per window reused over timesteps and candidates.',
                'Denoising reconstruction accesses known future labels; it is not label-free generation.',
                'Train controls are a small metadata-selected cohort; geometry/episode/domain composition may confound split differences.',
                'Oracle prefix edits use labels and may transfer error to a later junction.',
                'No training loss, threshold, decoder, sampling protocol or dataset change.'])
        write_json(run.output/'diagnosis.json',result)
        run.write_metrics(result)
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--formal',action='store_true')
    a=parser.parse_args();r=diagnose(a.config,a.output,a.formal)
    print(json.dumps({k:r[k] for k in ('run_sha','optimizer_updates','new_free_generation',
        'actual_model_forward_calls','wall_seconds','peak_allocated_bytes','paired_noisy_inputs')},indent=2))
if __name__=='__main__':main()
