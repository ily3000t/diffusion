"""Bounded reconstruction/gradient probes and paired DDIM-step validation.

No optimizer is created. Probe-only velocity penalties are never registered as
a training objective. Reconstruction uses known future labels and is explicitly
distinct from label-free generation.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import math
import sys
import time
import numpy as np
import torch
from sumodiff.data.dataset import WindowDataset, collate_numpy, write_json
from sumodiff.models import ConditionalDenoiser, resolve_model_config, prepare_conditioning
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.decoding import decode_future, observed_heading, DecoderConfig
from sumodiff.evaluation.motion import MotionConfig, evaluate_motion, summary
from sumodiff.experiments.recorder import RunRecorder, file_identity, load_config
from .checkpoint import load_checkpoint
from .config import resolve_diffusion_config
from .data import dataset_identity, labeled_cache, select_batch, input_files
from .epoch_training import epoch_batch
from .process import NoiseSchedule, normalize, denormalize
from .physical_losses import scene_huber
from .sampling import sample_config, task_seed
from .stage5b_report import generation_rows, paired_interval
from .training import draw_noise, synchronize
from .unified_losses import unified_objective, resolve_unified_loss, RoadGeometryCache

EXPECTED_CHECKPOINT='2a25293744e0f9ff8bec9a03b3a1c3e077275bc7c9cece4119cdda13f070900e'
DEFAULTS=dict(schema_version='sumodiff.velocity.validation.config.v1',seed=20261019,
    device='cuda:0',torch_threads=1,batch_size=4,train_windows=48,validation_windows=200,
    timesteps=[0,50,100,200,300],gradient_batches=16,forced_gradient_timestep=200,
    second_difference_unit_mps=.15,boundary_frames=4)

def config(value):
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown velocity diagnostic configuration')
    c=dict(DEFAULTS,**value)
    if c['schema_version']!=DEFAULTS['schema_version'] or c['device'] not in ('cpu','cuda:0'):
        raise ValueError('Unsupported velocity validation schema/device')
    for k in ('seed','torch_threads','batch_size','train_windows','validation_windows','gradient_batches','boundary_frames'):
        if type(c[k]) is not int or c[k]<1: raise ValueError(f'Invalid {k}')
    if c['seed']>=2**32 or c['batch_size']!=4 or c['validation_windows']!=200 or not 1<=c['train_windows']<=96:
        raise ValueError('Probe is bounded at B4, at most 96 train and exactly 200 validation windows')
    if not 1<=c['gradient_batches']<=16 or c['boundary_frames']!=4:
        raise ValueError('Probe is limited to 16 batches and a fixed first-four-frame boundary group')
    if (not isinstance(c['timesteps'],list) or len(c['timesteps'])>6 or not c['timesteps']
        or len(set(c['timesteps']))!=len(c['timesteps'])
        or any(type(t) is not int or not 0<=t<=300 for t in c['timesteps'])):
        raise ValueError('Probe timesteps must be unique low-noise integers in [0,300]')
    if type(c['forced_gradient_timestep']) is not int or not 1<=c['forced_gradient_timestep']<=300:
        raise ValueError('Invalid forced low-noise gradient timestep')
    if type(c['second_difference_unit_mps']) not in (int,float) or not math.isfinite(c['second_difference_unit_mps']) or c['second_difference_unit_mps']<=0:
        raise ValueError('Invalid second-difference probe unit')
    return c

from .raw_velocity_losses import velocity_residual

def velocity_probe_terms(raw,c,targets,timesteps,schedule,unit=.15,boundary=4):
    """Diagnostic Huber terms: per scene, alpha weighted, full-batch denominator."""
    if not math.isfinite(unit) or unit<=0 or not 1<=boundary<raw.shape[-2]:
        raise ValueError('Invalid probe normalization or boundary')
    residual,mask=velocity_residual(raw,c,targets)
    alpha=schedule.alpha_bar[timesteps].to(raw.dtype)
    eligible=(timesteps<=300).to(raw.dtype)
    def mean(r,m):
        return (scene_huber(r/unit,m,1.)*alpha*eligible).mean()
    return dict(raw_second=mean(residual,mask),
        raw_second_boundary=mean(residual[...,:boundary,:],mask[...,:boundary]),
        raw_second_interior=mean(residual[...,boundary:,:],mask[...,boundary:]))

def derivative_vectors(c,states):
    h=c['history'][...,-3:,:2].to(states.dtype).clone()
    h[...,-1,:]=c['initial_positions'].to(states.dtype)
    position=torch.cat((h,states[...,:2]),-2)
    v=torch.diff(position,dim=-2)/.1
    accel=torch.diff(v,dim=-2)/.1
    jerk=torch.diff(accel,dim=-2)/.1
    return v[...,2:,:],accel[...,1:,:],jerk

def gradient_vector(loss,parameters):
    if not loss.requires_grad:
        return torch.zeros(sum(p.numel() for p in parameters),device=parameters[0].device,dtype=torch.float64)
    gs=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
        for p,g in zip(parameters,gs)]).double()

def grad_stats(loss,parameters,reference):
    g=gradient_vector(loss,parameters)
    norm=torch.linalg.vector_norm(g)
    refnorm=torch.linalg.vector_norm(reference)
    return dict(loss=float(loss.detach()),parameter_gradient_norm=float(norm),
        norm_ratio_to_epsilon=float(norm/refnorm) if refnorm>0 else None,
        cosine_with_epsilon=float(torch.dot(g,reference)/norm/refnorm) if norm>0 and refnorm>0 else None)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def reconstruction_rows(model,cache,schedule,c,timestep,device):
    rows=[];pooled=defaultdict(list);counts=Counter();label_counts=Counter()
    for start in range(0,len(cache[2]),c['batch_size']):
        indices=torch.arange(start,min(start+c['batch_size'],len(cache[2])))
        cond,target,ids=select_batch(cache,indices,device)
        mask=cond['agent_mask']
        noise=torch.stack([torch.randn(target['future'][i].shape,
            generator=torch.Generator().manual_seed(task_seed(c['seed'],wid))) for i,wid in enumerate(ids)]).to(device)
        ts=torch.full((len(ids),),timestep,dtype=torch.long,device=device)
        clean=normalize(target['future'],schedule.config)
        noisy=torch.where(mask[...,None,None],schedule.add_noise(clean,noise,ts),0.)
        with torch.no_grad():
            epsilon=model(noisy,ts,cond)
            raw=denormalize(schedule.clean_from_noise(noisy,epsilon,ts),schedule.config)
            raw=torch.where(mask[...,None,None],raw,0.)
            decoded=decode_future(raw,cond['initial_positions'],
                observed_heading(cond['history'],cond['history_mask'],mask),mask)
            truth=decode_future(target['future'],cond['initial_positions'],
                observed_heading(cond['history'],cond['history_mask'],mask),mask)
            residual,fm=velocity_residual(raw,cond,target)
            _,_,j=derivative_vectors(cond,decoded.states)
            _,_,qj=derivative_vectors(cond,truth.states)
            rawv=torch.cat((cond['history'][...,-2:,2:4].to(raw.dtype),raw[...,2:4]),-2)
            raw_velocity_jerk=torch.diff(rawv,n=2,dim=-2)/.01
            field_mse=(epsilon-noise).square().mean(-2)
        for i,wid in enumerate(ids):
            a=mask[i].cpu().numpy()
            def values(x):
                return x[i][mask[i]].detach().cpu().numpy()
            per_field=field_mse[i][mask[i]].mean(0).cpu().tolist()
            r=dict(window_id=wid,timestep=timestep,epsilon_mse=float(np.mean(per_field)),
                channel_epsilon_mse=per_field)
            quantity={
                'raw_position_error_m':torch.linalg.vector_norm(raw[...,:2]-target['future'][...,:2],dim=-1),
                'raw_velocity_error_mps':torch.linalg.vector_norm(raw[...,2:4]-target['future'][...,2:4],dim=-1),
                'decoded_position_error_m':torch.linalg.vector_norm(decoded.states[...,:2]-truth.states[...,:2],dim=-1),
                'decoded_velocity_error_mps':torch.linalg.vector_norm(decoded.states[...,2:4]-truth.states[...,2:4],dim=-1),
                'raw_second_difference_error_mps':torch.linalg.vector_norm(residual,dim=-1),
                'raw_velocity_jerk_error_mps3':torch.linalg.vector_norm(residual/.01,dim=-1),
                'raw_velocity_field_jerk_mps3':torch.linalg.vector_norm(raw_velocity_jerk,dim=-1),
                'decoded_jerk_mps3':torch.linalg.vector_norm(j,dim=-1),
                'decoded_jerk_error_mps3':torch.linalg.vector_norm(j-qj,dim=-1),
                'position_correction_m':torch.linalg.vector_norm(decoded.position_correction,dim=-1)}
            for key,x in quantity.items():
                v=values(x)
                pooled[key].extend(v.ravel().tolist())
                r[key+'_p99']=float(np.quantile(v,.99))
                if key in ('decoded_jerk_mps3','raw_velocity_jerk_error_mps3'):
                    pooled[key+'_boundary'].extend(v[:,:4].ravel().tolist())
                    pooled[key+'_interior'].extend(v[:,4:].ravel().tolist())
            arrays={k:v[i].detach().cpu().numpy() for k,v in cond.items()}
            states=decoded.states[i].cpu().numpy()
            targetstates=truth.states[i].cpu().numpy()
            fmap=target['future_mask'][i].cpu().numpy()
            kw=dict(history=arrays['history'],history_mask=arrays['history_mask'],
                future_mask=fmap,initial_positions=arrays['initial_positions'],
                initial_heading=decoded.initial_heading[i].cpu().numpy(),agent_mask=a)
            motion=evaluate_motion(states=states,**kw)
            label=evaluate_motion(states=targetstates,**kw)
            r['motion_pass']=motion['hard_quality_pass']
            r['violation_counts']=motion['violation_counts']
            r['boundary_jerk_max']=motion['boundary']['jerk_mps3']['max']
            counts['motion_pass']+=motion['hard_quality_pass'] is True
            counts['comfort_assessed_agents']+=motion['comfort_assessed_agents']
            counts['comfort_pass_agents']+=motion['comfort_pass_agents']
            label_counts['motion_pass']+=label['hard_quality_pass'] is True
            for k,v in motion['violation_counts'].items(): counts[k+'_fail_scenes']+=v>0
            rows.append(r)
    return dict(planned_scenes=len(rows),counts=dict(counts),label_control=dict(label_counts),
        pooled={k:summary(v) for k,v in pooled.items()},
        mean_epsilon_mse=float(np.mean([r['epsilon_mse'] for r in rows])),
        channel_epsilon_mse=np.mean([r['channel_epsilon_mse'] for r in rows],axis=0).tolist(),
        label_conditioned_reconstruction=True,free_generation=False),rows

def diagnose(dataset,checkpoint,config_path,output,repository,command,formal=False):
    c=config(load_config(config_path))
    identity=file_identity(checkpoint)
    if identity['sha256']!=EXPECTED_CHECKPOINT: raise ValueError('Diagnosis requires the fixed 45000-step candidate')
    payload=load_checkpoint(checkpoint)
    if payload['step']!=45000: raise ValueError('Wrong checkpoint step')
    data=dataset_identity(dataset)
    if data['manifest']['sha256']!=payload['metadata']['data']['manifest']['sha256']:
        raise ValueError('Checkpoint/dataset mismatch')
    configure(c['seed'],c['torch_threads'])
    device=torch.device(c['device'])
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    caches={s:labeled_cache(dataset,s,c[k]) for s,k in
        [('train','train_windows'),('validation','validation_windows')]}
    if any(len(caches[s][2])!=c[k] for s,k in [('train','train_windows'),('validation','validation_windows')]):
        raise ValueError('Probe cohorts do not match requested sizes')
    train_ds=WindowDataset(dataset,'train',core_only=True)
    lookup={e['window_id']:i for i,e in enumerate(train_ds.entries)}
    all_ids=payload['metadata']['training_signature']['train_window_ids']
    audit_batches=[]
    for b in range(c['gradient_batches']):
        ix,epoch,eb=epoch_batch(45001+b,len(all_ids),4,payload['metadata']['training_config']['seed'])
        batchids=[all_ids[i] for i in ix.tolist()]
        audit_batches.append(dict(window_ids=batchids,epoch=epoch,epoch_batch=eb,
            samples=[train_ds[lookup[wid]] for wid in batchids]))
    audit_ids=[wid for b in audit_batches for wid in b['window_ids']]
    if len(set(audit_ids))!=len(audit_ids): raise ValueError('Audit batch identities duplicate')
    files=[config_path,*[v['location'] for v in data.values()],
        *input_files(train_ds,[lookup[wid] for wid in audit_ids],True)]
    for cache in caches.values():files.extend(cache[3])
    effective=dict(schema_version='sumodiff.velocity.validation.run.v1',probe=c,data=data,
        checkpoint=identity,training_sha=payload['metadata']['run_sha'],
        selected_window_ids={s:cache[2] for s,cache in caches.items()},gradient_window_ids=audit_ids,
        gradient_policy='same next-epoch IDs and checkpoint CPU-noise generator; eval-mode, locally padded B4, no optimizer',
        evaluation=dict(decoder=DecoderConfig().__dict__,motion=MotionConfig().__dict__),
        candidate_penalties='probe only, unweighted; not added to any training objective')
    with RunRecorder(output,repository,effective,command,{'reconstruction_noise':c['seed']},
        purpose='stage5c_low_noise_velocity_validation',data_files=sorted(set(map(str,files))),
        checkpoint=checkpoint,data_id=data['manifest']['id'],formal=formal) as run:
        model=ConditionalDenoiser(resolve_model_config(payload['metadata']['model_config'])).to(device)
        model.load_state_dict(payload['model'],strict=True)
        model.eval();model.requires_grad_(False)
        before=parameter_hash(model)
        schedule=NoiseSchedule(resolve_diffusion_config(payload['metadata']['diffusion_config']),device)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter()
        reconstruction={}
        for split,cache in caches.items():
            reconstruction[split]={}
            for timestep in c['timesteps']:
                result,rows=reconstruction_rows(model,cache,schedule,c,timestep,device)
                reconstruction[split][str(timestep)]=result
                write_json(run.output/f'reconstruction_{split}_t{timestep:03}.json',rows)
                print(f"reconstruction={split} t={timestep} motion={result['counts']['motion_pass']}/{len(rows)}",flush=True)
        model.requires_grad_(True)
        loss_config=resolve_unified_loss(payload['metadata']['training_config']['loss'])
        roads=RoadGeometryCache(dataset,audit_ids,loss_config)
        generator=torch.Generator();generator.set_state(payload['generator_state'])
        parameters=list(model.parameters())
        gradient_rows=[]
        for b,batch in enumerate(audit_batches):
            samples=batch['samples'];packed=collate_numpy(samples)
            cond=prepare_conditioning(packed,device)
            target={k:torch.as_tensor(v,device=device) for k,v in packed['targets'].items()}
            _,noise,natural=draw_noise(target,cond,schedule,generator)
            for kind,ts in [('natural',natural),('forced_low',torch.full_like(natural,c['forced_gradient_timestep']))]:
                noisy=torch.where(cond['agent_mask'][...,None,None],
                    schedule.add_noise(normalize(target['future'],schedule.config),noise,ts),0.)
                epsilon=model(noisy,ts,cond)
                terms=unified_objective(epsilon,noisy,noise,ts,cond,target,schedule,loss_config,
                    roads=roads.batch(batch['window_ids']),ramp=1.)
                raw=denormalize(schedule.clean_from_noise(noisy,epsilon,ts),schedule.config)
                probes=velocity_probe_terms(raw,cond,target,ts,schedule,c['second_difference_unit_mps'],c['boundary_frames'])
                reference=gradient_vector(terms['noise_mse'],parameters)
                fields={k:terms[k] for k in ('noise_mse','weighted_imitation','weighted_temporal',
                    'weighted_road','auxiliary_total')}
                fields.update(probes)
                stats={k:grad_stats(v,parameters,reference) for k,v in fields.items()}
                gradient_rows.append(dict(batch=b+1,kind=kind,window_ids=batch['window_ids'],
                    timesteps=ts.cpu().tolist(),eligible_scenes=terms['auxiliary_eligible_scenes'],components=stats))
            print(f'gradient_batches={b+1}/{len(audit_batches)}',flush=True)
        gradient_summary={}
        for kind in ('natural','forced_low'):
            selected=[r for r in gradient_rows if r['kind']==kind and r['eligible_scenes']]
            gradient_summary[kind]=dict(batches=c['gradient_batches'],eligible_batches=len(selected),
                components={k:dict(norm_ratio_median=float(np.median([r['components'][k]['norm_ratio_to_epsilon'] for r in selected])),
                    norm_ratio_max=float(max(r['components'][k]['norm_ratio_to_epsilon'] for r in selected)),
                    cosine_median=float(np.median([r['components'][k]['cosine_with_epsilon'] for r in selected
                        if r['components'][k]['cosine_with_epsilon'] is not None]))
                        if any(r['components'][k]['cosine_with_epsilon'] is not None for r in selected) else None)
                    for k in gradient_rows[0]['components']} if selected else None)
        synchronize(device)
        result=dict(schema_version='sumodiff.velocity.validation.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            checkpoint=identity,reconstruction=reconstruction,gradient_summary=gradient_summary,
            model_unchanged=parameter_hash(model)==before,optimizer_updates=0,
            parameter_grad_buffers_unset=all(p.grad is None for p in model.parameters()),
            wall_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
            interpretation='Conditional noised-label reconstruction and train-only gradient math, not generation quality or training improvement')
        if not result['model_unchanged'] or not result['parameter_grad_buffers_unset']:
            raise RuntimeError('Diagnosis changed model state')
        if file_identity(checkpoint)['sha256']!=identity['sha256']: raise RuntimeError('Checkpoint changed')
        write_json(run.output/'gradient_rows.json',gradient_rows)
        write_json(run.output/'velocity_summary.json',result)
        run.write_metrics(result)
    return result

def compare(dataset,checkpoint,original20,fresh20,fresh50,diagnostic,output,repository,command,formal=False):
    identity=file_identity(checkpoint)
    if identity['sha256']!=EXPECTED_CHECKPOINT:
        raise ValueError('Comparison requires the fixed 45000-step checkpoint')
    sources=[Path(p).resolve(strict=True) for p in (original20,fresh20,fresh50,diagnostic)]
    for p in sources:
        manifest=read(p/'manifest.json')
        if read(p/'status.json')['state']!='completed':
            raise ValueError('Comparison source is incomplete')
        if manifest['checkpoint']['sha256']!=identity['sha256']:
            raise ValueError('Source checkpoint differs')
        if formal and (not manifest['formal'] or manifest['git']['dirty']):
            raise ValueError('Formal comparison requires clean formal sources')
    settings=[load_config(p/'resolved_config.yaml')['sampling'] for p in sources[:3]]
    if settings[0]!=settings[1] or settings[0]['steps']!=20:
        raise ValueError('20-step replay differs from the original protocol')
    if dict(settings[2],steps=20)!=settings[1] or settings[2]['steps']!=50:
        raise ValueError('Only step count may change in the 50-step comparison')
    ds=WindowDataset(dataset,'validation')
    conditionings={e['window_id']:ds.inference(i)['conditioning'] for i,e in enumerate(ds.entries)}
    metadata={e['window_id']:e for e in ds.entries}
    old,os,of,noise=generation_rows(sources[0],conditionings,metadata)
    left,ls,lf,_=generation_rows(sources[1],conditionings,metadata,noise)
    right,rs,rf,_=generation_rows(sources[2],conditionings,metadata,noise)
    replay=[]
    for wid in noise:
        with np.load(sources[0]/'trajectories'/wid/'stages.npz',allow_pickle=False) as a:
            with np.load(sources[1]/'trajectories'/wid/'stages.npz',allow_pickle=False) as b:
                ok=all(np.array_equal(a[k],b[k]) for k in
                    ('initial_noise','raw_normalized_future','decoded_absolute_states'))
                replay.append(ok)
                if not ok: raise ValueError(f'20-step trajectory replay differs: {wid}')
    def details(source,rows):
        motion=Counter();comfort=Counter();families=defaultdict(lambda:dict(
            planned=0,quality=0,road_bad=0,route_bad=0,collision=0))
        samples=[];jerks=[[] for _ in range(40)]
        for row in rows:
            m=read(source/'trajectories'/row['window_id']/'metrics.json')['decoded']
            for k,v in m['motion']['violation_counts'].items():motion[k]+=v>0
            for k in ('comfort_assessed_agents','comfort_pass_agents','comfort_assessed_points','comfort_violation_points'):
                comfort[k]+=m['motion'][k]
            f=families[row['family']];f['planned']+=1;f['quality']+=int(row['quality_success'])
            f['road_bad']+=bool(row['road_violation']);f['route_bad']+=bool(row['route_violation'])
            f['collision']+=bool(row['collision']);samples.append(row['sampling_seconds'])
            with np.load(source/'trajectories'/row['window_id']/'stages.npz',allow_pickle=False) as a:
                from .quality_audit import physical_derivatives
                _,_,j=physical_derivatives(conditionings[row['window_id']],a['decoded_absolute_states'])
                value=np.linalg.norm(j,axis=-1)[a['agent_mask']]
                for t in range(40):jerks[t].extend(value[:,t].tolist())
        return dict(motion_fail_scenes=dict(motion),comfort=dict(comfort),families=dict(families),
            sampling_seconds=summary(samples),
            jerk_by_time=[dict(time_s=(t+1)/10,values=summary(v)) for t,v in enumerate(jerks)])
    ds20,ds50=details(sources[1],left),details(sources[2],right)
    diagnostic_result=read(sources[3]/'velocity_summary.json')
    if not diagnostic_result['model_unchanged'] or diagnostic_result['optimizer_updates']!=0:
        raise ValueError('Diagnostic model changed')
    keys=('quality_success','road_violation','route_violation','collision','decoded_interior_jerk_p99',
          'boundary_jerk_max','position_correction_max','sampling_seconds')
    intervals={k:paired_interval(left,right,k,20261020) for k in keys}
    files=[checkpoint,*of,*lf,*rf,*[p/'manifest.json' for p in sources],
           sources[3]/'status.json',sources[3]/'resolved_config.yaml',
           sources[3]/'velocity_summary.json',sources[3]/'gradient_rows.json']
    effective=dict(schema_version='sumodiff.ddim.velocity.report.config.v1',sources=list(map(str,sources)),
        checkpoint=identity,settings_20=settings[1],settings_50=settings[2],
        bootstrap_iterations=2000,bootstrap_seed=20261020,
        inference_only_intervention='DDIM 20 versus 50 on identical checkpoint, tasks and initial noise')
    with RunRecorder(output,repository,effective,command,{'bootstrap':20261020},
        purpose='stage5c_ddim_velocity_validation_report',data_files=sorted(set(map(str,files))),
        checkpoint=checkpoint,formal=formal) as run:
        result=dict(schema_version='sumodiff.ddim.velocity.report.v1',run_sha=run.manifest['git']['commit_sha'],
            checkpoint=identity,checkpoint_training_sha=load_checkpoint(checkpoint)['metadata']['run_sha'],
            verification=dict(original_20_bitwise_replays=sum(replay),noise_pairs=200,
                sampling_settings_differ_only_in_steps=True),
            counts={'steps20':ls['counts'],'steps50':rs['counts']},
            pooled={'steps20':ls['pooled'],'steps50':rs['pooled']},
            details={'steps20':ds20,'steps50':ds50},paired_intervals=intervals,
            reconstruction=diagnostic_result['reconstruction'],gradient_summary=diagnostic_result['gradient_summary'],
            diagnostic_identity=file_identity(sources[3]/'velocity_summary.json'),
            limitations=['Conditional validation only, one noise seed.',
                'Reconstruction uses noised future labels, not free-generation quality.',
                'Gradient probes do not update parameters or establish efficacy of a new loss.',
                'Train-only audit uses next-epoch IDs and locally padded B4, not an exact training replay.',
                'No attack roles or closed-loop result.'])
        write_json(run.output/'validation_summary.json',result)
        write_json(run.output/'scene_rows.json',{'steps20':left,'steps50':right})
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        names=list(ds20['families']);x=np.arange(len(names))
        for offset,key,color in [(-.18,'steps20','tab:blue'),(.18,'steps50','tab:orange')]:
            d=result['details'][key]
            axes[0].bar(x+offset,[100*d['families'][n]['quality']/d['families'][n]['planned'] for n in names],
                .36,label=key,color=color)
            axes[1].plot([v['time_s'] for v in d['jerk_by_time']],
                [v['values']['p99'] for v in d['jerk_by_time']],label=key,color=color)
        axes[0].set(xticks=x,xticklabels=[n.replace('_','\n') for n in names],ylabel='Quality pass (%)')
        axes[0].legend();axes[0].set_title('Paired 200-task free generation')
        axes[1].axhline(15,color='red',ls='--',label='Hard candidate limit')
        axes[1].set(yscale='log',xlabel='Future time (s)',ylabel='Jerk P99 (m/s^3)',title='Decoded jerk, no smoothing')
        axes[1].legend(loc='upper right')
        for split,color in [('train','tab:blue'),('validation','tab:orange')]:
            d=result['reconstruction'][split];ts=list(map(int,d))
            axes[2].plot(ts,[d[str(t)]['pooled']['decoded_jerk_mps3_interior']['p99'] for t in ts],
                '-o',label=split,color=color)
        axes[2].set(yscale='log',xlabel='Known-future corruption timestep',
            ylabel='Interior jerk P99 (m/s^3)',title='Label-conditioned reconstruction only')
        axes[2].legend()
        fig.savefig(run.output/'validation.png',dpi=170);fig.savefig(run.output/'validation.svg');plt.close(fig)
        result['figures']={name:file_identity(run.output/name) for name in ('validation.png','validation.svg')}
        write_json(run.output/'validation_summary.json',result)
        run.write_metrics(result)
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    for action in ('diagnose','compare'):
        a=sub.add_parser(action)
        for name in ('dataset','checkpoint','output'):a.add_argument('--'+name,type=Path,required=True)
        a.add_argument('--repository',type=Path,default=Path.cwd())
        a.add_argument('--formal',action='store_true')
        if action=='diagnose':a.add_argument('--config',type=Path,required=True)
        else:
            for name in ('original20','fresh20','fresh50','diagnostic'):
                a.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv)
    command=[sys.executable,*sys.orig_argv[1:]] if argv is None else [
        sys.executable,'-m','sumodiff.diffusion.velocity_validation',*argv]
    if args.action=='diagnose':
        result=diagnose(args.dataset,args.checkpoint,args.config,args.output,args.repository,command,args.formal)
        print(dict(model_unchanged=result['model_unchanged'],optimizer_updates=result['optimizer_updates'],
            wall_seconds=result['wall_seconds']))
    else:
        result=compare(args.dataset,args.checkpoint,args.original20,args.fresh20,args.fresh50,
            args.diagnostic,args.output,args.repository,command,args.formal)
        print(result['counts'])
    return 0

if __name__=='__main__':
    raise SystemExit(main())
