"""Bounded B16 joint-gradient calibration; never starts the 200-epoch run."""
from copy import deepcopy
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from sumodiff.data.dataset import WindowDataset,write_json
from sumodiff.models import ConditionalDenoiser,resolve_model_config
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.experiments.recorder import RunRecorder,load_config
from .epoch_training import epoch_config,verified_curated_dataset
from .boundary_validation import descriptors,make_cache
from .decoded_temporal_diagnostics import balanced_indices
from .raw_velocity_diagnostics import vector_statistics
from .velocity_validation import gradient_vector
from .raw_velocity_losses import resolve_raw_velocity_loss
from .unified_losses import RoadGeometryCache
from .jerk_tail import JerkTailConfig,fresh_objective
from .fresh_protocol import SCHEMA,recipe_id
from .data import dataset_identity,select_batch
from .config import resolve_diffusion_config
from .process import NoiseSchedule,normalize
from .training import draw_noise,synchronize
from .schedule import ScheduleConfig,UpdateSchedule

CANDIDATES=(1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3)
PARTS=('weighted_imitation','weighted_temporal','weighted_road','weighted_raw_boundary','weighted_raw_interior')

def candidate_ok(noise,baseline,tail,weight,clip):
    aux=baseline+weight*tail;total=noise+aux
    reference=vector_statistics(noise+baseline,noise,clip)
    a=vector_statistics(aux,noise,clip);t=vector_statistics(total,noise,clip)
    tail_stat=vector_statistics(weight*tail,noise,clip)
    # Numerical influence guards, not evidence of useful generated trajectories.
    ratio=tail_stat['norm_ratio_to_epsilon']
    cosine=t['cosine_with_epsilon'];base_cos=reference['cosine_with_epsilon']
    passed=(ratio is not None and ratio<=.25 and cosine is not None and base_cos is not None
            and cosine>=base_cos-.1
            and t['parameter_gradient_norm']<=1.25*reference['parameter_gradient_norm'])
    return dict(auxiliary=a,total=t,weighted_tail=tail_stat,passed=bool(passed))

def calibrate(dataset,model_path,config_path,output,repository,command,formal=False,development=False):
    source=load_config(config_path);c=epoch_config(source)
    if c['schema_version']!=SCHEMA or c['batch_size']!=16 or c['tail_weight'] is not None:
        raise ValueError('Calibration requires the unresolved fresh B16 recipe')
    if development and formal:raise ValueError('Development calibration cannot be formal')
    if not development:verified_curated_dataset(dataset,c)
    ds=WindowDataset(dataset,'train',core_only=True);meta=descriptors(ds)
    groups={name:balanced_indices(meta,16,c['seed'],dense)
            for name,dense in [('ordinary',False),('dense_intersection',True)]}
    selected=list(dict.fromkeys(i for values in groups.values() for i in values))
    cache=make_cache(ds,selected);local={old:new for new,old in enumerate(selected)}
    architecture=resolve_model_config(load_config(model_path));data=dataset_identity(dataset)
    effective=dict(schema_version='sumodiff.fresh.calibration.run.v1',training=c,model=architecture.to_dict(),
        data=data,development_only=development,training_window_ids=cache[2],
        candidate_tail_weights=list(CANDIDATES),states=['random_initialization','after_32_warmup_updates'],
        modes=['natural','fixed_t0','fixed_t50','fixed_t200'],
        guard='tail norm <=25% epsilon; total cosine drop <=0.1; total norm <=1.25 baseline')
    files=[config_path,model_path,*cache[3],*[v['location'] for v in data.values()]]
    with RunRecorder(output,repository,effective,command,{'model':c['seed'],'noise':c['noise_seed']},
            purpose='fresh_training_joint_calibration',data_files=files,data_id=data['manifest']['id'],formal=formal) as run:
        configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        model=ConditionalDenoiser(architecture).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['minimum_learning_rate'],weight_decay=c['weight_decay'])
        scheduler=UpdateSchedule(ScheduleConfig(**c['schedule']))
        diffusion=resolve_diffusion_config(c['diffusion']);schedule=NoiseSchedule(diffusion,device)
        raw=resolve_raw_velocity_loss(c['raw_loss']);tail=JerkTailConfig(**c['jerk_tail'])
        roads=RoadGeometryCache(dataset,cache[2],raw);params=list(model.parameters())
        generator=torch.Generator().manual_seed(c['noise_seed'])
        rows=[];timings=[];initial_hash=parameter_hash(model);started=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        for state in ('random_initialization','after_32_warmup_updates'):
            for name,indices in groups.items():
                cond,target,ids=select_batch(cache,np.asarray([local[i] for i in indices]),device)
                natural,noise,ts=draw_noise(target,cond,schedule,generator)
                for mode in ('natural','fixed_t0','fixed_t50','fixed_t200'):
                    times=ts if mode=='natural' else torch.full_like(ts,int(mode[7:]))
                    noisy=natural if mode=='natural' else torch.where(cond['agent_mask'][...,None,None],
                        schedule.add_noise(normalize(target['future'],diffusion),noise,times),0.)
                    terms=fresh_objective(model(noisy,times,cond),noisy,noise,times,cond,target,schedule,
                        raw,tail,1.,roads=roads.batch(ids),ramp=1.)
                    gn=gradient_vector(terms['noise_mse'],params)
                    gb=gradient_vector(terms['baseline_auxiliary'],params)
                    gj=gradient_vector(terms['tail'],params)
                    candidate={str(w):candidate_ok(gn,gb,gj,w,c['gradient_clip_norm']) for w in CANDIDATES}
                    # Direct autograd of the complete selected candidate verifies actual combination.
                    allowed=[w for w in CANDIDATES if candidate[str(w)]['passed']]
                    chosen=max(allowed) if allowed else CANDIDATES[0]
                    direct=gradient_vector(terms['noise_mse']+terms['baseline_auxiliary']+chosen*terms['tail'],params)
                    if not torch.allclose(direct,gn+gb+chosen*gj,rtol=2e-4,atol=1e-7):
                        raise RuntimeError('Combined gradient does not match direct autograd')
                    parts={k:vector_statistics(gradient_vector(terms[k],params),gn,c['gradient_clip_norm']) for k in PARTS}
                    rows.append(dict(state=state,group=name,mode=mode,window_ids=ids,timesteps=times.cpu().tolist(),
                        eligible_scenes=terms['auxiliary_eligible_scenes'],epsilon=vector_statistics(gn,gn,c['gradient_clip_norm']),
                        baseline_auxiliary=vector_statistics(gb,gn,c['gradient_clip_norm']),parts=parts,candidates=candidate,
                        direct_combination_verified=True,losses={k:float(v.detach()) for k,v in terms.items()
                            if isinstance(v,torch.Tensor) and v.ndim==0}))
                    print(f'calibration state={state} group={name} mode={mode}',flush=True)
                    del terms,gn,gb,gj,direct
            if state=='random_initialization':
                for step in range(1,33):
                    name='ordinary' if step%2 else 'dense_intersection'
                    cond,target,ids=select_batch(cache,np.asarray([local[i] for i in groups[name]]),device)
                    synchronize(device);tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                    noisy,noise,ts=draw_noise(target,cond,schedule,generator)
                    lr,ramp=scheduler.apply(optimizer,step)
                    terms=fresh_objective(model(noisy,ts,cond),noisy,noise,ts,cond,target,schedule,
                        raw,tail,CANDIDATES[0],roads=roads.batch(ids),ramp=ramp)
                    terms['total_loss'].backward()
                    norm=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip_norm'],error_if_nonfinite=True)
                    optimizer.step();scheduler.complete(step);synchronize(device)
                    timings.append(dict(step=step,seconds=time.perf_counter()-tick,gradient_norm=float(norm),
                        lr=lr,ramp=ramp,clip=float(norm)>c['gradient_clip_norm'],batch_size=16))
                    del terms
        usable=[w for w in CANDIDATES if all(r['candidates'][str(w)]['passed'] for r in rows)]
        chosen=max(usable) if usable else None
        warmed_hash=parameter_hash(model)
        cache_bytes=sum(v.numel()*v.element_size() for part in cache[:2] for v in part.values())
        stress=[]
        calibration_peak_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        if chosen is not None:
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            for name in groups:
                cond,target,ids=select_batch(cache,np.asarray([local[i] for i in groups[name]]),device)
                for mode in ('natural','fixed_t200'):
                    synchronize(device);tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                    noisy,noise,ts=draw_noise(target,cond,schedule,generator)
                    if mode=='fixed_t200':
                        ts=torch.full_like(ts,200)
                        noisy=torch.where(cond['agent_mask'][...,None,None],schedule.add_noise(normalize(target['future'],diffusion),noise,ts),0.)
                    for pg in optimizer.param_groups:pg['lr']=c['peak_learning_rate']
                    terms=fresh_objective(model(noisy,ts,cond),noisy,noise,ts,cond,target,schedule,
                        raw,tail,chosen,roads=roads.batch(ids),ramp=1.)
                    terms['total_loss'].backward()
                    norm=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip_norm'],error_if_nonfinite=True)
                    optimizer.step();synchronize(device)
                    stress.append(dict(group=name,mode=mode,seconds=time.perf_counter()-tick,gradient_norm=float(norm),
                        clipping_applied=float(norm)>c['gradient_clip_norm'],lr=c['peak_learning_rate'],auxiliary_ramp=1.,
                        loss=float(terms['total_loss'].detach()),batch_size=16))
                    del terms
        peak_alloc=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        peak_reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None
        report=dict(schema_version='sumodiff.fresh.calibration.v1',run_sha=run.manifest['git']['commit_sha'],
            development_only=development,recipe_sha256=recipe_id(c,architecture),
            dataset_hashes={k:v['sha256'] for k,v in data.items()},selected_tail_weight=chosen,passed=chosen is not None,
            rows=rows,warmup_updates=timings,initial_parameter_sha256=initial_hash,
            warmed_parameter_sha256=warmed_hash,stress_final_parameter_sha256=parameter_hash(model),schedule=scheduler.state_dict(),
            cached_tensor_bytes=cache_bytes,cache_windows=len(cache[2]),
            estimated_full_cohort_cache_bytes=cache_bytes*(c['expected_train_windows']+c['expected_validation_windows'])/len(cache[2]),
            peak_allocated_bytes=peak_alloc,peak_reserved_bytes=peak_reserved,
            calibration_peak_allocated_bytes=calibration_peak_allocated,full_strength_stress=stress,
            stress_semantics='four isolated private peak-LR/full-aux updates after frozen probes; not production training',
            optimizer_mean_seconds=float(np.mean([t['seconds'] for t in timings[1:]])),
            full_strength_optimizer_mean_seconds=float(np.mean([t['seconds'] for t in stress])) if stress else None,
            total_seconds=time.perf_counter()-started,training_labels_only=True,
            limitations=['32 tiny warmup updates do not establish learning or convergence',
                'gradient guards are numerical candidate selection, not generated quality validation',
                'all-strength probes intentionally differ from early scheduled auxiliary coefficients'])
        write_json(run.output/'calibration.json',report);run.write_metrics(report)
        if chosen is None:raise ValueError('No candidate passes joint gradient guards; report retained')
        if not development:
            resolved=deepcopy(source);resolved.update(tail_weight=chosen,calibration_report=str(run.output/'calibration.json'))
            (run.output/'training_config.yaml').write_text(yaml.safe_dump(resolved,sort_keys=True),encoding='utf-8')
    return report
