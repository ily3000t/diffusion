"""Bounded directional-condition check on existing training raw episodes; no training."""
from dataclasses import replace
from pathlib import Path
import json,sys,time
import numpy as np
import torch
from .config import resolve_model_config
from .denoiser import ConditionalDenoiser
from .inputs import prepare_conditioning
from .probes import invariance_probe,parameter_hash,profile,full_capacity_conditioning,seeded_noise
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from sumodiff.data.episodes import load_episode
from sumodiff.data.windows import build_window
from sumodiff.data.dataset import collate_numpy,write_json
from sumodiff.geometry.maps import RoadMap
from sumodiff.experiments.stage5a_plan import resolve_plan
from sumodiff.diffusion.checkpoint import load_checkpoint,restore_training
from sumodiff.diffusion.audit import exact_tree

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def resolve_probe(value):
    expected=dict(schema_version='sumodiff.directional.conditioning.probe.v1',
        collection='artifacts/runs/ramp-alignment-collect-20261014-r1',
        model_config='configs/model/control_directional.yaml',reference_tick=300,
        seed=20261014,noise_seed=20261015,device='cuda:0',iterations=2,warmups=1,synthetic_batch_size=4,
        legacy_checkpoints=['artifacts/runs/stage5a-fresh-recovery1-resume/checkpoint_step_075000.pt',
                            'artifacts/runs/control-short-full-20261014-r1/checkpoint_step_000016.pt'])
    if value!=expected or any(type(value[k]) is not int for k in ('reference_tick','seed','noise_seed','iterations','warmups','synthetic_batch_size')):
        raise ValueError('Directional probe protocol is frozen to five train inputs and a bounded GPU check')
    return value

def probe(config_path,output,formal=False):
    from .cli import configure
    cfg=resolve_probe(load_config(config_path));root=Path.cwd()
    source=Path(cfg['collection']);sm=read(source/'manifest.json');sc=load_config(source/'resolved_config.yaml')
    if read(source/'status.json')['state']!='completed' or not sm['formal'] or sm['git']['dirty']:
        raise ValueError('Require completed clean formal training source collection')
    if file_identity(source/'resolved_config.yaml')['sha256']!=sm['resolved_config_sha256']:
        raise ValueError('Source collection config drift')
    records=read(source/'collection_results.json')['cases']
    ids=[f'ramp_merge-g{g:02d}-combined-s{s}' for g in (0,1) for s in (20261105,20261106)]
    ids.append('ramp_merge-g00-control-s20261105')
    by_id={r['case_id']:r for r in records};tasks=[]
    files=[config_path,cfg['model_config'],*[source/n for n in ('manifest.json','status.json','resolved_config.yaml','collection_results.json')]]
    for case_id in ids:
        rec=by_id[case_id];declared=next(c for c in sc['cases'] if c['case_id']==case_id)
        if rec['split']!='train' or declared['split']!='train' or not rec['completed'] or not rec['eligible_for_normal_training']:
            raise ValueError('Only completed training-owned episodes may enter condition design')
        p=Path(rec['episode_manifest']);em=read(p);parent=p.parent
        if em['run_sha']!=sm['git']['commit_sha']:raise ValueError('Raw source code SHA differs')
        files += [p,parent/'manifest.json',parent/'status.json',parent/'resolved_config.yaml',rec['config']['location']]
        files += [v['location'] for v in em['outputs']]+[v['location'] for v in em['scene']['files']]
        tasks.append(dict(case_id=case_id,episode_manifest=str(p),reference_tick=cfg['reference_tick'],split='train'))
    files+=cfg['legacy_checkpoints']+['configs/experiments/stage5a.yaml']
    for identity in sm['data']['files']:
        actual=file_identity(identity['location'])
        if (actual['sha256'],actual['size_bytes'])!=(identity['sha256'],identity['size_bytes']):
            raise ValueError('Source collection input drift')
        files.append(identity['location'])
    architecture=resolve_model_config(load_config(cfg['model_config']))
    if not architecture.directional_lane_permissions:raise ValueError('Probe requires the explicit directional architecture')
    effective=dict(protocol=cfg,model=architecture.to_dict(),tasks=tasks,
        scope='conditioning/gradient/resources only; random new model, zero optimizer updates',
        raw_scope='authenticate full files; window features constructed from t0 prefix only; no stored label arrays')
    with RunRecorder(output,root,effective,[sys.executable,'-m','sumodiff.models',*sys.argv[1:]],
        {'model':cfg['seed'],'noise':cfg['noise_seed']},purpose='directional_lane_conditioning_validation',
        data_files=sorted(set(map(str,files))),formal=formal) as run:
        started=time.perf_counter();samples=[];input_checks=[]
        wc=resolve_plan(load_config('configs/experiments/stage5a.yaml'))['window']
        for task in tasks:
            ep=load_episode(task['episode_manifest'],'train');t0=task['reference_tick']
            # Prefix truncation makes future observations unavailable to build_window.
            prefix=replace(ep,frames=ep.frames[:t0+1]);road=RoadMap.read(ep.network_path)
            window,reason=build_window(prefix,t0,road,dict(wc,reference_policy='interaction_current'))
            if reason:raise ValueError('Fixed condition task unavailable: '+reason)
            sample=dict(conditioning=window.conditioning,input_metadata=window.input_metadata,exact_map=window.exact_map)
            c=prepare_conditioning(collate_numpy([sample]),config=architecture)
            blocked=int((~c['lane_change_permissions']&c['lane_mask'][...,None]).sum())
            legacy_rejected=False
            try:prepare_conditioning(collate_numpy([sample]))
            except NotImplementedError:legacy_rejected=True
            assert legacy_rejected==(task['case_id']!=ids[-1])
            assert blocked==(3 if legacy_rejected else 0)
            input_checks.append(dict(**task,active_agents=int(c['agent_mask'].sum()),lanes=int(c['lane_mask'].sum()),
                restricted_directions=blocked,legacy_rejected=legacy_rejected,future_observations_available=False))
            samples.append(sample)
        configure(cfg['seed'],1);model=ConditionalDenoiser(architecture);initial_hash=parameter_hash(model)
        legacy_checks=[];unrestricted=collate_numpy([samples[-1]])
        for path in cfg['legacy_checkpoints']:
            payload=load_checkpoint(path);old_config=resolve_model_config(payload['metadata']['model_config'])
            assert not old_config.directional_lane_permissions
            old_model=ConditionalDenoiser(old_config);old_model.load_state_dict(payload['model'],strict=True);old_model.eval()
            assert exact_tree(dict(old_model.state_dict()),payload['model'])
            cond=prepare_conditioning(unrestricted,config=old_config)
            noise,ts=seeded_noise(cond,cfg['noise_seed'],old_config.diffusion_embedding_steps,old_config.state_dimension)
            with torch.no_grad():prediction=old_model(noise,ts,cond)
            assert torch.isfinite(prediction).all()
            optimizer=torch.optim.AdamW(model.parameters())
            try:restore_training(payload,model,optimizer,torch.Generator(),'cpu')
            except ValueError as exc:
                if 'architecture' not in str(exc):raise
            else:raise AssertionError('Cross-version resume was accepted')
            assert parameter_hash(model)==initial_hash
            legacy_checks.append(dict(checkpoint=file_identity(path),schema=old_config.schema_version,
                step=payload['step'],strict_load=True,weights_bit_identical=True,finite_forward=True,
                prediction_shape=list(prediction.shape),cross_version_resume_rejected=True))
            del old_model,payload,optimizer
        device=torch.device(cfg['device'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
        # Reset initialization after old-model checks consumed the CPU RNG.
        configure(cfg['seed'],1);model=ConditionalDenoiser(architecture).to(device)
        assert parameter_hash(model)==initial_hash
        source_c=prepare_conditioning(collate_numpy(samples),config=architecture)
        c={k:v.to(device) for k,v in source_c.items()}
        checks,arrays=invariance_probe(model,c,cfg['noise_seed'])
        noise,ts=seeded_noise(c,cfg['noise_seed'],architecture.diffusion_embedding_steps,3)
        with torch.no_grad():
            original=model(noise,ts,c)
            modified={k:v.clone() for k,v in c.items()}
            # Flip each real restriction, leaving all geometry and current traffic unchanged.
            modified['lane_change_permissions'][c['lane_mask']]=True
            altered=model(noise,ts,modified)
            sensitivity=float((altered-original).abs().max())
            assert sensitivity>1e-8
        order=torch.arange(c['lane_mask'].shape[1]-1,-1,-1,device=device);changed={k:v.clone() for k,v in c.items()}
        for key in ('lane_polylines','lane_mask','lane_point_mask','lane_change_permissions'):changed[key]=c[key][:,order]
        for key in ('lane_adjacency','lane_rule_relations'):changed[key]=c[key][:,order][:,:,order]
        for key in ('route_lane_mask','route_lane_features'):changed[key]=c[key][:,:,order]
        with torch.no_grad():permuted=model(noise,ts,changed)
        lane_error=float((permuted-original).abs().max());assert lane_error<2e-5
        resources=[]
        for kind,condition in [('five_observed_train_windows',source_c),('synthetic_full_capacity_b4',full_capacity_conditioning(source_c,cfg['synthetic_batch_size']))]:
            for fusion in ('hierarchical','parallel'):
                record=profile(model,condition,device,fusion,cfg['noise_seed'],cfg['iterations'],cfg['warmups'])
                record['input_kind']=kind;resources.append(record)
        assert parameter_hash(model)==initial_hash
        np.savez_compressed(run.output/'untrained_outputs.npz',**arrays)
        result=dict(schema_version='sumodiff.directional.conditioning.results.v1',run_sha=run.manifest['git']['commit_sha'],
            model_config=architecture.to_dict(),source_sha=sm['git']['commit_sha'],input_checks=input_checks,
            legacy_checkpoints=legacy_checks,structural=checks,permission_flip_prediction_max_abs_difference=sensitivity,
            lane_permutation_max_abs_error=lane_error,parameters_unchanged=True,parameter_sha256=initial_hash,
            parameter_count=sum(p.numel() for p in model.parameters()),resource_cases=resources,
            optimizer_updates=0,stored_label_arrays_loaded=0,validation_episode_arrays_loaded=0,test_episode_arrays_loaded=0,
            elapsed_seconds=time.perf_counter()-started,quality_evaluation_performed=False,
            limitations=['random noise predictor; sensitivity is not learned compliance or quality',
                'synthetic capacity is a resource load, not a 12-car SUMO sample',
                'forward/backward memory excludes optimizer states; not formal training throughput'])
        write_json(run.output/'results.json',result);run.write_metrics(result)
    return result
