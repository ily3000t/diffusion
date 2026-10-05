"""Training-only label access, inference task selection, and scale provenance."""
from pathlib import Path
import json
import numpy as np
import torch
from sumodiff.data.dataset import WindowDataset,collate_numpy
from sumodiff.models import prepare_conditioning
from sumodiff.experiments.recorder import file_identity


def dataset_identity(directory):
    directory=Path(directory).resolve(strict=True)
    return dict(manifest=file_identity(directory/'dataset_manifest.json'),index=file_identity(directory/'windows.json'))


def choose_indices(dataset,limit,minimum_agents=1):
    # Round-robin road families using only pre-existing input metadata. No labels.
    families={}
    for i,entry in enumerate(dataset.entries):
        meta=dataset._json(entry,'input.json')
        if sum(v is not None for v in meta['agent_ids'])<minimum_agents: continue
        families.setdefault(meta['family'],[]).append(i)
    chosen=[]
    while any(families.values()) and (limit is None or len(chosen)<limit):
        for family in sorted(families):
            if families[family] and (limit is None or len(chosen)<limit): chosen.append(families[family].pop(0))
    return chosen


def input_files(dataset,indices,labels=False):
    names=('conditioning.npz','input.json','map.json')+ (('targets.npz','labels.json') if labels else ())
    return [dataset._path(dataset.entries[i],name) for i in indices for name in names]


def labeled_cache(directory,split,limit):
    if split not in ('train','validation'): raise ValueError('Training/probe cache cannot use test')
    ds=WindowDataset(directory,split,core_only=True)
    indices=choose_indices(ds,limit)
    if not indices: raise ValueError(f'Empty {split} dataset')
    samples=[ds[i] for i in indices]
    batch=collate_numpy(samples)
    conditioning=prepare_conditioning(batch,'cpu')
    targets={k:torch.as_tensor(v) for k,v in batch['targets'].items()}
    if not targets['future_mask'][conditioning['agent_mask']].all(): raise ValueError('Core training requires complete future labels')
    return conditioning,targets,[s['input_metadata']['window_id'] for s in samples],input_files(ds,indices,True)


def select_batch(cache,indices,device):
    c,targets,ids,_=cache
    return ({k:v[indices].to(device) for k,v in c.items()},
            {k:v[indices].to(device) for k,v in targets.items()},[ids[i] for i in indices.tolist()])


def scale_statistics(directory,config):
    rows=[];position=[];velocity=[];history_speed=[];files=[]
    for split in ('train','validation'):
        ds=WindowDataset(directory,split)
        valid_frames=0;core=0
        for i in range(len(ds)):
            sample=ds[i];c=sample['conditioning'];t=sample['targets']
            mask=c['agent_mask'][:,None]&t['future_mask']
            f=t['future'][mask]
            if not np.isfinite(f).all(): raise ValueError('Nonfinite valid labels')
            position.extend(np.linalg.norm(f[:,:2],axis=-1).tolist())
            velocity.extend(np.linalg.norm(f[:,2:4],axis=-1).tolist())
            hm=c['agent_mask'][:,None]&c['history_mask']
            history_speed.extend(np.linalg.norm(c['history'][hm][:,2:4],axis=-1).tolist())
            valid_frames+=len(f);core+=ds.entries[i]['core_training_eligible']
        files+=input_files(ds,range(len(ds)),True)
        rows.append(dict(split=split,windows=len(ds),core_windows=core,valid_future_agent_frames=valid_frames))
    def stats(values,unit):
        a=np.asarray(values)
        return dict(count=len(a),p50=float(np.quantile(a,.5)),p99=float(np.quantile(a,.99)),max=float(a.max()),fraction_above_scale=float((a>unit).mean()))
    return dict(schema_version='sumodiff.scale.audit.v1',dataset=dataset_identity(directory),diffusion=config.to_dict(),splits=rows,
        displacement_norm_m=stats(position,config.position_scale_m),speed_norm_mps=stats(velocity,config.velocity_scale_mps),
        history_speed_norm_mps=stats(history_speed,config.velocity_scale_mps),
        interpretation='fixed physical units, not range bounds; no clipping; no test data or per-scene statistics'),files


def verified_scale_audit(path,directory,config):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    actual=dataset_identity(directory)
    if value['schema_version']!='sumodiff.scale.audit.v1' or value['diffusion']!=config.to_dict(): raise ValueError('Scale audit/config mismatch')
    for key in actual:
        if actual[key]['sha256']!=value['dataset'][key]['sha256']: raise ValueError('Scale audit dataset mismatch')
    if [r['split'] for r in value['splits']]!=['train','validation']: raise ValueError('Scale audit must exclude test')
    return file_identity(path)
