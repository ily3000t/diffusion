"""Replay generated arrays from a relocated projection with no label files."""
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from sumodiff.data.dataset import WindowDataset,collate_numpy,write_json
from sumodiff.models import ConditionalDenoiser,resolve_model_config,prepare_conditioning
from sumodiff.models.cli import configure
from sumodiff.decoding import DecoderConfig,decode_future
from sumodiff.evaluation.metrics import _heading
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from .checkpoint import load_checkpoint
from .config import resolve_diffusion_config
from .data import choose_indices,input_files,dataset_identity
from .sampling import sample_config,task_seed
from .process import NoiseSchedule,ddim_sample,denormalize


def make_projection(dataset,indices,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    # Keep the original authenticated catalog but intentionally remove label
    # access; all input metadata for this split is needed for task selection.
    for name in ('dataset_manifest.json','windows.json'): shutil.copyfile(dataset.directory/name,destination/name)
    selected=set(indices);files=[]
    for i,entry in enumerate(dataset.entries):
        names=('input.json',)+ (('conditioning.npz','map.json') if i in selected else ())
        for name in names:
            source=dataset._path(entry,name);target=destination/entry['files'][name]['relative_path']
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            if file_identity(target)['sha256']!=entry['files'][name]['sha256']: raise ValueError('Projection copy integrity failure')
            files.append(target)
    if list(destination.rglob('targets.npz')) or list(destination.rglob('labels.json')): raise RuntimeError('Projection leaked label files')
    return files


def replay(dataset_path,checkpoint,source_run,output,repository,command,formal=False):
    source=Path(source_run);manifest=json.loads((source/'manifest.json').read_text());effective=load_config(source/'resolved_config.yaml')
    if json.loads((source/'status.json').read_text())['state']!='completed': raise ValueError('Source generation did not complete')
    if formal and (not manifest['formal'] or manifest['git']['dirty']): raise ValueError('Formal replay requires a clean formal source')
    if file_identity(source/'resolved_config.yaml')['sha256']!=manifest['resolved_config_sha256']: raise ValueError('Source config integrity failure')
    if file_identity(checkpoint)['sha256']!=manifest['checkpoint']['sha256']: raise ValueError('Source checkpoint integrity failure')
    c=sample_config(effective['sampling']);ds=WindowDataset(dataset_path,c['split']);indices=choose_indices(ds,c['limit'],c['minimum_agents'])
    tasks=[ds._json(ds.entries[i],'input.json') for i in indices]
    if [t['window_id'] for t in tasks]!=effective['task_window_ids']: raise ValueError('Replay tasks changed')
    data=dataset_identity(dataset_path)
    if any(data[k]['sha256']!=effective['data'][k]['sha256'] for k in data): raise ValueError('Replay dataset identity changed')
    catalog=json.loads((source/'trajectory_index.json').read_text())
    if [r['window_id'] for r in catalog]!=effective['task_window_ids']: raise ValueError('Source output catalog changed')
    output_files=[]
    for r in catalog:
        if r['generation_status']!='completed': raise ValueError('Replay expects completed source generation')
        for name,identity in r['files'].items():
            p=source/'trajectories'/r['window_id']/name
            if file_identity(p)['sha256']!=identity['sha256']: raise ValueError('Source trajectory integrity failure')
            output_files.append(p)
    payload=load_checkpoint(checkpoint);architecture=resolve_model_config(payload['metadata']['model_config']);diffusion=resolve_diffusion_config(payload['metadata']['diffusion_config'])
    files=[source/name for name in ('manifest.json','resolved_config.yaml','status.json','trajectory_index.json')]+output_files
    files+=input_files(ds,indices,False)
    files += [ds._path(e,'input.json') for e in ds.entries]+[v['location'] for v in data.values()]
    with RunRecorder(output,repository,dict(schema_version='sumodiff.inference.replay.config.v1',source_manifest=file_identity(source/'manifest.json'),sampling=c,data=data,
        checkpoint=file_identity(checkpoint),comparison='bitwise initial noise, normalized raw future, physical raw and decoded states; same runtime'),command,
        {'initial_noise':c['seed']},purpose='stage5_relocated_label_free_inference_replay',data_files=sorted(set(map(str,files))),checkpoint=checkpoint,formal=formal) as run:
        projection_files=make_projection(ds,indices,run.output/'input_only_projection');projection=WindowDataset(run.output/'input_only_projection',c['split'])
        selected=choose_indices(projection,c['limit'],c['minimum_agents'])
        if selected!=indices: raise ValueError('Projection task selection changed')
        configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
        if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        model=ConditionalDenoiser(architecture).to(device);model.load_state_dict(payload['model']);model.eval();model.requires_grad_(False)
        schedule=NoiseSchedule(diffusion,device);rows=[]
        for i,meta in zip(selected,tasks):
            source_sample=projection.inference(i);conditions=source_sample['conditioning'];heading=_heading(conditions)
            tensors=prepare_conditioning(collate_numpy([source_sample]),device);seed=task_seed(c['seed'],meta['window_id'])
            initial=torch.randn((1,len(conditions['agent_mask']),40,6),generator=torch.Generator().manual_seed(seed)).to(device)
            raw=ddim_sample(model,tensors,initial,schedule,c['steps'],c['eta'],generator=torch.Generator(device=device).manual_seed(seed))
            physical=denormalize(raw,diffusion)[0]
            decoded=decode_future(physical,tensors['initial_positions'][0],torch.as_tensor(heading,device=device),tensors['agent_mask'][0],DecoderConfig(**c['evaluation']['decoder']))
            actual=dict(initial_noise=initial[0].cpu().numpy(),raw_normalized_future=raw[0].cpu().numpy(),raw_future_delta=physical.cpu().numpy(),
                raw_absolute_states=decoded.raw_states.cpu().numpy(),decoded_absolute_states=decoded.states.cpu().numpy())
            with np.load(source/'trajectories'/meta['window_id']/'stages.npz',allow_pickle=False) as expected:
                rows.append(dict(window_id=meta['window_id'],equal={k:bool(np.array_equal(v,expected[k])) for k,v in actual.items()},
                    max_absolute_error={k:float(np.abs(v-expected[k]).max()) for k,v in actual.items()}))
        result=dict(schema_version='sumodiff.inference.replay.metrics.v1',run_sha=run.manifest['git']['commit_sha'],tasks=len(rows),
            source_run_sha=manifest['git']['commit_sha'],label_files_present=False,all_arrays_bitwise_equal=all(all(r['equal'].values()) for r in rows),
            projection_files=[file_identity(p) for p in projection_files],rows=rows,scope='label-free relocated projection, not new simulation data')
        write_json(run.output/'inference_replay.json',result);run.write_metrics(result)
        if not result['all_arrays_bitwise_equal']: raise RuntimeError('Label-free inference replay differed')
    return result
