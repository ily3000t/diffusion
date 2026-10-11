"""True epoch traversal and curated-data guards, preserving legacy step-training signatures."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import yaml
from sumodiff.data.dataset import WindowDataset,write_json
from sumodiff.experiments.recorder import file_identity

SCHEMA='sumodiff.base.epoch.config.v1'
EXTRA=dict(epochs=100,expected_train_windows=1200,expected_validation_windows=200,
    budget_kind='full',train_probe_limit=60,validation_every_epochs=1,checkpoint_every_epochs=5,
    monitor_every_epochs=10,monitor={})


def epoch_config(value):
    from .training import TRAIN_DEFAULTS,train_config
    if not isinstance(value,dict): raise ValueError('Epoch training config must be a mapping')
    from .fresh_protocol import SCHEMA as FRESH_SCHEMA, EXTRA as FRESH_EXTRA, resolve_extra
    fresh=value.get('schema_version')==FRESH_SCHEMA
    allowed=(set(TRAIN_DEFAULTS)-{'max_steps','validation_interval','checkpoint_interval','train_limit','validation_limit'})|set(EXTRA)
    if fresh:allowed|=set(FRESH_EXTRA)
    if set(value)-allowed or value.get('schema_version') not in (SCHEMA,FRESH_SCHEMA):
        raise ValueError('Unknown/unsupported epoch training options; step budgets/limits must not masquerade as epochs')
    base={k:v for k,v in value.items() if k in TRAIN_DEFAULTS}
    base['schema_version']=TRAIN_DEFAULTS['schema_version']
    base.update(train_limit=None,validation_limit=None)
    base.setdefault('require_probe_improvement',False)
    if fresh:
        if 'additional_losses_enabled' in base.get('diffusion',{}) and base['diffusion']['additional_losses_enabled'] is not True:
            raise ValueError('Fresh training requires its explicit auxiliary-loss capability')
        base['diffusion']=deepcopy(base.get('diffusion',{}));base['diffusion']['additional_losses_enabled']=False
    c=train_config(base)
    if fresh:c['diffusion']['additional_losses_enabled']=True
    extra=deepcopy(EXTRA);extra.update({k:v for k,v in value.items() if k in EXTRA})
    for k in ('epochs','expected_train_windows','expected_validation_windows','train_probe_limit',
              'validation_every_epochs','checkpoint_every_epochs'):
        if type(extra[k]) is not int or extra[k]<1: raise ValueError(f'Invalid {k}')
    if type(extra['monitor_every_epochs']) is not int or extra['monitor_every_epochs']<0:
        raise ValueError('monitor_every_epochs must be nonnegative; 0 explicitly disables it')
    if extra['budget_kind'] not in ('smoke','full'): raise ValueError('Invalid epoch budget kind')
    if extra['budget_kind']=='smoke' and (extra['epochs']>5 or extra['expected_train_windows']>60 or extra['expected_validation_windows']>30):
        raise ValueError('Smoke budget is limited to 5 epochs and 60/30 windows')
    from .sampling import sample_config
    monitor=deepcopy(extra['monitor']);monitor.update(device=c['device'],torch_threads=c['torch_threads'],split='validation')
    extra['monitor']=sample_config(monitor)
    if extra['monitor']['limit'] is None or extra['monitor']['limit']>(60 if fresh else 12) or extra['monitor']['eta']!=0 or extra['monitor']['steps']>100:
        raise ValueError('Periodic monitor requires bounded fixed validation tasks and deterministic DDIM')
    c.update(extra);c['schema_version']=SCHEMA;c['sampling_mode']='epoch_shuffle_without_replacement'
    steps=math.ceil(c['expected_train_windows']/c['batch_size'])
    c.update(max_steps=c['epochs']*steps,validation_interval=c['validation_every_epochs']*steps,
             checkpoint_interval=c['checkpoint_every_epochs']*steps,steps_per_epoch=steps)
    if fresh:
        c.update(resolve_extra(value,steps,c['epochs']));c['schema_version']=FRESH_SCHEMA
        c['formal_estimate_steps']=c['max_steps']
        c['learning_rate']=c['minimum_learning_rate']
    return c


def verified_curated_dataset(directory,c):
    directory=Path(directory).resolve(strict=True)
    m=json.loads((directory/'dataset_manifest.json').read_text(encoding='utf-8'))
    selection=file_identity(directory/'selection.json')
    if not m.get('curation',{}).get('label_quality_filtered') or selection['sha256']!=m['curation']['selection']['sha256']:
        raise ValueError('Epoch training requires an intact stage5A quality-curated dataset')
    s=json.loads((directory/'selection.json').read_text(encoding='utf-8'))
    fresh=c['schema_version']=='sumodiff.base.epoch.config.v2'
    expected_schema='sumodiff.stage5a.selection.v2' if fresh else 'sumodiff.stage5a.selection.v1'
    if s.get('schema_version')!=expected_schema or s['profile']!=('pilot' if c['budget_kind']=='smoke' else 'full'):
        raise ValueError('Pilot data cannot be used as a full-budget training cohort')
    ds=WindowDataset(directory)
    rows=s['selected']
    if len({r['window_id'] for r in rows})!=len(rows) or {r['window_id'] for r in rows}!={e['window_id'] for e in ds.entries}:
        raise ValueError('Selected-window index mismatch')
    if any(not r['eligible'] or not r['core_complete'] or r['reasons'] for r in rows):
        raise ValueError('Selection contains uncertified labels')
    by_id={e['window_id']:e for e in ds.entries}
    for row in rows:
        if any(row[k]!=by_id[row['window_id']][k] for k in ('split','geometry_id','episode_key','family')):
            raise ValueError('Selected-window metadata differs from the certified index')
    for split,quotas in s['quotas'].items():
        for family,n in quotas.items():
            if sum(r['split']==split and r['family']==family for r in rows)!=n:
                raise ValueError('Selected family counts differ from the declared quotas')
    populations=[('train',c['expected_train_windows']),('validation',c['expected_validation_windows'])]
    if fresh and c['budget_kind']=='full':populations.append(('test',c['expected_test_windows']))
    for split,n in populations:
        selected=[r for r in rows if r['split']==split]
        if len(selected)!=n or any(not e['core_training_eligible'] for e in ds.entries if e['split']==split):
            raise ValueError(f'Expected exactly {n} certified complete {split} windows')
        if any(not v['passed'] for v in s['selection'][split].values()):
            raise ValueError('Dataset quota/diversity report was not passed')
    if fresh and c['budget_kind']=='full':
        fraction=s.get('minimum_turning_fraction')
        if type(fraction) not in (int,float) or not 0<fraction<=1:raise ValueError('Missing certified turning coverage')
        for split,_ in populations:
            selected_turns=sum(r['split']==split and r['family']=='unsignalized_intersection' and r.get('actual_turning_agents',0)>0 for r in rows)
            required=math.ceil(s['quotas'][split]['unsignalized_intersection']*fraction)
            coverage=s.get('turning_coverage',{}).get(split,{})
            if selected_turns<required or not coverage.get('passed') or coverage.get('selected_windows')!=selected_turns:
                raise ValueError('Certified turning coverage does not match selected windows')
    permitted=('train','validation','test') if fresh else ('train','validation')
    if any(r['split'] not in permitted for r in rows):raise ValueError('Unexpected split')
    from itertools import combinations
    for left,right in combinations(permitted,2):
        for field in ('geometry_id','episode_key'):
            if {r[field] for r in rows if r['split']==left}&{r[field] for r in rows if r['split']==right}:
                raise ValueError(f'Curated {field} leakage between {left} and {right}')
    if {r['geometry_id'] for r in rows if r['split']=='train'} & {r['geometry_id'] for r in rows if r['split']=='validation'}:
        raise ValueError('Curated geometry leakage')
    if {r['episode_key'] for r in rows if r['split']=='train'} & {r['episode_key'] for r in rows if r['split']=='validation'}:
        raise ValueError('Curated episode leakage')
    return selection


def epoch_batch(step,n,batch_size,seed):
    """Stateless epoch permutation; resume at any committed step preserves coverage/order."""
    if type(step) is not int or step<1 or n<1 or batch_size<1: raise ValueError('Invalid epoch step/population')
    batches=math.ceil(n/batch_size)
    epoch=(step-1)//batches
    index=(step-1)%batches
    epoch_seed=int.from_bytes(hashlib.sha256(f'epoch:{seed}:{epoch}'.encode()).digest()[:8],'little')%(2**63-1)
    order=torch.randperm(n,generator=torch.Generator().manual_seed(epoch_seed))
    return order[index*batch_size:min((index+1)*batch_size,n)],epoch+1,index+1


def probe_subset(cache,limit):
    n=min(limit,len(cache[2]));indices=torch.arange(n)
    return ({k:v[indices] for k,v in cache[0].items()},{k:v[indices] for k,v in cache[1].items()},
            cache[2][:n],cache[3])


def periodic_sample(dataset,checkpoint,run,repository,c,epoch,formal):
    """Independent sampler resets global RNGs: snapshot/restore all of them around monitoring."""
    from .sampling import sample
    py_state=random.getstate();np_state=np.random.get_state();cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    config_path=run.output/'monitor_config.yaml'
    if not config_path.exists(): config_path.write_text(yaml.safe_dump(c['monitor'],sort_keys=True),encoding='utf-8')
    output=run.output/f'free_validation_epoch_{epoch:04d}'
    command=[sys.executable,'-m','sumodiff.diffusion','sample','--dataset',str(dataset),
        '--checkpoint',str(checkpoint),'--config',str(config_path),'--output',str(output)]
    if formal: command.append('--formal')
    started=time.perf_counter()
    try:
        result=sample(dataset,checkpoint,config_path,output,repository,command,formal)
        index=json.loads((output/'trajectory_index.json').read_text(encoding='utf-8'))
        max_correction=0.;max_displacement=0.
        for item in index:
            if item['generation_status']!='completed': continue
            with np.load(output/'trajectories'/item['window_id']/'stages.npz',allow_pickle=False) as arrays:
                # Persisted raw and decoded positions use the same selected agent set.
                mask=arrays['agent_mask']
                max_displacement=max(max_displacement,float(np.linalg.norm(arrays['raw_future_delta'][mask,:,:2],axis=-1).max()))
                max_correction=max(max_correction,float(np.linalg.norm(
                    arrays['decoded_absolute_states'][mask,:,:2]-arrays['raw_absolute_states'][mask,:,:2],axis=-1).max()))
        return dict(epoch=epoch,checkpoint=file_identity(checkpoint),run=str(output),
            raw_displacement_max_m=max_displacement,position_correction_max_m=max_correction,
            quality_pass_rate=result['decoded']['quality_pass_rate'],planned_tasks=result['planned_tasks'],
            generation_failures=result['generation_failures'],total_monitor_seconds=time.perf_counter()-started)
    finally:
        random.setstate(py_state);np.random.set_state(np_state);torch.set_rng_state(cpu)
        if cuda: torch.cuda.set_rng_state_all(cuda)
        if c['device'].startswith('cuda'): torch.cuda.reset_peak_memory_stats()
