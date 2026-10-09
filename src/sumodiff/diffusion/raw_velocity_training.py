"""150 -> 200 real epochs: inherit model/AdamW/RNG, restart only the schedule."""
from copy import deepcopy
import math
from pathlib import Path
import numpy as np
import torch
from .checkpoint import restore_training
from .config import resolve_diffusion_config
from .controlled_training import inherited_state
from .schedule import ScheduleConfig, UpdateSchedule, resolve_schedule
from .unified_training import train_unified, UnifiedProtocol, monitor_quality
from .raw_velocity_losses import resolve_raw_velocity_loss, raw_velocity_objective

SCHEMA='sumodiff.raw.velocity.train.config.v1'
DEFAULTS=dict(schema_version=SCHEMA,objective_name='raw_velocity',additional_epochs=50,
    expected_parent_step=45000,expected_parent_sha256='',expected_train_windows=1200,
    expected_validation_windows=200,batch_size=4,device='cuda:0',torch_threads=1,
    peak_learning_rate=2e-5,minimum_learning_rate=1e-6,warmup_epochs=2,
    raw_auxiliary_warmup_epochs=2,validation_every_epochs=1,checkpoint_every_epochs=10,
    monitor_every_epochs=10,monitor_tasks=20,loss={})

def raw_velocity_config(value,parent,*,prefix_updates=None):
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown stage5D continuation options')
    r=deepcopy(DEFAULTS);r.update(value)
    if r['schema_version']!=SCHEMA or r['objective_name']!='raw_velocity':
        raise ValueError('Unsupported stage5D schema/objective')
    for k in ('additional_epochs','expected_parent_step','expected_train_windows','expected_validation_windows',
              'batch_size','torch_threads','warmup_epochs','raw_auxiliary_warmup_epochs',
              'validation_every_epochs','checkpoint_every_epochs','monitor_every_epochs','monitor_tasks'):
        if type(r[k]) is not int or r[k]<1:raise ValueError(f'Invalid {k}')
    contract=dict(additional_epochs=50,expected_parent_step=45000,expected_train_windows=1200,
        expected_validation_windows=200,batch_size=4,warmup_epochs=2,raw_auxiliary_warmup_epochs=2,
        validation_every_epochs=1,checkpoint_every_epochs=10,monitor_every_epochs=10,monitor_tasks=20)
    if any(r[k]!=v for k,v in contract.items()):
        raise ValueError('Stage5D fixes 150->200 epochs, B4, two-epoch warmups and ten-epoch monitoring')
    sha=r['expected_parent_sha256']
    if not isinstance(sha,str) or len(sha)!=64 or any(x not in '0123456789abcdef' for x in sha):
        raise ValueError('Complete 150-epoch parent SHA-256 required')
    p=parent['metadata']['training_config']
    if (p.get('schema_version')!='sumodiff.unified.train.config.v1' or p.get('objective_name')!='unified'
        or p.get('epochs')!=150 or p.get('execution_budget_kind')!='fifty_epoch_unified'
        or parent['step']!=45000 or parent['step']!=150*p['steps_per_epoch'] or p['max_steps']!=45000
        or parent['metadata']['runtime']['snapshot_kind']!='training_recovery'):
        raise ValueError('Stage5D requires the completed 150-epoch unified recovery checkpoint')
    for k in ('batch_size','device','torch_threads','expected_train_windows','expected_validation_windows'):
        if r[k]!=p[k]:raise ValueError(f'{k} must retain the audited cohort/runtime')
    loss=resolve_raw_velocity_loss(r['loss']);inherited=loss.unified.to_dict()
    if dict(inherited,imitation_weight=p['loss']['imitation_weight'])!=p['loss']:
        raise ValueError('Keep existing temporal, road and normalization objectives')
    if inherited['imitation_weight']<=0:raise ValueError('Imitation must remain enabled')
    peak=r['peak_learning_rate']
    if type(peak) not in (int,float) or not math.isfinite(peak) or not 2e-5<=peak<=5e-5:
        raise ValueError('Stage5D peak lr must be in [2e-5,5e-5]')
    if type(r['minimum_learning_rate']) not in (int,float) or r['minimum_learning_rate']!=1e-6:
        raise ValueError('Stage5D starts and ends at the inherited 1e-6 lr')
    if not parent['optimizer']['state']:raise ValueError('Source AdamW moments must exist')
    settings=p['optimizer']
    for g in parent['optimizer']['param_groups']:
        if (g['lr']!=1e-6 or g['weight_decay']!=settings['weight_decay']
            or tuple(g['betas'])!=tuple(settings['betas']) or g['eps']!=settings['eps']):
            raise ValueError('Source optimizer differs from the audited parent')
    schedule=ScheduleConfig(peak_lr=peak,minimum_lr=1e-6,warmup_updates=600,
                            auxiliary_warmup_updates=600,total_updates=15000)
    if prefix_updates is not None and (type(prefix_updates) is not int or not 1<=prefix_updates<=600):
        raise ValueError('Stage5D short prefix must contain 1 through 600 updates')
    c=deepcopy(p)
    c.update(schema_version=SCHEMA,objective_name='raw_velocity',source_step=45000,source_epochs=150,
        fork_source_sha256=sha,additional_epochs=50,epochs=200,steps_per_epoch=300,
        loss=loss.to_dict(),schedule=schedule.to_dict(),
        initialization='inherit_model_AdamW_moments_and_noise_Torch_RNG_reset_scheduler_only',
        optimizer=deepcopy(settings),learning_rate=1e-6,nominal_additional_updates=15000,
        planned_additional_updates=prefix_updates if prefix_updates is not None else 15000,
        execution_budget_kind='preflight_prefix' if prefix_updates is not None else 'fifty_epoch_raw_velocity',
        validation_every_epochs=1,checkpoint_every_epochs=10,monitor_every_epochs=10,
        inherited_auxiliary_ramp=1.,raw_auxiliary_warmup_epochs=2,
        inherited_imitation_weight=p['loss']['imitation_weight'],validation_interval=300,
        checkpoint_interval=3000,formal_estimate_steps=15000)
    c['monitor']=deepcopy(p['monitor']);c['monitor']['limit']=20
    c['max_steps']=45000+c['planned_additional_updates']
    return c

def inherit_optimizer(parent,model,optimizer,generator,device):
    # AdamW's CPU step tensors may otherwise alias the loaded source payload.
    restore_training(deepcopy(parent),model,optimizer,generator,device)
    state=inherited_state(model,optimizer,generator)
    if len(optimizer.state)!=sum(p.requires_grad for p in model.parameters()):
        raise ValueError('Source AdamW must have moments for every trainable parameter')
    for v in optimizer.state.values():
        if set(v)!={'step','exp_avg','exp_avg_sq'} or float(v['step'])!=15000:
            raise ValueError('Expected 15000 source AdamW updates since stage5C reset')
        if not all(torch.isfinite(v[k]).all() for k in ('exp_avg','exp_avg_sq')):
            raise ValueError('Nonfinite inherited optimizer moments')
    return state

def validate_runtime(value):
    m=value['metadata'];c=m['training_config']
    if c.get('schema_version')!=SCHEMA or m['training_signature']['training']!=c:
        raise ValueError('Stage5D needs immutable complete training provenance')
    objective=resolve_raw_velocity_loss(c['loss'])
    if not objective.enabled or not m['diffusion_config']['additional_losses_enabled'] or c['diffusion']!=m['diffusion_config']:
        raise ValueError('Stage5D loss/checkpoint flag mismatch')
    if (c['source_step']!=45000 or c['source_epochs']!=150 or c['epochs']!=200
        or c['nominal_additional_updates']!=15000 or c['schedule']['total_updates']!=15000
        or c['max_steps']!=45000+c['planned_additional_updates'] or c['inherited_auxiliary_ramp']!=1.
        or not c['source_step']<=value['step']<=c['max_steps']):
        raise ValueError('Invalid stage5D source, schedule or budget')
    learning=UpdateSchedule(resolve_schedule(c['schedule']));runtime=m['runtime']
    learning.load_state_dict(runtime['scheduler_state'],expected_updates=value['step']-45000)
    if runtime['snapshot_kind'] not in ('training_recovery','monitor_assessment'):
        raise ValueError('Invalid stage5D snapshot')
    if value['step']>45000 and any(g['lr']!=learning.last_lr for g in value['optimizer']['param_groups']):
        raise ValueError('Stage5D optimizer lr/scheduler mismatch')
    expected=15000+value['step']-45000
    if not value['optimizer']['state'] or any(float(s['step'])!=expected for s in value['optimizer']['state'].values()):
        raise ValueError('Stage5D optimizer history was reset or lost')

def velocity_monitor_quality(directory,cache):
    from .quality_audit import physical_derivatives
    from .stage5b_report import read
    q=monitor_quality(directory,cache);by_id={wid:i for i,wid in enumerate(cache[2])}
    boundary=[];good=0;assessed=0;planned_agents=0;collisions=0;unknown=0;not_applicable=0
    for item in read(Path(directory)/'trajectory_index.json'):
        source=Path(directory)/'trajectories'/item['window_id'];m=read(source/'metrics.json')['decoded']
        selected=int(cache[0]['agent_mask'][by_id[item['window_id']]].sum());planned_agents+=selected
        motion=m.get('motion',{});good+=motion.get('comfort_pass_agents',0)
        assessed+=motion.get('comfort_assessed_agents',0)
        col=m.get('collisions',{}).get('groups',{}).get('any',{}).get('collision')
        collisions+=col is True;unknown+=col is None and selected>=2;not_applicable+=selected<2
        if item['generation_status']=='completed' and (source/'stages.npz').exists():
            i=by_id[item['window_id']];cond={k:v[i].numpy() for k,v in cache[0].items()}
            with np.load(source/'stages.npz',allow_pickle=False) as a:
                _,_,j=physical_derivatives(cond,a['decoded_absolute_states'])
            boundary.extend(np.linalg.norm(j[cond['agent_mask'],:4],axis=-1).ravel().tolist())
    return dict(q,decoded_boundary_jerk_p99=float(np.quantile(boundary,.99)) if boundary else None,
        decoded_boundary_jerk_max=max(boundary) if boundary else None,strict_comfort_pass_agents=good,
        strict_comfort_assessed_agents=assessed,strict_comfort_planned_agents=planned_agents,
        strict_comfort_unassessed_agents=planned_agents-assessed,collision_scenes=collisions,
        collision_unknown=unknown,collision_not_applicable_scenes=not_applicable)

def velocity_frontier(records):
    eligible=[r for r in records if r['quality']['unknown']==0 and r['quality']['failures']==0
        and r['quality']['collision_unknown']==0 and r['quality']['decoded_interior_jerk_p99'] is not None
        and r['quality']['decoded_boundary_jerk_p99'] is not None]
    def vector(r):
        q=r['quality'];n=q['planned'];a=q['strict_comfort_planned_agents']
        comfort=-q['strict_comfort_pass_agents']/a if a else 0.
        return (-q['quality_passes']/n,q['road_bad']/n,q['route_bad']/n,q['collision_scenes']/n,
                q['decoded_interior_jerk_p99'],q['decoded_boundary_jerk_p99'],comfort)
    return [r['checkpoint'] for r in eligible if not any(
        all(x<=y for x,y in zip(vector(o),vector(r))) and any(x<y for x,y in zip(vector(o),vector(r)))
        for o in eligible if o is not r)]

class RawVelocityProtocol(UnifiedProtocol):
    config=staticmethod(raw_velocity_config)
    @staticmethod
    def scale_diffusion(parent):
        # The original scale audit predates all auxiliary losses. This flag
        # changes training capabilities, not scales/noising; all other audited
        # fields and dataset/audit hashes must still match exactly.
        c=deepcopy(parent['metadata']['diffusion_config'])
        c['additional_losses_enabled']=False
        return resolve_diffusion_config(c)
    resolve_loss=staticmethod(resolve_raw_velocity_loss)
    objective=staticmethod(raw_velocity_objective)
    initialize=staticmethod(inherit_optimizer)
    monitor_quality=staticmethod(velocity_monitor_quality)
    frontier=staticmethod(velocity_frontier)
    max_prefix_updates=600
    run_schema='sumodiff.raw.velocity.train.run.v1'
    metrics_schema='sumodiff.raw.velocity.train.metrics.v1'
    purpose='stage5d_raw_velocity_continuation'
    full_budget_kind='fifty_epoch_raw_velocity'
    optimizer_reset=False
    loss_reduction='per-scene valid means; low-t alpha_bar; full batch denominator; inherited auxiliaries full-strength; raw-only ramp'

def train_raw_velocity(*args,**kwargs):
    return train_unified(*args,**kwargs,_protocol=RawVelocityProtocol())
