"""200 -> 225 epochs; preserve AdamW/RNG, warm only temporal reweighting."""
from copy import deepcopy
import math
from .raw_velocity_training import RawVelocityProtocol, inherit_optimizer
from .unified_training import train_unified
from .decoded_temporal_losses import resolve_decoded_temporal_loss, decoded_temporal_objective
from .schedule import ScheduleConfig, UpdateSchedule, resolve_schedule

SCHEMA='sumodiff.decoded.temporal.train.config.v1'
DEFAULTS=dict(schema_version=SCHEMA,objective_name='decoded_temporal',additional_epochs=25,
    expected_parent_step=60000,expected_parent_sha256='',expected_train_windows=1200,
    expected_validation_windows=200,batch_size=4,device='cuda:0',torch_threads=1,
    peak_learning_rate=2e-5,minimum_learning_rate=1e-6,warmup_epochs=1,
    temporal_reweight_warmup_epochs=1,validation_every_epochs=1,checkpoint_every_epochs=5,
    monitor_every_epochs=5,monitor_tasks=20,loss={})


def decoded_temporal_config(value,parent,*,prefix_updates=None):
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown stage5E continuation options')
    r=deepcopy(DEFAULTS);r.update(value)
    if r['schema_version']!=SCHEMA or r['objective_name']!='decoded_temporal':
        raise ValueError('Unsupported stage5E schema/objective')
    contract=dict(additional_epochs=25,expected_parent_step=60000,expected_train_windows=1200,
        expected_validation_windows=200,batch_size=4,warmup_epochs=1,temporal_reweight_warmup_epochs=1,
        validation_every_epochs=1,checkpoint_every_epochs=5,monitor_every_epochs=5,monitor_tasks=20)
    for k,v in contract.items():
        if type(r[k]) is not int or r[k]!=v:
            raise ValueError('Stage5E fixes 200->225 epochs, B4, one-epoch warmups and five-epoch monitoring')
    if type(r['torch_threads']) is not int or not 1<=r['torch_threads']<=4:
        raise ValueError('Invalid thread count')
    sha=r['expected_parent_sha256']
    if not isinstance(sha,str) or len(sha)!=64 or any(x not in '0123456789abcdef' for x in sha):
        raise ValueError('Complete 200-epoch parent SHA-256 required')
    p=parent['metadata']['training_config']
    if (p.get('schema_version')!='sumodiff.raw.velocity.train.config.v1' or p.get('objective_name')!='raw_velocity'
        or parent['step']!=60000 or p.get('epochs')!=200 or p.get('max_steps')!=60000
        or p.get('steps_per_epoch')!=300 or p.get('execution_budget_kind')!='fifty_epoch_raw_velocity'
        or parent['metadata']['runtime']['snapshot_kind']!='training_recovery'):
        raise ValueError('Stage5E requires the completed 200-epoch stage5D recovery checkpoint')
    for k in ('batch_size','device','torch_threads','expected_train_windows','expected_validation_windows'):
        if r[k]!=p[k]:raise ValueError(f'{k} must retain the audited parent runtime/cohort')
    loss=resolve_decoded_temporal_loss(r['loss'])
    if loss.inherited.to_dict()!=p['loss']:
        raise ValueError('Keep all inherited raw velocity, imitation, road and normalization objectives')
    if not math.isclose(loss.interior_weight,.9*loss.inherited.unified.temporal_weight,rel_tol=0,abs_tol=1e-15):
        raise ValueError('Stage5E preserves the inherited interior contribution')
    if loss.boundary_weight<.1*loss.inherited.unified.temporal_weight:
        raise ValueError('Stage5E boundary candidate cannot silently weaken the inherited boundary')
    peak=r['peak_learning_rate']
    if type(peak) not in (int,float) or not math.isfinite(peak) or not 1e-5<=peak<=2e-5:
        raise ValueError('Stage5E peak lr bounded at [1e-5,2e-5]')
    if type(r['minimum_learning_rate']) not in (int,float) or r['minimum_learning_rate']!=1e-6:
        raise ValueError('Stage5E starts and ends at inherited 1e-6')
    settings=p['optimizer']
    if not parent['optimizer']['state'] or any(float(x['step'])!=30000 for x in parent['optimizer']['state'].values()):
        raise ValueError('Expected 30000 accumulated parent AdamW updates')
    for g in parent['optimizer']['param_groups']:
        if (g['lr']!=1e-6 or g['weight_decay']!=settings['weight_decay']
            or tuple(g['betas'])!=tuple(settings['betas']) or g['eps']!=settings['eps']):
            raise ValueError('Parent AdamW settings changed')
    learning=ScheduleConfig(peak_lr=peak,minimum_lr=1e-6,warmup_updates=300,
        auxiliary_warmup_updates=300,total_updates=7500)
    if prefix_updates is not None and (type(prefix_updates) is not int or not 1<=prefix_updates<=300):
        raise ValueError('Stage5E prefix bounded at 1 through 300 updates')
    c=deepcopy(p);c.pop('raw_auxiliary_warmup_epochs',None)
    c.update(schema_version=SCHEMA,objective_name='decoded_temporal',source_step=60000,source_epochs=200,
        fork_source_sha256=sha,additional_epochs=25,epochs=225,steps_per_epoch=300,loss=loss.to_dict(),
        schedule=learning.to_dict(),initialization='inherit_model_AdamW_moments_and_noise_Torch_RNG_reset_scheduler_only',
        optimizer=deepcopy(settings),learning_rate=1e-6,nominal_additional_updates=7500,
        planned_additional_updates=prefix_updates if prefix_updates is not None else 7500,
        execution_budget_kind='preflight_prefix' if prefix_updates is not None else 'twenty_five_epoch_decoded_temporal',
        validation_every_epochs=1,checkpoint_every_epochs=5,monitor_every_epochs=5,
        inherited_auxiliary_ramp=1.,inherited_raw_auxiliary_ramp=1.,temporal_reweight_warmup_epochs=1,
        inherited_optimizer_updates=30000,validation_interval=300,checkpoint_interval=1500,
        formal_estimate_steps=7500,epoch_sampling='all_1200_without_replacement_no_density_reweighting')
    c['monitor']=deepcopy(p['monitor']);c['monitor']['limit']=20
    c['max_steps']=60000+c['planned_additional_updates']
    return c


def initialize(parent,model,optimizer,generator,device):
    return inherit_optimizer(parent,model,optimizer,generator,device,expected_updates=30000)


def validate_runtime(value):
    m=value['metadata'];c=m['training_config']
    if c.get('schema_version')!=SCHEMA or m['training_signature']['training']!=c:
        raise ValueError('Stage5E requires complete immutable provenance')
    loss=resolve_decoded_temporal_loss(c['loss'])
    if not loss.enabled or not m['diffusion_config']['additional_losses_enabled'] or c['diffusion']!=m['diffusion_config']:
        raise ValueError('Stage5E objective/checkpoint mismatch')
    if (c['source_step']!=60000 or c['source_epochs']!=200 or c['epochs']!=225
        or c['additional_epochs']!=25 or c['nominal_additional_updates']!=7500
        or c['schedule']['total_updates']!=7500 or c['inherited_optimizer_updates']!=30000
        or c['max_steps']!=60000+c['planned_additional_updates']
        or c['inherited_auxiliary_ramp']!=1. or c['inherited_raw_auxiliary_ramp']!=1.
        or not 60000<=value['step']<=c['max_steps']):
        raise ValueError('Invalid stage5E epoch, schedule or budget')
    if c['execution_budget_kind']=='twenty_five_epoch_decoded_temporal':
        if c['max_steps']!=67500:raise ValueError('Wrong full stage5E budget')
    elif c['execution_budget_kind']!='preflight_prefix' or not 1<=c['planned_additional_updates']<=300:
        raise ValueError('Invalid stage5E short budget')
    learning=UpdateSchedule(resolve_schedule(c['schedule']));runtime=m['runtime']
    learning.load_state_dict(runtime['scheduler_state'],expected_updates=value['step']-60000)
    if runtime['snapshot_kind'] not in ('training_recovery','monitor_assessment'):
        raise ValueError('Invalid stage5E snapshot')
    if value['step']>60000 and any(g['lr']!=learning.last_lr for g in value['optimizer']['param_groups']):
        raise ValueError('Stage5E optimizer lr differs from saved scheduler')
    expected=30000+value['step']-60000
    if not value['optimizer']['state'] or any(float(x['step'])!=expected for x in value['optimizer']['state'].values()):
        raise ValueError('Stage5E AdamW history was reset or lost')


class DecodedTemporalProtocol(RawVelocityProtocol):
    config=staticmethod(decoded_temporal_config)
    resolve_loss=staticmethod(resolve_decoded_temporal_loss)
    objective=staticmethod(decoded_temporal_objective)
    initialize=staticmethod(initialize)
    max_prefix_updates=300
    run_schema='sumodiff.decoded.temporal.train.run.v1'
    metrics_schema='sumodiff.decoded.temporal.train.metrics.v1'
    purpose='stage5e_decoded_boundary_continuation'
    full_budget_kind='twenty_five_epoch_decoded_temporal'
    loss_reduction='separate per-scene 4/36 decoded temporal means; low-t alpha_bar/full batch; inherited auxiliaries full; ramp only temporal reweight delta'


def train_decoded_temporal(*args,**kwargs):
    return train_unified(*args,**kwargs,_protocol=DecodedTemporalProtocol())
