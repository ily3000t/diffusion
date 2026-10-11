"""Fixed from-scratch recipe, calibration locks and strict schedule recovery."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from .raw_velocity_losses import resolve_raw_velocity_loss
from .jerk_tail import JerkTailConfig
from .schedule import ScheduleConfig, UpdateSchedule
from sumodiff.experiments.recorder import file_identity

SCHEMA = 'sumodiff.base.epoch.config.v2'
EXTRA = dict(expected_test_windows=600,lr_warmup_epochs=10,auxiliary_warmup_epochs=10,
             minimum_learning_rate=1e-6,peak_learning_rate=1e-4,
             raw_loss={},jerk_tail={},tail_weight=None,calibration_report=None)

def resolve_extra(value,steps,epochs):
    from .raw_velocity_losses import RawVelocityLossConfig
    c=deepcopy(EXTRA);c.update({k:value[k] for k in EXTRA if k in value})
    for k in ('expected_test_windows','lr_warmup_epochs','auxiliary_warmup_epochs'):
        if type(c[k]) is not int or c[k]<1:raise ValueError(f'Invalid {k}')
    raw=resolve_raw_velocity_loss(c['raw_loss'])
    # This revision implements one fixed candidate, not a silent alternative objective.
    if raw.to_dict()!=RawVelocityLossConfig().to_dict():
        raise ValueError('Fresh V2 requires the declared epoch200 auxiliary objective')
    tail=JerkTailConfig(**c['jerk_tail'])
    c['raw_loss']=raw.to_dict();c['jerk_tail']=tail.to_dict()
    if c['tail_weight'] is not None and (type(c['tail_weight']) not in (int,float)
            or not math.isfinite(c['tail_weight']) or c['tail_weight']<=0):
        raise ValueError('Invalid tail weight')
    if c['calibration_report'] is not None and not isinstance(c['calibration_report'],str):
        raise ValueError('Calibration report must be an explicit path')
    c['schedule']=ScheduleConfig(peak_lr=c['peak_learning_rate'],minimum_lr=c['minimum_learning_rate'],
        warmup_updates=c['lr_warmup_epochs']*steps,total_updates=epochs*steps,
        auxiliary_warmup_updates=c['auxiliary_warmup_epochs']*steps).to_dict()
    c['initialization']='random_new_model_and_adamw'
    return c

def recipe_id(c,model):
    ignored={'tail_weight','calibration_report','calibration_identity'}
    payload=dict(training={k:v for k,v in c.items() if k not in ignored},model=model.to_dict())
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def verify_calibration(c,model,data):
    if c['tail_weight'] is None or c['calibration_report'] is None:
        raise ValueError('Run Calibrate on the new dataset before Train; tail weight is unresolved')
    path=Path(c['calibration_report']).resolve(strict=True)
    report=json.loads(path.read_text(encoding='utf-8'))
    if (report.get('schema_version')!='sumodiff.fresh.calibration.v1' or not report.get('passed')
            or report.get('development_only') or report['recipe_sha256']!=recipe_id(c,model)
            or report['dataset_hashes']!={k:v['sha256'] for k,v in data.items()}
            or report['selected_tail_weight']!=c['tail_weight']):
        raise ValueError('Calibration does not certify this exact fresh recipe/model/dataset')
    status=json.loads((path.parent/'status.json').read_text(encoding='utf-8'))
    manifest=json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
    if status['state']!='completed' or not manifest['formal'] or manifest['git']['dirty']:
        raise ValueError('Production calibration requires a completed clean formal run')
    if json.loads((path.parent/'metrics.json').read_text(encoding='utf-8'))!=report:
        raise ValueError('Calibration report differs from the recorded metrics')
    from sumodiff.experiments.recorder import load_config
    expected={k:v for k,v in c.items() if k!='calibration_identity'}
    expected.update(tail_weight=None,calibration_report=None)
    if load_config(path.parent/'resolved_config.yaml')['training']!=expected:
        raise ValueError('Calibrated recipe differs from the actual calibration run')
    return file_identity(path)

def validate_runtime(payload):
    c=payload['metadata']['training_config'];m=payload['metadata']
    if c['schema_version']!=SCHEMA or c['initialization']!='random_new_model_and_adamw':
        raise ValueError('Invalid fresh checkpoint protocol')
    if m['training_signature']['training']!=c or c['diffusion']!=m['diffusion_config']:
        raise ValueError('Fresh checkpoint training recipe mismatch')
    if c['tail_weight'] is None or not c.get('calibration_identity'):
        raise ValueError('Uncalibrated fresh checkpoint')
    raw=resolve_raw_velocity_loss(c['raw_loss']);JerkTailConfig(**c['jerk_tail'])
    if not c['diffusion']['additional_losses_enabled'] or not math.isfinite(c['tail_weight']) or c['tail_weight']<=0:
        raise ValueError('Fresh objective capability/weight mismatch')
    expected_schedule=ScheduleConfig(peak_lr=c['peak_learning_rate'],minimum_lr=c['minimum_learning_rate'],
        warmup_updates=c['lr_warmup_epochs']*c['steps_per_epoch'],total_updates=c['epochs']*c['steps_per_epoch'],
        auxiliary_warmup_updates=c['auxiliary_warmup_epochs']*c['steps_per_epoch'])
    if c['schedule']!=expected_schedule.to_dict():raise ValueError('Fresh schedule differs from the epoch protocol')
    scheduler=UpdateSchedule(expected_schedule)
    scheduler.load_state_dict(m['runtime']['update_schedule'],expected_updates=payload['step'])
    if any(g['lr']!=scheduler.last_lr for g in payload['optimizer']['param_groups']):
        raise ValueError('Fresh optimizer/scheduler learning rate mismatch')
    if any(int(s['step'])!=payload['step'] for s in payload['optimizer']['state'].values()):
        raise ValueError('Fresh AdamW update count mismatch')
    if c['max_steps']!=c['epochs']*c['steps_per_epoch'] or c['schedule']['total_updates']!=c['max_steps']:
        raise ValueError('Fresh checkpoint budget mismatch')

def seal_test(dataset,checkpoint,config_path,validation_run,output,repository,command,formal=False):
    from .sampling import sample_config
    from .data import dataset_identity
    from .checkpoint import load_checkpoint
    from .epoch_training import verified_curated_dataset
    from sumodiff.experiments.recorder import RunRecorder,load_config
    from sumodiff.data.dataset import write_json
    c=sample_config(load_config(config_path));p=load_checkpoint(checkpoint)
    training=p['metadata']['training_config']
    if training['schema_version']!=SCHEMA or c['split']!='test' or c['limit']!=training['expected_test_windows']:
        raise ValueError('Seal requires the fresh model and complete predefined test budget')
    verified_curated_dataset(dataset,training)
    v=Path(validation_run).resolve(strict=True)
    m=json.loads((v/'manifest.json').read_text(encoding='utf-8'))
    status=json.loads((v/'status.json').read_text(encoding='utf-8'))
    cfg=load_config(v/'resolved_config.yaml')
    metrics=json.loads((v/'metrics.json').read_text(encoding='utf-8'))
    identity=file_identity(checkpoint);data=dataset_identity(dataset)
    if (not m['formal'] or m['git']['dirty'] or status['state']!='completed'
            or cfg['checkpoint']['sha256']!=identity['sha256'] or cfg['data']!=data
            or cfg['sampling']!={**c,'split':'validation'}
            or metrics['planned_tasks']!=training['expected_validation_windows']):
        raise ValueError('Select a model using completed full formal validation with the same sampling protocol')
    doc=dict(schema_version='sumodiff.fresh.test.seal.v1',checkpoint=identity,
        dataset_hashes={k:d['sha256'] for k,d in data.items()},sampling=c,
        validation_manifest=file_identity(v/'manifest.json'),validation_metrics=file_identity(v/'metrics.json'),
        model_selection='explicit checkpoint chosen after validation, before test; test never tunes the recipe')
    with RunRecorder(output,repository,doc,command,{'test_noise':c['seed']},purpose='freeze_fresh_test_protocol',
            data_files=[config_path,v/'manifest.json',v/'metrics.json',v/'resolved_config.yaml'],
            checkpoint=checkpoint,formal=formal) as run:
        write_json(run.output/'test_seal.json',doc);run.write_metrics(dict(sealed=True))
    return dict(sealed=True)

def verify_test_seal(path,dataset,checkpoint,c):
    from .data import dataset_identity
    if path is None:raise ValueError('Fresh test requires Seal after complete validation')
    path=Path(path).resolve(strict=True);doc=json.loads(path.read_text(encoding='utf-8'))
    data=dataset_identity(dataset)
    m=json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
    status=json.loads((path.parent/'status.json').read_text(encoding='utf-8'))
    from sumodiff.experiments.recorder import load_config
    if doc!=load_config(path.parent/'resolved_config.yaml'):
        raise ValueError('Test seal differs from the recorded frozen configuration')
    if (doc.get('schema_version')!='sumodiff.fresh.test.seal.v1' or not m['formal'] or m['git']['dirty']
            or status['state']!='completed' or doc['checkpoint']['sha256']!=file_identity(checkpoint)['sha256']
            or doc['dataset_hashes']!={k:v['sha256'] for k,v in data.items()} or doc['sampling']!=c):
        raise ValueError('Test model/data/config differ from the completed frozen seal')
    return file_identity(path)

