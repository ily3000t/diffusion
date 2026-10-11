from copy import deepcopy
import json
import pytest
import torch
from sumodiff.experiments.recorder import load_config
from sumodiff.experiments.stage5a_plan import resolve_plan,jobs,protocol_id
from sumodiff.diffusion.epoch_training import epoch_config
from sumodiff.diffusion.fresh_protocol import SCHEMA,verify_calibration,verify_test_seal
from sumodiff.diffusion.schedule import ScheduleConfig,UpdateSchedule
from sumodiff.diffusion.training import training_signature
from sumodiff.diffusion.checkpoint import save_checkpoint,load_checkpoint,restore_training
from sumodiff.models import ModelConfig

def config():
    return epoch_config(load_config('configs/train/stage5a_epochs.yaml'))

def test_current_recipe_is_fresh_fixed_75000_updates():
    c=config()
    assert c['schema_version']==SCHEMA and c['max_steps']==75000 and c['steps_per_epoch']==375
    assert c['batch_size']==16 and c['epochs']==200
    assert [c['expected_train_windows'],c['expected_validation_windows'],c['expected_test_windows']]==[6000,600,600]
    schedule=UpdateSchedule(ScheduleConfig(**c['schedule']))
    assert schedule.learning_rate(1)==1e-6
    assert schedule.learning_rate(3750)==1e-4
    assert schedule.learning_rate(75000)==1e-6
    assert schedule.auxiliary_ramp(3750)==1
    assert c['raw_loss']['unified']['temporal_weight']==.001
    with pytest.raises(ValueError,match='Calibrate'):verify_calibration(c,ModelConfig(),{})

def test_three_splits_preassigned_disjoint_and_old_protocol_preserved():
    c=resolve_plan(load_config('configs/experiments/stage5a.yaml'));schedule=jobs(c)
    assert len(schedule)==216 and len(jobs(c,True))==6
    assert len({j['seed'] for j in schedule})==216
    shapes={s:{json.dumps(j['scenario']['geometry'],sort_keys=True) for j in schedule if j['split']==s}
        for s in ('train','validation','test')}
    assert not shapes['train']&shapes['validation'] and not shapes['train']&shapes['test'] and not shapes['validation']&shapes['test']
    assert protocol_id(c)!=protocol_id(resolve_plan({}))
    assert len(jobs(resolve_plan({})))==78
    pilot=jobs(c,True)
    assert next(j for j in pilot if j['split']=='train' and j['scenario']['family']=='unsignalized_intersection')['geometry_slot'].endswith('g06')
    assert next(j for j in pilot if j['split']=='validation' and j['scenario']['family']=='unsignalized_intersection')['geometry_slot'].endswith('g02')
    assert all(j['scenario']['traffic']['end_seconds']==30 for j in pilot if j['scenario']['family']=='unsignalized_intersection')
    for split,n in [('train',144),('validation',36),('test',36)]:
        assert sum(j['split']==split for j in schedule)==n

def test_fresh_signature_cannot_extend_budget_or_change_objective_silently():
    c=config();model=ModelConfig();data={'manifest':{'sha256':'d'},'index':{'sha256':'i'}};scale={'sha256':'s'}
    a=training_signature(model,c,data,scale,['t'],['v'])
    for key,value in [('epochs',250),('tail_weight',1e-5),('max_steps',100000),('monitor_every_epochs',5)]:
        other=deepcopy(c);other[key]=value
        assert a!=training_signature(model,other,data,scale,['t'],['v'])

def test_fresh_checkpoint_restores_adamw_rng_and_both_ramps_exactly(tmp_path):
    c=config();c.update(tail_weight=1e-6,calibration_identity={'sha256':'unit-fixture'})
    architecture=ModelConfig();data={'manifest':{'sha256':'data'},'index':{'sha256':'index'}};scale={'sha256':'scale'}
    torch.manual_seed(7);model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
    gen=torch.Generator().manual_seed(9);scheduler=UpdateSchedule(ScheduleConfig(**c['schedule']))
    def update(m,o,g,s):
        n=s.completed_updates+1;lr,ramp=s.apply(o,n)
        x=torch.randn(4,3,generator=g);y=torch.randn(4,2,generator=g)
        o.zero_grad();loss=(m(x)-y).square().mean()*(1+ramp);loss.backward();o.step();s.complete(n)
        return float(loss)
    update(model,optimizer,gen,scheduler)
    meta=dict(model_config=architecture.to_dict(),diffusion_config=c['diffusion'],training_config=c,
        training_signature=training_signature(architecture,c,data,scale,['t'],['v']),
        data=data,scale_audit=scale,run_sha='fixture',run_branch='test',run_dirty=False,parent_checkpoint=None,
        runtime=dict(torch=str(torch.__version__),update_schedule=scheduler.state_dict()))
    path=tmp_path/'fresh.pt';save_checkpoint(path,model,optimizer,gen,1,meta)
    expected=update(model,optimizer,gen,scheduler);state=deepcopy(model.state_dict())
    payload=load_checkpoint(path,signature=meta['training_signature'])
    m=torch.nn.Linear(3,2);o=torch.optim.AdamW(m.parameters());g=torch.Generator()
    restore_training(payload,m,o,g,'cpu');s=UpdateSchedule(ScheduleConfig(**c['schedule']))
    s.load_state_dict(payload['metadata']['runtime']['update_schedule'],o,1)
    assert update(m,o,g,s)==expected and torch.equal(g.get_state(),gen.get_state())
    assert s.state_dict()==scheduler.state_dict()
    for k,v in state.items():assert torch.equal(v,m.state_dict()[k])
    bad=deepcopy(payload);bad['metadata']['runtime']['update_schedule']['completed_updates']=2
    torch.save(bad,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='mismatch'):load_checkpoint(tmp_path/'bad.pt')

def test_test_evaluation_requires_a_seal_before_any_tasks_are_read():
    with pytest.raises(ValueError,match='Seal'):verify_test_seal(None,'not-read','not-read',{})
