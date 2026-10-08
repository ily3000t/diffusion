from copy import deepcopy
import pytest
import torch
from sumodiff.diffusion.schedule import ScheduleConfig,UpdateSchedule


def test_real_update_endpoints_and_auxiliary_ramp():
    s=UpdateSchedule(ScheduleConfig())
    assert s.learning_rate(1)==1e-6
    assert s.learning_rate(600)==1e-4
    assert s.learning_rate(15000)==1e-6
    assert 1e-6<s.learning_rate(7800)<1e-4
    assert s.auxiliary_ramp(1)==1/1500
    assert s.auxiliary_ramp(1500)==s.auxiliary_ramp(15000)==1
    for k in (0,15001):
        with pytest.raises(ValueError):s.learning_rate(k)


def test_checkpoint_recovery_does_not_restart_learning_rate_or_auxiliary_ramp():
    c=ScheduleConfig(warmup_updates=3,total_updates=12,auxiliary_warmup_updates=5)
    uninterrupted=UpdateSchedule(c)
    net=torch.nn.Linear(2,2);opt=torch.optim.AdamW(net.parameters(),lr=1e-4)
    for i in range(1,8):
        uninterrupted.apply(opt,i);uninterrupted.complete(i)
    state=deepcopy(uninterrupted.state_dict())
    restored=UpdateSchedule(c);restored.load_state_dict(state,opt,expected_updates=7)
    assert uninterrupted.apply(opt,8)==restored.apply(opt,8)
    assert restored.apply(opt,8)[1]==1
    with pytest.raises(ValueError,match='consecutive'):restored.apply(opt,9)
    restored.complete(8)
    assert restored.state_dict()['completed_updates']==8


def test_scheduler_state_rejects_config_step_lr_and_ramp_drift():
    s=UpdateSchedule(ScheduleConfig())
    net=torch.nn.Linear(2,2);opt=torch.optim.AdamW(net.parameters(),lr=1e-4)
    for i in range(1,13):
        s.apply(opt,i);s.complete(i)
    state=s.state_dict()
    for key,value in [('last_lr',1e-4),('last_auxiliary_ramp',.5),('config_sha256','bad')]:
        bad=dict(state);bad[key]=value
        with pytest.raises(ValueError):UpdateSchedule(s.config).load_state_dict(bad,opt)
    with pytest.raises(ValueError,match='update mismatch'):
        UpdateSchedule(s.config).load_state_dict(state,opt,expected_updates=13)
    opt.param_groups[0]['lr']=1e-4
    with pytest.raises(ValueError,match='Optimizer'):
        UpdateSchedule(s.config).load_state_dict(state,opt)
