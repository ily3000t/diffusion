from copy import deepcopy
import pytest
import torch
from test_raw_velocity_training import parent150,request as request150,Toy
from test_physical_losses import fixture
from sumodiff.models import ModelConfig
from sumodiff.diffusion.raw_velocity_training import raw_velocity_config
from sumodiff.diffusion.raw_velocity_losses import resolve_raw_velocity_loss
from sumodiff.diffusion.unified_training import unified_signature,new_optimizer,optimize_batch
from sumodiff.diffusion.decoded_temporal_training import DEFAULTS,decoded_temporal_config,initialize,validate_runtime
from sumodiff.diffusion.decoded_temporal_losses import DecodedTemporalLossConfig,decoded_temporal_objective
from sumodiff.diffusion.schedule import UpdateSchedule,resolve_schedule
from sumodiff.diffusion.checkpoint import load_checkpoint,save_checkpoint,restore_training
from sumodiff.diffusion.process import NoiseSchedule
from sumodiff.diffusion.audit import exact_tree


def parent200():
    p=parent150();old=raw_velocity_config(request150(p),p)
    p['step']=60000;m=p['metadata'];m.update(training_config=old,diffusion_config=old['diffusion'])
    learning=UpdateSchedule(resolve_schedule(old['schedule']));learning.completed_updates=15000
    learning.last_lr=learning.learning_rate(15000);learning.last_auxiliary_ramp=1.
    m['runtime'].update(scheduler_state=learning.state_dict(),best_epsilon=dict(step=60000,mse=.001,checkpoint=None))
    m['training_signature']=unified_signature(ModelConfig(),old,m['data'],m['scale_audit'],['train'],['val'])
    for x in p['optimizer']['state'].values():x['step'].fill_(30000)
    return p


def request(p):
    return dict(DEFAULTS,expected_parent_sha256='3'*64,
        loss=DecodedTemporalLossConfig(inherited=resolve_raw_velocity_loss(p['metadata']['training_config']['loss'])).to_dict())


def metadata(c,p,learning):
    m=deepcopy(p['metadata']);m.update(training_config=c,diffusion_config=c['diffusion'],
        training_signature=unified_signature(ModelConfig(),c,m['data'],m['scale_audit'],['train'],['val']))
    m['runtime']['scheduler_state']=learning.state_dict();return m


def test_protocol_preserves_parent_and_starts_new_25_epoch_schedule():
    p=parent200();c=decoded_temporal_config(request(p),p)
    assert c['source_step']==60000 and c['max_steps']==67500 and c['epochs']==225
    assert c['schedule']['total_updates']==7500 and c['schedule']['warmup_updates']==300
    assert c['monitor_every_epochs']==5 and 'raw_auxiliary_warmup_epochs' not in c
    model=Toy();opt=new_optimizer(model,c);g=torch.Generator();initialize(p,model,opt,g,'cpu')
    assert exact_tree(opt.state_dict(),p['optimizer'])
    for x in opt.state.values():x['step'].add_(1)
    assert all(float(x['step'])==30000 for x in p['optimizer']['state'].values())
    short=decoded_temporal_config(request(p),p,prefix_updates=16)
    assert short['max_steps']==60016 and short['schedule']['total_updates']==7500


@pytest.mark.parametrize('change',[{'expected_parent_step':45000},{'additional_epochs':50},{'batch_size':8},
    {'monitor_every_epochs':10},{'warmup_epochs':2},{'minimum_learning_rate':True},{'peak_learning_rate':5e-5},
    {'expected_parent_sha256':'bad'}])
def test_wrong_old_or_unbounded_protocol_rejected(change):
    p=parent200();r=request(p);r.update(change)
    with pytest.raises(ValueError):decoded_temporal_config(r,p)


def test_inherited_loss_and_optimizer_cannot_silently_change():
    p=parent200();r=request(p);r['loss']['inherited']['interior_weight']*=2
    with pytest.raises(ValueError,match='Keep'):decoded_temporal_config(r,p)
    r=request(p);r['loss']['interior_weight']*=2
    with pytest.raises(ValueError,match='interior'):decoded_temporal_config(r,p)
    p['optimizer']['state'][0]['step'].fill_(1)
    with pytest.raises(ValueError,match='30000'):decoded_temporal_config(request(p),p)


def test_real_resume_preserves_AdamW_noise_and_both_ramps(tmp_path):
    p=parent200();c=decoded_temporal_config(request(p),p,prefix_updates=8)
    obj=DecodedTemporalLossConfig(inherited=resolve_raw_velocity_loss(p['metadata']['training_config']['loss']))
    net=Toy();opt=new_optimizer(net,c);g=torch.Generator();initialize(p,net,opt,g,'cpu')
    learning=UpdateSchedule(resolve_schedule(c['schedule']));cond,target=fixture(batch=2,count=40);noise=NoiseSchedule()
    for u in range(1,9):
        terms,*_=optimize_batch(net,opt,cond,target,noise,g,1.,obj,None,learning,u,objective_fn=decoded_temporal_objective)
        assert terms['raw_auxiliary_ramp']==1. and terms['temporal_reweight_ramp']==u/300
        if u==4:save_checkpoint(tmp_path/'mid.pt',net,opt,g,60004,metadata(c,p,learning))
    mid=load_checkpoint(tmp_path/'mid.pt');other=Toy();oopt=new_optimizer(other,c);og=torch.Generator()
    restore_training(mid,other,oopt,og,'cpu');resumed=UpdateSchedule(resolve_schedule(c['schedule']))
    resumed.load_state_dict(mid['metadata']['runtime']['scheduler_state'],oopt,expected_updates=4)
    for u in range(5,9):optimize_batch(other,oopt,cond,target,noise,og,1.,obj,None,resumed,u,objective_fn=decoded_temporal_objective)
    assert exact_tree(net.state_dict(),other.state_dict()) and exact_tree(opt.state_dict(),oopt.state_dict())
    assert torch.equal(g.get_state(),og.get_state()) and learning.state_dict()==resumed.state_dict()
    mid['optimizer']['state'][0]['step'].fill_(4)
    with pytest.raises(ValueError,match='reset or lost'):validate_runtime(mid)


def test_checkpoint_signature_locks_group_weights_and_stage5e_budget(tmp_path):
    p=parent200();c=decoded_temporal_config(request(p),p,prefix_updates=8)
    net=Toy();opt=new_optimizer(net,c);g=torch.Generator();initialize(p,net,opt,g,'cpu')
    learning=UpdateSchedule(resolve_schedule(c['schedule']));m=metadata(c,p,learning)
    save_checkpoint(tmp_path/'initial.pt',net,opt,g,60000,m)
    other=deepcopy(m['training_signature']);other['training']['loss']['boundary_weight']=.003
    with pytest.raises(ValueError,match='signature'):load_checkpoint(tmp_path/'initial.pt',signature=other)
