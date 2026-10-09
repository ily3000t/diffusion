from copy import deepcopy
import pytest
import torch
from test_controlled_training import parent_fixture
from test_unified_training import requested
from test_physical_losses import fixture
from sumodiff.models import ModelConfig
from sumodiff.diffusion.process import NoiseSchedule
from sumodiff.diffusion.unified_training import unified_config,unified_signature,new_optimizer,optimize_batch
from sumodiff.diffusion.raw_velocity_training import DEFAULTS,raw_velocity_config,inherit_optimizer,validate_runtime
from sumodiff.diffusion.raw_velocity_losses import RawVelocityLossConfig,raw_velocity_objective
from sumodiff.diffusion.unified_losses import resolve_unified_loss
from sumodiff.diffusion.schedule import UpdateSchedule,resolve_schedule
from sumodiff.diffusion.checkpoint import save_checkpoint,load_checkpoint,restore_training
from sumodiff.diffusion.audit import exact_tree

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(6,6,dtype=torch.float64)
    def forward(self,x,t,c):
        assert 'future' not in c and 'future_mask' not in c
        return self.linear(x)

def parent150():
    p=parent_fixture();old=unified_config(requested(),p)
    p['step']=45000;m=p['metadata'];m.update(training_config=old,diffusion_config=old['diffusion'])
    learning=UpdateSchedule(resolve_schedule(old['schedule']));learning.completed_updates=15000
    learning.last_lr=learning.learning_rate(15000);learning.last_auxiliary_ramp=1.
    m['runtime'].update(snapshot_kind='training_recovery',scheduler_state=learning.state_dict(),
        torch=str(torch.__version__),best_epsilon=dict(step=45000,mse=.001,checkpoint=None),quality_history=[])
    m['training_signature']=unified_signature(ModelConfig(),old,m['data'],m['scale_audit'],['train'],['val'])
    source=Toy();opt=new_optimizer(source,old)
    source.linear(torch.ones(1,6,dtype=torch.float64)).square().sum().backward();opt.step()
    for v in opt.state.values():v['step'].fill_(15000)
    p.update(model=deepcopy(source.state_dict()),optimizer=deepcopy(opt.state_dict()),
        generator_state=torch.Generator().manual_seed(8).get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=[])
    return p

def request(p):
    loss=RawVelocityLossConfig(unified=resolve_unified_loss(p['metadata']['training_config']['loss']))
    return dict(DEFAULTS,expected_parent_sha256='2'*64,loss=loss.to_dict())

def metadata(c,p,learning):
    m=deepcopy(p['metadata'])
    m.update(training_config=c,diffusion_config=c['diffusion'],
        training_signature=unified_signature(ModelConfig(),c,m['data'],m['scale_audit'],['train'],['val']))
    m['runtime']['scheduler_state']=learning.state_dict()
    return m

def test_stage5d_keeps_full_state_and_has_new_epoch_range():
    p=parent150();c=raw_velocity_config(request(p),p)
    assert c['source_step']==45000 and c['max_steps']==60000 and c['epochs']==200
    assert c['schedule']['peak_lr']==2e-5 and c['monitor_every_epochs']==10
    assert c['schedule']['auxiliary_warmup_updates']==600 and c['inherited_auxiliary_ramp']==1
    net=Toy();opt=new_optimizer(net,c);g=torch.Generator()
    inherit_optimizer(p,net,opt,g,'cpu')
    assert exact_tree(opt.state_dict(),p['optimizer'])
    for v in opt.state.values():v['step'].add_(1)
    assert all(float(v['step'])==15000 for v in p['optimizer']['state'].values())
    for v in opt.state.values():v['step'].sub_(1)
    assert exact_tree(net.state_dict(),p['model']) and torch.equal(g.get_state(),p['generator_state'])
    prefix=raw_velocity_config(request(p),p,prefix_updates=600)
    assert prefix['max_steps']==45600 and prefix['schedule']['total_updates']==15000

@pytest.mark.parametrize('change',[{'expected_parent_step':30000},{'additional_epochs':10},
    {'batch_size':8},{'monitor_every_epochs':5},{'peak_learning_rate':1e-4},
    {'minimum_learning_rate':True},{'raw_auxiliary_warmup_epochs':5},{'expected_parent_sha256':'bad'}])
def test_old_or_unbounded_protocol_changes_rejected(change):
    p=parent150();r=request(p);r.update(change)
    with pytest.raises(ValueError):raw_velocity_config(r,p)

def test_parent_and_existing_non_imitation_objectives_cannot_silently_change():
    p=parent150();r=request(p);r['loss']['unified']['temporal_weight']=.5
    with pytest.raises(ValueError,match='Keep'):raw_velocity_config(r,p)
    r=request(p);r['loss']['unified']['imitation_weight']/=2
    assert raw_velocity_config(r,p)['loss']['unified']['imitation_weight']>0
    p['step']=30000
    with pytest.raises(ValueError,match='150-epoch'):raw_velocity_config(request(p),p)

def test_real_training_and_checkpoint_resume_preserve_moments_ramps_and_noise(tmp_path):
    p=parent150();c=raw_velocity_config(request(p),p,prefix_updates=8)
    net=Toy();opt=new_optimizer(net,c);g=torch.Generator()
    inherit_optimizer(p,net,opt,g,'cpu')
    learning=UpdateSchedule(resolve_schedule(c['schedule']))
    cond,target=fixture(batch=2,count=40);schedule=NoiseSchedule()
    obj=RawVelocityLossConfig(unified=resolve_unified_loss(p['metadata']['training_config']['loss']))
    for u in range(1,9):
        optimize_batch(net,opt,cond,target,schedule,g,1.,obj,None,learning,u,objective_fn=raw_velocity_objective)
        if u==4:save_checkpoint(tmp_path/'mid.pt',net,opt,g,45004,metadata(c,p,learning))
    mid=load_checkpoint(tmp_path/'mid.pt');other=Toy();otheropt=new_optimizer(other,c);og=torch.Generator()
    restore_training(mid,other,otheropt,og,'cpu')
    resumed=UpdateSchedule(resolve_schedule(c['schedule']))
    resumed.load_state_dict(mid['metadata']['runtime']['scheduler_state'],otheropt,expected_updates=4)
    for u in range(5,9):
        optimize_batch(other,otheropt,cond,target,schedule,og,1.,obj,None,resumed,u,objective_fn=raw_velocity_objective)
    assert exact_tree(net.state_dict(),other.state_dict()) and exact_tree(opt.state_dict(),otheropt.state_dict())
    assert torch.equal(g.get_state(),og.get_state()) and learning.state_dict()==resumed.state_dict()
    mid['optimizer']['state'][0]['step'].fill_(4)
    with pytest.raises(ValueError,match='reset or lost'):validate_runtime(mid)

def test_recovery_signature_includes_new_weights_and_budget(tmp_path):
    p=parent150();c=raw_velocity_config(request(p),p,prefix_updates=8)
    net=Toy();opt=new_optimizer(net,c);g=torch.Generator();inherit_optimizer(p,net,opt,g,'cpu')
    learning=UpdateSchedule(resolve_schedule(c['schedule']))
    path=tmp_path/'initial.pt';m=metadata(c,p,learning)
    save_checkpoint(path,net,opt,g,45000,m)
    other=deepcopy(m['training_signature']);other['training']['loss']['interior_weight']=1e-4
    with pytest.raises(ValueError,match='signature'):load_checkpoint(path,signature=other)
