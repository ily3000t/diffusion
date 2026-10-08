from copy import deepcopy
import pytest
import torch
from sumodiff.models import ModelConfig
from sumodiff.diffusion import DiffusionConfig, NoiseSchedule, masked_noise_mse
from sumodiff.diffusion.epoch_training import epoch_config
from sumodiff.diffusion.training import train_config, training_signature, draw_noise
from sumodiff.diffusion.checkpoint import load_checkpoint, save_checkpoint
from sumodiff.diffusion.audit import exact_tree
from sumodiff.diffusion.controlled_training import controlled_config, optimize_batch, DEFAULTS
from sumodiff.diffusion.physical_losses import PhysicalLossConfig
from test_physical_losses import fixture


def parent_fixture():
    c=epoch_config(dict(schema_version='sumodiff.base.epoch.config.v1',seed=20261012,
        noise_seed=20261013,validation_seed=20261014,epochs=100))
    model=ModelConfig();data={'manifest':{'sha256':'data'},'index':{'sha256':'index'}};scale={'sha256':'scale'}
    m=dict(model_config=model.to_dict(),diffusion_config=c['diffusion'],training_config=c,
        training_signature=training_signature(model,c,data,scale,['train'],['val']),data=data,scale_audit=scale,
        run_sha='parent-code',run_branch='feature',run_dirty=False,parent_checkpoint=None,
        runtime={'torch':str(torch.__version__),'device':'cuda:0'})
    return dict(step=30000,metadata=m,optimizer=dict(param_groups=[dict(lr=.0003,weight_decay=.0001)]))


def requested(arm='A'):
    pattern={'A':(0.,0.,0.),'B':(.01,0.,0.),'C':(.01,.01,0.),'D':(.01,.01,.01)}[arm]
    return dict(DEFAULTS,arm=arm,expected_parent_sha256='1'*64,
        physical_loss=PhysicalLossConfig(consistency_weight=pattern[0],velocity_difference_weight=pattern[1],
                                        boundary_weight=pattern[2]).to_dict())


@pytest.mark.parametrize('arm',['A','B','C','D'])
def test_budget_and_inherited_contract(arm):
    p=parent_fixture();c=controlled_config(requested(arm),p)
    assert c['epochs']==110 and c['max_steps']==33000 and c['planned_additional_updates']==3000
    for key in ['learning_rate','seed','noise_seed','validation_seed','batch_size','weight_decay','gradient_clip_norm']:
        assert c[key]==p['metadata']['training_config'][key]
    assert c['diffusion']['additional_losses_enabled']==(arm!='A')
    probe=controlled_config(requested(arm),p,prefix_updates=12)
    assert probe['max_steps']==30012 and probe['execution_budget_kind']=='preflight_prefix'
    assert probe['nominal_additional_updates']==3000


@pytest.mark.parametrize('change',[dict(learning_rate=.0001),dict(batch_size=8),dict(additional_epochs=11),
    dict(expected_parent_step=24000),dict(device='cpu'),dict(expected_train_windows=1000),
    dict(expected_parent_sha256=''),dict(arm='missing'),dict(physical_loss={})])
def test_contract_rejects_unpaired_changes(change):
    c=requested('B');c.update(change)
    with pytest.raises(ValueError):controlled_config(c,parent_fixture())


def test_auxiliary_flag_cannot_silently_enter_legacy_training():
    assert DiffusionConfig(additional_losses_enabled=True).additional_losses_enabled
    with pytest.raises(ValueError,match='controlled'):train_config({'diffusion':{'additional_losses_enabled':True}})


def test_loss_changes_require_new_identity_and_strict_resume_still_rejects(tmp_path):
    p=parent_fixture();a=controlled_config(requested('A'),p);b=controlled_config(requested('B'),p)
    model=ModelConfig();m=deepcopy(p['metadata'])
    def metadata(c):
        out=deepcopy(m);out.update(training_config=c,diffusion_config=c['diffusion'],
            training_signature=training_signature(model,c,m['data'],m['scale_audit'],['train'],['val']))
        return out
    assert metadata(a)['training_signature']!=metadata(b)['training_signature']
    net=torch.nn.Linear(6,6);opt=torch.optim.AdamW(net.parameters());g=torch.Generator().manual_seed(8)
    path=tmp_path/'B.pt';save_checkpoint(path,net,opt,g,30012,metadata(b))
    assert load_checkpoint(path)['metadata']['diffusion_config']['additional_losses_enabled']
    with pytest.raises(ValueError,match='signature'):load_checkpoint(path,signature=metadata(a)['training_signature'])
    corrupt=deepcopy(metadata(b));corrupt['training_config']['schema_version']='sumodiff.base.epoch.config.v1'
    path=tmp_path/'bad.pt';save_checkpoint(path,net,opt,g,30012,corrupt)
    with pytest.raises(ValueError,match='provenance'):load_checkpoint(path)


def test_A_step_matches_the_existing_optimizer_algorithm_and_rng():
    class ToyDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(6,6,dtype=torch.float64)
        def forward(self,x,t,c):
            assert 'future_mask' not in c and 'future' not in c
            return self.linear(x)
    torch.manual_seed(73)
    legacy=ToyDenoiser();actual=deepcopy(legacy)
    first=torch.optim.AdamW(legacy.parameters(),lr=.0003,weight_decay=.0001)
    second=torch.optim.AdamW(actual.parameters(),lr=.0003,weight_decay=.0001)
    g1=torch.Generator().manual_seed(19);g2=torch.Generator().manual_seed(19)
    c,targets=fixture(batch=2,agents=2);schedule=NoiseSchedule()
    for _ in range(4):
        legacy.train();first.zero_grad(set_to_none=True)
        noisy,noise,time=draw_noise(targets,c,schedule,g1)
        loss=masked_noise_mse(legacy(noisy,time,c),noise,c['agent_mask'],targets['future_mask'])
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(legacy.parameters(),1.,error_if_nonfinite=True)
        first.step()
        terms,actual_norm,actual_noise,actual_time=optimize_batch(actual,second,c,targets,schedule,g2,1.,PhysicalLossConfig())
        assert torch.equal(loss,terms['noise_mse']) and torch.equal(norm,actual_norm)
        assert torch.equal(noise,actual_noise) and torch.equal(time,actual_time)
        assert exact_tree(legacy.state_dict(),actual.state_dict())
        assert exact_tree(first.state_dict(),second.state_dict())
        assert torch.equal(g1.get_state(),g2.get_state())
