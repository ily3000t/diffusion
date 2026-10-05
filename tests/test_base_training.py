from copy import deepcopy
import pytest
import torch
from sumodiff.models import ModelConfig
from sumodiff.diffusion import DiffusionConfig
from sumodiff.diffusion.training import train_config,training_signature
from sumodiff.diffusion.checkpoint import save_checkpoint,load_checkpoint,restore_training


def metadata(config):
    model=ModelConfig();diffusion=DiffusionConfig();data={'manifest':{'sha256':'data'},'index':{'sha256':'index'}};scale={'sha256':'scale'}
    return dict(model_config=model.to_dict(),diffusion_config=diffusion.to_dict(),training_config=config,
        training_signature=training_signature(model,config,data,scale,['train'],['val']),data=data,scale_audit=scale,
        run_sha='code',run_branch='feature',run_dirty=False,parent_checkpoint=None,runtime={'torch':str(torch.__version__),'device':'cpu'})


def test_adamw_checkpoint_continuation_matches_uninterrupted(tmp_path):
    torch.manual_seed(43);model=torch.nn.Linear(3,2);opt=torch.optim.AdamW(model.parameters());gen=torch.Generator().manual_seed(22)
    def step(model,opt,gen):
        x=torch.randn(4,3,generator=gen);y=torch.randn(4,2,generator=gen);opt.zero_grad();loss=(model(x)-y).square().mean();loss.backward();opt.step();return float(loss)
    step(model,opt,gen);meta=metadata(train_config({}));path=tmp_path/'checkpoint.pt';save_checkpoint(path,model,opt,gen,1,meta)
    expected=step(model,opt,gen);expected_state=deepcopy(model.state_dict());expected_rng=gen.get_state().clone()
    loaded=load_checkpoint(path,signature=meta['training_signature']);other=torch.nn.Linear(3,2);otheropt=torch.optim.AdamW(other.parameters());g=torch.Generator()
    restore_training(loaded,other,otheropt,g,'cpu');actual=step(other,otheropt,g)
    assert expected==actual and torch.equal(expected_rng,g.get_state())
    for key,value in expected_state.items(): assert torch.equal(value,other.state_dict()[key])
    assert otheropt.state_dict()['param_groups']==opt.state_dict()['param_groups']
    for state,otherstate in zip(opt.state.values(),otheropt.state.values()):
        for key,value in state.items(): assert torch.equal(value,otherstate[key])
    with pytest.raises(FileExistsError): save_checkpoint(path,other,otheropt,g,2,meta)
    with pytest.raises(ValueError,match='signature'): load_checkpoint(path,signature={})
    with pytest.raises(NotImplementedError,match='cfg'): load_checkpoint(path,requested={'cfg':True})


def test_old_checkpoints_and_missing_metadata_are_rejected(tmp_path):
    path=tmp_path/'old.pt';torch.save({'model_state_dict':{}},path)
    with pytest.raises(ValueError,match='incompatible'): load_checkpoint(path)


def test_resume_identity_allows_more_steps_but_not_changed_data_or_units():
    a=train_config({});b=deepcopy(a);b['max_steps']=480
    assert metadata(a)['training_signature']==metadata(b)['training_signature']
    for key,value in [('batch_size',2),('learning_rate',.001),('noise_seed',30)]:
        b=deepcopy(a);b[key]=value;assert metadata(a)['training_signature']!=metadata(b)['training_signature']
    b=deepcopy(a);b['diffusion']['position_scale_m']=20
    assert metadata(a)['training_signature']!=metadata(b)['training_signature']


@pytest.mark.parametrize('options',[{'max_steps':True},{'batch_size':0},{'precision':'amp'},{'diffusion':{'cfg':True}}])
def test_invalid_train_options(options):
    with pytest.raises((ValueError,NotImplementedError)): train_config(options)


def test_checkpoint_tree_audit_catches_optimizer_and_rng_changes():
    from sumodiff.diffusion.audit import exact_tree
    a={'model':torch.ones(2),'optimizer':[torch.zeros(2)],'rng':torch.tensor([1],dtype=torch.uint8)}
    assert exact_tree(a,deepcopy(a))
    for key in ('model','rng'):
        b=deepcopy(a);b[key][0]+=1;assert not exact_tree(a,b)
    b=deepcopy(a);b['optimizer'][0][0]+=1;assert not exact_tree(a,b)
