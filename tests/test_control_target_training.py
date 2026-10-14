from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch
from test_models import conditioning, small_config, vehicle_permutation
from test_base_sampling import Oracle
from sumodiff.models import ModelConfig, ConditionalDenoiser
from sumodiff.diffusion.config import DiffusionConfig, ControlDiffusionConfig, validate_architecture
from sumodiff.diffusion.process import normalize,denormalize,NoiseSchedule,ddim_sample,masked_noise_mse
from sumodiff.diffusion.representation import training_target,decode_prediction
from sumodiff.diffusion.control_training import SCHEMA,control_config,control_objective
from sumodiff.diffusion.training import draw_noise,training_signature
from sumodiff.diffusion.checkpoint import save_checkpoint,load_checkpoint,restore_training
from sumodiff.diffusion.schedule import ScheduleConfig,UpdateSchedule
from sumodiff.decoding import DecoderConfig

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def setup():
    c=conditioning()
    d=ControlDiffusionConfig(additional_losses_enabled=True)
    controls=torch.zeros(1,4,40,3)
    controls[:,:2,:,0]=.1;controls[:,:2,:,1]=.05;controls[:,:2,:,2]=.02
    out=decode_prediction(controls,c,d)
    t=dict(future=out.raw_future.float(),future_mask=c['agent_mask'][...,None].expand(1,4,40).clone())
    return c,d,controls,t


def model_config():
    return replace(small_config(),schema_version='sumodiff.model.control.config.v1',state_dimension=3)


def recipe():
    return control_config(dict(schema_version=SCHEMA,max_steps=5,device='cpu',
        unified_loss=dict(road_weight=0),tail_weight=.0003))


def test_control_schemas_and_physical_scales_are_explicit():
    d=ControlDiffusionConfig()
    x=torch.tensor([[[[4.,1.,.2]]]])
    torch.testing.assert_close(normalize(x,d),torch.full_like(x,2))
    torch.testing.assert_close(denormalize(normalize(x,d),d),x)
    with pytest.raises(ValueError):normalize(torch.zeros(1,1,40,6),d)
    with pytest.raises(ValueError):validate_architecture(ModelConfig(),d)
    with pytest.raises(NotImplementedError):ModelConfig(state_dimension=3)
    validate_architecture(model_config(),d)
    for bad in [dict(max_steps=65),dict(raw_loss={}),dict(diffusion=DiffusionConfig().to_dict())]:
        with pytest.raises(ValueError):control_config(dict(schema_version=SCHEMA,**bad))


def test_label_transform_roundtrip_padding_and_missing_horizon():
    c,d,controls,t=setup()
    c['history'][:,2:]=float('nan');c['initial_positions'][:,2:]=float('nan');c['attributes'][:,2:]=float('nan')
    t['future'][:,2:]=float('nan')
    z=training_target(t,c,d)
    assert not z[:,2:].any()
    rebuilt=decode_prediction(denormalize(z,d),c,d)
    torch.testing.assert_close(rebuilt.states[:,:2,:2],decode_prediction(controls,c,d).states[:,:2,:2],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(rebuilt.raw_future[:,:2,:,:2],t['future'][:,:2,:,:2].double(),atol=1e-6,rtol=1e-6)
    t['future_mask'][0,0,-1]=False
    with pytest.raises(ValueError,match='complete'):training_target(t,c,d)
    with pytest.raises(ValueError,match='LS decoder'):decode_prediction(controls,c,d,DecoderConfig(position_weight=2))


def test_rollout_gradient_and_joint_scene_denominator():
    c,d,controls,t=setup()
    z=training_target(t,c,d)
    schedule=NoiseSchedule(d)
    time=torch.tensor([200]);noise=torch.randn_like(z)
    noisy=schedule.add_noise(z,noise,time)
    epsilon=(noise+.1).requires_grad_()
    cfg=recipe()
    out=control_objective(epsilon,noisy,noise,time,c,t,schedule,cfg)
    out['total_loss'].backward()
    assert torch.isfinite(epsilon.grad).all() and epsilon.grad[:,:2].abs().sum()>0
    assert not epsilon.grad[:,2:].any()
    pair_c={k:v.repeat(2,*([1]*(v.ndim-1))) for k,v in c.items()}
    pair_t={k:v.repeat(2,*([1]*(v.ndim-1))) for k,v in t.items()}
    pair_z=z.repeat(2,1,1,1);pair_noise=noise.repeat(2,1,1,1);pair_epsilon=epsilon.detach().repeat(2,1,1,1)
    times=torch.tensor([200,900]);pair_noisy=schedule.add_noise(pair_z,pair_noise,times)
    both=control_objective(pair_epsilon,pair_noisy,pair_noise,times,pair_c,pair_t,schedule,cfg)
    torch.testing.assert_close(both['auxiliary_total'],out['auxiliary_total']/2)


def test_exact_control_reconstruction_has_negligible_paired_auxiliary_loss():
    c,d,_,t=setup();z=training_target(t,c,d);schedule=NoiseSchedule(d)
    time=torch.tensor([200]);noise=torch.randn_like(z);noisy=schedule.add_noise(z,noise,time)
    out=control_objective(noise,noisy,noise,time,c,t,schedule,recipe())
    assert float(out['auxiliary_total'])<1e-10


@pytest.mark.parametrize('fusion',['hierarchical','parallel'])
def test_three_channel_network_permutation_padding_and_backward(fusion):
    c,d,_,t=setup()
    model=ConditionalDenoiser(replace(model_config(),fusion=fusion)).eval()
    z=training_target(t,c,d);time=torch.tensor([200]);order=torch.tensor([2,1,3,0])
    expected=model(z,time,c)
    actual=model(z[:,order],time,vehicle_permutation(c,order))
    torch.testing.assert_close(actual,expected[:,order],atol=2e-6,rtol=2e-5)
    assert expected.shape==(1,4,40,3)
    expected.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_three_channel_ddim_uses_same_rollout_without_label_condition():
    c,d,_,t=setup();z=training_target(t,c,d);schedule=NoiseSchedule(d)
    model=Oracle(schedule,z);noise=torch.randn_like(z);noise[:,2:]=float('nan')
    result=ddim_sample(model,c,noise,schedule,20)
    torch.testing.assert_close(result[:,:2],z[:,:2],atol=1e-6,rtol=1e-6)
    out=decode_prediction(denormalize(result,d),c,d)
    assert model.calls==20 and not out.heading_fallback_mask.any()
    assert not out.position_correction.any()
    torch.testing.assert_close(out.states,out.raw_states)


def test_control_checkpoint_resume_preserves_actual_model_optimizer_rng_and_schedule(tmp_path):
    c,d,_,t=setup();cfg=recipe();arch=model_config()
    torch.manual_seed(42);model=ConditionalDenoiser(arch)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    gen=torch.Generator().manual_seed(51);schedule=NoiseSchedule(d);lr=UpdateSchedule(ScheduleConfig(**cfg['schedule']))
    def update(step,m,o,g,scheduler):
        m.train();o.zero_grad(set_to_none=True)
        noisy,noise,time=draw_noise(t,c,schedule,g)
        rate,ramp=scheduler.apply(o,step)
        out=control_objective(m(noisy,time,c),noisy,noise,time,c,t,schedule,cfg,ramp=ramp)
        out['total_loss'].backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        o.step();scheduler.complete(step)
        return float(out['total_loss'])
    update(1,model,opt,gen,lr);update(2,model,opt,gen,lr)
    data=dict(manifest=dict(sha256='fixture'),index=dict(sha256='fixture-index'));scale=dict(sha256='fixture-scale')
    metadata=dict(model_config=arch.to_dict(),diffusion_config=d.to_dict(),training_config=cfg,
        training_signature=training_signature(arch,cfg,data,scale,['train'],['validation']),
        data=data,scale_audit=scale,run_sha='fixture',run_branch='fixture',run_dirty=True,parent_checkpoint=None,
        runtime=dict(torch=str(torch.__version__),device='cpu',update_schedule=lr.state_dict()))
    path=tmp_path/'control.pt';save_checkpoint(path,model,opt,gen,2,metadata)
    expected=update(3,model,opt,gen,lr)
    expected_state=deepcopy(model.state_dict());expected_opt=deepcopy(opt.state_dict())
    payload=load_checkpoint(path,signature=metadata['training_signature'])
    other=ConditionalDenoiser(arch);otheropt=torch.optim.AdamW(other.parameters());g=torch.Generator()
    restore_training(payload,other,otheropt,g,'cpu')
    otherlr=UpdateSchedule(ScheduleConfig(**cfg['schedule']))
    otherlr.load_state_dict(payload['metadata']['runtime']['update_schedule'],otheropt,2)
    assert update(3,other,otheropt,g,otherlr)==expected
    for key,value in expected_state.items():assert torch.equal(value,other.state_dict()[key])
    from sumodiff.diffusion.audit import exact_tree
    assert exact_tree(expected_opt,otheropt.state_dict())
    assert torch.equal(gen.get_state(),g.get_state()) and lr.state_dict()==otherlr.state_dict()
    bad=deepcopy(payload);bad['metadata']['diffusion_config']=DiffusionConfig().to_dict()
    torch.save(bad,tmp_path/'wrong.pt')
    with pytest.raises(ValueError,match='dimension'):load_checkpoint(tmp_path/'wrong.pt')


def test_joint_probe_measures_cancellation_instead_of_adding_norm_ratios():
    from sumodiff.diffusion.control_calibration import gradient_vector,cosine
    p=torch.nn.Parameter(torch.tensor([1.,2.]))
    noise=p[0];first=2*p[1];second=-2*p[1]
    ge=gradient_vector(noise,[p]);g1=gradient_vector(first,[p]);g2=gradient_vector(second,[p])
    ga=gradient_vector(first+second,[p]);gt=gradient_vector(noise+first+second,[p])
    assert float(ga.norm())==0 and float((g1.norm()+g2.norm())/ge.norm())==4
    torch.testing.assert_close(gt,ge+ga)
    assert cosine(gt,ge)==1 and cosine(ga,ge) is None


def test_control_test_sampling_rejects_before_opening_dataset(tmp_path,monkeypatch):
    from sumodiff.diffusion import sampling
    cfg=tmp_path/'sample.json';cfg.write_text(json.dumps(dict(split='test',device='cpu')))
    payload=dict(metadata=dict(model_config=model_config().to_dict(),diffusion_config=ControlDiffusionConfig().to_dict()))
    monkeypatch.setattr(sampling,'load_checkpoint',lambda *args,**kwargs:payload)
    monkeypatch.setattr(sampling,'WindowDataset',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('test read')))
    with pytest.raises(NotImplementedError,match='frozen'):
        sampling.sample('unopened-dataset','fixture-checkpoint',cfg,tmp_path/'unused',Path.cwd(),['fixture'])
