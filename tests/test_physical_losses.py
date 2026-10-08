from copy import deepcopy
import pytest
import torch
from sumodiff.diffusion import NoiseSchedule, DiffusionConfig, normalize, masked_noise_mse
from sumodiff.diffusion.physical_losses import PhysicalLossConfig, physical_objective, boundary_vectors


def fixture(batch=1, agents=2, count=6, dtype=torch.float64):
    dt=.1
    initial=torch.zeros(batch, agents, 2, dtype=dtype)
    initial[:, :, 1]=torch.arange(agents, dtype=dtype)*4
    k=torch.arange(-20,1,dtype=dtype)
    history=torch.zeros(batch,agents,21,6,dtype=dtype)
    history[...,0]=k*dt*10+k*(k-1)/2*dt**2*2
    history[...,1]=initial[:,:,1,None]
    history[...,2]=10+k*dt*2
    history[...,5]=1
    # q(-1)=-1 and q(-2)=-1.98 -> observed interval v0=10, a0=2.
    history[..., -3, 0]=-1.98;history[..., -2, 0]=-1;history[..., -1, 0]=0
    c=dict(agent_mask=torch.ones(batch,agents,dtype=torch.bool),history=history,
           history_mask=torch.ones(batch,agents,21,dtype=torch.bool),initial_positions=initial)
    k=torch.arange(1,count+1,dtype=dtype)
    future=torch.zeros(batch,agents,count,6,dtype=dtype)
    future[...,0]=k*dt*10+k*(k+1)/2*dt**2*2
    future[...,2]=10+k*dt*2
    future[...,5]=1
    targets=dict(future=future,future_mask=torch.ones(batch,agents,count,dtype=torch.bool))
    return c,targets


def inputs(c,t,timestep=200):
    s=NoiseSchedule();clean=normalize(t['future'],s.config)
    noise=torch.randn(clean.shape,dtype=clean.dtype,generator=torch.Generator().manual_seed(7))
    times=torch.full((len(clean),),timestep,dtype=torch.long)
    noisy=s.add_noise(clean,noise,times)
    return s,noise,times,noisy


def test_A_is_bitwise_legacy_mse_and_gradient():
    c,t=fixture()
    s,noise,ts,noisy=inputs(c,t)
    p=(noise+.13).detach().requires_grad_(True)
    expected=masked_noise_mse(p,noise,c['agent_mask'],t['future_mask'])
    expected_gradient=torch.autograd.grad(expected,p)[0]
    loss=physical_objective(p,noisy,noise,ts,c,t,s,PhysicalLossConfig())
    assert torch.equal(loss['total_loss'],expected)
    assert loss['physical_total']==0 and loss['auxiliary_eligible_scenes']==0
    assert torch.equal(torch.autograd.grad(loss['total_loss'],p)[0],expected_gradient)


def test_exact_noise_has_zero_auxiliary_losses_for_nonzero_acceleration():
    c,t=fixture();s,noise,ts,noisy=inputs(c,t)
    cfg=PhysicalLossConfig(consistency_weight=.01,velocity_difference_weight=.01,boundary_weight=.01)
    result=physical_objective(noise,noisy,noise,ts,c,t,s,cfg)
    for name in ('consistency','velocity_difference','boundary'):
        assert result[name] < 1e-20
    assert torch.diff(t['future'][...,2],dim=-1).abs().min()>.1


def test_inconsistency_supplies_position_and_velocity_gradients():
    c,t=fixture();s,noise,ts,noisy=inputs(c,t)
    p=noise.clone();p[...,2]+=.2;p.requires_grad_(True)
    result=physical_objective(p,noisy,noise,ts,c,t,s,PhysicalLossConfig(consistency_weight=.01))
    assert result['consistency']>0
    g=torch.autograd.grad(result['consistency'],p)[0]
    assert g[...,:2].abs().sum()>0 and g[...,2:4].abs().sum()>0
    assert torch.isfinite(g).all()


def test_velocity_difference_detects_temporal_change_not_constant_bias():
    c,t=fixture();s,noise,ts,noisy=inputs(c,t)
    cfg=PhysicalLossConfig(velocity_difference_weight=.01)
    p=noise.clone();p[...,2]+=.15
    first=physical_objective(p,noisy,noise,ts,c,t,s,cfg)
    p[...,2]+=.1*torch.arange(p.shape[-2],dtype=p.dtype)
    second=physical_objective(p,noisy,noise,ts,c,t,s,cfg)
    assert first['velocity_difference']<1e-20 and second['velocity_difference']>0


def test_boundary_gradients_and_missing_context_rejection():
    c,t=fixture();s,noise,ts,noisy=inputs(c,t)
    p=noise.clone();p[...,0,0]+=.1;p.requires_grad_(True)
    cfg=PhysicalLossConfig(boundary_weight=.01)
    out=physical_objective(p,noisy,noise,ts,c,t,s,cfg)
    assert out['boundary']>0
    gradient=torch.autograd.grad(out['boundary'],p)[0]
    assert gradient[...,:2].abs().sum()>0 and gradient[...,2:4].abs().sum()>0
    c['history_mask'][...,-2]=False
    with pytest.raises(ValueError,match='complete'):physical_objective(p,noisy,noise,ts,c,t,s,cfg)


def test_high_noise_has_exact_mse_and_batch_denominator_is_preserved():
    c,t=fixture(batch=2);s,noise,ts,noisy=inputs(c,t,timestep=800)
    cfg=PhysicalLossConfig(consistency_weight=.01)
    p=(noise+.03).detach().requires_grad_(True)
    out=physical_objective(p,noisy,noise,ts,c,t,s,cfg)
    assert out['auxiliary_eligible_scenes']==0
    assert torch.equal(out['total_loss'],out['noise_mse'])
    ts[0]=200;noisy=s.add_noise(normalize(t['future'],s.config),noise,ts)
    out=physical_objective(p,noisy,noise,ts,c,t,s,cfg)
    single=physical_objective(p[:1],noisy[:1],noise[:1],ts[:1],
        {k:v[:1] for k,v in c.items()},{k:v[:1] for k,v in t.items()},s,cfg)
    assert out['auxiliary_eligible_scenes']==1
    torch.testing.assert_close(out['consistency'],single['consistency']/2)


def test_poisoned_agent_padding_has_zero_gradient_and_no_effect():
    c,t=fixture();s,noise,ts,noisy=inputs(c,t)
    c['agent_mask'][:,1]=False;c['history_mask'][:,1]=False
    c['history'][:,1]=float('nan');c['initial_positions'][:,1]=float('nan')
    t['future'][:,1]=float('nan')
    p=(noise+.02).clone();p[:,1]=float('nan');p.requires_grad_(True)
    cfg=PhysicalLossConfig(consistency_weight=.01,velocity_difference_weight=.01,boundary_weight=.01)
    out=physical_objective(p,noisy,noise,ts,c,t,s,cfg)
    assert torch.isfinite(out['total_loss'])
    grad=torch.autograd.grad(out['total_loss'],p)[0]
    assert torch.isfinite(grad).all() and not grad[:,1].any()


def test_combined_loss_gradcheck():
    c,t=fixture(agents=1,count=3);s,noise,ts,noisy=inputs(c,t)
    cfg=PhysicalLossConfig(consistency_weight=.01,velocity_difference_weight=.01,boundary_weight=.01)
    p=(noise+.01).detach().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x:physical_objective(x,noisy,noise,ts,c,t,s,cfg)['total_loss'],
                                   (p,),atol=2e-5,rtol=1e-3)


@pytest.mark.parametrize('bad',[dict(consistency_weight=-1),dict(huber_beta=0),
    dict(max_auxiliary_timestep=True),dict(dt=.2)])
def test_invalid_physical_configuration_is_explicit(bad):
    with pytest.raises((ValueError,NotImplementedError)):PhysicalLossConfig(**bad)
