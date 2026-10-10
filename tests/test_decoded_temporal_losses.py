from copy import deepcopy
import pytest
import torch
from test_physical_losses import fixture, inputs
from sumodiff.diffusion.raw_velocity_losses import RawVelocityLossConfig, raw_velocity_objective
from sumodiff.diffusion.unified_losses import UnifiedLossConfig
from sumodiff.diffusion.decoded_temporal_losses import (
    DecodedTemporalLossConfig, resolve_decoded_temporal_loss, decoded_temporal_objective)


def config():
    return DecodedTemporalLossConfig(inherited=RawVelocityLossConfig(
        unified=UnifiedLossConfig(road_weight=0.)))


def evaluate(c, t, cfg=None, ramp=1., timestep=200):
    s, noise, ts, noisy = inputs(c, t, timestep)
    pred = (noise+.013*torch.randn(noise.shape, dtype=noise.dtype,
        generator=torch.Generator().manual_seed(3))).detach().requires_grad_(True)
    out = decoded_temporal_objective(pred,noisy,noise,ts,c,t,s,cfg or config(),ramp=ramp)
    return out,pred,(s,noise,ts,noisy)


def test_zero_ramp_reproduces_parent_total_and_gradient():
    c,t=fixture(batch=2,count=40)
    new,p,(s,n,ts,x)=evaluate(c,t,ramp=0.)
    old=raw_velocity_objective(p,x,n,ts,c,t,s,config().inherited,ramp=1.)
    torch.testing.assert_close(new['total_loss'],old['total_loss'],rtol=1e-12,atol=1e-14)
    a=torch.autograd.grad(new['total_loss'],p,retain_graph=True)[0]
    b=torch.autograd.grad(old['total_loss'],p)[0]
    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-13)
    assert new['raw_auxiliary_ramp']==new['inherited_auxiliary_ramp']==1.
    assert new['effective_decoded_boundary_weight']==.0001


def test_groups_replace_old_temporal_and_keep_other_objectives():
    c,t=fixture(count=40);out,_,_=evaluate(c,t)
    assert out['effective_decoded_boundary_weight']==pytest.approx(.001)
    assert out['effective_decoded_interior_weight']==pytest.approx(.0009)
    torch.testing.assert_close(out['temporal'],.1*out['decoded_temporal_boundary']+.9*out['decoded_temporal_interior'])
    expected=sum(out[k] for k in ('weighted_imitation','weighted_road','weighted_raw_boundary',
        'weighted_raw_interior','weighted_decoded_boundary','weighted_decoded_interior'))
    torch.testing.assert_close(out['auxiliary_total'],expected)
    torch.testing.assert_close(out['weighted_temporal'],out['weighted_decoded_boundary']+out['weighted_decoded_interior'])


def test_boundary_gradient_reaches_later_raw_predictions_through_global_decoder():
    c,t=fixture(count=40);out,p,_=evaluate(c,t)
    grad=torch.autograd.grad(out['decoded_temporal_boundary'],p)[0]
    assert grad[...,8:,0:2].abs().max()>0
    assert torch.isfinite(grad).all()


def test_invalid_slot_nan_is_excluded_before_decoder_and_derivatives():
    c,t=fixture(agents=2,count=40);c['agent_mask'][:,1]=False
    c['history'][:,1]=float('nan');t['future'][:,1]=float('nan');t['future_mask'][:,1]=False
    out,p,_=evaluate(c,t)
    assert torch.isfinite(out['total_loss'])
    grad=torch.autograd.grad(out['total_loss'],p)[0]
    assert torch.equal(grad[:,1],torch.zeros_like(grad[:,1]))


def test_low_t_gate_full_batch_denominator_and_no_eligible_are_explicit():
    c,t=fixture(batch=2,count=40);out,_,_=evaluate(c,t,timestep=500)
    assert out['auxiliary_eligible_scenes']==0 and out['auxiliary_total']==0
    out,_,_=evaluate(c,t,timestep=0)
    assert out['auxiliary_eligible_scenes']==2 and out['decoded_temporal_boundary']>=0


@pytest.mark.parametrize('change',[{'boundary_weight':0},{'interior_weight':float('nan')},
    {'boundary_frames':8},{'boundary_weight':.1},{'schema_version':'old'}])
def test_invalid_loss_options_rejected(change):
    value=config().to_dict();value.update(change)
    with pytest.raises(ValueError):resolve_decoded_temporal_loss(value)


def test_group_extension_requires_complete_40_future_points():
    c,t=fixture(count=40);t['future_mask'][...,7]=False
    with pytest.raises(ValueError,match='complete'):evaluate(c,t)
    c,t=fixture(count=6)
    with pytest.raises(ValueError,match='40'):evaluate(c,t)


def test_label_acceleration_is_supervised_without_forcing_zero_derivatives():
    c,t=fixture(count=40);s,n,ts,x=inputs(c,t)
    out=decoded_temporal_objective(n,x,n,ts,c,t,s,config())
    assert out['decoded_temporal_boundary']<1e-18 and out['decoded_temporal_interior']<1e-18
    assert out['raw_second_boundary']<1e-18 and out['raw_second_interior']<1e-18


def test_mixed_timestep_keeps_whole_batch_denominator():
    from sumodiff.diffusion.process import normalize
    c,t=fixture(batch=2,count=40);s,n,_,_=inputs(c,t)
    ts=torch.tensor([0,500]);x=s.add_noise(normalize(t['future'],s.config),n,ts);p=(n+.01).requires_grad_(True)
    both=decoded_temporal_objective(p,x,n,ts,c,t,s,config())
    one=decoded_temporal_objective(p[:1],x[:1],n[:1],ts[:1],{k:v[:1] for k,v in c.items()},
        {k:v[:1] for k,v in t.items()},s,config())
    assert both['auxiliary_eligible_scenes']==1
    torch.testing.assert_close(both['auxiliary_total'],one['auxiliary_total']/2,rtol=1e-10,atol=1e-13)
