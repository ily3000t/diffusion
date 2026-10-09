from copy import deepcopy
import pytest
import torch
from test_physical_losses import fixture, inputs
from sumodiff.diffusion.raw_velocity_losses import (
    RawVelocityLossConfig, resolve_raw_velocity_loss, raw_velocity_objective)
from sumodiff.diffusion.unified_losses import UnifiedLossConfig, unified_objective

def configuration():
    return RawVelocityLossConfig(unified=UnifiedLossConfig(road_weight=0))

def test_exact_accelerating_labels_have_zero_added_supervision():
    c,t=fixture(count=40);s,n,ts,x=inputs(c,t)
    out=raw_velocity_objective(n,x,n,ts,c,t,s,configuration())
    assert out['raw_second_boundary']<1e-22 and out['raw_second_interior']<1e-22
    assert torch.diff(t['future'][...,2],dim=-1).abs().min()>.1

def test_raw_ramp_never_weakens_inherited_objectives():
    c,t=fixture(count=40);s,n,ts,x=inputs(c,t)
    e=(n+.05).requires_grad_()
    old=unified_objective(e,x,n,ts,c,t,s,configuration().unified,ramp=1.)
    off=raw_velocity_objective(e,x,n,ts,c,t,s,configuration(),ramp=0.)
    on=raw_velocity_objective(e,x,n,ts,c,t,s,configuration(),ramp=.25)
    assert torch.equal(off['total_loss'],old['total_loss'])
    assert torch.equal(on['inherited_auxiliary_total'],old['auxiliary_total'])
    assert on['inherited_auxiliary_ramp']==1 and on['raw_auxiliary_ramp']==.25

def test_boundary_and_interior_stencils_have_separate_weighted_gradients():
    c,t=fixture(count=40);s,n,ts,x=inputs(c,t)
    e=n.clone();e[...,0,2]+=.2;e.requires_grad_()
    out=raw_velocity_objective(e,x,n,ts,c,t,s,configuration())
    assert out['raw_second_boundary']>0 and out['raw_second_interior']<1e-22
    g=torch.autograd.grad(out['raw_auxiliary_total'],e)[0]
    assert g[...,2:4].abs().sum()>0 and g[...,:2].abs().sum()==0
    e=n.clone();e[...,10,3]+=.2;e.requires_grad_()
    out=raw_velocity_objective(e,x,n,ts,c,t,s,configuration())
    assert out['raw_second_boundary']<1e-22 and out['raw_second_interior']>0

def test_mixed_timestep_uses_full_batch_and_separate_scene_means():
    c,t=fixture(batch=2,count=40);s,n,ts,x=inputs(c,t)
    ts[1]=800;x=s.add_noise(t['future']/t['future'].new_tensor([50,50,20,20,1,1]),n,ts)
    e=n.clone();e[...,10,2]+=.2
    out=raw_velocity_objective(e,x,n,ts,c,t,s,configuration())
    one=raw_velocity_objective(e[:1],x[:1],n[:1],ts[:1],
        {k:v[:1] for k,v in c.items()},{k:v[:1] for k,v in t.items()},s,configuration())
    torch.testing.assert_close(out['raw_auxiliary_total'],one['raw_auxiliary_total']/2)

def test_added_loss_masks_nan_padding_before_derivatives():
    c,t=fixture(count=40);s,n,ts,x=inputs(c,t)
    c['agent_mask'][:,1]=False
    e=n.clone();e[:,0,10,2]+=.2;e.requires_grad_()
    first=raw_velocity_objective(e,x,n,ts,c,t,s,configuration())
    c['history'][:,1]=float('nan');t['future'][:,1]=float('nan')
    e2=e.detach().clone();e2[:,1]=float('nan');e2.requires_grad_()
    second=raw_velocity_objective(e2,x,n,ts,c,t,s,configuration())
    assert torch.equal(first['raw_auxiliary_total'],second['raw_auxiliary_total'])
    g=torch.autograd.grad(second['raw_auxiliary_total'],e2)[0]
    assert torch.isfinite(g).all() and g[:,1].abs().sum()==0

@pytest.mark.parametrize('change',[{'interior_weight':0},{'boundary_weight':True},
    {'second_difference_unit_mps':float('inf')},{'boundary_frames':5},{'unified':{'imitation_weight':-1}},
    {'future_exit_condition':True}])
def test_invalid_or_silent_loss_variants_rejected(change):
    cfg=configuration().to_dict();cfg.update(change)
    with pytest.raises(ValueError):resolve_raw_velocity_loss(cfg)
