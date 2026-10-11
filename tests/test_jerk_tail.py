from dataclasses import replace
import numpy as np
import pytest
import torch
from test_physical_losses import fixture,inputs
from sumodiff.diffusion.jerk_tail import JerkTailConfig,physical_jerk,jerk_tail_scenes,fresh_objective
from sumodiff.diffusion.raw_velocity_losses import RawVelocityLossConfig
from sumodiff.diffusion.unified_losses import UnifiedLossConfig
from sumodiff.diffusion.quality_audit import physical_derivatives

def states(c,t):
    p=t['future'].clone();p[...,:2]+=c['initial_positions'][...,None,:]
    return p

def test_matches_independent_evaluation_all_40_stencils():
    c,t=fixture(batch=2,count=40);p=states(c,t)
    p[...,0,0]+=.02;p[...,18,1]+=.03;p[...,-1,0]+=.04
    j,mask=physical_jerk(p,c,t['future_mask'])
    for s in range(2):
        cn={k:v[s].numpy() for k,v in c.items()}
        expected=physical_derivatives(cn,p[s].numpy())[2]
        np.testing.assert_allclose(j[s].numpy(),expected,atol=1e-10,rtol=1e-10)
    assert j.shape==(2,2,40,2) and mask.all()
    assert j[...,0,:].norm()>0 and j[...,18,:].norm()>0 and j[...,-1,:].norm()>0

def test_per_scene_topk_includes_valid_zeros():
    c,t=fixture(batch=2,count=40);p=states(c,t)
    p[0,0,-1,0]+=.03
    result,mean,top,k=jerk_tail_scenes(p,c,t['future_mask'],JerkTailConfig())
    assert k==[8,8]  # 80 valid vehicle-frames per scene; not nonzero exceedances.
    torch.testing.assert_close(top[0],mean[0]*10)
    assert result[1]==0 and result[0]>mean[0]

def test_nan_padding_has_zero_gradient_and_zero_jerk_is_finite():
    c,t=fixture(count=40);c['agent_mask'][:,1]=False
    p=states(c,t);c['history'][:,1]=float('nan');c['initial_positions'][:,1]=float('nan')
    p[:,1]=float('nan');p.requires_grad_()
    loss=jerk_tail_scenes(p,c,t['future_mask'],JerkTailConfig())[0].sum()
    loss.backward()
    assert torch.isfinite(p.grad).all() and not p.grad[:,1].any()

def test_complete_trajectory_required_and_tail_parameters_checked():
    c,t=fixture(count=40);t['future_mask'][0,0,-1]=False
    with pytest.raises(ValueError,match='complete'):physical_jerk(states(c,t),c,t['future_mask'])
    for kwargs in ({'top_fraction':2},{'threshold_mps3':True},{'huber_beta':0}):
        with pytest.raises(ValueError):JerkTailConfig(**kwargs)

def test_all_auxiliary_components_share_one_ramp_and_mixed_t_uses_full_batch():
    c,t=fixture(batch=2,count=40);s,n,ts,x=inputs(c,t)
    cfg=RawVelocityLossConfig(unified=UnifiedLossConfig(road_weight=0))
    p=(n+.05).requires_grad_()
    on=fresh_objective(p,x,n,ts,c,t,s,cfg,JerkTailConfig(),1e-5,ramp=1.)
    half=fresh_objective(p,x,n,ts,c,t,s,cfg,JerkTailConfig(),1e-5,ramp=.5)
    off=fresh_objective(p,x,n,ts,c,t,s,cfg,JerkTailConfig(),1e-5,ramp=0.)
    assert torch.equal(off['total_loss'],off['noise_mse'])
    for k in ('weighted_imitation','weighted_temporal','weighted_raw_boundary','weighted_raw_interior','weighted_tail','auxiliary_total'):
        torch.testing.assert_close(half[k],on[k]/2)
    ts[1]=800;x=s.add_noise(t['future']/t['future'].new_tensor([50,50,20,20,1,1]),n,ts)
    both=fresh_objective(p,x,n,ts,c,t,s,cfg,JerkTailConfig(),1e-5)
    one=fresh_objective(p[:1],x[:1],n[:1],ts[:1],{k:v[:1] for k,v in c.items()},
        {k:v[:1] for k,v in t.items()},s,cfg,JerkTailConfig(),1e-5)
    torch.testing.assert_close(both['tail'],one['tail']/2)
    both['total_loss'].backward();assert torch.isfinite(p.grad).all()
