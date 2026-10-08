from copy import deepcopy
import numpy as np
import pytest
from shapely.geometry import box,Polygon,mapping
import torch
from test_physical_losses import fixture,inputs
from sumodiff.diffusion.unified_losses import (
    UnifiedLossConfig,unified_objective,prepare_roads,outside_distance,body_points)
from sumodiff.diffusion.process import masked_noise_mse


def setup(batch=1,agents=2):
    c,t = fixture(batch,agents)
    c['attributes'] = torch.tensor([4.,1.8,1.,0.],dtype=torch.float64).expand(batch,agents,4).clone()
    cfg = UnifiedLossConfig(road_numerical_tolerance_m=0.)
    road,_ = prepare_roads(dict(drivable=mapping(box(-20,-20,40,20)),
                               route_corridors=[mapping(box(-20,-20,40,20))]*agents),cfg)
    s,n,ts,x = inputs(c,t)
    return c,t,cfg,[road]*batch,s,n,ts,x


def test_exact_noise_and_accelerating_labels_have_zero_losses():
    c,t,cfg,roads,s,n,ts,x=setup()
    assert torch.diff(t['future'][...,2],dim=-1).abs().min()>.1
    out=unified_objective(n,x,n,ts,c,t,s,cfg,roads=roads)
    for key in ('imitation','temporal','road','auxiliary_total'):
        assert out[key]<1e-20
    assert out['heading_fallback_frames']==0


def test_high_noise_and_disabled_objective_preserve_epsilon_exactly():
    c,t,cfg,roads,s,n,ts,x=setup()
    ts[:]=800;x=s.add_noise(__import__('sumodiff.diffusion',fromlist=['normalize']).normalize(t['future'],s.config),n,ts)
    p=(n+.03).requires_grad_()
    base=masked_noise_mse(p,n,c['agent_mask'],t['future_mask'])
    out=unified_objective(p,x,n,ts,c,t,s,cfg)
    assert torch.equal(base,out['total_loss']) and out['auxiliary_eligible_scenes']==0
    ts[:]=200
    out=unified_objective(p,x,n,ts,c,t,s,UnifiedLossConfig(imitation_weight=0,temporal_weight=0,road_weight=0))
    assert torch.equal(base,out['total_loss'])


def test_full_batch_denominator_and_ramp_are_preserved():
    c,t,cfg,roads,s,n,ts,x=setup(batch=2)
    ts[1]=800
    from sumodiff.diffusion.process import normalize
    x=s.add_noise(normalize(t['future'],s.config),n,ts)
    p=(n+.03).requires_grad_()
    out=unified_objective(p,x,n,ts,c,t,s,cfg,roads=roads,ramp=.4)
    one=unified_objective(p[:1],x[:1],n[:1],ts[:1],{k:v[:1] for k,v in c.items()},
                         {k:v[:1] for k,v in t.items()},s,cfg,roads=roads[:1],ramp=.4)
    assert out['auxiliary_eligible_scenes']==1
    assert torch.allclose(out['auxiliary_total'],one['auxiliary_total']/2)
    assert torch.allclose(out['weighted_imitation'],.4*cfg.imitation_weight*out['imitation'])


def test_decoder_losses_supply_position_velocity_and_heading_gradients():
    c,t,cfg,roads,s,n,ts,x=setup()
    p=n.clone();p[...,0]+=.08;p[...,1]+=.04;p[...,4]+=.07
    p.requires_grad_()
    out=unified_objective(p,x,n,ts,c,t,s,cfg,roads=roads)
    g=torch.autograd.grad(out['auxiliary_total'],p)[0]
    assert torch.isfinite(g).all()
    for subset in (slice(0,2),slice(2,4),slice(4,6)):
        assert g[...,subset].abs().sum()>0
    assert out['lateral']>out['longitudinal']


def test_heading_amplitude_is_supervised_before_normalization():
    c,t,cfg,roads,s,n,ts,x=setup()
    a=s.alpha_bar[ts].to(n.dtype)
    p=n.clone()
    # Make reconstructed raw heading 2*(0,1); decoder normalizes to (0,1).
    p[...,5]-=torch.sqrt(a/(1-a))[:,None,None]
    p.requires_grad_()
    out=unified_objective(p,x,n,ts,c,t,s,cfg,roads=roads)
    assert out['heading']>0 and out['lateral']<1e-20
    assert torch.autograd.grad(out['heading'],p)[0][...,5].abs().sum()>0


def test_temporal_supervision_includes_history_future_boundary():
    c,t,cfg,roads,s,n,ts,x=setup()
    p=n.clone();p[...,0,2]+=.4;p.requires_grad_()
    out=unified_objective(p,x,n,ts,c,t,s,cfg,roads=roads)
    assert out['first_difference']>0 and out['second_difference']>0
    c['history_mask'][...,-2]=False
    with pytest.raises(ValueError,match='complete'):
        unified_objective(p,x,n,ts,c,t,s,cfg,roads=roads)


def test_inactive_nan_padding_cannot_affect_losses_or_gradients():
    c,t,cfg,roads,s,n,ts,x=setup()
    c['agent_mask'][:,1]=False
    for k in ('history','attributes','initial_positions'):
        c[k][:,1]=float('nan')
    t['future'][:,1]=float('nan')
    p=n.clone();p[...,0]+=.03;p[:,1]=float('nan');p.requires_grad_()
    out=unified_objective(p,x,n,ts,c,t,s,cfg,roads=roads)
    g=torch.autograd.grad(out['total_loss'],p)[0]
    assert torch.isfinite(out['total_loss']) and torch.isfinite(g).all()
    assert g[:,1].abs().sum()==0


def test_road_distance_value_and_gradient_match_geometry_and_finite_difference():
    geometry=box(-2,-1,2,1)
    p=torch.tensor([[2.4,.3],[0.,0.]],dtype=torch.float64,requires_grad=True)
    d,mask=outside_distance(p,geometry)
    assert mask.tolist()==[True,False]
    assert torch.allclose(d,torch.tensor([.4,0.],dtype=d.dtype))
    g=torch.autograd.grad(d.sum(),p)[0]
    assert torch.allclose(g,torch.tensor([[1.,0.],[0.,0.]],dtype=g.dtype))
    for axis in range(2):
        step=torch.zeros_like(p);step[0,axis]=1e-6
        numerical=(outside_distance(p.detach()+step,geometry)[0][0]-outside_distance(p.detach()-step,geometry)[0][0])/2e-6
        assert abs(numerical-g[0,axis])<1e-7


def test_holes_legal_route_union_and_body_rotation_are_respected():
    from shapely.ops import unary_union
    cfg=UnifiedLossConfig(road_numerical_tolerance_m=0)
    polygon=Polygon([(-8,-8),(8,-8),(8,8),(-8,8)],holes=[[(-1,-1),(1,-1),(1,1),(-1,1)]])
    p=torch.tensor([[.2,0.],[3.,0.]],dtype=torch.float64,requires_grad=True)
    d,mask=outside_distance(p,polygon)
    assert mask.tolist()==[True,False] and abs(float(d[0])-.8)<1e-10
    road,_=prepare_roads(dict(drivable=mapping(polygon),
        route_corridors=[mapping(unary_union([box(-10,-3,10,0),box(-10,0,10,3)]))]),cfg)
    assert not outside_distance(torch.tensor([[0.,2.]]),road['routes'][0])[1].any()
    states=torch.tensor([[[[0.,0.,0.,0.,1.,0.]]]],dtype=torch.float64)
    points=body_points(states,torch.tensor([[[4.,2.]]]))
    assert torch.allclose(points[...,0].abs().max(),torch.tensor(1.,dtype=points.dtype))
    assert torch.allclose(points[...,1].abs().max(),torch.tensor(2.,dtype=points.dtype))


def test_body_road_proxy_gradient_reaches_orientation():
    from sumodiff.diffusion.unified_losses import road_proxy
    states=torch.tensor([[[[0.,.4,0.,0.,.2,np.sqrt(.96)]]]],dtype=torch.float64,requires_grad=True)
    c=dict(agent_mask=torch.tensor([[True]]),attributes=torch.tensor([[[4.,1.8,1.,1.]]]))
    cfg=UnifiedLossConfig(road_numerical_tolerance_m=0)
    road,_=prepare_roads(dict(drivable=mapping(box(-10,-1.1,10,1.1)),
                             route_corridors=[mapping(box(-10,-1.1,10,1.1))]),cfg)
    loss,_=road_proxy(states,c,[road],cfg)
    gradient=torch.autograd.grad(loss.sum(),states)[0]
    assert gradient[...,1].abs().sum()>0 and gradient[...,4:6].abs().sum()>0
    assert torch.isfinite(gradient).all()


def test_missing_maps_and_unsupported_loss_switches_are_explicit():
    c,t,cfg,roads,s,n,ts,x=setup()
    with pytest.raises(ValueError,match='vector maps'):
        unified_objective(n+.2,x,n,ts,c,t,s,cfg)
    from sumodiff.diffusion.unified_losses import resolve_unified_loss
    with pytest.raises(ValueError,match='Unknown'):
        resolve_unified_loss({'raw_consistency_weight':.01})
