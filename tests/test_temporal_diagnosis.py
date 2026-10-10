import pytest
import torch
from test_physical_losses import fixture, inputs
from sumodiff.diffusion.temporal_diagnosis import resolve_probe, residual_curves, SPANS, LOSS_FIELDS
from sumodiff.diffusion.decoded_temporal_losses import DecodedTemporalLossConfig, decoded_temporal_objective
from sumodiff.diffusion.raw_velocity_losses import RawVelocityLossConfig
from sumodiff.diffusion.unified_losses import UnifiedLossConfig

def test_velocity_derivative_units_and_endpoint_alignment():
    c,t=fixture(agents=1,count=40)
    q=t['future'].clone();p=q.clone()
    p[...,39,2]+=2.
    result=residual_curves(p,p,q,c,t)
    assert result['decoded_acceleration_error'][0,0,39]==pytest.approx(20.)
    assert result['decoded_velocity_jerk_error'][0,0,39]==pytest.approx(200.)
    assert result['raw_velocity_jerk_error'][0,0,39]==pytest.approx(200.)
    assert result['decoded_velocity_jerk_error'][...,:39].count_nonzero()==0

def test_boundary_is_observed_history_not_wrapped_future():
    c,t=fixture(agents=1,count=40)
    q=t['future'].clone();p=q.clone();p[...,0,2]+=1.
    result=residual_curves(p,p,q,c,t)
    assert result['decoded_velocity_jerk_error'][0,0,:3].tolist()==pytest.approx([100.,200.,100.])
    assert result['decoded_acceleration_error'][0,0,:2].tolist()==pytest.approx([10.,10.])
    assert result['decoded_velocity_jerk_error'][0,0,36:].count_nonzero()==0

def test_padding_nan_does_not_change_valid_curves():
    c,t=fixture(agents=2,count=40)
    c['agent_mask'][:,1]=False;c['history'][:,1]=float('nan')
    t['future'][:,1]=float('nan');t['future_mask'][:,1]=False
    q=t['future'].clone();p=q.clone();p[:,0,39,2]+=2.
    result=residual_curves(p,p,q,c,t)
    assert all(torch.isfinite(v).all() for v in result.values())
    assert all(v[:,1].count_nonzero()==0 for v in result.values())

def test_scene_losses_reproduce_actual_batch_objective():
    c,t=fixture(batch=3,count=40)
    s,n,ts,x=inputs(c,t)
    cfg=DecodedTemporalLossConfig(inherited=RawVelocityLossConfig(unified=UnifiedLossConfig(road_weight=0.)))
    p=n+.01*torch.randn(n.shape,dtype=n.dtype,generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        full=decoded_temporal_objective(p,x,n,ts,c,t,s,cfg)
        parts=[decoded_temporal_objective(p[i:i+1],x[i:i+1],n[i:i+1],ts[i:i+1],
            {k:v[i:i+1] for k,v in c.items()},{k:v[i:i+1] for k,v in t.items()},s,cfg) for i in range(3)]
    for k in LOSS_FIELDS:
        torch.testing.assert_close(full[k],torch.stack([part[k] for part in parts]).mean(),rtol=1e-10,atol=1e-13)
    assert all(not full[k].requires_grad for k in LOSS_FIELDS)

def test_periods_partition_and_interior_overlap_are_explicit():
    indexes=list(range(40))
    assert indexes[SPANS['boundary']]+indexes[SPANS['middle']]+indexes[SPANS['endpoint']]==indexes
    assert indexes[SPANS['interior']]==indexes[SPANS['middle']]+indexes[SPANS['endpoint']]

@pytest.mark.parametrize('bad',[{'timesteps':[0,0]},{'timesteps':[301]},{'timesteps':[True]},
    {'timesteps':[0,50,100,200,300]},{'seed':-1},{'device':'automatic'},{'train_limit':12}])
def test_unsupported_or_unbounded_protocol_rejected(bad):
    with pytest.raises(ValueError):resolve_probe(bad)
