import pytest
import torch
from test_physical_losses import fixture
from sumodiff.diffusion.process import NoiseSchedule
from sumodiff.diffusion.velocity_validation import (
    config,velocity_residual,velocity_probe_terms,derivative_vectors,gradient_vector,grad_stats)

def test_accelerating_label_has_zero_probe_without_forcing_zero_acceleration():
    c,t=fixture(batch=2,count=40)
    raw=t['future'].clone().requires_grad_()
    terms=velocity_probe_terms(raw,c,t,torch.tensor([50,200]),NoiseSchedule())
    assert torch.diff(raw[...,2],dim=-1).abs().max()>0
    assert all(float(v)==0 for v in terms.values())

def test_first_velocity_perturbation_has_boundary_gradient_and_no_interior_penalty():
    c,t=fixture(count=40)
    raw=t['future'].clone()
    raw[...,0,2]+=.4
    raw.requires_grad_()
    terms=velocity_probe_terms(raw,c,t,torch.tensor([200]),NoiseSchedule())
    assert terms['raw_second_boundary']>0 and terms['raw_second_interior']==0
    g=torch.autograd.grad(terms['raw_second'],raw)[0]
    assert torch.isfinite(g).all() and g[...,2:4].abs().sum()>0
    assert g[...,:2].abs().sum()==0 and g[...,4:6].abs().sum()==0

def test_inactive_nan_padding_has_zero_effect_and_zero_gradient():
    c,t=fixture(count=40)
    c['agent_mask'][:,1]=False
    raw=t['future'].clone();raw[:,0,:,2]+=.2;raw.requires_grad_()
    reference=velocity_probe_terms(raw,c,t,torch.tensor([200]),NoiseSchedule())['raw_second']
    c['history'][:,1]=float('nan');t['future'][:,1]=float('nan')
    bad=raw.detach().clone();bad[:,1]=float('nan');bad.requires_grad_()
    loss=velocity_probe_terms(bad,c,t,torch.tensor([200]),NoiseSchedule())['raw_second']
    assert torch.equal(loss,reference)
    g=torch.autograd.grad(loss,bad)[0]
    assert torch.isfinite(g).all() and g[:,1].abs().sum()==0

def test_full_batch_denominator_keeps_high_noise_scene_in_denominator():
    c,t=fixture(batch=2,count=40)
    raw=t['future'].clone();raw[...,0,2]+=.2
    schedule=NoiseSchedule()
    times=torch.tensor([200,800])
    mixed=velocity_probe_terms(raw,c,t,times,schedule)['raw_second']
    one=velocity_probe_terms(raw[:1],{k:v[:1] for k,v in c.items()},
        {k:v[:1] for k,v in t.items()},times[:1],schedule)['raw_second']
    assert torch.equal(mixed,one/2)

def test_constant_future_velocity_bias_is_penalized_at_boundary_only():
    c,t=fixture(count=40)
    raw=t['future'].clone();raw[...,2]+=.3
    residual,_=velocity_residual(raw,c,t)
    assert residual[...,:2,:].abs().max()>0
    assert residual[...,2:,:].abs().max()<1e-14

def test_missing_history_and_future_are_rejected():
    c,t=fixture(count=40)
    t['future_mask'][...,10]=False
    with pytest.raises(ValueError,match='complete'):velocity_residual(t['future'],c,t)
    t['future_mask'][...,10]=True;c['history_mask'][...,-2]=False
    with pytest.raises(ValueError,match='complete'):velocity_residual(t['future'],c,t)

def test_position_derivatives_recover_constant_velocity():
    c,t=fixture(count=40)
    k=torch.arange(-20,1,dtype=torch.float64)
    c['history'][...,0]=k
    c['history'][...,1]=c['initial_positions'][...,1,None]
    states=t['future'].clone()
    states[...,0]=torch.arange(1,41,dtype=torch.float64)
    states[...,1]=c['initial_positions'][...,1,None]
    v,a,j=derivative_vectors(c,states)
    assert torch.allclose(v[...,0],torch.full_like(v[...,0],10.))
    assert a.abs().max()<1e-10 and j.abs().max()<1e-9

def test_gradient_statistics_do_not_populate_parameter_grad_buffers():
    model=torch.nn.Linear(2,1,dtype=torch.float64)
    parameters=list(model.parameters())
    loss=model(torch.ones(3,2,dtype=torch.float64)).square().mean()
    reference=gradient_vector(loss,parameters)
    stats=grad_stats(loss,parameters,reference)
    assert stats['norm_ratio_to_epsilon']==pytest.approx(1)
    assert stats['cosine_with_epsilon']==pytest.approx(1)
    assert all(p.grad is None for p in parameters)

@pytest.mark.parametrize('value',[
    {'timesteps':[301]}, {'timesteps':[True]}, {'timesteps':[0,0]},
    {'gradient_batches':17}, {'validation_windows':6},
    {'second_difference_unit_mps':0}, {'future_exit_condition':True}])
def test_unbounded_or_unsupported_diagnostic_settings_rejected(value):
    with pytest.raises(ValueError):config(value)
