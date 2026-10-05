import pytest
import torch
from sumodiff.diffusion import DiffusionConfig,NoiseSchedule,normalize,denormalize,masked_noise_mse


def test_physical_scales_do_not_clip_or_change_heading():
    x=torch.tensor([[[[150.,-100.,40.,-30.,.6,.8]]]],dtype=torch.float64)
    c=DiffusionConfig();z=normalize(x,c)
    torch.testing.assert_close(z,torch.tensor([[[[3.,-2.,2.,-1.5,.6,.8]]]],dtype=x.dtype))
    torch.testing.assert_close(denormalize(z,c),x)


def test_scene_equal_weight_and_masked_nan_gradient():
    p=torch.zeros(2,2,3,6,requires_grad=True);noise=torch.ones_like(p)
    a=torch.tensor([[1,0],[1,1]],dtype=torch.bool);f=torch.ones(2,2,3,dtype=torch.bool);f[0,0,1:]=False
    noise[0]=2;noise[0,1]=float('nan')
    loss=masked_noise_mse(p,noise,a,f)
    assert loss.item()==2.5
    loss.backward();assert not p.grad[0,1].any() and not p.grad[0,0,1:].any()
    with pytest.raises(ValueError,match='no supervised'): masked_noise_mse(p,noise,a,torch.zeros_like(f))


def test_ddim_oracle_recovers_clean_without_clipping():
    c=DiffusionConfig();s=NoiseSchedule(c);x=torch.randn(2,3,40,6,dtype=torch.float64)*4
    e=torch.randn_like(x);sequence=s.timesteps(20)
    noisy=s.add_noise(x,e,torch.full((2,),sequence[0],dtype=torch.long))
    for i,current in enumerate(sequence):
        t=torch.full((2,),current,dtype=torch.long)
        estimate=s.clean_from_noise(noisy,e,t)
        torch.testing.assert_close(estimate,x,atol=1e-11,rtol=1e-11)
        previous=sequence[i+1] if i+1<len(sequence) else -1
        noisy=s.transition(noisy,estimate,e,current,previous)
        if previous>=0: torch.testing.assert_close(noisy,s.add_noise(x,e,torch.full((2,),previous,dtype=torch.long)),atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(noisy,x,atol=1e-11,rtol=1e-11)
    assert sequence[0]==999 and sequence[-1]==0 and len(set(sequence))==20


def test_updated_clean_state_requires_updated_noise_and_stochastic_variance():
    s=NoiseSchedule();x=torch.randn(1,2,40,6,dtype=torch.float64);e=torch.randn_like(x);t=torch.tensor([500])
    noisy=s.add_noise(x,e,t);changed=x+.2;updated=s.noise_from_clean(noisy,changed,t)
    torch.testing.assert_close(s.add_noise(changed,updated,t),noisy)
    assert not torch.allclose(s.transition(noisy,changed,updated,500,400),s.transition(noisy,changed,e,500,400))
    z=torch.ones_like(x);out=s.transition(noisy,x,e,500,400,1.,z);a=s.alpha_bar[500];ap=s.alpha_bar[400]
    sigma=((1-ap)/(1-a)*(1-a/ap)).sqrt()
    torch.testing.assert_close(out,ap.sqrt()*x+(1-ap-sigma**2).sqrt()*e+sigma*z)
    torch.testing.assert_close(s.transition(noisy,x,e,500,-1,1.),x)


@pytest.mark.parametrize('key',['cfg','partial_diffusion','rolling_generation','additional_losses_enabled'])
def test_unsupported_diffusion_options_raise(key):
    with pytest.raises(NotImplementedError): DiffusionConfig(**{key:True})


def test_invalid_timesteps_are_explicit():
    s=NoiseSchedule()
    for steps in (0,1001,True):
        with pytest.raises(ValueError): s.timesteps(steps)
    with pytest.raises(ValueError): s.transition(torch.ones(1),torch.ones(1),torch.ones(1),20,20)
