import pytest
import torch
from sumodiff.diffusion.diagnostics import channel_mse,noise_for_ids,diagnostic_config
from sumodiff.diffusion import NoiseSchedule,DiffusionConfig,normalize,denormalize


def test_channels_have_equal_scene_weight_and_ignore_padding_nan():
    p=torch.zeros(2,2,3,6);q=torch.ones_like(p);q[0]=2
    mask=torch.ones(2,2,3,dtype=torch.bool);mask[0,1]=False;mask[0,0,1:]=False;q[~mask]=float('nan')
    mse=channel_mse(p,q,mask)
    torch.testing.assert_close(mse[0],torch.full((6,),4.,dtype=torch.float64));torch.testing.assert_close(mse[1],torch.ones(6,dtype=torch.float64))
    assert mse.mean().item()==2.5
    with pytest.raises(ValueError): channel_mse(p,q,torch.zeros_like(mask))


def test_noise_stable_under_batch_order_and_different_repeats():
    first=noise_for_ids(['a','b'],(2,40,6),20,0)
    reversed_noise=noise_for_ids(['b','a'],(2,40,6),20,0)
    assert torch.equal(first,reversed_noise.flip(0))
    assert torch.equal(first[0],noise_for_ids(['a'],(2,40,6),20,0)[0])
    assert not torch.equal(first,noise_for_ids(['a','b'],(2,40,6),20,1))


def test_known_epsilon_error_has_expected_state_channel_amplification():
    config=DiffusionConfig();schedule=NoiseSchedule(config);physical=torch.randn(2,2,40,6,dtype=torch.float64)
    clean=normalize(physical,config);noise=torch.randn_like(clean);t=torch.tensor([999,500]);bias=torch.tensor([.01,.02,.03,.04,.05,.06],dtype=torch.float64)
    noisy=schedule.add_noise(clean,noise,t);recovered=denormalize(schedule.clean_from_noise(noisy,noise+bias,t),config)
    a=schedule.alpha_bar[t][:,None,None,None];units=torch.tensor([50.,50.,20.,20.,1.,1.],dtype=torch.float64)
    expected=physical-((1-a)/a).sqrt()*bias*units
    torch.testing.assert_close(recovered,expected,atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('value',[{'timesteps':[1000]},{'sampling_steps':[1000]},{'noise_repeats':10},{'free_sampling':{'limit':None}},{'future_mask':True}])
def test_unbounded_or_unknown_diagnostics_rejected(value):
    with pytest.raises(ValueError): diagnostic_config(value)
