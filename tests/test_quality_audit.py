import numpy as np
import pytest
import torch
from sumodiff.decoding import DecoderConfig, decode_future
from sumodiff.diffusion import NoiseSchedule, DiffusionConfig, normalize
from sumodiff.diffusion.quality_audit import independent_position_solve, current_acceleration_control, hybrid_inputs, physical_derivatives, denoising_trace
from sumodiff.evaluation.motion import MotionConfig


def history(v0=10., accel=2.):
    k=np.arange(-20,1,dtype=np.float64)
    h=np.zeros((1,21,6));h[0,:,0]=k*.1*v0+k*(k+1)/2*.01*accel
    h[0,:,2]=v0+k*.1*accel;h[0,:,5]=1.
    return {'history':h,'history_mask':np.ones((1,21),bool),
            'initial_positions':np.zeros((1,2)),'agent_mask':np.ones(1,bool)}


@pytest.mark.parametrize('position_weight,velocity_weight',[(1.,1.),(.3,2.),(3.,.4)])
def test_stacked_residual_lstsq_matches_decoder_with_poisoned_padding(position_weight,velocity_weight):
    rng=np.random.default_rng(7);raw=rng.normal(size=(3,40,6));raw[2]=np.nan
    raw[:2,:,4:6]=[0.,1.]
    mask=np.array([True,True,False]);initial=np.array([[4.,5.],[-3.,7.],[np.nan,np.nan]])
    heading=np.array([[0.,1.],[0.,1.],[np.nan,np.nan]])
    config=DecoderConfig(position_weight=position_weight,velocity_weight=velocity_weight)
    reference=independent_position_solve(raw,mask,config)
    actual=decode_future(torch.tensor(raw),torch.tensor(initial),torch.tensor(heading),torch.tensor(mask),config)
    np.testing.assert_allclose(actual.states.numpy()[:2,:,:2]-initial[:2,None],reference[:2],atol=1e-10,rtol=1e-10)
    assert np.all(reference[2]==0)
    raw[0,0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite active'):independent_position_solve(raw,mask,config)


def test_current_control_uses_interval_mean_boundary_not_instantaneous_formula():
    c=history();future=current_acceleration_control(c)
    states=future.copy();states[:,:,:2]+=c['initial_positions'][:,None]
    velocity,acceleration,jerk=physical_derivatives(c,states)
    assert future[0,0,0]==pytest.approx(1.02)
    np.testing.assert_allclose(velocity,states[:,:,2:4],atol=1e-11)
    np.testing.assert_allclose(acceleration[:,:,0],2.,atol=1e-10)
    np.testing.assert_allclose(jerk,0.,atol=1e-9)
    c['history_mask'][0,-2]=False
    with pytest.raises(ValueError,match='observed positions'):current_acceleration_control(c)


def test_oracle_channel_controls_do_not_change_source_and_keep_other_channels():
    raw=np.full((2,40,6),3.);label=np.full_like(raw,9.)
    out=hybrid_inputs(raw,label)
    for name,channels in [('oracle_position',range(2)),('oracle_velocity',range(2,4)),('oracle_heading',range(4,6)),('oracle_position_velocity',range(4))]:
        for i in range(6):assert np.all(out[name][...,i]==(9. if i in channels else 3.))
    assert np.all(raw==3.) and np.all(label==9.)


def test_known_clean_oracle_replays_ddim_and_keeps_future_out_of_conditions():
    c=history(accel=0.);schedule=NoiseSchedule(DiffusionConfig())
    future=torch.tensor(current_acceleration_control(c),dtype=torch.float32)[None]
    clean=normalize(future,schedule.config)
    class Model:
        def condition_encoder(self,tensors):
            assert set(tensors)=={'agent_mask'}
            return torch.zeros(1,1,128)
        def unet(self,value,t,condition,mask):
            return schedule.noise_from_clean(value,clean,t)
    tensors={'agent_mask':torch.ones(1,1,dtype=torch.bool)}
    initial=torch.randn(1,1,40,6,generator=torch.Generator().manual_seed(20))
    result,rows,estimates,epsilon=denoising_trace(Model(),tensors,initial,schedule,c,DecoderConfig(),MotionConfig(),20)
    np.testing.assert_allclose(result,clean[0].numpy(),atol=2e-5,rtol=1e-5)
    assert len(rows)==20 and rows[-1]['timestep']==0 and estimates.shape==(20,1,40,6)
