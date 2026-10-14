import numpy as np
import pytest
from sumodiff.decoding.control_scale_review import body_motion


def test_body_frame_straight_stop_and_lateral_velocity():
    control=np.zeros((2,40,3))
    result=body_motion(control,np.array([8.,0.]))
    assert not result['heading_span_rad'].any() and not result['lateral_excursion_m'].any()
    control[0,:,2]=.25
    result=body_motion(control,np.array([8.,0.]))
    np.testing.assert_allclose(result['lateral_excursion_m'],[1.,0.],atol=1e-12)


def test_body_frame_known_turn_uses_midpoint_basis_and_start():
    controls=np.zeros((1,10,3));controls[...,1]=1.
    result=body_motion(controls,np.array([2.]))
    angles=np.arange(1,11)*.1
    expected=.2*np.sin(angles-.05).sum()
    np.testing.assert_allclose(result['heading_span_rad'],[1.],atol=1e-12)
    np.testing.assert_allclose(result['lateral_excursion_m'],[expected],atol=1e-12)
    with pytest.raises(ValueError):body_motion(controls,np.array([np.nan]))


def test_velocity_reference_is_observed_interval_speed():
    controls=np.zeros((1,4,3));controls[...,0]=-1.;controls[...,1]=.5
    result=body_motion(controls,np.array([2.]))
    speeds=2.-np.arange(1,5)*.1
    expected=.1*(speeds*np.sin(np.arange(1,5)*.05-.025)).sum()
    np.testing.assert_allclose(result['lateral_excursion_m'],[expected],atol=1e-12)
