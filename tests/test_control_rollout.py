import math
import pytest
import torch
from sumodiff.decoding.controls import (
    ControlConfig, initial_control_state, inverse_label_controls, rollout_controls)


def observed(config=None, *, yaw=0., speed=8., length=4.):
    h = torch.zeros(2, 21, 6, dtype=torch.float64)
    t = torch.arange(-20, 1, dtype=h.dtype) * .1
    h[..., 0] = speed * t * math.cos(yaw)
    h[..., 1] = speed * t * math.sin(yaw)
    h[..., 2] = speed * math.cos(yaw)
    h[..., 3] = speed * math.sin(yaw)
    h[..., 4], h[..., 5] = math.sin(yaw), math.cos(yaw)
    return h, torch.ones(2, 21, dtype=torch.bool), h[:, -1, :2].clone(), torch.full((2,), length, dtype=h.dtype), torch.ones(2, dtype=torch.bool)


def initialize(config=None, **kwargs):
    return initial_control_state(*observed(config, **kwargs), config)


@pytest.mark.parametrize('point', ['center', 'front'])
@pytest.mark.parametrize('basis', ['midpoint', 'endpoint'])
@pytest.mark.parametrize('lateral', [False, True])
def test_roundtrip_and_center_velocity(point, basis, lateral):
    config = ControlConfig(reference_point=point, integration_heading=basis, lateral_velocity=lateral)
    init = initialize(config, yaw=3.12)
    controls = torch.zeros(2, 40, config.channels, dtype=torch.float64)
    controls[..., 0] = .3
    controls[..., 1] = .25
    if lateral:
        controls[..., 2] = .15
    out = rollout_controls(controls, init)
    inverse = inverse_label_controls(out.states, torch.ones(2, 40, dtype=torch.bool), init)
    torch.testing.assert_close(inverse.controls, controls, atol=1e-11, rtol=0)
    replay = rollout_controls(inverse.controls, init)
    torch.testing.assert_close(replay.states, out.states, atol=1e-11, rtol=0)
    torch.testing.assert_close(out.positions_with_start[:, 0], init.center_position)
    torch.testing.assert_close(torch.diff(out.positions_with_start, dim=-2)/.1, out.states[..., 2:4])
    assert out.states.dtype == torch.float64


def test_independent_constant_and_known_turn_reference_offset():
    init = initialize()
    out = rollout_controls(torch.zeros(2, 5, 2), init)
    expected = torch.arange(1, 6, dtype=torch.float64) * .8
    torch.testing.assert_close(out.states[..., 0], expected.expand(2, 5))
    assert not out.states[..., 1].any()
    config = ControlConfig(reference_point='front')
    init = initialize(config)
    ctrl = torch.zeros(2, 5, 2, dtype=torch.float64)
    ctrl[..., 1] = 1.
    out = rollout_controls(ctrl, init)
    angles = torch.arange(1, 6, dtype=torch.float64) * .1
    mid = angles - .05
    front = torch.tensor([2., 0.]) + .8 * torch.cumsum(torch.stack((torch.cos(mid), torch.sin(mid)), -1), 0)
    expected = front - 2 * torch.stack((torch.cos(angles), torch.sin(angles)), -1)
    torch.testing.assert_close(out.states[0, :, :2], expected)
    torch.testing.assert_close(out.states[0, :, 4], torch.sin(angles))


def test_lateral_residual_is_explicit_not_silently_repaired():
    init3 = initialize(ControlConfig(lateral_velocity=True))
    ctrl = torch.zeros(2, 8, 3, dtype=torch.float64)
    ctrl[..., 2] = .4
    labels = rollout_controls(ctrl, init3).states
    init2 = initialize()
    inverse2 = inverse_label_controls(labels, torch.ones(2, 8, dtype=torch.bool), init2)
    torch.testing.assert_close(inverse2.lateral_interval_velocity, torch.full((2, 8), .4, dtype=torch.float64))
    out2 = rollout_controls(inverse2.controls, init2)
    assert torch.linalg.vector_norm(out2.states[..., :2]-labels[..., :2], dim=-1).max().item() == pytest.approx(.32)
    inverse3 = inverse_label_controls(labels, torch.ones(2, 8, dtype=torch.bool), init3)
    torch.testing.assert_close(rollout_controls(inverse3.controls, init3).states, labels)


def test_padding_nan_and_gradient_are_isolated():
    h, hm, p0, lengths, mask = observed()
    mask[1] = False
    h[1], p0[1], lengths[1] = float('nan'), float('nan'), float('nan')
    init = initial_control_state(h, hm, p0, lengths, mask)
    ctrl = torch.zeros(2, 4, 2, dtype=torch.float64)
    ctrl[1] = float('nan')
    ctrl.requires_grad_(True)
    out = rollout_controls(ctrl, init)
    out.states.square().sum().backward()
    assert torch.isfinite(ctrl.grad).all() and not ctrl.grad[1].any()
    assert not out.states[1].any()
    inv = inverse_label_controls(out.states, mask[:, None].expand(2, 4), init)
    assert not inv.controls[1].any()


@pytest.mark.parametrize('lateral', [False, True])
def test_gradcheck(lateral):
    init = initialize(ControlConfig(lateral_velocity=lateral, reference_point='front'))
    ctrl = torch.full((2, 3, init.config.channels), .12, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda c: rollout_controls(c, init).states, (ctrl,), atol=1e-5, rtol=1e-3)


def test_common_frame_and_agent_permutation():
    config = ControlConfig(lateral_velocity=True)
    h, hm, p0, length, mask = observed()
    h[1, :, :2] += torch.tensor([2., 3.])
    p0[1] += torch.tensor([2., 3.])
    ctrl = torch.randn(2, 5, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(8))*.1
    init = initial_control_state(h, hm, p0, length, mask, config)
    original = rollout_controls(ctrl, init).states
    perm = torch.tensor([1, 0])
    permuted = rollout_controls(ctrl[perm], initial_control_state(h[perm], hm[perm], p0[perm], length[perm], mask[perm], config)).states
    torch.testing.assert_close(permuted, original[perm])
    angle = .7
    rot = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=h.dtype)
    shift = torch.tensor([7., -4.])
    new_h = h.clone()
    new_h[..., :2] = h[..., :2] @ rot.T + shift
    new_h[..., 2:4] = h[..., 2:4] @ rot.T
    yaw = torch.atan2(h[..., 4], h[..., 5]) + angle
    new_h[..., 4:6] = torch.stack((torch.sin(yaw), torch.cos(yaw)), -1)
    transformed = rollout_controls(ctrl, initial_control_state(new_h, hm, p0@rot.T+shift, length, mask, config)).states
    torch.testing.assert_close(transformed[..., :2], original[..., :2]@rot.T+shift)
    torch.testing.assert_close(transformed[..., 2:4], original[..., 2:4]@rot.T)
    torch.testing.assert_close(transformed[..., 4:6], torch.stack((torch.sin(torch.atan2(original[..., 4], original[..., 5])+angle), torch.cos(torch.atan2(original[..., 4], original[..., 5])+angle)), -1))


def test_history_only_initialization_and_label_velocity_audit():
    h, hm, p0, length, mask = observed()
    h[..., 2:4] = 999.  # Initial interval speed must use past positions.
    init = initial_control_state(h, hm, p0, length, mask)
    torch.testing.assert_close(init.forward_interval_speed, torch.full((2,), 8., dtype=h.dtype))
    labels = rollout_controls(torch.zeros(2, 5, 2), init).states
    altered = labels.clone()
    altered[..., 2:4] += 7
    inverse = inverse_label_controls(altered, torch.ones(2, 5, dtype=torch.bool), init)
    torch.testing.assert_close(inverse.controls, torch.zeros_like(inverse.controls), atol=1e-11, rtol=0)
    torch.testing.assert_close(inverse.label_velocity_residual, torch.full((2, 5, 2), 7., dtype=h.dtype))


def test_invalid_inputs_no_future_fallback():
    h, hm, p0, length, mask = observed()
    hm[0, -2] = False
    with pytest.raises(ValueError, match='Both final observed'):
        initial_control_state(h, hm, p0, length, mask)
    hm[0, -2] = True
    h[0, -1, 4:6] = 0
    with pytest.raises(ValueError, match='heading must be nonzero'):
        initial_control_state(h, hm, p0, length, mask)
    init = initialize()
    with pytest.raises(ValueError, match='complete active'):
        inverse_label_controls(torch.zeros(2, 4, 6), torch.zeros(2, 4, dtype=torch.bool), init)
    with pytest.raises(ValueError, match='channel count'):
        rollout_controls(torch.zeros(2, 4, 3), init)
    ctrl = torch.zeros(2, 4, 2)
    ctrl[..., 1] = 40
    with pytest.raises(ValueError, match='below pi'):
        rollout_controls(ctrl, init)
    ctrl[..., 1] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        rollout_controls(ctrl, init)
    with pytest.raises(ValueError, match='float32 or float64'):
        ControlConfig(compute_dtype='float16')
    with pytest.raises(NotImplementedError):
        ControlConfig(dt=.2)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_low_precision_promotes(dtype):
    init = initialize()
    result = rollout_controls(torch.zeros(2, 5, 2, dtype=dtype), init)
    assert result.states.dtype == torch.float64 and torch.isfinite(result.states).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_autocast_keeps_computation_precision_and_gradients():
    args = tuple(x.cuda() for x in observed())
    init = initial_control_state(*args, ControlConfig(lateral_velocity=True))
    ctrl = torch.full((2, 40, 3), .02, device='cuda', requires_grad=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        states = rollout_controls(ctrl, init).states
    assert states.dtype == torch.float64
    states.square().mean().backward()
    assert torch.isfinite(ctrl.grad).all() and ctrl.grad.abs().sum() > 0
