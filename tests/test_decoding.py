import numpy as np
import pytest
import torch
from sumodiff.decoding import DecoderConfig, decode_future, observed_heading


def constant(t=40, dtype=torch.float64):
    raw = torch.zeros(1, t, 6, dtype=dtype)
    raw[..., 0] = torch.arange(1, t+1, dtype=dtype)
    raw[..., 2] = 10
    raw[..., 5] = 1
    return raw, torch.tensor([[13., -4.]], dtype=dtype), torch.tensor([[0., 1.]], dtype=dtype), torch.tensor([True])


def test_consistent_constant_stop_and_turn_unchanged():
    raw, start, heading, mask = constant()
    for mode in ('constant', 'stop', 'turn'):
        candidate = raw.clone()
        if mode == 'stop':
            candidate[..., :4] = 0
        elif mode == 'turn':
            angles = torch.arange(1, 41, dtype=raw.dtype) * .02
            p = torch.stack((10*torch.sin(angles), 10*(1-torch.cos(angles))), dim=-1)
            candidate[0, :, :2] = p
            candidate[0, :, 2:4] = torch.diff(torch.cat((torch.zeros(1, 2, dtype=raw.dtype), p)), dim=0) / .1
            candidate[0, :, 4:6] = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
        decoded = decode_future(candidate, start, heading, mask)
        torch.testing.assert_close(decoded.states[..., :2], candidate[..., :2]+start[:, None], atol=1e-11, rtol=0)
        torch.testing.assert_close(decoded.states[..., 2:], candidate[..., 2:], atol=1e-10, rtol=0)
        torch.testing.assert_close(decoded.positions_with_start[:, 0], start)
        assert not decoded.heading_fallback_mask.any()


def test_inconsistent_state_reduces_defined_objective_and_respects_anchor():
    raw, start, heading, mask = constant(t=5)
    raw[..., 2] = 0
    decoded = decode_future(raw, start, heading, mask)
    displacement = decoded.states[..., :2] - start[:, None]
    before = ((raw[..., 2:4] - torch.diff(torch.cat((torch.zeros(1, 1, 2), raw[..., :2]), dim=1), dim=1)/.1)**2).sum()
    after = ((displacement-raw[..., :2])**2).sum()+((decoded.states[..., 2:4]-raw[..., 2:4])**2).sum()
    assert after < before
    torch.testing.assert_close(decoded.positions_with_start[:, 0], start)
    torch.testing.assert_close(torch.diff(decoded.positions_with_start, dim=1)/.1, decoded.states[..., 2:4])


def test_heading_fallback_carries_last_valid_and_counts():
    raw, start, heading, mask = constant(t=4)
    raw[0, 0, 4:6] = 0
    raw[0, 1, 4:6] = torch.tensor([2., 0.])
    raw[0, 2, 4:6] = 1e-12
    raw[0, 3, 4:6] = 0
    decoded = decode_future(raw, start, heading, mask)
    torch.testing.assert_close(decoded.states[0, :, 4:6], torch.tensor([[0.,1.],[1.,0.],[1.,0.],[1.,0.]], dtype=torch.float64))
    assert decoded.heading_fallback_mask.sum() == 3
    assert decoded.heading_angle_valid.sum() == 1
    with pytest.raises(ValueError, match='Observed initial heading'):
        decode_future(raw, start, heading*0, mask)


def test_gradcheck_positions_velocity_heading_and_padding():
    raw, start, heading, mask = constant(t=3)
    raw[..., 4] = .2
    raw.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: decode_future(x,start,heading,mask).states, (raw,), atol=2e-5, rtol=1e-3)
    padded = torch.cat((raw.detach(), torch.full_like(raw, float('nan')))).requires_grad_(True)
    padded_start = torch.cat((start, torch.full_like(start, float('nan'))))
    padded_heading = torch.cat((heading, torch.full_like(heading, float('nan'))))
    decoded = decode_future(padded,padded_start,padded_heading,torch.tensor([True,False]))
    decoded.states.sum().backward()
    assert torch.isfinite(padded.grad).all() and not padded.grad[1].any()
    assert not decoded.states[1].any()
    torch.testing.assert_close(decoded.states[0], decode_future(raw.detach(),start,heading,mask).states[0])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_no_half_precision_linear_solve(dtype):
    raw, start, heading, mask = constant(dtype=dtype)
    decoded = decode_future(raw,start,heading,mask)
    assert decoded.states.dtype == torch.float64 and torch.isfinite(decoded.states).all()


def test_missing_t0_heading_and_bad_input_are_explicit():
    raw,start,heading,mask = constant()
    history = torch.zeros(1,21,6)
    history[...,5] = 1
    history_mask = torch.ones(1,21,dtype=torch.bool)
    torch.testing.assert_close(observed_heading(history,history_mask,mask), history[:,-1,4:6])
    history_mask[:,-1] = False
    with pytest.raises(ValueError, match='Missing t0 heading'):
        observed_heading(history,history_mask,mask)
    raw[...,0] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        decode_future(raw,start,heading,mask)
    with pytest.raises(ValueError, match='never half'):
        DecoderConfig(solve_dtype='float16')
