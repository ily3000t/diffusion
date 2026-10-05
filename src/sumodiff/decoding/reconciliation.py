"""Differentiable position/velocity reconciliation in physical units."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class DecoderConfig:
    dt: float = .1
    position_weight: float = 1.
    velocity_weight: float = 1.
    position_unit_m: float = 1.
    velocity_unit_mps: float = 1.
    heading_epsilon: float = 1e-6
    solve_dtype: str = 'float64'

    def __post_init__(self):
        for key in ('dt', 'position_weight', 'velocity_weight', 'position_unit_m', 'velocity_unit_mps', 'heading_epsilon'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.dt != .1:
            raise NotImplementedError('Only dt=0.1 is supported by v1')
        if self.solve_dtype not in ('float32', 'float64'):
            raise ValueError('Linear solve requires float32 or float64, never half precision')


@dataclass
class DecodedTrajectory:
    raw_future: torch.Tensor
    raw_states: torch.Tensor
    states: torch.Tensor
    positions_with_start: torch.Tensor
    initial_heading: torch.Tensor
    position_correction: torch.Tensor
    velocity_correction: torch.Tensor
    heading_correction: torch.Tensor
    heading_angle_correction: torch.Tensor
    heading_angle_valid: torch.Tensor
    raw_velocity_residual: torch.Tensor
    raw_heading_norm_error: torch.Tensor
    heading_fallback_mask: torch.Tensor
    agent_mask: torch.Tensor


def _finite_active(tensor, mask, name):
    if not torch.isfinite(tensor[mask]).all():
        raise ValueError(f'Non-finite {name} for an active agent')


def observed_heading(history, history_mask, agent_mask):
    """Use t0 observed heading; never inspect labels or infer an exit time.

    If t0's full state is unavailable, the caller must supply its known current
    heading separately. Do not invent it from zero padding or future motion.
    """
    if history.shape[:-2] != agent_mask.shape or history.shape[-1] != 6 or history_mask.shape != history.shape[:-1]:
        raise ValueError('Invalid history/mask shapes')
    if not history_mask[..., -1][agent_mask].all():
        raise ValueError('Missing t0 heading: explicitly supply the currently observed heading')
    return history[..., -1, 4:6]


def decode_future(raw_future, initial_positions, initial_heading, agent_mask, config=None):
    """Reconcile a full generated horizon WITHOUT a future-label mask.

    q0=0 is fixed. Minimize alpha*||q-qhat||^2 + beta*||Dq/dt-vhat||^2,
    where alpha=position_weight/position_unit_m^2 and
    beta=velocity_weight/velocity_unit_mps^2. No smoothing/dynamics/road loss.
    Output is float32 or float64 according to solve_dtype, even for half input.
    The same differentiable decoder is intended for base and later guidance.
    """
    config = config or DecoderConfig()
    if not isinstance(raw_future, torch.Tensor) or not raw_future.is_floating_point() or raw_future.ndim < 3:
        raise TypeError('raw_future must be a floating tensor [...,N,T,6]')
    leading, count = raw_future.shape[:-2], raw_future.shape[-2]
    if raw_future.shape[-1] != 6 or count < 1 or agent_mask.shape != leading or agent_mask.dtype != torch.bool:
        raise ValueError('Invalid future or boolean agent-mask shape')
    if initial_positions.shape != (*leading, 2) or initial_heading.shape != (*leading, 2):
        raise ValueError('Initial position/heading shapes must match agents')
    if any(value.device != raw_future.device for value in (initial_positions, initial_heading, agent_mask)):
        raise ValueError('All decoder inputs must share one device')
    _finite_active(raw_future, agent_mask, 'future')
    _finite_active(initial_positions, agent_mask, 'initial positions')
    _finite_active(initial_heading, agent_mask, 'initial heading')
    dtype = getattr(torch, config.solve_dtype)
    # Mask before all arithmetic so NaNs in padding cannot enter a solve/gradient.
    clean = torch.where(agent_mask[..., None, None], raw_future, 0.).to(dtype)
    start = torch.where(agent_mask[..., None], initial_positions, 0.).to(dtype)
    heading0 = torch.where(agent_mask[..., None], initial_heading, 0.).to(dtype)
    norm0 = torch.linalg.vector_norm(heading0, dim=-1, keepdim=True)
    if (norm0[..., 0][agent_mask] < config.heading_epsilon).any():
        raise ValueError('Observed initial heading must have a nonzero sin/cos vector')
    heading0 = heading0 / norm0.clamp_min(config.heading_epsilon)
    with torch.autocast(device_type=raw_future.device.type, enabled=False):
        difference = torch.eye(count, device=raw_future.device, dtype=dtype)
        if count > 1:
            difference = difference - torch.diag(torch.ones(count-1, device=raw_future.device, dtype=dtype), diagonal=-1)
        alpha = config.position_weight / config.position_unit_m**2
        beta = config.velocity_weight / config.velocity_unit_mps**2
        matrix = alpha * torch.eye(count, device=raw_future.device, dtype=dtype) + beta / config.dt**2 * difference.T @ difference
        rhs = alpha * clean[..., :2] + beta / config.dt * difference.T @ clean[..., 2:4]
        displacement = torch.linalg.solve(matrix, rhs)
        velocity = difference @ displacement / config.dt
        raw_velocity = difference @ clean[..., :2] / config.dt
    norms = torch.linalg.vector_norm(clean[..., 4:6], dim=-1, keepdim=True)
    fallback = (norms[..., 0] < config.heading_epsilon) & agent_mask[..., None]
    normalized = clean[..., 4:6] / norms.clamp_min(config.heading_epsilon)
    headings, previous = [], heading0
    for step in range(count):
        previous = torch.where(fallback[..., step, None], previous, normalized[..., step, :])
        headings.append(previous)
    heading = torch.stack(headings, dim=-2)
    position = displacement + start[..., None, :]
    raw_states = torch.cat((clean[..., :2] + start[..., None, :], clean[..., 2:]), dim=-1)
    states = torch.cat((position, velocity, heading), dim=-1)
    raw_states = torch.where(agent_mask[..., None, None], raw_states, 0.)
    states = torch.where(agent_mask[..., None, None], states, 0.)
    angle_valid = agent_mask[..., None] & ~fallback
    safe_raw_heading = torch.where(angle_valid[..., None], normalized, torch.tensor([0., 1.], dtype=dtype, device=clean.device))
    safe_heading = torch.where(agent_mask[..., None, None], heading, torch.tensor([0., 1.], dtype=dtype, device=clean.device))
    delta_angle = torch.atan2(safe_heading[..., 0], safe_heading[..., 1]) - torch.atan2(safe_raw_heading[..., 0], safe_raw_heading[..., 1])
    delta_angle = torch.where(angle_valid, torch.atan2(torch.sin(delta_angle), torch.cos(delta_angle)), 0.)
    return DecodedTrajectory(raw_future=raw_future, raw_states=raw_states, states=states,
        positions_with_start=torch.cat((start[..., None, :], states[..., :2]), dim=-2), initial_heading=heading0,
        position_correction=states[..., :2] - raw_states[..., :2],
        velocity_correction=states[..., 2:4] - clean[..., 2:4], heading_correction=states[..., 4:6] - clean[..., 4:6],
        heading_angle_correction=delta_angle, heading_angle_valid=angle_valid,
        raw_velocity_residual=clean[..., 2:4] - raw_velocity,
        raw_heading_norm_error=torch.where(agent_mask[..., None], (norms[..., 0] - 1.).abs(), 0.),
        heading_fallback_mask=fallback, agent_mask=agent_mask)
