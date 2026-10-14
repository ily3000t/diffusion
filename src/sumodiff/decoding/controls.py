"""Interval control inversion and shared differentiable trajectory rollout.

Controls are sampled interval quantities, not continuous actuator commands.
The optional lateral interval velocity is an explicit reconstruction reference,
not a no-slip vehicle model. No clipping, smoothing, or road enforcement.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class ControlConfig:
    dt: float = .1
    reference_point: str = 'center'
    integration_heading: str = 'midpoint'
    lateral_velocity: bool = False
    compute_dtype: str = 'float64'
    heading_epsilon: float = 1e-6

    def __post_init__(self):
        if type(self.dt) not in (int, float) or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('dt must be finite and positive')
        if self.dt != .1:
            raise NotImplementedError('Only dt=0.1 is supported by this diagnostic')
        if self.reference_point not in ('center', 'front'):
            raise ValueError('Unknown reference point')
        if self.integration_heading not in ('midpoint', 'endpoint'):
            raise ValueError('Unknown integration heading')
        if type(self.lateral_velocity) is not bool:
            raise TypeError('lateral_velocity must be boolean')
        if self.compute_dtype not in ('float32', 'float64'):
            raise ValueError('Rollout requires float32 or float64 computation')
        if type(self.heading_epsilon) not in (int, float) or not math.isfinite(self.heading_epsilon) or self.heading_epsilon <= 0:
            raise ValueError('heading_epsilon must be finite and positive')

    @property
    def channels(self):
        return 3 if self.lateral_velocity else 2


@dataclass(frozen=True)
class InitialControlState:
    center_position: torch.Tensor
    reference_position: torch.Tensor
    yaw: torch.Tensor
    forward_interval_speed: torch.Tensor
    length: torch.Tensor
    agent_mask: torch.Tensor
    config: ControlConfig


@dataclass
class ControlTrajectory:
    states: torch.Tensor
    positions_with_start: torch.Tensor
    reference_positions_with_start: torch.Tensor
    forward_interval_speed: torch.Tensor
    yaw: torch.Tensor
    agent_mask: torch.Tensor


@dataclass
class InvertedControls:
    controls: torch.Tensor
    lateral_interval_velocity: torch.Tensor
    label_velocity_residual: torch.Tensor
    label_heading_norm_error: torch.Tensor


def _floating(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f'{name} must be a floating tensor')


def _active_finite(value, mask, name):
    if not torch.isfinite(value[mask]).all():
        raise ValueError(f'Non-finite active {name}')


def _prefix_sum(value, dim):
    # Torch 2.5 CUDA cumsum rejects strict determinism. A fixed triangular
    # linear map defines the same finite sum, with deterministic GEMM/backward.
    if value.is_cuda and torch.are_deterministic_algorithms_enabled():
        x = value.movedim(dim, -1)
        triangle = torch.ones((x.shape[-1], x.shape[-1]), device=x.device, dtype=x.dtype).tril()
        return torch.nn.functional.linear(x, triangle).movedim(-1, dim)
    return torch.cumsum(value, dim=dim)


def _direction(yaw):
    return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)


def _side(yaw):
    return torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)


def _wrapped(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _safe_yaw(sincos, mask, epsilon):
    norm = torch.linalg.vector_norm(sincos, dim=-1)
    if (norm[mask] < epsilon).any():
        raise ValueError('Active observed/label heading must be nonzero')
    safe = torch.where(mask[..., None], sincos,
                       torch.tensor([0., 1.], device=sincos.device, dtype=sincos.dtype))
    return torch.atan2(safe[..., 0], safe[..., 1]), norm


def initial_control_state(history, history_mask, initial_positions, lengths, agent_mask, config=None):
    """Read only the last two observed poses to initialize an interval speed.

    history[..., -1] is t0. No future input or SUMO instantaneous speed is read.
    All agents share the same fixed local frame; lengths are observed attributes.
    """
    config = config or ControlConfig()
    for name, value in (('history', history), ('initial_positions', initial_positions), ('lengths', lengths)):
        _floating(value, name)
    if not isinstance(agent_mask, torch.Tensor) or agent_mask.dtype != torch.bool:
        raise ValueError('agent_mask must be boolean')
    leading = history.shape[:-2]
    if history.ndim < 3 or history.shape[-1] != 6 or history.shape[-2] < 2:
        raise ValueError('history must have shape [...,N,H>=2,6]')
    if agent_mask.shape != leading or initial_positions.shape != (*leading, 2) or lengths.shape != leading:
        raise ValueError('Initial shapes must match agents')
    if history_mask.shape != history.shape[:-1] or history_mask.dtype != torch.bool:
        raise ValueError('Invalid boolean history_mask')
    if any(v.device != history.device for v in (history_mask, initial_positions, lengths, agent_mask)):
        raise ValueError('Initial inputs must share one device')
    if not history_mask[..., -2:][agent_mask].all():
        raise ValueError('Both final observed poses are required; no future fallback')
    for value, name in ((history[..., -2:, :], 'history'), (initial_positions, 'positions'), (lengths, 'lengths')):
        _active_finite(value, agent_mask, name)
    if (lengths[agent_mask] <= 0).any():
        raise ValueError('Active vehicle lengths must be positive')
    dtype = getattr(torch, config.compute_dtype)
    h = torch.where(agent_mask[..., None, None], history[..., -2:, :], 0.).to(dtype)
    p0 = torch.where(agent_mask[..., None], initial_positions, 0.).to(dtype)
    length = torch.where(agent_mask, lengths, 0.).to(dtype)
    if not torch.allclose(h[..., -1, :2][agent_mask], p0[agent_mask], atol=1e-5, rtol=0):
        raise ValueError('Observed t0 position and supplied initial position disagree')
    with torch.autocast(device_type=history.device.type, enabled=False):
        valid = agent_mask[..., None].expand(*leading, 2)
        yaw, _ = _safe_yaw(h[..., 4:6], valid, config.heading_epsilon)
        turn = _wrapped(yaw[..., 1] - yaw[..., 0])
        if (turn[agent_mask].abs() >= math.pi - 1e-8).any():
            raise ValueError('Ambiguous observed yaw increment near pi')
        offset = length / 2 if config.reference_point == 'front' else torch.zeros_like(length)
        previous = h[..., 0, :2] + offset[..., None] * _direction(yaw[..., 0])
        current = p0 + offset[..., None] * _direction(yaw[..., 1])
        basis = yaw[..., 0] + turn / 2 if config.integration_heading == 'midpoint' else yaw[..., 1]
        u0 = ((current - previous) * _direction(basis)).sum(-1) / config.dt
    return InitialControlState(p0, current, yaw[..., 1], u0, length, agent_mask, config)


def rollout_controls(controls, initial):
    """Integrate interval acceleration/yaw rate (and optional lateral velocity).

    u[k]=u[k-1]+dt*a[k], theta[k]=theta[k-1]+dt*omega[k].
    Reference displacement=dt*(u[k]*d(basis)+r[k]*n(basis)).
    r is zero for the two-control candidate. Center velocity is always the
    backward center displacement. Padding is masked before arithmetic.
    """
    _floating(controls, 'controls')
    config, mask = initial.config, initial.agent_mask
    if controls.ndim < 3 or controls.shape[:-2] != mask.shape or controls.shape[-1] != config.channels or controls.shape[-2] < 1:
        raise ValueError('Controls must match initial agents, horizon and explicit channel count')
    if controls.device != initial.center_position.device:
        raise ValueError('Controls and initial state must share one device')
    _active_finite(controls, mask, 'controls')
    dtype = getattr(torch, config.compute_dtype)
    values = torch.where(mask[..., None, None], controls, 0.).to(dtype)
    # Avoid aliasing multiple revolutions into a shortest-arc offline heading.
    if (values[..., 1][mask].abs() * config.dt >= math.pi).any():
        raise ValueError('Yaw increment must be strictly below pi per interval')
    with torch.autocast(device_type=controls.device.type, enabled=False):
        yaw = initial.yaw[..., None] + config.dt * _prefix_sum(values[..., 1], dim=-1)
        u = initial.forward_interval_speed[..., None] + config.dt * _prefix_sum(values[..., 0], dim=-1)
        basis = yaw - values[..., 1] * config.dt / 2 if config.integration_heading == 'midpoint' else yaw
        velocity = u[..., None] * _direction(basis)
        if config.lateral_velocity:
            velocity = velocity + values[..., 2, None] * _side(basis)
        reference = initial.reference_position[..., None, :] + config.dt * _prefix_sum(velocity, dim=-2)
        offset = initial.length / 2 if config.reference_point == 'front' else torch.zeros_like(initial.length)
        position = reference - offset[..., None, None] * _direction(yaw)
        positions = torch.cat((initial.center_position[..., None, :], position), dim=-2)
        center_velocity = torch.diff(positions, dim=-2) / config.dt
        states = torch.cat((position, center_velocity, torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)), dim=-1)
        states = torch.where(mask[..., None, None], states, 0.)
        positions = torch.where(mask[..., None, None], positions, 0.)
        reference = torch.cat((initial.reference_position[..., None, :], reference), dim=-2)
        reference = torch.where(mask[..., None, None], reference, 0.)
    return ControlTrajectory(states, positions, reference, torch.where(mask[..., None], u, 0.),
                             torch.where(mask[..., None], yaw, 0.), mask)


def inverse_label_controls(states, future_mask, initial):
    """Offline label transform ONLY; not an inference condition.

    Labels are absolute local center states, not relative-position targets.
    Incomplete active horizons are rejected. Two controls discard the reported
    lateral residual; three controls explicitly preserve it. Original label
    velocity is audited, never substituted into the inverse calculation.
    """
    _floating(states, 'states')
    config, mask = initial.config, initial.agent_mask
    if states.ndim < 3 or states.shape[:-2] != mask.shape or states.shape[-1] != 6 or states.shape[-2] < 1:
        raise ValueError('Absolute states must match agents and have six channels')
    if future_mask.shape != states.shape[:-1] or future_mask.dtype != torch.bool:
        raise ValueError('Invalid boolean label future_mask')
    if states.device != initial.center_position.device or future_mask.device != states.device:
        raise ValueError('Label inputs and initial state must share one device')
    if not future_mask[mask].all():
        raise ValueError('Control inversion requires a complete active label horizon')
    _active_finite(states, mask, 'label states')
    clean = torch.where(mask[..., None, None], states, 0.).to(getattr(torch, config.compute_dtype))
    valid = mask[..., None].expand(states.shape[:-1])
    with torch.autocast(device_type=states.device.type, enabled=False):
        yaw, norms = _safe_yaw(clean[..., 4:6], valid, config.heading_epsilon)
        previous_yaw = torch.cat((initial.yaw[..., None], yaw[..., :-1]), dim=-1)
        turn = _wrapped(yaw - previous_yaw)
        if (turn[mask].abs() >= math.pi - 1e-8).any():
            raise ValueError('Ambiguous label yaw increment near pi')
        basis = previous_yaw + turn / 2 if config.integration_heading == 'midpoint' else yaw
        offset = initial.length / 2 if config.reference_point == 'front' else torch.zeros_like(initial.length)
        reference = clean[..., :2] + offset[..., None, None] * _direction(yaw)
        reference_with_start = torch.cat((initial.reference_position[..., None, :], reference), dim=-2)
        interval_velocity = torch.diff(reference_with_start, dim=-2) / config.dt
        speed = (interval_velocity * _direction(basis)).sum(-1)
        lateral = (interval_velocity * _side(basis)).sum(-1)
        previous_speed = torch.cat((initial.forward_interval_speed[..., None], speed[..., :-1]), dim=-1)
        channels = [(speed - previous_speed) / config.dt, turn / config.dt]
        if config.lateral_velocity:
            channels.append(lateral)
        controls = torch.stack(channels, dim=-1)
        center_velocity = torch.diff(torch.cat((initial.center_position[..., None, :], clean[..., :2]), dim=-2), dim=-2) / config.dt
        residual = clean[..., 2:4] - center_velocity
    return InvertedControls(torch.where(mask[..., None, None], controls, 0.),
                            torch.where(valid, lateral, 0.),
                            torch.where(valid[..., None], residual, 0.),
                            torch.where(valid, (norms - 1).abs(), 0.))
