"""Masked auxiliary objectives for a controlled epsilon-model continuation."""
from dataclasses import asdict, dataclass
import math
import torch
import torch.nn.functional as F
from sumodiff.decoding import DecoderConfig, decode_future, observed_heading
from .process import masked_noise_mse, denormalize


@dataclass(frozen=True)
class PhysicalLossConfig:
    schema_version: str = 'sumodiff.physical.loss.config.v1'
    consistency_weight: float = 0.
    velocity_difference_weight: float = 0.
    boundary_weight: float = 0.
    max_auxiliary_timestep: int = 300
    huber_beta: float = 1.
    dt: float = .1
    velocity_unit_mps: float = 20.
    acceleration_unit_mps2: float = 6.
    jerk_unit_mps3: float = 15.

    def __post_init__(self):
        if self.schema_version != 'sumodiff.physical.loss.config.v1':
            raise ValueError('Unsupported physical loss schema')
        for name in ('consistency_weight', 'velocity_difference_weight', 'boundary_weight'):
            x = getattr(self, name)
            if type(x) not in (float, int) or not math.isfinite(x) or x < 0:
                raise ValueError(f'Invalid {name}')
        for name in ('huber_beta', 'dt', 'velocity_unit_mps', 'acceleration_unit_mps2', 'jerk_unit_mps3'):
            x = getattr(self, name)
            if type(x) not in (float, int) or not math.isfinite(x) or x <= 0:
                raise ValueError(f'Invalid {name}')
        if type(self.max_auxiliary_timestep) is not int or not 0 <= self.max_auxiliary_timestep < 1000:
            raise ValueError('Auxiliary timestep must be an integer in [0,999]')
        if self.dt != .1:
            raise NotImplementedError('Physical objectives support dt=0.1 only')

    @property
    def enabled(self):
        return bool(self.consistency_weight or self.velocity_difference_weight or self.boundary_weight)

    def to_dict(self):
        return asdict(self)


def resolve_physical_config(value):
    if not isinstance(value, dict) or set(value)-set(PhysicalLossConfig.__dataclass_fields__):
        raise ValueError('Unknown physical loss options')
    return PhysicalLossConfig(**value)


def scene_huber(residual, mask, beta):
    if mask.dtype != torch.bool or mask.shape != residual.shape[:-1]:
        raise ValueError('Invalid physical loss mask')
    safe = torch.where(mask[..., None], residual, 0.)
    if not torch.isfinite(safe).all():
        raise ValueError('Nonfinite supervised physical residual')
    counts = mask.sum(tuple(range(1, mask.ndim))) * residual.shape[-1]
    if (counts == 0).any():
        raise ValueError('No supervised physical elements in a selected scene')
    dims = tuple(range(1, residual.ndim))
    return F.smooth_l1_loss(safe, torch.zeros_like(safe), reduction='none', beta=beta).sum(dims)/counts


def boundary_vectors(states, conditioning, dt):
    """First two future interval v/a/jerk; shared fixed observed position at t0."""
    history = conditioning['history'][..., -3:, :2].to(states.dtype).clone()
    history[..., -1, :] = conditioning['initial_positions'].to(states.dtype)
    points = torch.cat((history, states[..., :2]), dim=-2)
    velocity = torch.diff(points, dim=-2)/dt
    acceleration = torch.diff(velocity, dim=-2)/dt
    jerk = torch.diff(acceleration, dim=-2)/dt
    return velocity[..., 2:4, :], acceleration[..., 1:3, :], jerk[..., :2, :]


def physical_objective(epsilon, noisy, noise, timestep, conditioning, targets, schedule, config):
    """No RNG, extra network call, clipping, sampler change, or label conditioning.

    Auxiliary terms use only t<=max and alpha_bar weighting. Their means retain
    the whole minibatch denominator. High-t scenes keep their epsilon objective.
    """
    mse = masked_noise_mse(epsilon, noise, conditioning['agent_mask'], targets['future_mask'])
    zero = mse.new_zeros(())
    result = dict(noise_mse=mse, consistency=zero, velocity_difference=zero,
                  boundary=zero, physical_total=zero, total_loss=mse,
                  auxiliary_eligible_scenes=0)
    if not config.enabled:
        return result  # A keeps exactly the legacy epsilon computation graph.
    if config.max_auxiliary_timestep >= schedule.config.training_steps:
        raise ValueError('Auxiliary timestep outside this noise schedule')
    eligible = torch.nonzero(timestep <= config.max_auxiliary_timestep).flatten()
    result['auxiliary_eligible_scenes'] = len(eligible)
    if not len(eligible):
        return result
    c = {k:v[eligible] for k,v in conditioning.items()}
    future = targets['future'][eligible]
    a = c['agent_mask']
    fm = targets['future_mask'][eligible] & a[..., None]
    clean = schedule.clean_from_noise(noisy[eligible], epsilon[eligible], timestep[eligible])
    # Clear label/agent padding before derivatives; padding has no gradient.
    physical = denormalize(torch.where(fm[..., None], clean, 0.), schedule.config)
    truth = torch.where(fm[..., None], future, 0.).to(physical.dtype)
    count = len(epsilon)
    alpha = schedule.alpha_bar[timestep[eligible]].to(physical.dtype).detach()
    def aggregate(per_scene):
        return (per_scene*alpha).sum()/count

    if config.consistency_weight:
        previous = torch.cat((torch.zeros_like(physical[..., :1, :2]), physical[..., :-1, :2]), dim=-2)
        previous_valid = torch.cat((c['history_mask'][..., -1:], fm[..., :-1]), dim=-1)
        valid = fm & previous_valid
        residual = ((physical[..., :2]-previous)/config.dt-physical[..., 2:4])/config.velocity_unit_mps
        result['consistency'] = aggregate(scene_huber(residual, valid, config.huber_beta))
    if config.velocity_difference_weight:
        valid = fm[..., 1:] & fm[..., :-1]
        difference = (torch.diff(physical[..., 2:4], dim=-2)-torch.diff(truth[..., 2:4], dim=-2))/config.dt
        result['velocity_difference'] = aggregate(scene_huber(difference/config.acceleration_unit_mps2, valid, config.huber_beta))
    if config.boundary_weight:
        if not c['history_mask'][a][..., -3:].all() or not fm[a].all():
            raise ValueError('Boundary decoder supervision requires complete curated H+F')
        decoder = DecoderConfig(dt=config.dt, solve_dtype='float64')
        heading = observed_heading(c['history'], c['history_mask'], a)
        prediction = decode_future(physical, c['initial_positions'], heading, a, decoder)
        label = decode_future(truth, c['initial_positions'], heading, a, decoder)
        predicted = boundary_vectors(prediction.states, c, config.dt)
        expected = boundary_vectors(label.states, c, config.dt)
        valid = a[..., None].expand(*a.shape, 2)
        terms = [scene_huber((x-y)/unit, valid, config.huber_beta)
                 for x,y,unit in zip(predicted, expected,
                      (config.velocity_unit_mps, config.acceleration_unit_mps2, config.jerk_unit_mps3))]
        result['boundary'] = aggregate(sum(terms)/3)
    result['physical_total'] = (config.consistency_weight*result['consistency']
        + config.velocity_difference_weight*result['velocity_difference']
        + config.boundary_weight*result['boundary'])
    result['total_loss'] = mse + result['physical_total']
    return result
