"""Stage5D raw velocity curvature supervision against normal labels."""
from dataclasses import asdict, dataclass, field
import math
import torch
from .physical_losses import scene_huber
from .process import denormalize
from .unified_losses import UnifiedLossConfig, resolve_unified_loss, unified_objective


@dataclass(frozen=True)
class RawVelocityLossConfig:
    schema_version: str = 'sumodiff.raw.velocity.loss.config.v1'
    unified: UnifiedLossConfig = field(default_factory=UnifiedLossConfig)
    interior_weight: float = 5e-5
    boundary_weight: float = 1e-5
    second_difference_unit_mps: float = .15
    boundary_frames: int = 4

    def __post_init__(self):
        if self.schema_version != 'sumodiff.raw.velocity.loss.config.v1':
            raise ValueError('Unsupported raw velocity loss schema')
        if not isinstance(self.unified, UnifiedLossConfig) or not self.unified.enabled:
            raise ValueError('Raw velocity continuation requires the existing unified objectives')
        for name in ('interior_weight', 'boundary_weight', 'second_difference_unit_mps'):
            x = getattr(self, name)
            if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
                raise ValueError(f'Invalid {name}')
        if type(self.boundary_frames) is not int or self.boundary_frames != 4:
            raise ValueError('Stage5D uses first four future frames as the boundary group')

    @property
    def road_weight(self):
        return self.unified.road_weight

    @property
    def road_numerical_tolerance_m(self):
        return self.unified.road_numerical_tolerance_m

    @property
    def enabled(self):
        return True

    def to_dict(self):
        return asdict(self)


def resolve_raw_velocity_loss(value):
    if not isinstance(value, dict) or set(value)-set(RawVelocityLossConfig.__dataclass_fields__):
        raise ValueError('Unknown raw velocity loss options')
    c = dict(value)
    if 'unified' in c:
        c['unified'] = resolve_unified_loss(c['unified'])
    return RawVelocityLossConfig(**c)


def velocity_residual(raw, conditioning, targets):
    """Second difference of v(-dt), v(0), v(dt), ... minus the same label stencil.

    The 40 results are indexed by their newest future observation. Dividing by
    dt**2 gives a velocity-field jerk error. Historical velocities never use
    future labels. Invalid slots are masked before arithmetic.
    """
    active = conditioning['agent_mask']
    mask = targets['future_mask'] & active[..., None]
    if not mask[active].all() or not conditioning['history_mask'][active][..., -2:].all():
        raise ValueError('Raw velocity supervision requires complete future and last two history observations')
    if raw.shape != targets['future'].shape or raw.shape[-2:] != (40, 6):
        raise ValueError('Raw velocity supervision requires fixed 40 by 6 futures')
    raw = torch.where(active[..., None, None], raw, 0.)
    truth = torch.where(active[..., None, None], targets['future'], 0.).to(raw.dtype)
    observed = torch.where(active[..., None, None], conditioning['history'][..., -2:, 2:4], 0.).to(raw.dtype)
    p = torch.cat((observed, raw[..., 2:4]), -2)
    q = torch.cat((observed, truth[..., 2:4]), -2)
    residual = torch.diff(p, n=2, dim=-2)-torch.diff(q, n=2, dim=-2)
    if not torch.isfinite(residual[mask]).all():
        raise ValueError('Nonfinite active velocity curvature')
    return residual, mask


def raw_velocity_objective(epsilon, noisy, noise, timestep, conditioning, targets,
                           schedule, config, *, roads=None, ramp=1.):
    if type(ramp) not in (int, float) or not math.isfinite(ramp) or not 0 <= ramp <= 1:
        raise ValueError('Raw auxiliary ramp must be in [0,1]')
    # Inherited objectives are already calibrated/trained: no second warmup.
    out = unified_objective(epsilon, noisy, noise, timestep, conditioning, targets,
                            schedule, config.unified, roads=roads, ramp=1.)
    legacy = out['auxiliary_total']
    zero = out['noise_mse'].new_zeros(())
    out.update(inherited_auxiliary_total=legacy, inherited_auxiliary_ramp=1.,
               raw_auxiliary_ramp=float(ramp), raw_second_boundary=zero,
               raw_second_interior=zero, weighted_raw_boundary=zero,
               weighted_raw_interior=zero, raw_auxiliary_total=zero)
    eligible = torch.nonzero(timestep <= config.unified.max_auxiliary_timestep).flatten()
    if len(eligible) and ramp:
        c = {k: v[eligible] for k, v in conditioning.items()}
        t = {k: v[eligible] for k, v in targets.items()}
        clean = schedule.clean_from_noise(noisy[eligible], epsilon[eligible], timestep[eligible])
        raw = denormalize(torch.where(c['agent_mask'][..., None, None], clean, 0.), schedule.config)
        residual, mask = velocity_residual(raw, c, t)
        alpha = schedule.alpha_bar[timestep[eligible]].to(raw.dtype).detach()
        def aggregate(r, m):
            return (scene_huber(r/config.second_difference_unit_mps, m,
                               config.unified.huber_beta)*alpha).sum()/len(epsilon)
        k = config.boundary_frames
        out['raw_second_boundary'] = aggregate(residual[..., :k, :], mask[..., :k])
        out['raw_second_interior'] = aggregate(residual[..., k:, :], mask[..., k:])
        out['weighted_raw_boundary'] = ramp*config.boundary_weight*out['raw_second_boundary']
        out['weighted_raw_interior'] = ramp*config.interior_weight*out['raw_second_interior']
        out['raw_auxiliary_total'] = out['weighted_raw_boundary']+out['weighted_raw_interior']
    out['auxiliary_total'] = legacy+out['raw_auxiliary_total']
    out['total_loss'] = out['noise_mse']+out['auxiliary_total']
    return out
