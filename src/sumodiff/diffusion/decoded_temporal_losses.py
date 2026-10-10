"""Reweight decoded boundary/interior supervision without duplicating old losses."""
from dataclasses import asdict, dataclass, field
import math
from .raw_velocity_losses import RawVelocityLossConfig, resolve_raw_velocity_loss, raw_velocity_objective


@dataclass(frozen=True)
class DecodedTemporalLossConfig:
    schema_version: str = 'sumodiff.decoded.temporal.loss.config.v1'
    inherited: RawVelocityLossConfig = field(default_factory=RawVelocityLossConfig)
    boundary_weight: float = .001
    interior_weight: float = .0009
    boundary_frames: int = 4

    def __post_init__(self):
        if self.schema_version != 'sumodiff.decoded.temporal.loss.config.v1':
            raise ValueError('Unsupported decoded temporal loss schema')
        if not isinstance(self.inherited, RawVelocityLossConfig) or self.inherited.unified.temporal_weight <= 0:
            raise ValueError('Existing decoded temporal supervision must remain enabled')
        if type(self.boundary_frames) is not int or self.boundary_frames != 4:
            raise ValueError('Stage5E fixes four boundary and 36 interior observations')
        for k in ('boundary_weight', 'interior_weight'):
            x = getattr(self, k)
            if type(x) not in (int, float) or not math.isfinite(x) or not 0 < x <= .01:
                raise ValueError(f'Invalid {k}; bounded at 0.01')

    @property
    def road_weight(self):
        return self.inherited.road_weight

    @property
    def road_numerical_tolerance_m(self):
        return self.inherited.road_numerical_tolerance_m

    @property
    def enabled(self):
        return True

    def to_dict(self):
        return asdict(self)


def resolve_decoded_temporal_loss(value):
    if not isinstance(value, dict) or set(value)-set(DecodedTemporalLossConfig.__dataclass_fields__):
        raise ValueError('Unknown decoded temporal loss options')
    c = dict(value)
    if 'inherited' in c:
        c['inherited'] = resolve_raw_velocity_loss(c['inherited'])
    return DecodedTemporalLossConfig(**c)


def decoded_temporal_objective(epsilon, noisy, noise, timestep, conditioning, targets,
                               schedule, config, *, roads=None, ramp=1.):
    if type(ramp) not in (int, float) or not math.isfinite(ramp) or not 0 <= ramp <= 1:
        raise ValueError('Temporal reweight ramp must be in [0,1]')
    # Compute the shared prediction/label decoder once in unified_objective.
    # Stage5D raw terms, imitation and roads remain full strength from update 1.
    out = raw_velocity_objective(epsilon, noisy, noise, timestep, conditioning, targets,
        schedule, config.inherited, roads=roads, ramp=1., _temporal_groups=True)
    old_temporal = out['weighted_temporal']
    sf = config.inherited.unified.second_difference_factor
    boundary = out['first_difference_boundary']+sf*out['second_difference_boundary']
    interior = out['first_difference_interior']+sf*out['second_difference_interior']
    base = config.inherited.unified.temporal_weight
    # Complete 40-point labels make 4/40 and 36/40 exactly the old full mean.
    wb = base*.1+ramp*(config.boundary_weight-base*.1)
    wi = base*.9+ramp*(config.interior_weight-base*.9)
    weighted_boundary, weighted_interior = wb*boundary, wi*interior
    weighted_temporal = weighted_boundary+weighted_interior
    out.update(decoded_temporal_boundary=boundary, decoded_temporal_interior=interior,
        weighted_decoded_boundary=weighted_boundary, weighted_decoded_interior=weighted_interior,
        inherited_weighted_temporal=old_temporal, weighted_temporal=weighted_temporal,
        temporal_reweight_delta=weighted_temporal-old_temporal, temporal_reweight_ramp=float(ramp),
        effective_decoded_boundary_weight=float(wb), effective_decoded_interior_weight=float(wi))
    # Replace old temporal, never append it a second time.
    out['auxiliary_total'] = (out['weighted_imitation']+out['weighted_road']+
        out['weighted_raw_boundary']+out['weighted_raw_interior']+weighted_temporal)
    out['total_loss'] = out['noise_mse']+out['auxiliary_total']
    return out
