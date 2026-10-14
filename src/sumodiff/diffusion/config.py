"""Fixed units and an explicit noise schedule; unsupported research switches fail."""
from dataclasses import asdict, dataclass
import math

@dataclass(frozen=True)
class DiffusionConfig:
    schema_version: str = 'sumodiff.diffusion.config.v1'
    training_steps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    position_scale_m: float = 50.0
    velocity_scale_mps: float = 20.0
    objective: str = 'epsilon'
    cfg: bool = False
    partial_diffusion: bool = False
    rolling_generation: bool = False
    additional_losses_enabled: bool = False

    def __post_init__(self):
        if self.schema_version != 'sumodiff.diffusion.config.v1' or self.objective != 'epsilon':
            raise ValueError('Unsupported diffusion schema/objective')
        if type(self.training_steps) is not int or self.training_steps < 2:
            raise ValueError('training_steps must be an integer >=2')
        for key in ('beta_start','beta_end','position_scale_m','velocity_scale_mps'):
            v=getattr(self,key)
            if type(v) not in (int,float) or not math.isfinite(v) or v<=0: raise ValueError(f'Invalid {key}')
        if not self.beta_start < self.beta_end < 1: raise ValueError('Require 0 < beta_start < beta_end < 1')
        for key in ('cfg','partial_diffusion','rolling_generation'):
            if type(getattr(self,key)) is not bool: raise ValueError(f'{key} must be boolean')
            if getattr(self,key): raise NotImplementedError(f'{key} is not supported by the base checkpoint')

        if type(self.additional_losses_enabled) is not bool:
            raise ValueError('additional_losses_enabled must be boolean')

    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class ControlDiffusionConfig(DiffusionConfig):
    schema_version: str = 'sumodiff.control.diffusion.config.v1'
    acceleration_scale_mps2: float = 2.
    yaw_rate_scale_radps: float = .5
    lateral_velocity_scale_mps: float = .1

    def __post_init__(self):
        base = {key:getattr(self,key) for key in DiffusionConfig.__dataclass_fields__}
        base['schema_version'] = 'sumodiff.diffusion.config.v1'
        DiffusionConfig(**base)
        if self.schema_version != 'sumodiff.control.diffusion.config.v1':
            raise ValueError('Unsupported control diffusion schema')
        if self.position_scale_m != 50. or self.velocity_scale_mps != 20.:
            raise ValueError('Control diffusion uses explicit control scales; legacy state scales must stay at defaults')
        for key in ('acceleration_scale_mps2','yaw_rate_scale_radps','lateral_velocity_scale_mps'):
            v = getattr(self,key)
            if type(v) not in (int,float) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'Invalid {key}')


def is_control(config):
    return isinstance(config,ControlDiffusionConfig)


def target_dimension(config):
    return 3 if is_control(config) else 6


def validate_architecture(model,config):
    if model.state_dimension != target_dimension(config):
        raise ValueError('Model output dimension and diffusion representation mismatch')
    if model.diffusion_embedding_steps != config.training_steps:
        raise ValueError('Model embedding/schedule mismatch')


def resolve_diffusion_config(value):
    if not isinstance(value,dict):raise ValueError('Unknown diffusion options')
    cls = ControlDiffusionConfig if value.get('schema_version')=='sumodiff.control.diffusion.config.v1' else DiffusionConfig
    if set(value)-set(cls.__dataclass_fields__):raise ValueError('Unknown diffusion options')
    return cls(**value)
