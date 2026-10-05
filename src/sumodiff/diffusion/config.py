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
        for key in ('cfg','partial_diffusion','rolling_generation','additional_losses_enabled'):
            if type(getattr(self,key)) is not bool: raise ValueError(f'{key} must be boolean')
            if getattr(self,key): raise NotImplementedError(f'{key} is not supported by the base checkpoint')

    def to_dict(self): return asdict(self)


def resolve_diffusion_config(value):
    if not isinstance(value,dict) or set(value)-set(DiffusionConfig.__dataclass_fields__): raise ValueError('Unknown diffusion options')
    return DiffusionConfig(**value)
