"""Explicit stage4 architecture and conditioning feature units."""
from dataclasses import asdict,dataclass
import math


@dataclass(frozen=True)
class ModelConfig:
    schema_version: str = 'sumodiff.model.config.v1'
    condition_dimension: int = 128
    attention_heads: int = 4
    fusion: str = 'hierarchical'
    history_points: int = 21
    future_points: int = 40
    state_dimension: int = 6
    max_agents: int = 12
    raster_size: int = 256
    polyline_points: int = 64
    unet_channels: tuple = (64,128,256)
    raster_channels: tuple = (16,32,64,128,128)
    diffusion_embedding_steps: int = 1000
    condition_position_unit_m: float = 50.
    condition_velocity_unit_mps: float = 20.
    condition_length_unit_m: float = 10.
    condition_width_unit_m: float = 5.
    condition_priority_unit: float = 3.
    attack_role_embedding: bool = False
    cfg: bool = False
    partial_diffusion: bool = False
    rolling_generation: bool = False

    def __post_init__(self):
        if self.schema_version!='sumodiff.model.config.v1':raise ValueError('Unsupported model schema')
        for key in ('condition_dimension','attention_heads','max_agents','diffusion_embedding_steps'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError(f'Invalid {key}')
        if self.condition_dimension%self.attention_heads or self.condition_dimension%2:
            raise ValueError('Condition dimension must be even and divisible by attention heads')
        if self.fusion not in ('hierarchical','parallel'):raise ValueError('Unsupported fusion')
        if (self.history_points,self.future_points,self.state_dimension,self.raster_size,self.polyline_points)!=(21,40,6,256,64):
            raise NotImplementedError('Stage4 supports only the specified 21/40, six-state, 256 raster, 64-point interface')
        for name,count in (('unet_channels',3),('raster_channels',5)):
            values=getattr(self,name)
            if len(values)!=count or any(type(c) is not int or c<8 or c%8 for c in values):
                raise ValueError(f'{name} requires {count} positive multiples of eight')
        if self.unet_channels[-1]%self.attention_heads:raise ValueError('Bottleneck channels must divide into heads')
        for name in ('condition_position_unit_m','condition_velocity_unit_mps','condition_length_unit_m','condition_width_unit_m','condition_priority_unit'):
            if type(getattr(self,name)) not in (int,float) or not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f'Invalid {name}')
        for name in ('attack_role_embedding','cfg','partial_diffusion','rolling_generation'):
            if type(getattr(self,name)) is not bool:raise ValueError(f'{name} must be boolean')
            if getattr(self,name):raise NotImplementedError(f'{name} is not implemented/supported')

    def to_dict(self):
        result=asdict(self)
        for name in ('unet_channels','raster_channels'):result[name]=list(result[name])
        return result


def resolve_model_config(supplied):
    if not isinstance(supplied,dict) or set(supplied)-set(ModelConfig.__dataclass_fields__):
        raise ValueError('Unknown model options')
    return ModelConfig(**supplied)
