"""Resolved evaluation settings; thresholds are explicit engineering candidates."""
from copy import deepcopy
from dataclasses import asdict
from sumodiff.decoding import DecoderConfig
from .motion import MotionConfig
from .roads import RoadConfig
from .collisions import CollisionConfig

DEFAULTS = dict(schema_version='sumodiff.evaluation.config.v1',decoder=asdict(DecoderConfig()),
    motion=asdict(MotionConfig()),road=asdict(RoadConfig()),collision=asdict(CollisionConfig()),
    seed=20261005,torch_threads=1,cuda_smoke=False)


def resolve_config(supplied):
    if not isinstance(supplied,dict) or set(supplied)-set(DEFAULTS):
        raise ValueError('Unknown evaluation options')
    result = deepcopy(DEFAULTS)
    for key,value in supplied.items():
        if isinstance(result[key],dict):
            if not isinstance(value,dict) or set(value)-set(result[key]):
                raise ValueError(f'Unknown {key} options')
            result[key].update(value)
        else:
            result[key] = value
    if result['schema_version'] != DEFAULTS['schema_version'] or type(result['seed']) is not int or not 0 <= result['seed'] < 2**32:
        raise ValueError('Invalid evaluation schema or seed')
    if type(result['torch_threads']) is not int or result['torch_threads'] < 1 or type(result['cuda_smoke']) is not bool:
        raise ValueError('Invalid thread count or cuda_smoke flag')
    DecoderConfig(**result['decoder']); MotionConfig(**result['motion']); RoadConfig(**result['road']); CollisionConfig(**result['collision'])
    return result
