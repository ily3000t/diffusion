"""Strict fixed-horizon window configuration, without fitted normalization."""
from copy import deepcopy
import math

DEFAULTS = dict(schema_version='sumodiff.window.config.v1', dt_seconds=.1,
    history_points=21, future_points=40, reference_stride_ticks=10, max_agents=12,
    selection_radius_m=80., reference_policy='first_current_id',
    raster_size=256, map_extent_m=[-192., -192., 192., 192.], polyline_points=64,
    coverage_speed_bound_mps=25., coverage_body_margin_m=8.)


def resolve_config(supplied):
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown window options: {sorted(unknown)}')
    config = {**deepcopy(DEFAULTS), **deepcopy(supplied)}
    if config['schema_version'] != DEFAULTS['schema_version'] or config['dt_seconds'] != .1 or config['history_points'] != 21 or config['future_points'] != 40:
        raise NotImplementedError('Only the v1 0.1s / 21-history / 40-future contract is supported')
    if config['reference_policy'] != 'first_current_id':
        raise NotImplementedError('Unsupported reference selection policy')
    for key in ('reference_stride_ticks', 'max_agents', 'raster_size', 'polyline_points'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['max_agents'] > 12 or config['raster_size'] != 256 or config['polyline_points'] < 2:
        raise ValueError('Unsupported agent cap, raster size, or polyline point count')
    for key in ('selection_radius_m', 'coverage_speed_bound_mps', 'coverage_body_margin_m'):
        if type(config[key]) not in (float, int) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    extent = config['map_extent_m']
    if not isinstance(extent, list) or len(extent) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in extent):
        raise ValueError('Expected four finite map bounds')
    required = config['selection_radius_m'] + config['coverage_speed_bound_mps'] * .1 * 40 + config['coverage_body_margin_m']
    if any(x < required for x in (-extent[0], -extent[1], extent[2], extent[3])):
        raise ValueError(f'Map extent insufficient for configured speed/selection/body envelope: {required}m')
    return config
