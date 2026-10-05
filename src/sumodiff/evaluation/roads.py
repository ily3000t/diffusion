"""Full oriented-body coverage against vector geometry, including holes."""
from dataclasses import dataclass
import math
import numpy as np
import shapely
from shapely.geometry import box, shape
from sumodiff.geometry.boxes import box_corners


@dataclass(frozen=True)
class RoadConfig:
    numerical_tolerance_m: float = 1e-6

    def __post_init__(self):
        if not math.isfinite(self.numerical_tolerance_m) or self.numerical_tolerance_m < 0:
            raise ValueError('Road tolerance must be finite and nonnegative')


def _coverage(geometry, bodies, tolerance):
    prepared = geometry.buffer(tolerance) if tolerance else geometry
    shapely.prepare(prepared)
    outside = ~shapely.covers(prepared,bodies)
    outside_areas = shapely.area(shapely.difference(bodies[outside],prepared))
    return outside, dict(evaluated_body_frames=len(bodies), violation_body_frames=int(outside.sum()),
        violation_fraction=float(outside.mean()) if len(bodies) else None,
        max_outside_area_m2=float(np.max(outside_areas)) if len(outside_areas) else (0. if len(bodies) else None))


def evaluate_roads(poses, sizes, mask, agent_mask, exact_map, map_extent_m, config=None):
    config = config or RoadConfig()
    poses,sizes,mask,agents = np.asarray(poses,float),np.asarray(sizes,float),np.asarray(mask,bool),np.asarray(agent_mask,bool)
    if poses.ndim != 3 or poses.shape[-1] != 3 or mask.shape != poses.shape[:2] or sizes.shape != (len(poses),2) or agents.shape != (len(poses),):
        raise ValueError('Invalid road metric shapes')
    mask = mask & agents[:,None]
    if not np.isfinite(poses[mask]).all() or not np.isfinite(sizes[agents]).all() or (sizes[agents] <= 0).any():
        raise ValueError('Invalid active body geometry')
    extent = np.asarray(map_extent_m,float)
    if extent.shape != (4,) or not np.isfinite(extent).all() or extent[2] <= extent[0] or extent[3] <= extent[1]:
        raise ValueError('Invalid map coverage extent')
    drivable = shape(exact_map['drivable'])
    if drivable.is_empty or not drivable.is_valid or drivable.geom_type not in ('Polygon','MultiPolygon'):
        raise ValueError('A valid vector drivable area is required')
    locations = np.argwhere(mask)
    if len(locations):
        corners = box_corners(poses[mask],sizes[locations[:,0]])
        bodies = shapely.polygons(corners)
    else:
        bodies = np.array([],dtype=object)
    road_bad,road = _coverage(drivable,bodies,config.numerical_tolerance_m)
    map_bad,coverage = _coverage(box(*extent),bodies,config.numerical_tolerance_m)
    route_bad = np.zeros(len(bodies),bool)
    for slot in np.flatnonzero(agents):
        if slot >= len(exact_map['route_corridors']):
            raise ValueError('Missing route corridor for active slot')
        corridor = shape(exact_map['route_corridors'][slot])
        if corridor.is_empty or not corridor.is_valid:
            raise ValueError('Invalid route corridor')
        subset = locations[:,0] == slot if len(locations) else np.zeros(0,bool)
        route_bad[subset], _ = _coverage(corridor,bodies[subset],config.numerical_tolerance_m)
    def agents_with_violation(flags):
        return len(set(locations[flags,0].tolist())) if len(locations) else 0
    road['violating_agents'], coverage['violating_agents'] = agents_with_violation(road_bad),agents_with_violation(map_bad)
    route = dict(evaluated_body_frames=len(bodies),violation_body_frames=int(route_bad.sum()),
                 violation_fraction=float(route_bad.mean()) if len(bodies) else None,violating_agents=agents_with_violation(route_bad))
    observed_bad = bool(road_bad.any() or map_bad.any() or route_bad.any())
    full = bool(mask[agents].all()) if agents.any() else False
    quality = False if observed_bad else (True if full else None)
    return dict(schema_version='sumodiff.roads.v1', road=road, route=route, map_coverage=coverage,
        geometry_quality_pass=quality, numerical_tolerance_m=config.numerical_tolerance_m,
        full_future_observed=full, evaluation='whole polygon coverage at supplied future frames; raster not used')
