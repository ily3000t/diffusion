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
    geometry_repair: str = "strict"
    max_repair_area_change_m2: float = 1e-6
    max_repair_hausdorff_m: float = 1e-6

    def __post_init__(self):
        for name in ('numerical_tolerance_m','max_repair_area_change_m2','max_repair_hausdorff_m'):
            value = getattr(self,name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.geometry_repair not in ('strict','bounded_make_valid'):
            raise ValueError('Unsupported geometry repair policy')


def _checked_geometry(serialized, name, config):
    original = shape(serialized)
    # Legal clipping and linework repair may both contain polygon+line parts.
    # Keep the point set intact; require positive area and all original bounds.
    if (original.geom_type not in ('Polygon','MultiPolygon','GeometryCollection') or original.is_empty
        or not np.isfinite(shapely.get_coordinates(original)).all() or original.area <= 0):
        raise ValueError(f'{name} must be a finite nonempty polygon area')
    if original.is_valid:
        return original, None
    reason = shapely.is_valid_reason(original)
    if config.geometry_repair == 'strict':
        raise ValueError(f'Invalid {name}: {reason}')
    # Preserve line/point components from linework repair. Discarding them can
    # move the point-set boundary by a lane half-width even if area is unchanged.
    repaired = shapely.make_valid(original,method='linework',keep_collapsed=True)
    area_change = abs(repaired.area-original.area)
    distance = original.hausdorff_distance(repaired)
    if (repaired.is_empty or not repaired.is_valid or repaired.area <= 0
        or not math.isfinite(area_change) or not math.isfinite(distance)
        or area_change > config.max_repair_area_change_m2
        or distance > config.max_repair_hausdorff_m):
        raise ValueError(f'{name} repair exceeds numerical bounds: area={area_change}, distance={distance}')
    record = dict(geometry=name,reason=str(reason),method='linework_keep_collapsed',
        original_type=original.geom_type,repaired_type=repaired.geom_type,
        area_change_m2=float(area_change),hausdorff_m=float(distance),
        component_types=[g.geom_type for g in shapely.get_parts(repaired)])
    return repaired, record


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
    drivable,repair = _checked_geometry(exact_map['drivable'],'drivable',config)
    repairs = [repair] if repair is not None else []
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
        corridor,repair = _checked_geometry(exact_map['route_corridors'][slot],f'route:{slot}',config)
        if repair is not None:
            repairs.append(repair)
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
        geometry_repairs=repairs,geometry_repair_policy=config.geometry_repair,
        full_future_observed=full, evaluation='whole polygon coverage at supplied future frames; raster not used')
