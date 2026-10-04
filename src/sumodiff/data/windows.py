"""Masked windows selected from t0 alone; labels are a separate structure."""
from dataclasses import dataclass
from collections import Counter
import numpy as np
from shapely.geometry import Point, box
from sumodiff.geometry.coordinates import FixedFrame
from sumodiff.geometry.maps import rasterize


@dataclass
class Window:
    conditioning: dict
    targets: dict
    input_metadata: dict
    label_metadata: dict
    exact_map: dict


def candidate_ticks(episode, config):
    # t0 on the 1s grid; 21 preceding ticks include the extra velocity sample.
    first = ((config['history_points'] + config['reference_stride_ticks'] - 1) // config['reference_stride_ticks']) * config['reference_stride_ticks']
    return range(first, len(episode.frames), config['reference_stride_ticks'])


def _state(episode, vehicle_id, tick, frame):
    if tick <= 0 or tick >= len(episode.frames):
        return None, 'outside_episode'
    current = episode.frames[tick].get(vehicle_id)
    previous = episode.frames[tick - 1].get(vehicle_id)
    if current is None:
        return None, 'vehicle_missing'
    if previous is None:
        return None, 'previous_velocity_sample_missing'
    position = frame.positions(current.center)
    velocity = frame.vectors((current.center - previous.center) / episode.dt)
    return np.r_[position, velocity, frame.heading(current.yaw)], None


def build_window(episode, tick, road, config):
    if tick not in candidate_ticks(episode, config):
        raise ValueError('Reference tick must be on the configured grid with the extra preceding history sample')
    current = episode.frames[tick]
    if not current:
        return None, 'no_current_reference_vehicle'
    reference = sorted(current)[0]
    reference_observation = current[reference]
    frame = FixedFrame(reference_observation.center, reference_observation.yaw)
    ranked = sorted((np.linalg.norm(obs.center - frame.origin), vehicle_id)
                    for vehicle_id, obs in current.items() if vehicle_id != reference)
    near = [(distance, vid) for distance, vid in ranked if distance <= config['selection_radius_m']]
    ids = [reference] + [vid for _, vid in near[:config['max_agents'] - 1]]
    n, h, f = config['max_agents'], config['history_points'], config['future_points']
    history, future = np.zeros((n, h, 6), np.float32), np.zeros((n, f, 6), np.float32)
    agent_mask, history_mask, future_mask = np.zeros(n, bool), np.zeros((n, h), bool), np.zeros((n, f), bool)
    attributes, initial_positions = np.zeros((n, 4), np.float32), np.zeros((n, 2), np.float32)
    history_reasons, future_reasons = Counter(), Counter()
    routes, route_indices, types = [], [], []
    coverage, body_coverage, offroad, max_speed = 0, 0, 0, 0.
    coverage_area = box(*config['map_extent_m'])
    for slot, vehicle_id in enumerate(ids):
        obs = current[vehicle_id]
        agent_mask[slot] = True
        initial_positions[slot] = frame.positions(obs.center)
        attributes[slot] = [obs.raw['length_m'], obs.raw['width_m'], 1., slot == 0]
        routes.append(list(obs.raw['route_edges']))
        route_indices.append(obs.raw['route_index'])
        types.append(obs.raw['type_id'])
        for j, t in enumerate(range(tick - h + 1, tick + 1)):
            state, reason = _state(episode, vehicle_id, t, frame)
            if reason:
                history_reasons[reason] += 1
            else:
                history[slot, j], history_mask[slot, j] = state, True
        for j, t in enumerate(range(tick + 1, tick + f + 1)):
            state, reason = _state(episode, vehicle_id, t, frame)
            if reason:
                future_reasons[reason] += 1
            else:
                future[slot, j], future_mask[slot, j] = state, True
                future[slot, j, :2] -= initial_positions[slot]
                observed = episode.frames[t][vehicle_id]
                max_speed = max(max_speed, float(np.linalg.norm(state[2:4])))
                # Independent flags: a point outside the crop may still be on the road.
                coverage += not coverage_area.covers(Point(*state[:2]))
                direction = np.array([state[5], state[4]])
                sideways = np.array([-direction[1], direction[0]])
                corners = np.array([state[:2] + a * observed.raw['length_m'] / 2 * direction +
                                    b * observed.raw['width_m'] / 2 * sideways for a in (-1, 1) for b in (-1, 1)])
                bounds = config['map_extent_m']
                body_coverage += bool((corners[:, 0] < bounds[0]).any() or (corners[:, 0] > bounds[2]).any() or
                                      (corners[:, 1] < bounds[1]).any() or (corners[:, 1] > bounds[3]).any())
                offroad += not road.drivable.covers(Point(*observed.center))
    polylines, adjacency, valid_routes, exact = road.local(frame, routes, config['polyline_points'])
    route_lane_mask = np.zeros((n, len(road.lanes)), bool)
    route_lane_mask[:len(ids)] = valid_routes
    conditioning = dict(history=history, agent_mask=agent_mask, history_mask=history_mask,
        attributes=attributes, initial_positions=initial_positions,
        map_raster=rasterize(exact, config['map_extent_m'], config['raster_size']),
        lane_polylines=polylines, lane_mask=np.ones(len(road.lanes), bool),
        lane_point_mask=np.ones(polylines.shape[:2], bool), lane_adjacency=adjacency, route_lane_mask=route_lane_mask)
    input_metadata = dict(schema_version='sumodiff.window.input.v1',
        window_id=f"{episode.key.split(':')[1]}-t{tick}", episode_key=episode.key,
        source_episode_id=episode.manifest['episode_id'], geometry_id=episode.manifest['geometry_id'],
        split=episode.split, family=episode.manifest['family'], reference_tick=tick, reference_time_s=tick * episode.dt,
        dt_seconds=episode.dt, fixed_frame=frame.to_dict(), agent_ids=ids + [None] * (n - len(ids)),
        planned_routes=routes + [None] * (n - len(ids)), route_indices_at_t0=route_indices + [None] * (n - len(ids)),
        sumo_type_ids=types + [None] * (n - len(ids)), reference_id=reference,
        current_vehicle_count=len(current), excluded_vehicle_count=len(current) - len(ids),
        excluded_by_radius=len(ranked) - len(near), excluded_by_capacity=max(0, len(near) - (n - 1)),
        map_extent_m=config['map_extent_m'], raster_row_direction='positive_y_to_negative_y')
    label_metadata = dict(schema_version='sumodiff.window.labels.v1',
        history_missing_reasons=dict(history_reasons), future_missing_reasons=dict(future_reasons),
        complete_history=bool(history_mask[agent_mask].all()), complete_future=bool(future_mask[agent_mask].all()),
        core_training_eligible=bool(history_mask[agent_mask].all() and future_mask[agent_mask].all()),
        valid_future_points=int(future_mask.sum()), future_center_out_of_map=coverage,
        future_body_out_of_map=body_coverage, future_center_offroad=offroad, future_max_speed_mps=max_speed)
    return Window(conditioning, dict(future=future, future_mask=future_mask), input_metadata, label_metadata, exact), None


def audit_calibration_episodes(episodes, config):
    """Check the preconfigured envelope using train/validation only, never test."""
    report = {'used_splits': ['train', 'validation'], 'max_interval_speed_mps': 0., 'max_half_diagonal_m': 0., 'records': 0}
    for episode in episodes:
        if episode.split == 'test':
            continue
        for tick in range(1, len(episode.frames)):
            for vid, obs in episode.frames[tick].items():
                report['records'] += 1
                report['max_half_diagonal_m'] = max(report['max_half_diagonal_m'], .5 * float(np.hypot(obs.raw['length_m'], obs.raw['width_m'])))
                previous = episode.frames[tick - 1].get(vid)
                if previous is not None:
                    report['max_interval_speed_mps'] = max(report['max_interval_speed_mps'], float(np.linalg.norm(obs.center - previous.center) / episode.dt))
    report['speed_bound_pass'] = report['max_interval_speed_mps'] <= config['coverage_speed_bound_mps']
    report['body_margin_pass'] = report['max_half_diagonal_m'] <= config['coverage_body_margin_m']
    if not report['records'] or not report['speed_bound_pass'] or not report['body_margin_pass']:
        raise ValueError(f'Train/validation coverage envelope audit failed: {report}')
    return report
