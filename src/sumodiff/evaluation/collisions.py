"""Frame and between-frame collision checks, with explicit unknown outcomes."""
from dataclasses import dataclass
import math
import numpy as np
from sumodiff.geometry.boxes import box_axes, interpolate_pose, separation_margin, wrap_angle


@dataclass(frozen=True)
class CollisionConfig:
    contact_tolerance_m: float = 1e-9
    rotation_zero_tolerance_rad: float = 1e-12
    rotating_time_resolution_s: float = 1e-4
    max_depth: int = 12
    max_evaluations_per_interval: int = 256

    def __post_init__(self):
        for key in ('contact_tolerance_m', 'rotation_zero_tolerance_rad', 'rotating_time_resolution_s'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('max_depth', 'max_evaluations_per_interval'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')


def _translation_interval(a0, a1, b0, b1, size_a, size_b, tolerance):
    axes_a, axes_b = box_axes(a0), box_axes(b0)
    axes = np.vstack((axes_a, axes_b))
    extent = np.abs(axes @ axes_a.T) @ (np.asarray(size_a)/2) + np.abs(axes @ axes_b.T) @ (np.asarray(size_b)/2) + tolerance
    offset = axes @ (b0[:2] - a0[:2])
    rate = axes @ ((b1[:2]-b0[:2]) - (a1[:2]-a0[:2]))
    low, high = 0., 1.
    for center, derivative, radius in zip(offset, rate, extent):
        if abs(derivative) < 1e-15:
            if abs(center) > radius:
                return None
            continue
        first, last = sorted(((-radius-center)/derivative, (radius-center)/derivative))
        low, high = max(low, first), min(high, last)
        if low > high:
            return None
    return low


def check_interval(a0, a1, b0, b1, size_a, size_b, dt=.1, config=None):
    """Linear centers, shortest-arc yaw interpolation.

    Fixed orientation uses exact swept SAT. Rotating boxes use midpoint SAT and
    a conservative translation+rotation motion bound to certify separation;
    unresolved leaves are unknown, never silently marked collision-free.
    """
    config = config or CollisionConfig()
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Positive dt required')
    delta_a, delta_b = float(wrap_angle(a1[2]-a0[2])), float(wrap_angle(b1[2]-b0[2]))
    if max(abs(delta_a), abs(delta_b)) <= config.rotation_zero_tolerance_rad:
        fraction = _translation_interval(a0,a1,b0,b1,size_a,size_b,config.contact_tolerance_m)
        return dict(contact=fraction is not None, fraction=fraction, method='exact_swept_translation', evaluations=0)
    for fraction in (0., 1.):
        if separation_margin(interpolate_pose(a0,a1,fraction), interpolate_pose(b0,b1,fraction),size_a,size_b) <= config.contact_tolerance_m:
            return dict(contact=True, fraction=fraction, method='rotating_endpoint', evaluations=0)
    relative_motion = np.linalg.norm((b1[:2]-b0[:2])-(a1[:2]-a0[:2]))
    angular_motion = np.linalg.norm(size_a)/2*abs(delta_a) + np.linalg.norm(size_b)/2*abs(delta_b)
    evaluations = 0
    def visit(low, high, depth):
        nonlocal evaluations
        if evaluations >= config.max_evaluations_per_interval:
            return None, None
        middle = (low+high)/2
        evaluations += 1
        a, b = interpolate_pose(a0,a1,middle), interpolate_pose(b0,b1,middle)
        gap = separation_margin(a,b,size_a,size_b)
        if gap <= config.contact_tolerance_m:
            return True, middle
        if gap > (relative_motion+angular_motion)*(high-low)/2 + config.contact_tolerance_m:
            return False, None
        if depth >= config.max_depth or (high-low)*dt <= config.rotating_time_resolution_s:
            return None, None
        left, left_fraction = visit(low,middle,depth+1)
        if left is True:
            return True, left_fraction
        right, right_fraction = visit(middle,high,depth+1)
        if right is True:
            return True, right_fraction
        return (False if left is False and right is False else None), None
    contact, fraction = visit(0.,1.,0)
    return dict(contact=contact, fraction=fraction, method='bounded_rotating_interpolation', evaluations=evaluations)


def _group(pairs):
    if not pairs:
        return dict(pair_count=0, collision=None, observed_collision=None, collision_pair_fraction=None, observed_collision_pair_fraction=None, unknown_pairs=0)
    observed = any(pair['observed_collision'] for pair in pairs)
    unknown = sum(pair['collision'] is None for pair in pairs)
    outcome = True if observed else (None if unknown else False)
    return dict(pair_count=len(pairs), collision=outcome, observed_collision=observed,
                collision_pair_fraction=(sum(pair['observed_collision'] for pair in pairs)/len(pairs) if not unknown else None),
                observed_collision_pair_fraction=sum(pair['observed_collision'] for pair in pairs)/len(pairs), unknown_pairs=unknown)


def evaluate_collisions(poses, sizes, mask, agent_mask, target_pair=None, dt=.1, config=None):
    """poses[N,T+1,3] includes t0; only selected agents are assessed.

    target_pair=(attacker_slot,target_slot), optional and assigned externally.
    Missing labels do not bridge intervals. Rates over partially observed
    trajectories are observed lower bounds and unknown counts are retained.
    """
    config = config or CollisionConfig()
    poses, sizes, mask, agent_mask = np.asarray(poses,float), np.asarray(sizes,float), np.asarray(mask,bool), np.asarray(agent_mask,bool)
    if poses.ndim != 3 or poses.shape[-1] != 3 or sizes.shape != (len(poses),2) or mask.shape != poses.shape[:2] or agent_mask.shape != (len(poses),):
        raise ValueError('Invalid collision input shapes')
    active = np.flatnonzero(agent_mask).tolist()
    if not np.isfinite(poses[mask & agent_mask[:,None]]).all() or not np.isfinite(sizes[agent_mask]).all() or (sizes[agent_mask] <= 0).any():
        raise ValueError('Invalid active collision geometry')
    if target_pair is not None:
        if len(target_pair) != 2 or target_pair[0] == target_pair[1] or any(type(i) is not int or i not in active for i in target_pair):
            raise ValueError('Target task requires two distinct active slots')
        target_pair = tuple(target_pair)
    pairs = []
    for a_index, a in enumerate(active):
        for b in active[a_index+1:]:
            shared = mask[a] & mask[b]
            initial = bool(separation_margin(poses[a,0],poses[b,0],sizes[a],sizes[b]) <= config.contact_tolerance_m) if shared[0] else None
            frame_hits = [t for t in np.flatnonzero(shared) if t > 0 and separation_margin(poses[a,t],poses[b,t],sizes[a],sizes[b]) <= config.contact_tolerance_m]
            hits, unknown, interval_count, calls = [], 0, 0, 0
            for t in range(len(shared)-1):
                if not shared[t] or not shared[t+1]:
                    continue
                result = check_interval(poses[a,t],poses[a,t+1],poses[b,t],poses[b,t+1],sizes[a],sizes[b],dt,config)
                interval_count += 1
                calls += result['evaluations']
                if result['contact'] is True:
                    hits.append(dict(time_s=(t+result['fraction'])*dt, method=result['method']))
                elif result['contact'] is None:
                    unknown += 1
            # The assessed horizon includes the observed start; initial contacts
            # are flagged separately and cannot qualify as newly caused events.
            observed = bool(initial is True or frame_hits or hits)
            earliest = ([dict(time_s=0.,method='initial_frame')] if initial is True else []) + hits + [dict(time_s=t*dt,method='future_frame') for t in frame_hits]
            outcome = True if observed else (None if unknown or not shared.all() else False)
            pairs.append(dict(slots=[a,b], initial_contact=initial, observed_collision=observed, collision=outcome,
                first_detected_contact=min(earliest,key=lambda x:x['time_s']) if earliest else None,
                future_frame_collision=bool(frame_hits), checked_intervals=interval_count,
                missing_intervals=len(shared)-1-interval_count, unresolved_intervals=unknown,
                adaptive_sat_evaluations=calls))
    def is_target(pair):
        return target_pair is not None and set(pair['slots']) == set(target_pair)
    groups = dict(any=_group(pairs), non_target=_group([p for p in pairs if not is_target(p)]),
                  target=_group([p for p in pairs if is_target(p)]))
    if target_pair is None:
        groups.update(background_background=_group([]), attacker_background=_group([]), target_background=_group([]))
    else:
        attacker, target = target_pair
        backgrounds = set(active)-set(target_pair)
        groups.update(background_background=_group([p for p in pairs if set(p['slots']).issubset(backgrounds)]),
            attacker_background=_group([p for p in pairs if attacker in p['slots'] and set(p['slots']) & backgrounds]),
            target_background=_group([p for p in pairs if target in p['slots'] and set(p['slots']) & backgrounds]))
    target_records = [p for p in pairs if is_target(p)]
    new_target = None if not target_records or target_records[0]['collision'] is None or target_records[0]['initial_contact'] is None else bool(target_records[0]['collision'] and not target_records[0]['initial_contact'])
    return dict(schema_version='sumodiff.collisions.v1', groups=groups, pairs=pairs, new_target_event=new_target,
        target_pair=list(target_pair) if target_pair else None,
        checked_intervals=sum(p['checked_intervals'] for p in pairs), unresolved_intervals=sum(p['unresolved_intervals'] for p in pairs),
        adaptive_sat_evaluations=sum(p['adaptive_sat_evaluations'] for p in pairs), contact_includes_touching=True,
        interpolation='linear center, shortest-arc yaw; exact translation or bounded rotation')
