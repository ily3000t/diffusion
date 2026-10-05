"""Independent scene metrics and explicit rates, without a scenario score."""
from dataclasses import asdict
import numpy as np
from .motion import MotionConfig, evaluate_motion, summary
from .roads import RoadConfig, evaluate_roads
from .collisions import CollisionConfig, evaluate_collisions


def _heading(conditioning, explicit=None):
    agents = np.asarray(conditioning['agent_mask'],bool)
    if explicit is not None:
        return np.asarray(explicit,float)
    if 'initial_heading' in conditioning:
        return np.asarray(conditioning['initial_heading'],float)
    hm = np.asarray(conditioning['history_mask'],bool)
    if not hm[:,-1][agents].all():
        raise ValueError('Missing t0 heading: supply its currently observed value explicitly')
    return np.asarray(conditioning['history'],float)[:,-1,4:6]


def _gate(values):
    if any(value is False for value in values):
        return False
    return True if all(value is True for value in values) else None


def applicability(agents, target_pair):
    count = int(np.asarray(agents,bool).sum())
    pairs = count*(count-1)//2
    has_target = target_pair is not None
    return dict(any_pair=pairs>0, non_target_pair=pairs-int(has_target)>0, target_pair=has_target,
        background_background=has_target and count>=4, attacker_background=has_target and count>=3,
        target_background=has_target and count>=3)


def evaluate_trajectory(conditioning, input_metadata, exact_map, states, *, future_mask=None,
                        initial_heading=None, target_pair=None, motion_config=None, road_config=None, collision_config=None):
    agents = np.asarray(conditioning['agent_mask'],bool)
    states = np.asarray(states,float)
    if states.ndim != 3 or states.shape[0] != len(agents) or states.shape[-1] != 6:
        raise ValueError('Physical states must have shape [N,T,6]')
    if not agents.any():
        raise ValueError('A trajectory task must have at least one selected agent')
    fm = np.broadcast_to(agents[:,None],states.shape[:2]).copy() if future_mask is None else np.asarray(future_mask,bool)&agents[:,None]
    if fm.shape != states.shape[:2] or not np.isfinite(states[fm]).all():
        raise ValueError('Invalid active trajectory or future mask')
    initial = np.asarray(conditioning['initial_positions'],float)
    heading = _heading(conditioning,initial_heading)
    if initial.shape != (len(agents),2) or heading.shape != (len(agents),2):
        raise ValueError('Invalid currently observed pose shape')
    if not np.isfinite(initial[agents]).all() or not np.isfinite(heading[agents]).all() or (np.linalg.norm(heading[agents],axis=-1)<1e-6).any():
        raise ValueError('Invalid currently observed pose')
    motion = evaluate_motion(conditioning['history'],conditioning['history_mask'],states,fm,initial,heading,agents,
        input_metadata['dt_seconds'],motion_config)
    poses = np.concatenate((states[...,:2],np.arctan2(states[...,4],states[...,5])[...,None]),axis=-1)
    geometric_mask = fm & (np.linalg.norm(states[...,4:6],axis=-1)>=1e-6)
    current = np.concatenate((initial,np.arctan2(heading[:,0],heading[:,1])[:,None]),axis=-1)
    all_poses = np.concatenate((current[:,None],poses),axis=1)
    all_mask = np.concatenate((agents[:,None],geometric_mask),axis=1)
    sizes = np.asarray(conditioning['attributes'],float)[:,:2]
    collisions = evaluate_collisions(all_poses,sizes,all_mask,agents,target_pair,input_metadata['dt_seconds'],collision_config)
    roads = evaluate_roads(poses,sizes,geometric_mask,agents,exact_map,input_metadata['map_extent_m'],road_config)
    non_target = collisions['groups']['non_target']
    safety_pass = True if non_target['pair_count']==0 else (None if non_target['collision'] is None else not non_target['collision'])
    quality = _gate([motion['hard_quality_pass'],roads['geometry_quality_pass'],safety_pass])
    event = collisions['new_target_event']
    effective = None if event is None or quality is None else bool(event and quality)
    return dict(schema_version='sumodiff.scene.metrics.v1',status='completed',
        window_id=input_metadata.get('window_id'),planned_active_agents=int(agents.sum()),
        excluded_vehicle_count=input_metadata.get('excluded_vehicle_count'),task_applicability=applicability(agents,target_pair),
        target_pair=list(target_pair) if target_pair is not None else None,
        collisions=collisions,roads=roads,motion=motion,quality_pass=quality,effective_target_event=effective,
        quality_conditions=['candidate hard motion and consistency checks','whole-body road/route/map coverage','no non-target collision'],
        scope='selected agents; offline interpolation; no dynamics execution or closed-loop attack')


def correction_summary(decoded):
    mask = decoded.agent_mask.detach().cpu().numpy()
    def scalar_values(tensor):
        return tensor.detach().cpu().numpy()
    return dict(schema_version='sumodiff.decoder.corrections.v1',
        position_correction_m=summary(np.linalg.norm(scalar_values(decoded.position_correction),axis=-1)[mask]),
        velocity_correction_mps=summary(np.linalg.norm(scalar_values(decoded.velocity_correction),axis=-1)[mask]),
        heading_vector_correction=summary(np.linalg.norm(scalar_values(decoded.heading_correction),axis=-1)[mask]),
        heading_angle_correction_rad=summary(np.abs(scalar_values(decoded.heading_angle_correction))[scalar_values(decoded.heading_angle_valid)]),
        raw_velocity_residual_mps=summary(np.linalg.norm(scalar_values(decoded.raw_velocity_residual),axis=-1)[mask]),
        raw_heading_norm_error=summary(scalar_values(decoded.raw_heading_norm_error)[mask]),
        heading_fallback_count=int(scalar_values(decoded.heading_fallback_mask).sum()),
        selected_agent_count=int(mask.sum()))


def aggregate_scenes(records):
    """All planned applicable tasks stay in denominators, including failures.

    Unknown/failed tasks contribute no confirmed success and are reported.
    This is an observed lower-bound event rate, never an estimate imputed from
    dropped invalid samples. No applicable task => None, including zero pairs.
    """
    total = len(records)
    result = dict(planned_scenes=total,failed_scenes=sum(r['status']=='failed' for r in records),
        quality_pass_rate=sum(r.get('quality_pass') is True for r in records)/total if total else None,
        quality_unknown_scenes=sum(r.get('quality_pass') is None for r in records),event_rates={})
    names = {'any':'any_pair','target':'target_pair','non_target':'non_target_pair',
             'background_background':'background_background','attacker_background':'attacker_background','target_background':'target_background'}
    for name, eligibility in names.items():
        applicable = [r for r in records if r['task_applicability'][eligibility]]
        outcomes = [r.get('collisions',{}).get('groups',{}).get(name,{}).get('collision') for r in applicable]
        result['event_rates'][name] = dict(denominator=len(applicable),confirmed_positive=sum(x is True for x in outcomes),
            confirmed_negative=sum(x is False for x in outcomes),unknown_or_failed=sum(x is None for x in outcomes),
            confirmed_event_rate=sum(x is True for x in outcomes)/len(applicable) if applicable else None)
    target_records = [r for r in records if r['task_applicability']['target_pair']]
    result['effective_target_event_rate'] = sum(r.get('effective_target_event') is True for r in target_records)/len(target_records) if target_records else None
    result['effective_target_denominator'] = len(target_records)
    return result
