"""Label-only quality audit; completeness, physical checks and conversions are distinct."""
from collections import defaultdict, Counter, deque
import hashlib
import math
import numpy as np
from shapely.geometry import mapping
from sumodiff.evaluation.motion import MotionConfig, evaluate_motion
from sumodiff.evaluation.roads import RoadConfig, evaluate_roads
from sumodiff.evaluation.collisions import CollisionConfig
from sumodiff.evaluation.metrics import evaluate_trajectory, _gate


def independent_state(episode, vehicle_id, tick, origin, rotation):
    """Rebuild directly from raw front bumper/nav angle, without the production converter."""
    if tick < 1 or tick >= len(episode.frames):
        return None
    observations = [episode.frames[t].get(vehicle_id) for t in (tick-1,tick)]
    if any(o is None for o in observations):
        return None
    centers = []
    for obs in observations:
        alpha = math.radians(obs.raw['navigation_angle_deg'])
        direction = np.array([math.sin(alpha),math.cos(alpha)])
        centers.append(np.asarray(obs.raw['front_position_m']) - obs.raw['length_m']*.5*direction)
    obs = observations[-1]
    direction = rotation @ np.array([math.sin(alpha),math.cos(alpha)])
    return np.r_[rotation @ (centers[-1]-origin), rotation @ ((centers[-1]-centers[0])/episode.dt),
                 direction[1],direction[0]]


def compact_metrics(m):
    return dict(quality_pass=m['quality_pass'],roads=m['roads'],
        motion=dict(violation_counts=m['motion']['violation_counts'],hard_quality_pass=m['motion']['hard_quality_pass'],
          speed_mps=m['motion']['speed_mps'],accel_norm_mps2=m['motion']['accel_norm_mps2'],
          jerk_norm_mps3=m['motion']['jerk_norm_mps3'],boundary=m['motion']['boundary'],
          comfort_pass_fraction=m['motion']['observed_comfort_pass_fraction_agents']),
        collision=m['collisions']['groups']['non_target'])


def audit_window(window, episode, road, plan):
    c=window.conditioning;meta=window.input_metadata;agents=c['agent_mask']
    row=dict(window_id=meta['window_id'],family=meta['family'],split=meta['split'],
        episode_key=meta['episode_key'],geometry_id=meta['geometry_id'],reference_tick=meta['reference_tick'],
        selected_agents=int(agents.sum()),core_complete=window.label_metadata['core_training_eligible'],
        eligible=False,reasons=[],coordinate_errors=None)
    if not row['core_complete']:
        row['reasons'].append('incomplete_history_or_future')
        return row
    if row['selected_agents'] < plan['minimum_agents']:
        row['reasons'].append('too_few_agents')
    t0=meta['reference_tick'];start=t0-20
    # Three position-prefix points define jerk at the FIRST checked history point.
    ticks=range(start-3,t0+41)
    ids=meta['agent_ids'];n=len(ids)
    canonical=np.zeros((n,len(ticks),6))
    origin=np.asarray(meta['fixed_frame']['origin_world_m'])
    yaw=meta['fixed_frame']['yaw_world_rad']
    rotation=np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])
    for a in np.flatnonzero(agents):
        for j,tick in enumerate(ticks):
            value=independent_state(episode,ids[a],tick,origin,rotation)
            if value is None:
                row['reasons'].append('missing_quality_derivative_context')
                return row
            canonical[a,j]=value
    h=canonical[:,3:24];future=canonical[:,24:]
    saved_h=c['history'].astype(float)
    saved_f=window.targets['future'].astype(float).copy()
    saved_f[...,:2]+=c['initial_positions'][:,None,:]
    all_canonical=np.concatenate((h,future),axis=1)
    all_saved=np.concatenate((saved_h,saved_f),axis=1)
    err=np.abs(all_saved[agents]-all_canonical[agents])
    errors=dict(position_m=float(err[...,:2].max()),velocity_mps=float(err[...,2:4].max()),
                heading=float(err[...,4:6].max()))
    row['coordinate_errors']=errors
    limits=[('position_m','position_roundtrip_tolerance_m'),('velocity_mps','velocity_roundtrip_tolerance_mps'),
            ('heading','heading_roundtrip_tolerance')]
    if any(errors[k]>plan[threshold] for k,threshold in limits):
        row['reasons'].append('coordinate_or_precision_mismatch')
    ev=plan['evaluation']
    kwargs=dict(motion_config=MotionConfig(**ev['motion']),road_config=RoadConfig(**ev['road']),
                collision_config=CollisionConfig(**ev['collision']))
    # Evaluate all 21 history+40 future points, not just future labels.
    pc=dict(c,history=canonical[:,:3],history_mask=np.broadcast_to(agents[:,None],(n,3)).copy(),
            initial_positions=canonical[:,2,:2],initial_heading=canonical[:,2,4:6])
    raw=evaluate_trajectory(pc,meta,window.exact_map,canonical[:,3:],**kwargs)
    poses=np.concatenate((all_canonical[...,:2],
        np.arctan2(all_canonical[...,4],all_canonical[...,5])[...,None]),axis=-1)
    mask=np.broadcast_to(agents[:,None],poses.shape[:2]).copy()
    whole=evaluate_roads(poses,c['attributes'][:,:2],mask,agents,window.exact_map,meta['map_extent_m'],kwargs['road_config'])
    raw['roads']=whole
    # Quality prefix is artificial; reported boundary is the actual history/future t0.
    raw['motion']['boundary']=evaluate_motion(h,c['history_mask'],future,
        np.broadcast_to(agents[:,None],future.shape[:2]),c['initial_positions'],h[:,-1,4:6],
        agents,episode.dt,kwargs['motion_config'])['boundary']
    raw['quality_pass']=_gate([raw['motion']['hard_quality_pass'],whole['geometry_quality_pass'],
        not raw['collisions']['groups']['non_target']['collision'] if raw['collisions']['groups']['non_target']['collision'] is not None else None])
    row['raw64']=compact_metrics(raw)
    # Preserve the production float32 label checks too; neither result is repaired/smoothed.
    sc=dict(pc,history=canonical[:,:3].copy())
    stored=evaluate_trajectory(sc,meta,window.exact_map,all_saved,**kwargs)
    stored['motion']['boundary']=evaluate_motion(saved_h,c['history_mask'],saved_f,
        np.broadcast_to(agents[:,None],future.shape[:2]),c['initial_positions'],saved_h[:,-1,4:6],
        agents,episode.dt,kwargs['motion_config'])['boundary']
    stored_poses=np.concatenate((all_saved[...,:2],np.arctan2(all_saved[...,4],all_saved[...,5])[...,None]),axis=-1)
    stored['roads']=evaluate_roads(stored_poses,c['attributes'][:,:2],mask,agents,window.exact_map,meta['map_extent_m'],kwargs['road_config'])
    stored['quality_pass']=_gate([stored['motion']['hard_quality_pass'],stored['roads']['geometry_quality_pass'],
        not stored['collisions']['groups']['non_target']['collision'] if stored['collisions']['groups']['non_target']['collision'] is not None else None])
    row['stored32']=compact_metrics(stored)
    world_xy=all_canonical[...,:2] @ rotation + origin
    world_yaw=np.arctan2(all_canonical[...,4],all_canonical[...,5])+yaw
    world_poses=np.concatenate((world_xy,world_yaw[...,None]),axis=-1)
    world_map=dict(drivable=mapping(road.drivable),
        route_corridors=[mapping(road.route_area(route)) if route is not None else None for route in meta['planned_routes']])
    world=evaluate_roads(world_poses,c['attributes'][:,:2],mask,agents,world_map,[-1e7,-1e7,1e7,1e7],kwargs['road_config'])
    row['world_local_geometry_counts_match']=all(world[k]['violation_body_frames']==whole[k]['violation_body_frames'] for k in ('road','route'))
    if not row['world_local_geometry_counts_match']:
        row['reasons'].append('world_local_geometry_mismatch')
    for name,result in [('raw64',raw),('stored32',stored)]:
        if result['motion']['hard_quality_pass'] is not True: row['reasons'].append(name+':motion')
        for kind in ('road','route','map_coverage'):
            if result['roads'][kind]['violation_body_frames']: row['reasons'].append(name+':'+kind)
        if result['collisions']['groups']['non_target']['collision'] is not False:
            row['reasons'].append(name+':collision_or_unknown')
    permitted=[road.route_lane_ids(route) if route is not None else set() for route in meta['planned_routes']]
    row['assigned_lane_outside_known_route_frames']=sum(episode.frames[t][ids[a]].raw['lane_id'] not in permitted[a]
        for a in np.flatnonzero(agents) for t in range(start,t0+41))
    row['actual_turning_agents']=sum(float(np.ptp(np.unwrap(world_yaw[a])))>.15 for a in np.flatnonzero(agents))
    # Peak source attribution: compare vector front/center and reported scalar acceleration.
    source_peaks=[]
    for a in np.flatnonzero(agents):
        series=np.array([episode.frames[t][ids[a]].raw['front_position_m'] for t in ticks])
        front_jerks=np.linalg.norm(np.diff(series,n=3,axis=0),axis=-1)/episode.dt**3
        center_jerks=np.linalg.norm(np.diff(canonical[a,:,:2],n=3,axis=0),axis=-1)/episode.dt**3
        k=int(np.argmax(center_jerks));tick=list(ticks)[k+3]
        o=episode.frames[tick][ids[a]];prev=episode.frames[tick-1][ids[a]]
        source_peaks.append(dict(vehicle_id=ids[a],tick=tick,road_id=o.raw['road_id'],lane_id=o.raw['lane_id'],
          center_jerk_mps3=float(center_jerks[k]),front_jerk_mps3=float(front_jerks[k]),
          reported_scalar_jerk_mps3=abs(o.raw['acceleration_mps2']-prev.raw['acceleration_mps2'])/episode.dt))
    row['peak_source_attribution']=max(source_peaks,key=lambda v:v['center_jerk_mps3'])
    row['eligible']=not row['reasons']
    return row


def select_balanced(rows, quotas, seed, min_geometries, min_episodes, max_episode_fraction):
    """Round-robin geometry and episode, hashed time order, with explicit insufficiency."""
    chosen=[];summary={}
    for split,by_family in quotas.items():
        summary[split]={}
        for family,quota in by_family.items():
            pool=[r for r in rows if r['split']==split and r['family']==family and r['eligible']]
            grouped=defaultdict(lambda:defaultdict(list))
            for row in pool: grouped[row['geometry_id']][row['episode_key']].append(row)
            groups={}
            for g,eps in sorted(grouped.items()):
                groups[g]={e:deque(sorted(v,key=lambda r:hashlib.sha256(f"{seed}:{r['window_id']}".encode()).digest())) for e,v in sorted(eps.items())}
            epcounts=Counter();selected=[];cap=math.ceil(quota*max_episode_fraction)
            while len(selected)<quota:
                progress=False
                for g,eps in groups.items():
                    available=[e for e,q in eps.items() if q and epcounts[e]<cap]
                    if not available: continue
                    e=min(available,key=lambda e:(epcounts[e],e))
                    selected.append(eps[e].popleft());epcounts[e]+=1;progress=True
                    if len(selected)==quota: break
                if not progress: break
            geos={r['geometry_id'] for r in selected};episodes={r['episode_key'] for r in selected}
            passed=len(selected)==quota and len(geos)>=min_geometries and len(episodes)>=min_episodes
            summary[split][family]=dict(requested=quota,eligible_pool=len(pool),selected=len(selected),
                selected_geometry_groups=len(geos),selected_episodes=len(episodes),episode_counts=dict(epcounts),
                max_windows_per_episode=cap,passed=passed)
            chosen.extend(selected)
    return chosen,summary
