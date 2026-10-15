"""Frozen motion-balanced source recipe; old plans remain byte-compatible."""
from copy import deepcopy
import hashlib,json,math
from sumodiff.simulation.config import resolve_config
from .stage5a_plan import FAMILIES

SCHEMA='sumodiff.stage5a.config.v3'
POLICIES=('first_current_id','interaction_current')
OBSOLETE=('pilot_demand_seconds','pilot_intersection_train_geometry','pilot_intersection_validation_geometry','pilot_intersection_demand_seconds','minimum_turning_fraction')
MOTION=dict(reference_policies=list(POLICIES),reference_fraction=.5,
    minimum_motion_fraction=dict(three_lane_straight=dict(lane_change=.2),
        ramp_merge=dict(ramp_exit=.2,lane_change=.1),unsignalized_intersection=dict(turn=.1)),
    unique_events_train=dict(lane_change_straight=100,ramp_exit=100,lane_change_ramp=50,turn=50),
    unique_events_holdout=dict(lane_change_straight=10,ramp_exit=10,lane_change_ramp=5,turn=5),
    max_windows_per_event=3,one_view_per_episode_tick=True,selection_attempts=4,
    minimum_motion_geometries=3,minimum_motion_episodes=4,
    pilot_geometry_indices=dict(three_lane_straight=[0,1],ramp_merge=[0,1],unsignalized_intersection=[6,7]),
    pilot_reference_ticks=list(range(30,951,40)),pilot_windows_per_family=8,
    pilot_minimum_unique_per_category=1,
    lateral_accel_mps2=.3,lateral_speed_mps=.8,transition_length_m=80.,alignment_length_m=10.)

def resolve_motion_plan(value):
    from .stage5a_plan import resolve_plan
    supplied=deepcopy(value)
    if any(k in supplied for k in OBSOLETE):raise ValueError('Motion v3 replaces legacy pilot/turn options with its explicit motion recipe')
    motion=supplied.pop('motion',{})
    if not isinstance(motion,dict) or set(motion)-set(MOTION):raise ValueError('Unknown motion-data options')
    resolved=deepcopy(MOTION);resolved.update(motion)
    # The first recipe is fixed before any holdout data are observed.
    if json.dumps(resolved,sort_keys=True)!=json.dumps(MOTION,sort_keys=True) or type(resolved['max_windows_per_event']) is not int or type(resolved['selection_attempts']) is not int or type(resolved['one_view_per_episode_tick']) is not bool:
        raise ValueError('Motion coverage recipe v1 is frozen; changes need a new protocol')
    supplied['schema_version']='sumodiff.stage5a.config.v2'
    base=resolve_plan(supplied)
    for key in ('train_geometries','train_seeds_per_geometry','validation_geometries','test_geometries',
                'validation_seeds_per_geometry','test_seeds_per_geometry'):
        expected=dict(train_geometries=8,train_seeds_per_geometry=6,validation_geometries=3,test_geometries=3,
                      validation_seeds_per_geometry=4,test_seeds_per_geometry=4)[key]
        if base[key]!=expected:raise ValueError('Motion source geometry/episode budget is frozen')
    if any(base[s+'_quotas']!={f:(2000 if s=='train' else 200) for f in FAMILIES} for s in ('train','validation','test')):
        raise ValueError('Motion recipe retains 6000/600/600 window budgets')
    if base['demand_seconds']!=180 or base['max_episode_seconds']!=600:
        raise ValueError('Full source duration is frozen')
    for family,indices in resolved['pilot_geometry_indices'].items():
        if any(type(i) is not int or not 0<=i<base['train_geometries'] for i in indices):
            raise ValueError('Pilot geometry must belong to train')
    for key in OBSOLETE:base.pop(key,None)
    base.update(schema_version=SCHEMA,motion=resolved)
    return base

def base_plan(c):
    result=deepcopy(c);result.pop('motion');result['schema_version']='sumodiff.stage5a.config.v2'
    return result

def motion_scenario(c,family,split,index,seed_index,pilot=False):
    from .stage5a_plan import scenario
    result=scenario(base_plan(c),family,split,index,seed_index,pilot=False)
    if family in ('three_lane_straight','ramp_merge'):
        result['driver'].update(lane_change_model='SL2015',lcAccelLat=.3,maxSpeedLat=.8)
        result['simulation'].update(lane_change_duration=0.,lateral_resolution=result['geometry']['lane_width']/4)
    if family=='ramp_merge':
        result['schema_version']='sumodiff.scenario.v2'
        result['geometry'].update(transition_length=80.,alignment_length=10.)
        result['network']['protect_acceleration_lane']=True
    if pilot:
        result['traffic']['end_seconds']=30.
        result['simulation']['max_seconds']=240.
    return resolve_config(result)

def motion_jobs(c,pilot=False):
    from .stage5a_plan import jobs
    if not pilot:
        result=jobs(base_plan(c))
        for job in result:
            # Indices are encoded by the stable original job ID.
            parts=job['job_id'].split('-');gi=int(parts[-2][1:]);si=int(parts[-1][1:])
            job['job_id']='motion-'+job['job_id']
            job['scenario']=motion_scenario(c,job['scenario']['family'],job['split'],gi,si)
        return result
    result=[]
    for fi,family in enumerate(FAMILIES):
        for gi in c['motion']['pilot_geometry_indices'][family]:
            seed=(c['seed']+fi*10000+gi*100)%2**32
            result.append(dict(job_id=f'motion-train-{family}-g{gi:02d}-s00',split='train',seed=seed,
                geometry_slot=f'train-{family}-g{gi:02d}',scenario=motion_scenario(c,family,'train',gi,0,True)))
    return result

def motion_protocol_id(c):
    return 'sha256:'+hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def requirements(c,split,family,pilot=False):
    quota=c['motion']['pilot_windows_per_family'] if pilot else c[split+'_quotas'][family]
    fractions=c['motion']['minimum_motion_fraction'][family]
    unique=c['motion']['unique_events_train' if split=='train' else 'unique_events_holdout']
    aliases=dict(lane_change='lane_change_straight' if family=='three_lane_straight' else 'lane_change_ramp',ramp_exit='ramp_exit',turn='turn')
    return dict(quota=quota,policy_quotas={p:quota//2 for p in POLICIES},
        minimum_windows={k:(1 if pilot else math.ceil(quota*v)) for k,v in fractions.items()},
        minimum_unique={k:(1 if pilot else unique[aliases[k]]) for k in fractions},
        minimum_geometries=2 if pilot else c['minimum_geometries_per_family'],
        minimum_episodes=2 if pilot else c['minimum_episodes_per_family'],
        minimum_motion_geometries=1 if pilot else c['motion']['minimum_motion_geometries'],
        minimum_motion_episodes=1 if pilot else c['motion']['minimum_motion_episodes'],
        max_episode_windows=math.ceil(quota*(.6 if pilot else c['max_episode_fraction'])),
        max_windows_per_event=c['motion']['max_windows_per_event'],attempts=c['motion']['selection_attempts'])
