"""Predeclared stage5A geometry/split/seed plan; quotas are not convergence claims."""
from copy import deepcopy
import hashlib
import json
import math
from sumodiff.data.config import resolve_config as window_config
from sumodiff.evaluation.config import resolve_config as evaluation_config
from sumodiff.simulation.config import resolve_config as scenario_config, strict_merge

FAMILIES = ('three_lane_straight', 'ramp_merge', 'unsignalized_intersection')
DEFAULTS = dict(schema_version='sumodiff.stage5a.config.v1', seed=20261010,
    train_geometries=5, validation_geometries=2, train_seeds_per_geometry=4, validation_seeds_per_geometry=3,
    demand_seconds=120., pilot_demand_seconds=18., max_episode_seconds=600.,
    train_quotas={f:400 for f in FAMILIES},
    validation_quotas=dict(three_lane_straight=67, ramp_merge=67, unsignalized_intersection=66),
    minimum_agents=2, minimum_geometries_per_family=2, minimum_episodes_per_family=4,
    max_episode_fraction=.25, selection_seed=20261011,
    pilot_minimum_eligible_per_family=3, pilot_minimum_retention_fraction=.1,
    position_roundtrip_tolerance_m=.0001, velocity_roundtrip_tolerance_mps=.0001,
    heading_roundtrip_tolerance=.000002, window={}, evaluation={},
    driver=dict(car_follow_model='IDM', sigma=0., accel=2., decel=3.5, tau=1.5,
                minGap=3., speedFactor=.95, speedDev=.08),
    runtime=dict(sumo_home=None, client_source='sumo_home_tools'))


def resolve_plan(value):
    supplied = deepcopy(value)
    supplied.pop('window', None); supplied.pop('evaluation', None)
    c = strict_merge(DEFAULTS, supplied, 'stage5a')
    if c['schema_version'] != DEFAULTS['schema_version']:
        raise ValueError('Unsupported stage5A schema')
    for k in ('seed','selection_seed'):
        if type(c[k]) is not int or not 0 <= c[k] < 2**32: raise ValueError(f'Invalid {k}')
    for k in ('train_geometries','validation_geometries','train_seeds_per_geometry','validation_seeds_per_geometry',
              'minimum_agents','minimum_geometries_per_family','minimum_episodes_per_family','pilot_minimum_eligible_per_family'):
        if type(c[k]) is not int or c[k] < 1: raise ValueError(f'Invalid {k}')
    if not 2 <= c['minimum_agents'] <= 12: raise ValueError('Stage5A requires at least two agents')
    if c['minimum_geometries_per_family'] < 2 or min(c['train_geometries'], c['validation_geometries']) < c['minimum_geometries_per_family']:
        raise ValueError('Each full split/family needs multiple geometry groups')
    if c['train_geometries'] > 10 or c['validation_geometries'] > 10:
        raise ValueError('This finite geometry generator supports <=10 variants per split')
    for k in ('demand_seconds','pilot_demand_seconds','max_episode_seconds',
              'position_roundtrip_tolerance_m','velocity_roundtrip_tolerance_mps','heading_roundtrip_tolerance'):
        if type(c[k]) not in (int,float) or not math.isfinite(c[k]) or c[k] <= 0: raise ValueError(f'Invalid {k}')
    if c['pilot_demand_seconds'] > 30 or c['max_episode_seconds'] <= c['demand_seconds']:
        raise ValueError('Pilot is limited to 30s demand; full episodes need a clearance budget')
    for k in ('max_episode_fraction','pilot_minimum_retention_fraction'):
        if type(c[k]) not in (int,float) or not 0 < c[k] <= 1: raise ValueError(f'Invalid {k}')
    for k in ('train_quotas','validation_quotas'):
        if set(c[k]) != set(FAMILIES) or any(type(v) is not int or v < 1 for v in c[k].values()):
            raise ValueError('Each split requires three positive integer family quotas')
    c['window'] = window_config(value.get('window', {}))
    c['evaluation'] = evaluation_config(value.get('evaluation', {}))
    c['evaluation']['cuda_smoke'] = False
    # Validate actual vehicle parameters, rather than accepting unused YAML keys.
    scenario_config(dict(family='three_lane_straight', driver=c['driver'], runtime=c['runtime']))
    return c


def protocol_id(c):
    # Changing bulk budget/seed count/quotas does not alter the label-quality pilot protocol.
    keys = ('window','evaluation','driver','runtime','minimum_agents','pilot_demand_seconds',
            'pilot_minimum_eligible_per_family','pilot_minimum_retention_fraction',
            'position_roundtrip_tolerance_m','velocity_roundtrip_tolerance_mps','heading_roundtrip_tolerance')
    payload = dict(version='stage5a.geometry.v1', **{k:c[k] for k in keys})
    return 'sha256:' + hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def scenario(c, family, split, index, seed_index, *, pilot=False):
    # Split difference changes actual shape/width, not merely seed or filename.
    j = index * 2 + (1 if split == 'validation' else 0)
    if family == 'three_lane_straight':
        geometry = dict(core_length=400.+25*j, entry_buffer=300., exit_buffer=300.,
                        lane_width=3.5+.02*j, speed_limit=16., rotation_degrees=3.*j,
                        segmented_straight=False)
        rates = dict(main=1100.+100*(seed_index%3))
    elif family == 'ramp_merge':
        geometry = dict(approach_length=350.+15*j, merge_length=180.+12*j, exit_buffer=350.,
                        ramp_offset=45.+2*j, lane_width=3.5+.02*j, speed_limit=14., rotation_degrees=3.*j)
        rates = dict(main=850.+100*(seed_index%3), ramp=350.+50*(seed_index%3))
    else:
        geometry = dict(approach_length=300.+12*j, lane_width=3.5+.02*j, speed_limit=8.,
                        north_skew_degrees=(-1)**j*(2.+j), rotation_degrees=3.*j, junction_radius=14.+j)
        names = ('west_east','east_west','north_south','south_north',
                 'west_north','north_east','east_south','south_west',
                 'west_south','south_east','east_north','north_west')
        rates = {name:100. if k<4 else 60.+10*(seed_index%3) for k,name in enumerate(names)}
    driver = deepcopy(c['driver'])
    driver['tau'] += .1*(seed_index%3)
    driver['speedFactor'] -= .03*(seed_index%3)
    return scenario_config(dict(family=family,geometry=geometry,driver=driver,runtime=c['runtime'],
        traffic=dict(end_seconds=c['pilot_demand_seconds'] if pilot else c['demand_seconds'],route_rates_per_hour=rates),
        simulation=dict(max_seconds=c['max_episode_seconds'],lane_change_duration=3.)))


def jobs(c, pilot=False):
    result = []
    for split in ('train','validation'):
        ng = 1 if pilot else c[f'{split}_geometries']
        ns = 1 if pilot else c[f'{split}_seeds_per_geometry']
        for fi, family in enumerate(FAMILIES):
            for gi in range(ng):
                for si in range(ns):
                    # Non-overlapping seeds across split/family/geometry/repetition.
                    seed = (c['seed'] + (100000 if split=='validation' else 0) + fi*10000+gi*100+si) % 2**32
                    name = f'{split}-{family}-g{gi:02d}-s{si:02d}'
                    result.append(dict(job_id=name,split=split,seed=seed,geometry_slot=f'{split}-{family}-g{gi:02d}',
                        scenario=scenario(c,family,split,gi,si,pilot=pilot)))
    return result
