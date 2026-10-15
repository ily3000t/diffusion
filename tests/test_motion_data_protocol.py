from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from sumodiff.experiments.recorder import load_config,file_identity
from sumodiff.experiments.stage5a_plan import resolve_plan,jobs,protocol_id
from sumodiff.experiments.motion_data_plan import requirements,POLICIES
from sumodiff.data.motion_selection import select_motion,MotionIndex
from sumodiff.experiments.motion_data import window_signature,verify_sources

def plan():
    return resolve_plan(load_config(Path(__file__).parents[1]/'configs/experiments/motion_data.yaml'))

def test_recipe_preserves_full_budget_and_preassigns_three_disjoint_splits():
    c=plan();schedule=jobs(c);pilot=jobs(c,True)
    assert len(schedule)==216 and Counter(j['split'] for j in schedule)==dict(train=144,validation=36,test=36)
    assert len(pilot)==6 and {j['split'] for j in pilot}=={'train'}
    assert len({j['seed'] for j in schedule})==216
    geos={s:{json.dumps(j['scenario']['geometry'],sort_keys=True) for j in schedule if j['split']==s} for s in ('train','validation','test')}
    assert all(not geos[a]&geos[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    assert all(j['scenario']['traffic']['end_seconds']==30 and j['scenario']['simulation']['max_seconds']==240 for j in pilot)
    for j in schedule:
        family=j['scenario']['family']
        assert j['scenario']['driver']['car_follow_model']=='IDM'
        if family!='unsignalized_intersection':assert j['scenario']['driver']['lane_change_model']=='SL2015'
        if family=='ramp_merge':
            assert j['scenario']['geometry']['alignment_length']==10
            assert j['scenario']['network']['protect_acceleration_lane'] is True
    r=requirements(c,'train','ramp_merge')
    assert r['quota']==2000 and r['policy_quotas']==dict.fromkeys(POLICIES,1000)
    assert r['minimum_windows']==dict(ramp_exit=400,lane_change=200)
    assert requirements(c,'validation','unsignalized_intersection')['minimum_windows']['turn']==20

@pytest.mark.parametrize('key,value',[('max_windows_per_event',4),('pilot_minimum_unique_per_category',True),('reference_fraction',.7),('selection_attempts',0)])
def test_motion_contract_changes_require_an_explicit_new_recipe(key,value):
    c=load_config(Path(__file__).parents[1]/'configs/experiments/motion_data.yaml')
    c['motion']={key:value}
    with pytest.raises(ValueError,match='frozen'):resolve_plan(c)

def test_obsolete_pilot_knobs_and_wrong_quota_are_not_silently_ignored():
    c=load_config(Path(__file__).parents[1]/'configs/experiments/motion_data.yaml')
    c['minimum_turning_fraction']=.9
    with pytest.raises(ValueError,match='replaces'):resolve_plan(c)
    c.pop('minimum_turning_fraction');c['train_quotas']['ramp_merge']=400
    with pytest.raises(ValueError,match='6000'):resolve_plan(c)

def test_protocol_identity_binds_selection_and_quality_and_old_v2_remains_unchanged():
    c=plan();other=deepcopy(c);other['selection_seed']+=1
    assert protocol_id(c)!=protocol_id(other)
    other=deepcopy(c);other['evaluation']['motion']['max_jerk_mps3']+=1
    assert protocol_id(c)!=protocol_id(other)
    legacy=resolve_plan(load_config(Path(__file__).parents[1]/'configs/experiments/stage5a.yaml'))
    assert legacy['schema_version']=='sumodiff.stage5a.config.v2' and 'motion' not in legacy
    assert len(jobs(legacy,True))==6 and {j['split'] for j in jobs(legacy,True)}=={'train','validation'}

def selection_fixture():
    rows=[]
    for geo in range(2):
        for episode in range(2):
            ep=f'g{geo}e{episode}'
            for tick in range(20):
                tokens=[ep+':'+str(tick//3)] if 3<=tick<12 else []
                for policy in POLICIES:
                    rows.append(dict(window_id=f'{ep}:{tick}:{policy}',episode_key=ep,geometry_id=f'g{geo}',reference_tick=tick,
                        reference_policy=policy,eligible=True,motion_event_ids=dict(lane_change=tokens,ramp_exit=[],turn=[])))
    req=dict(quota=16,policy_quotas=dict.fromkeys(POLICIES,8),minimum_windows=dict(lane_change=4),
        minimum_unique=dict(lane_change=3),minimum_geometries=2,minimum_episodes=4,
        minimum_motion_geometries=2,minimum_motion_episodes=2,max_episode_windows=4,max_windows_per_event=2,attempts=4)
    return rows,req

def test_selection_is_stable_and_respects_views_policy_episode_and_event_caps():
    rows,req=selection_fixture();selected,report=select_motion(rows,req,21)
    assert report['passed'] and len(selected)==16
    reordered,again=select_motion(list(reversed(rows)),req,21)
    assert [r['window_id'] for r in selected]==[r['window_id'] for r in reordered]
    assert report==again
    assert Counter(r['reference_policy'] for r in selected)==dict.fromkeys(POLICIES,8)
    assert len({(r['episode_key'],r['reference_tick']) for r in selected})==16
    assert max(Counter(r['episode_key'] for r in selected).values())<=4
    counts=Counter(e for r in selected for e in r['motion_event_ids']['lane_change'])
    assert max(counts.values())<=2 and len(counts)>=3

def test_quota_failure_retains_actual_counts_without_repeat_or_quiet_relaxation():
    rows,req=selection_fixture()
    for r in rows:r['motion_event_ids']['lane_change']=[]
    selected,report=select_motion(rows,req,21)
    assert not report['passed'] and report['categories']['lane_change']['eligible_unique']==0
    assert report['categories']['lane_change']['selected_unique']==0
    assert len({r['window_id'] for r in selected})==len(selected)
    assert report['requirement']==req
    for r in rows:r['eligible']=False
    selected,report=select_motion(rows,req,21)
    assert not selected and not report['passed']

def test_duplicate_candidates_are_rejected():
    rows,req=selection_fixture()
    with pytest.raises(ValueError,match='Duplicate'):select_motion(rows+[rows[0]],req,21)

def test_window_signature_binds_labels_and_conditions_without_file_timestamps(tmp_path):
    from test_stage5a import synthetic_window
    w,_,_,_=synthetic_window(tmp_path)
    original=window_signature(w);copy=deepcopy(w)
    copy.conditioning=dict(reversed(list(copy.conditioning.items())))
    assert window_signature(copy)==original
    copy.targets['future'][0,0,0]+=.1
    assert window_signature(copy)!=original
    copy=deepcopy(w);copy.input_metadata['reference_policy']='interaction_current'
    assert window_signature(copy)!=original

def raw_fixture(tmp_path,family,turn=False,internal=True):
    frames=[]
    for tick in range(61):
        vehicles=[]
        for vid in ('main','ramp') if family=='ramp_merge' else ('turner',):
            route=['ramp_in','acceleration','main_out'] if vid=='ramp' else ['main_in','acceleration','main_out']
            road='acceleration' if tick<20 else ':merge_end_0' if tick<25 else 'main_out'
            lane=road+'_0'
            if turn:
                route=['west_in','north_out'];road=':center_0' if 10<=tick<=40 and internal else 'west_in'
                lane=road+'_0'
            theta=.01*tick if turn else 0.
            center=np.array([.5*tick,0.])
            front=center+2*np.array([np.cos(theta),np.sin(theta)])
            raw=dict(front_position_m=front.tolist(),navigation_angle_deg=float(90-np.rad2deg(theta)),
                length_m=4.,acceleration_mps2=0.,route_edges=route,road_id=road,lane_id=lane)
            vehicles.append(dict(vehicle_id=vid,sumo_raw=raw))
        frames.append(dict(schema_version='sumodiff.raw.frame.v1',tick=tick,time_seconds=tick*.1,vehicles=vehicles))
    (tmp_path/'frames.jsonl').write_text('\n'.join(json.dumps(x) for x in frames),encoding='utf-8')
    (tmp_path/'lanechanges.xml').write_text('<lanechanges><change id="main" time="1.0" from="a_1" to="a_0"/><change id="ramp" time="1.5" from="a_1" to="a_0"/></lanechanges>')
    ep=SimpleNamespace(key='ep',manifest={'family':family},manifest_path=tmp_path/'episode_manifest.json')
    road=SimpleNamespace(connections=[dict(sumo_raw={'from':'west_in','to':'north_out','dir':'l'})],
        route_lane_ids=lambda route:{'west_in_0',':center_0_0','north_out_0'})
    return MotionIndex(ep,road)

def test_ramp_motion_uses_mainline_changes_actual_exit_and_strict_future_interval(tmp_path):
    index=raw_fixture(tmp_path,'ramp_merge')
    flags,events=index.labels(['main','ramp'],9,True)
    assert flags['lane_change'] and flags['ramp_exit']
    assert len(events['lane_change'])==len(events['ramp_exit'])==1
    assert index.labels(['main','ramp'],10,True)[0]['lane_change'] is False
    assert index.labels(['main','ramp'],9,False)==(None,None)
    assert len({e['event_id'] for e in index.catalog if e['kind']=='ramp_exit'})==1

@pytest.mark.parametrize('internal,expected',[(True,True),(False,False)])
def test_turn_requires_observed_heading_and_known_legal_internal_lane(tmp_path,internal,expected):
    index=raw_fixture(tmp_path,'unsignalized_intersection',turn=True,internal=internal)
    flags,events=index.labels(['turner'],0,True)
    assert flags['turn']==expected
    assert len(events['turn'])==int(expected)
    if expected:
        first=index.labels(['turner'],0,True)[1]['turn']
        second=index.labels(['turner'],10,True)[1]['turn']
        assert first==second  # one vehicle maneuver, not a new event for every window

def test_source_plan_mismatch_is_rejected_before_episode_loading(tmp_path):
    c=plan();sources=[]
    for i,job in enumerate(jobs(c,True)):
        p=tmp_path/str(i);p.mkdir()
        scenario=deepcopy(job['scenario']);scenario['runtime']['sumo_home']='test-sumo'
        actual=dict(scenario=scenario,recording=dict(seed=job['seed']))
        import yaml
        (p/'resolved_config.yaml').write_text(yaml.safe_dump(actual))
        (p/'manifest.json').write_text(json.dumps(dict(formal=True,git={'dirty':False},resolved_config_sha256=file_identity(p/'resolved_config.yaml')['sha256'])))
        (p/'status.json').write_text(json.dumps(dict(state='completed')))
        (p/'episode_manifest.json').write_text(json.dumps(dict(seed=job['seed'])))
        sources.append(dict(manifest=str(p/'episode_manifest.json'),split=job['split'],family=job['scenario']['family']))
    assert len(verify_sources(sources,c,True,True))==18
    with pytest.raises(ValueError,match='population'):verify_sources(sources[:-1],c,True,True)
    sources[0]['split']='test'
    with pytest.raises(ValueError,match='preassigned'):verify_sources(sources,c,True,True)
