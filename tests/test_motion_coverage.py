import json
import math
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
from sumodiff.data.motion_coverage import (future_motion,indices_at,lane_changes,
    logged_future_events,ramp_exit_events,raw_tracks,add_stage,source_case,read_quality_rows)
from sumodiff.experiments.motion_coverage import validate_config,authenticated_json
from collections import defaultdict,Counter
from sumodiff.experiments.recorder import file_identity


def track(yaw=0.):
    ticks=np.arange(70)
    p=np.stack((ticks*.7,np.zeros_like(ticks)),axis=-1)
    return dict(ticks=ticks,center=p,front=p+np.array([2.,0.]),yaw=np.full(70,yaw),
        route_edges=['entry','core','exit'],road=['core']*70,lane=['core_0']*70,
        reported_accel=np.zeros(70))


def test_future_own_body_motion_is_rigid_transform_invariant():
    t=track();t['center'][:,1]=np.sin(np.arange(70)*.1)
    expected=future_motion(t,24)
    angle=.7;rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    moved=deepcopy(t);moved['center']=t['center']@rotation.T+np.array([123.,-50.]);moved['yaw']+=angle
    actual=future_motion(moved,24)
    assert actual['heading_span_rad']==pytest.approx(expected['heading_span_rad'])
    assert actual['lateral_excursion_m']==pytest.approx(expected['lateral_excursion_m'],abs=1e-12)


def test_heading_wrap_and_fixed_t0_are_handled():
    t=track();t['yaw']=np.deg2rad(np.linspace(170,200,70))
    t['yaw']=np.arctan2(np.sin(t['yaw']),np.cos(t['yaw']))
    result=future_motion(t,25)
    assert result['heading_span_rad']==pytest.approx(np.deg2rad(30*40/69))
    assert future_motion(track(),25)['lateral_excursion_m']==0.


def test_future_gaps_and_lifecycle_boundaries_are_unknown():
    t=track()
    assert indices_at(t,-1,10) is None
    assert future_motion(t,40) is None
    t['ticks']=np.delete(t['ticks'],35)
    for name in ['center','front','yaw','reported_accel']:t[name]=np.delete(t[name],35,axis=0)
    assert future_motion(t,20) is None


def test_lane_change_events_are_logged_crossings_with_exact_future_interval(tmp_path):
    p=tmp_path/'lane.xml'
    p.write_text('<lanechanges><changeStarted id="a"/><change id="a" time="2.0" from="x_1" to="x_0" reason="speedGain"/><changeEnded id="a"/></lanechanges>')
    events=lane_changes(p)['a']
    assert logged_future_events(events,20)==[]
    assert len(logged_future_events(events,19,1))==1
    p.write_text('<lanechanges><change id="a" time="2.05" from="x_1" to="x_0"/></lanechanges>')
    with pytest.raises(ValueError,match='grid'):lane_changes(p)


def test_ramp_exit_does_not_count_main_route_or_internal_link_as_merging():
    t=track();t['road']=['acceleration']*30+[':merge_0']*5+['main_out']*35
    assert ramp_exit_events(t)==[]
    t['route_edges']=['ramp_in','acceleration','main_out']
    assert ramp_exit_events(t)==[35]


def test_raw_center_is_rebuilt_from_front_navigation_without_saved_targets(tmp_path):
    p=tmp_path/'frames.jsonl'
    raw=dict(front_position_m=[10.,20.],navigation_angle_deg=0.,length_m=4.,
        route_edges=['a','b'],road_id='a',lane_id='a_0',acceleration_mps2=0.,vehicle_class='passenger')
    rows=[dict(schema_version='sumodiff.raw.frame.v1',tick=0,time_seconds=0.,vehicles=[]),
          dict(schema_version='sumodiff.raw.frame.v1',tick=1,time_seconds=.1,
               vehicles=[dict(vehicle_id='v',sumo_raw=raw)])]
    p.write_text('\n'.join(json.dumps(r) for r in rows))
    tracks,n=raw_tracks(p)
    assert n==2
    np.testing.assert_allclose(tracks['v']['center'][0],[10,18])
    assert tracks['v']['yaw'][0]==pytest.approx(math.pi/2)
    rows[1]['tick']=2;p.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='time grid'):raw_tracks(p)


def test_stage_unknown_denominators_and_event_occurrences_are_kept():
    counts=defaultdict(Counter)
    row=dict(core_complete=False,selected_agents=2,eligible=False)
    flags=add_stage(counts,row,[None,None],False,[('a',dict(tick=10))],[])
    assert counts['candidate']['windows']==1
    assert counts['candidate']['unknown_future_agent_windows']==2
    assert counts['candidate']['windows_without_any_complete_future_agent']==1
    assert flags['logged_lane_change'] and not flags['heading5']
    assert 'eligible' not in counts and 'selected' not in counts


def test_raw_jerk_trace_has_full_history_future_context():
    t=track();t['center'][35:,1]=.1;t['front']=t['center']+np.array([2.,0.])
    row=dict(reference_tick=25,window_id='w',family='synthetic',episode_key='ep',
             eligible=False,reasons=['raw64:motion'],raw64_jerk_max=200.)
    result=source_case(row,dict(agent_ids=['v',None]),dict(v=t),dict(v=[dict(tick=37)]))
    assert len(result['times_relative_t0_s'])==64 and len(result['physical_jerk_norm_mps3'])==61
    assert result['raw_center_jerk_max_mps3']==pytest.approx(200.)
    assert result['raw_front_jerk_max_mps3']==pytest.approx(200.)
    assert result['peak_vehicle_id']=='v'


def test_only_training_quality_rows_are_returned(tmp_path):
    p=tmp_path/'quality.jsonl'
    r=dict(window_id='w',family='x',split='train',episode_key='ep',geometry_id='g',
        reference_tick=30,selected_agents=2,core_complete=False,eligible=False,reasons=['incomplete'])
    p.write_text(json.dumps(dict(split='test'))+'\n'+json.dumps(r)+'\n'+json.dumps(dict(split='validation')))
    out=read_quality_rows(p)
    assert list(out)==['ep'] and out['ep'][0]['window_id']=='w'


def test_authenticated_metadata_rejects_byte_drift(tmp_path):
    p=tmp_path/'meta.json';p.write_text('{"split":"train"}')
    identity=file_identity(p)
    assert authenticated_json(p,identity)['split']=='train'
    p.write_text('{"split":"test"}')
    with pytest.raises(ValueError,match='changed'):authenticated_json(p,identity)


def test_protocol_rejects_test_selector_and_unbounded_cases():
    cfg=dict(schema_version='sumodiff.training.motion.coverage.config.v1',preparation='fixture',
        expected_data_id='fixture',reference_scales='fixture',reference_motion='fixture',
        expected_train_episodes=1,expected_train_candidates=1,expected_selected_windows=1,seed=1,case_limit_per_bucket=2)
    validate_config(cfg)
    with pytest.raises(ValueError):validate_config(dict(cfg,split='test'))
    with pytest.raises(ValueError):validate_config(dict(cfg,case_limit_per_bucket=100))


def test_event_opportunities_separate_radius_and_vehicle_capacity():
    from sumodiff.data.motion_coverage import event_opportunities
    a=track();b=track();b['center']+=np.array([1000.,0.])
    c=track();c['center']+=np.array([3.,0.])
    result=event_opportunities([('a',35),('b',35),('c',35)],dict(a=a,b=b,c=c),25,[17.5,0.],{'a'})
    assert len(result['actor_present'])==3
    assert result['inside_radius']=={('a',35),('c',35)}
    assert result['selected']=={('a',35)}
