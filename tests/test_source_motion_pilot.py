from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import pytest
from sumodiff.simulation.config import resolve_config as scenario_config
from sumodiff.simulation.scenes import layout,quintic_points
from sumodiff.data.config import resolve_config as window_config
from sumodiff.data.episodes import Episode,Observation
from sumodiff.data.selection import select_current
from sumodiff.data.windows import build_window
from sumodiff.geometry.maps import RoadMap
from sumodiff.experiments.source_motion_pilot import protocol,make_cases,raw_motion_summary

def sl():
    return dict(driver=dict(lane_change_model='SL2015',lcAccelLat=.3,maxSpeedLat=.8),
                simulation=dict(lane_change_duration=0.,lateral_resolution=.875))

def test_original_simulation_defaults_still_use_legacy_models():
    c=scenario_config(dict(family='ramp_merge'))
    assert c['driver']['lane_change_model']=='LC2013'
    assert c['driver']['lcAccelLat'] is None and c['driver']['jerkmax'] is None
    assert c['simulation']['lane_change_duration']==3
    assert c['geometry']['transition_length']==0
    nodes,_,links,_=layout(c)
    assert 'shape' not in next(n for n in nodes if n['id']=='merge_end')
    assert all('shape' not in link for link in links)

@pytest.mark.parametrize('override',[
    dict(driver=dict(lcAccelLat=.3)),
    dict(driver=dict(jerkmax=3)),
    dict(driver=dict(lane_change_model='SL2015')),
    dict(driver=dict(lane_change_model='SL2015',lcAccelLat=.3,maxSpeedLat=.8),simulation=dict(lateral_resolution=.875)),
    dict(simulation=dict(lateral_resolution=.875)),
    dict(driver=dict(car_follow_model='EIDM',jerkmax=float('nan'))),
    dict(geometry=dict(transition_length=80)),
    dict(geometry=dict(transition_length=300),network=dict(output_precision=8)),
])
def test_incompatible_motion_or_geometry_parameters_are_rejected(override):
    with pytest.raises(ValueError):
        scenario_config(dict(family='ramp_merge',**override))

def test_explicit_sublane_and_eidm_are_supported():
    c=scenario_config(dict(family='ramp_merge',**sl()))
    assert c['simulation']['lateral_resolution']==.875
    e=scenario_config(dict(family='unsignalized_intersection',driver=dict(car_follow_model='EIDM',jerkmax=3)))
    assert e['driver']['car_follow_model']=='EIDM'

def test_quintic_geometry_has_small_endpoint_slopes_and_correct_lane_connections():
    c=scenario_config(dict(family='ramp_merge',geometry=dict(transition_length=80),network=dict(output_precision=8)))
    nodes,edges,links,routes=layout(c)
    node=next(n for n in nodes if n['id']=='merge_end')
    assert len(node['shape'].split())==4
    link=next(l for l in links if l['from']=='acceleration' and l['fromLane']=='0')
    curve=np.array([list(map(float,p.split(','))) for p in link['shape'].split()])
    np.testing.assert_allclose(curve[[0,-1]],[[80,-12.25],[160,-8.75]])
    assert abs((curve[1,1]-curve[0,1])/.25)<1e-5
    assert abs((curve[-1,1]-curve[-2,1])/.25)<1e-5
    for l in links:
        if l['from']=='acceleration' and l['fromLane']!='0':
            xy=np.array([list(map(float,p.split(','))) for p in l['shape'].split()])
            assert np.ptp(xy[:,1])==0
    ramp=next(e for e in edges if e['id']=='ramp_in')
    points=np.array([list(map(float,p.split(','))) for p in ramp['shape'].split()])
    np.testing.assert_allclose(points[-1],[0,-10.5])
    assert routes['ramp']['edges']==['ramp_in','acceleration','main_out']

def test_transition_rotation_is_a_rigid_transform():
    a=scenario_config(dict(family='ramp_merge',geometry=dict(transition_length=80),network=dict(output_precision=8)))
    b=deepcopy(a);b['geometry']['rotation_degrees']=90
    la=next(l for l in layout(a)[2] if l['from']=='acceleration' and l['fromLane']=='0')
    lb=next(l for l in layout(b)[2] if l['from']=='acceleration' and l['fromLane']=='0')
    parse=lambda l:np.array([list(map(float,p.split(','))) for p in l['shape'].split()])
    np.testing.assert_allclose(parse(lb),parse(la)@np.array([[0,1],[-1,0]]),atol=1e-8)

def obs(x,route=None,index=0):
    return Observation(np.array([float(x),0.]),0.,dict(route_edges=route or ['main'],route_index=index,
        length_m=4.7,width_m=1.8,type_id='passenger'))

def test_density_reference_does_not_remain_on_lexical_outlier():
    current={'a':obs(-200),'b':obs(0),'c':obs(20),'d':obs(40)}
    cfg=window_config(dict(reference_policy='interaction_current'))
    ref,ids,excluded=select_current(current,'three_lane_straight',None,cfg)
    assert ref=='c' and set(ids)=={'b','c','d'} and excluded['excluded_by_radius']==1

def test_ramp_reference_uses_known_route_and_does_not_follow_departed_merge():
    route=['ramp_in','acceleration','main_out']
    current={'a':obs(119,['main_in','acceleration','main_out'],1),'b':obs(110,route,1),'c':obs(120,route,2)}
    road=SimpleNamespace(junctions=[dict(sumo_raw=dict(id='merge_end',x='120',y='0'))])
    cfg=window_config(dict(reference_policy='interaction_current'))
    assert select_current(current,'ramp_merge',road,cfg)[0]=='b'

def test_intersection_reference_uses_planned_legal_turn_and_explicit_map():
    current={'a':obs(1,['west_in','east_out']),'b':obs(5,['west_in','north_out'])}
    road=SimpleNamespace(connections=[dict(sumo_raw=dict(fromLane='0',toLane='0',dir='l',**{'from':'west_in','to':'north_out'}))],
        junctions=[dict(sumo_raw=dict(id='center',x='0',y='0'))])
    cfg=window_config(dict(reference_policy='interaction_current'))
    assert select_current(current,'unsignalized_intersection',road,cfg)[0]=='b'
    road.junctions=[]
    with pytest.raises(ValueError,match='declared map'):select_current(current,'unsignalized_intersection',road,cfg)

def test_new_reference_window_does_not_read_future_and_has_distinct_identity(tmp_path):
    net=tmp_path/'network.net.xml'
    net.write_text('<net><edge id="main"><lane id="main_0" index="0" speed="20" width="4" length="1000" shape="-500,0 500,0"/></edge></net>')
    frames=[{vid:obs(t*.5+x) for vid,x in [('a',-200),('b',0),('c',20),('d',40)]} for t in range(101)]
    ep=Episode('sha256:'+'a'*64,dict(dt=.1,episode_id='fixture',family='three_lane_straight',geometry_id='g'),net,'train',frames,net,[])
    road=RoadMap.read(net);cfg=window_config(dict(reference_policy='interaction_current'))
    w,_=build_window(ep,30,road,cfg)
    changed=deepcopy(ep)
    for current in changed.frames[31:]:
        for o in current.values():o.center[:]=np.nan;o.raw['route_edges']=['poison']
    # Corrupt finite future poses so label extraction remains defined; conditioning must be identical.
    for current in changed.frames[31:]:
        for o in current.values():o.center[:]=1e4
    v,_=build_window(changed,30,road,cfg)
    assert w.input_metadata==v.input_metadata
    for key in w.conditioning:np.testing.assert_array_equal(w.conditioning[key],v.conditioning[key])
    old,_=build_window(ep,30,road,window_config({}))
    assert old.input_metadata['window_id']!=w.input_metadata['window_id']
    assert old.input_metadata['reference_id']=='a' and w.input_metadata['reference_id']=='c'

def test_pilot_pairs_only_predeclared_training_geometries_and_equal_seed_demand():
    c,base=protocol(Path(__file__).parents[1]/'configs/experiments/source_motion_pilot.yaml')
    cases=make_cases(c,base)
    assert len(cases)==14 and {x['split'] for x in cases}=={'train'}
    for family in {x['family'] for x in cases}:
        group=[x for x in cases if x['family']==family]
        assert len({str(x['scenario']['traffic']) for x in group})==1
        assert all(x['training_geometry_index']<base['train_geometries'] for x in group)
    assert all(not any(x['scenario']['validation'].values()) for x in cases)

def test_raw_derivatives_do_not_bridge_lifecycle_holes_and_empty_is_na():
    ticks=np.array([0,1,2,3,8,9,10])
    tr=dict(ticks=ticks,center=np.c_[ticks*.1,np.zeros(len(ticks))],reported_accel=np.zeros(len(ticks)))
    r=raw_motion_summary({'v':tr},{},{'v':[]},None,'three_lane_straight')
    assert r['distributions']['all']['count']==1
    assert r['distributions']['all']['max']==pytest.approx(0,abs=1e-12)
    assert r['distributions']['lane_change_context']['max'] is None
