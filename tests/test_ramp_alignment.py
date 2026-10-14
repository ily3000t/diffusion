from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from sumodiff.simulation.config import resolve_config
from sumodiff.simulation.scenes import layout
from sumodiff.experiments.source_motion_pilot import protocol,make_cases,build_reference
from sumodiff.geometry.maps import RoadMap
from sumodiff.models.inputs import lane_rules

def config(**values):
    c=dict(schema_version='sumodiff.scenario.v2',family='ramp_merge',
        geometry=dict(transition_length=80,alignment_length=10),network=dict(output_precision=8,protect_acceleration_lane=True),
        driver=dict(lane_change_model='SL2015',lcAccelLat=.3,maxSpeedLat=.8),
        simulation=dict(lane_change_duration=0.,lateral_resolution=.875))
    for key,value in values.items():
        c[key].update(value)
    return resolve_config(c)

def test_v1_contract_does_not_gain_silent_defaults():
    c=resolve_config(dict(family='ramp_merge'))
    assert 'alignment_length' not in c['geometry']
    assert 'protect_acceleration_lane' not in c['network']
    with pytest.raises(ValueError,match='Unknown'):
        resolve_config(dict(family='ramp_merge',geometry=dict(alignment_length=10)))

@pytest.mark.parametrize('override',[
    dict(geometry=dict(alignment_length=-1)),
    dict(geometry=dict(alignment_length=float('nan'))),
    dict(geometry=dict(alignment_length=150)),
    dict(geometry=dict(transition_length=0)),
    dict(network=dict(protect_acceleration_lane=1)),
    dict(driver=dict(lane_change_model='LC2013',lcAccelLat=None),simulation=dict(lane_change_duration=3.,lateral_resolution=0)),
])
def test_invalid_alignment_or_rule_combinations_are_rejected(override):
    with pytest.raises(ValueError):config(**override)

def test_other_road_families_cannot_claim_scenario_v2_support():
    with pytest.raises(NotImplementedError):resolve_config(dict(schema_version='sumodiff.scenario.v2',family='three_lane_straight'))

def test_alignment_adds_a_real_straight_tail_without_widening_network():
    c=config();nodes,edges,links,_=layout(c)
    ramp=next(e for e in edges if e['id']=='ramp_in')
    xy=np.array([list(map(float,p.split(','))) for p in ramp['shape'].split()])
    np.testing.assert_allclose(xy[-2:],[[-10,-10.5],[0,-10.5]],atol=1e-8)
    zero=config(geometry=dict(alignment_length=0),network=dict(protect_acceleration_lane=False))
    n,e,l,_=layout(zero)
    assert nodes==n
    assert {x['id']:x['width'] for x in edges}=={x['id']:x['width'] for x in e}
    assert {x['id']:x['numLanes'] for x in edges}=={x['id']:x['numLanes'] for x in e}

def test_only_acceleration_lane_and_converging_internal_pair_are_restricted():
    _,edges,links,_=layout(config())
    lanes={e['id']:e.get('lanes',[]) for e in edges}
    assert lanes['acceleration']==[dict(index='1',changeRight='emergency')]
    assert not any(v for k,v in lanes.items() if k!='acceleration')
    restricted=[l for l in links if 'changeLeft' in l or 'changeRight' in l]
    assert len(restricted)==2
    assert next(l for l in restricted if l['fromLane']=='0')['changeLeft']=='emergency'
    assert next(l for l in restricted if l['fromLane']=='1')['changeRight']=='emergency'
    assert all(l['from']=='acceleration' and l['to']=='main_out' for l in restricted)

def test_control_geometry_and_v1_scene_layout_are_identical():
    c=config(geometry=dict(alignment_length=0),network=dict(protect_acceleration_lane=False))
    legacy=deepcopy(c);legacy['schema_version']='sumodiff.scenario.v1'
    legacy['geometry'].pop('alignment_length');legacy['network'].pop('protect_acceleration_lane')
    assert layout(c)==layout(resolve_config(legacy))

def test_map_retains_directional_rules_and_model_rejects_unencoded_restriction(tmp_path):
    p=tmp_path/'network.net.xml'
    p.write_text('<net><edge id="road"><lane id="road_0" index="0" speed="20" width="4" length="10" shape="0,0 10,0" changeRight="emergency"/></edge></net>')
    road=RoadMap.read(p)
    assert road.lanes[0]['change_right_classes']==['emergency']
    with pytest.raises(NotImplementedError,match='explicit model rule encoding'):
        lane_rules(dict(lanes=road.lanes,connections=[],junctions=[]))
    road.lanes[0]['change_right_classes']=['passenger']
    assert lane_rules(dict(lanes=road.lanes,connections=[],junctions=[])).shape==(1,1,2)

def test_fixed_pilot_has_two_training_geometries_and_separates_factors():
    root=Path(__file__).parents[1];c,b=protocol(root/'configs/experiments/ramp_alignment_pilot.yaml')
    cases=make_cases(c,b);assert len(cases)==12
    assert {x['split'] for x in cases}=={'train'}
    assert {x['training_geometry_index'] for x in cases}=={0,1}
    assert c['reference_ticks']==list(range(30,961,30))
    g0=[x for x in cases if x['training_geometry_index']==0]
    assert {x['variant'] for x in g0}=={'control','permissions','alignment','combined'}
    assert {x['variant'] for x in cases if x['training_geometry_index']==1}=={'control','combined'}
    assert len({json.dumps(x['scenario']['traffic'],sort_keys=True) for x in g0})==1

def test_unavailable_fixed_reference_is_reported_instead_of_shifted():
    c=dict(history_points=21,reference_stride_ticks=10)
    assert build_reference(SimpleNamespace(frames=[{}]*20),30,None,c)==(None,'outside_episode')
    ep=SimpleNamespace(frames=[{}]*31)
    assert build_reference(ep,30,None,c)==(None,'no_current_reference_vehicle')

def test_original_pilot_still_matches_its_frozen_case_snapshot():
    root=Path(__file__).parents[1];c,b=protocol(root/'configs/experiments/source_motion_pilot.yaml')
    cases=make_cases(c,b);assert len(cases)==14
    assert all(x['scenario']['schema_version']=='sumodiff.scenario.v1' for x in cases)


def test_frozen_geometry_indices_reject_booleans():
    from sumodiff.experiments.source_motion_pilot import ramp_protocol
    root=Path(__file__).parents[1];c,_=protocol(root/'configs/experiments/ramp_alignment_pilot.yaml')
    c['train_geometry_indices']=[False,True]
    with pytest.raises(ValueError,match='geometry indices'):ramp_protocol(c)


def test_failed_collection_keeps_all_fixed_quality_slots(tmp_path,monkeypatch):
    import yaml
    from sumodiff.experiments import source_motion_pilot as pilot
    from sumodiff.experiments.recorder import file_identity
    root=Path(__file__).parents[1];config_path=root/'configs/experiments/ramp_alignment_pilot.yaml'
    c,b=protocol(config_path);cases=make_cases(c,b)
    parent=tmp_path/'collection';parent.mkdir();records=[]
    for case in cases:
        path=parent/(case['case_id']+'.yaml');path.write_text(yaml.safe_dump(case['scenario']))
        records.append(dict(case_id=case['case_id'],family=case['family'],variant=case['variant'],seed=case['seed'],
            split='train',completed=False,eligible_for_normal_training=False,config=file_identity(path),
            episode_manifest=str(parent/case['case_id']/'episode_manifest.json')))
    (parent/'resolved_config.yaml').write_text(yaml.safe_dump(dict(protocol=c,cases=cases)))
    (parent/'manifest.json').write_text(json.dumps(dict(resolved_config_sha256=file_identity(parent/'resolved_config.yaml')['sha256'])))
    (parent/'status.json').write_text(json.dumps(dict(state='completed')))
    (parent/'collection_results.json').write_text(json.dumps(dict(cases=records)))
    class Recorder:
        def __init__(self,output,*a,**kw):
            self.output=Path(output);self.output.mkdir();self.manifest=dict(git=dict(commit_sha='fixture'))
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def write_metrics(self,*a):pass
    monkeypatch.setattr(pilot,'RunRecorder',Recorder)
    result=pilot.audit(config_path,parent,tmp_path/'audit',False)
    assert result['planned_cases']==result['failed_cases']==12
    assert result['quality_planned_windows']==result['quality_unavailable_windows']==768
    assert result['quality_assessed_windows']==0
    rows=[json.loads(line) for line in (tmp_path/'audit/window_quality.jsonl').read_text().splitlines()]
    assert len(rows)==768 and all(r['motion_flags'] is None and not r['eligible'] for r in rows)
