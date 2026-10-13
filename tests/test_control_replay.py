from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from shapely.geometry import box, mapping
from sumodiff.data.dataset import save_window, write_json
from sumodiff.decoding.controls import ControlConfig
from sumodiff.decoding.control_replay import (
    run, validate_protocol, validate_plan, physical_jerk, compatibility_gate)
from sumodiff.evaluation.motion import MotionConfig
from sumodiff.evaluation.roads import RoadConfig
from sumodiff.evaluation.collisions import CollisionConfig
from sumodiff.experiments.recorder import file_identity


def protocol(dataset, plan):
    return dict(schema_version='sumodiff.control_replay.protocol.v1',
                dataset=str(dataset), task_plan=str(plan), seed=44,
                methods={'two': asdict(ControlConfig()), 'three': asdict(ControlConfig(lateral_velocity=True))},
                gate=dict(max_position_error_m=.05, max_heading_error_rad=1e-5, max_velocity_error_mps=.05),
                evaluation=dict(motion=asdict(MotionConfig()), road=asdict(RoadConfig()), collision=asdict(CollisionConfig())))


def fixture(tmp_path, incomplete=False):
    dataset = tmp_path/'dataset'
    dataset.mkdir()
    history = np.zeros((2, 21, 6), np.float32)
    history[..., 0] = np.arange(-20, 1) * .8
    history[1, :, 1] = 10.
    history[..., 2], history[..., 5] = 8., 1.
    c = dict(history=history, history_mask=np.ones((2,21),bool),
             initial_positions=history[:,-1,:2].copy(), agent_mask=np.ones(2,bool),
             attributes=np.array([[4,2,1,1],[4,2,1,0]],np.float32))
    future = np.zeros((2,40,6),np.float32)
    future[...,0] = np.arange(1,41)*.8
    future[...,2],future[...,5] = 8.,1.
    fm=np.ones((2,40),bool)
    if incomplete:
        fm[0,-1]=False
    meta=dict(window_id='tiny',split='train',episode_key='ep',geometry_id='geo',
              dt_seconds=.1,map_extent_m=[-100,-100,100,100],excluded_vehicle_count=0)
    geom=mapping(box(-100,-100,100,100))
    window=SimpleNamespace(conditioning=c,targets=dict(future=future,future_mask=fm),
                           input_metadata=meta,label_metadata=dict(core_training_eligible=not incomplete),
                           exact_map=dict(drivable=geom,route_corridors=[geom,geom]))
    entry=save_window(dataset/'windows'/'tiny',window)
    write_json(dataset/'windows.json',[entry])
    write_json(dataset/'dataset_manifest.json',dict(schema_version='sumodiff.dataset.v1',
        window_index=file_identity(dataset/'windows.json'),data_id='synthetic-fixture'))
    plan=tmp_path/'tasks.json'
    write_json(plan,dict(schema_version='sumodiff.control_replay.tasks.v1',
                         tasks=[dict(window_id='tiny',split='train',group='ordinary')]))
    cfg=tmp_path/'config.yaml'
    cfg.write_text(yaml.safe_dump(protocol(dataset,plan)),encoding='utf-8')
    return cfg,c,future


@pytest.mark.parametrize('incomplete',[False,True])
def test_recorded_real_entry_preserves_failure_denominator(tmp_path,monkeypatch,incomplete):
    monkeypatch.setattr('sumodiff.experiments.recorder.collect_environment',lambda:dict(test_fixture=True))
    cfg,c,f=fixture(tmp_path,incomplete)
    output=tmp_path/'run'
    result=run(cfg,output)
    assert result['planned']==1 and result['optimizer_updates']==result['denoiser_calls']==result['test_arrays_loaded']==0
    assert result['gate_pass_all_tasks']=={'two':not incomplete,'three':not incomplete}
    assert json.loads((output/'status.json').read_text())['state']=='completed'
    assert result['summary']['train/ordinary']['two']['failed']==int(incomplete)
    assert result['summary']['train/ordinary']['two']['planned']==1
    assert result['summary']['train/ordinary']['label']['gate_pass'] is None
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['checkpoint'] is None and manifest['formal'] is False
    resolved=yaml.safe_load((output/'resolved_config.yaml').read_text())
    assert resolved['evaluation']['motion']['comfort_jerk_mps3']==5
    if not incomplete:
        with np.load(output/'000_train.npz') as saved:
            assert 'label_states' in saved and 'two_controls' in saved and 'three_states' in saved
    with pytest.raises(FileExistsError):
        run(cfg,output)


def test_invalid_scope_options_and_duplicate_task_rejected():
    task=dict(window_id='x',split='test',group='x')
    with pytest.raises(ValueError,match='Test tasks'):
        validate_plan(dict(schema_version='sumodiff.control_replay.tasks.v1',tasks=[task]))
    task['split']='train'
    with pytest.raises(ValueError,match='Duplicate'):
        validate_plan(dict(schema_version='sumodiff.control_replay.tasks.v1',tasks=[task,task]))
    config=protocol('a','b')
    config['unimplemented']=True
    with pytest.raises(ValueError,match='incomplete'):
        validate_protocol(config)
    config=protocol('a','b')
    config['gate']['max_position_error_m']=0
    with pytest.raises(ValueError,match='positive'):
        validate_protocol(config)


def test_jerk_is_physical_and_includes_history():
    c=dict(history=np.zeros((1,21,6)),history_mask=np.ones((1,21),bool),
           initial_positions=np.zeros((1,2)),agent_mask=np.ones(1,bool))
    t=np.arange(-20,1)*.1
    c['history'][0,:,0]=t**3
    states=np.zeros((1,40,6))
    states[0,:,0]=(np.arange(1,41)*.1)**3
    jerk=physical_jerk(c,states)
    np.testing.assert_allclose(jerk[0,:,0],6,atol=1e-10)
    assert jerk.shape==(1,40,2)


def test_gate_compares_agent_identity_and_new_collision_pairs():
    original=dict(quality_pass=True,motion=dict(violation_counts={'jerk':0}),
                  roads={n:dict(violation_body_frames=0) for n in ('road','route','map_coverage')},
                  collisions=dict(pairs=[dict(slots=[0,1],collision=False)]))
    replay=deepcopy(original)
    error=dict(max_position_error_m=0,max_velocity_error_mps=0,max_heading_error_rad=0)
    limits=protocol('a','b')['gate']
    gate=compatibility_gate(original,replay,{'0':True,'1':False},{'0':False,'1':True},error,limits)
    assert gate['reasons']==['new_strict_comfort_failure']
    assert gate['newly_uncomfortable_slots']==['0']
    replay['collisions']['pairs'][0]['collision']=None
    gate=compatibility_gate(original,replay,{'0':True},{'0':True},error,limits)
    assert 'new_collision_or_unknown' in gate['reasons']

@pytest.mark.parametrize('reference',['center','front'])
@pytest.mark.parametrize('basis',['midpoint','endpoint'])
def test_independent_numpy_rollout_known_curve(reference,basis):
    from sumodiff.decoding.replay_audit import numpy_rollout
    h=np.zeros((1,21,6))
    h[0,:,0]=np.arange(-20,1)*.8
    h[...,2],h[...,5]=8.,1.
    control=np.zeros((1,5,2))
    control[...,1]=1.
    config=asdict(ControlConfig(reference_point=reference,integration_heading=basis))
    actual=numpy_rollout(h,np.zeros((1,2)),np.array([4.]),np.array([True]),control,config)
    angles=np.arange(1,6)*.1
    bases=angles-.05 if basis=='midpoint' else angles
    offset=2. if reference=='front' else 0.
    expected=np.array([offset,0])+np.cumsum(.8*np.stack((np.cos(bases),np.sin(bases)),-1),axis=0)-offset*np.stack((np.cos(angles),np.sin(angles)),-1)
    np.testing.assert_allclose(actual[0,:,:2],expected,atol=1e-13)


def test_independent_audit_refuses_development_source(tmp_path):
    from sumodiff.decoding.replay_audit import run as audit
    source=tmp_path/'source'
    source.mkdir()
    write_json(source/'manifest.json',dict(formal=False,git=dict(dirty=False)))
    write_json(source/'status.json',dict(state='completed'))
    write_json(source/'results.json',{})
    (source/'resolved_config.yaml').write_text('purpose: fixture\n',encoding='utf-8')
    with pytest.raises(ValueError,match='completed clean formal'):
        audit(source,tmp_path/'out')
