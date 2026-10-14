from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from sumodiff.data.dataset import save_window, write_json
from sumodiff.experiments.recorder import file_identity, load_config
from sumodiff.decoding.controls import ControlConfig
from sumodiff.decoding.control_scale_audit import (
    validate_config, distribution, select_scales, window_controls, run)

CONFIG=load_config(Path('configs/experiments/control_scale_audit.yaml'))


def small_fixture(tmp_path,incomplete=False):
    directory=tmp_path/'dataset';directory.mkdir()
    h=np.zeros((2,21,6),np.float32);h[...,0]=np.arange(-20,1)*.8
    h[1,:,1]=10;h[...,2]=8;h[...,5]=1
    c=dict(history=h,history_mask=np.ones((2,21),bool),agent_mask=np.ones(2,bool),
           initial_positions=h[:,-1,:2].copy(),attributes=np.array([[4,2,1,1],[4,2,1,0]],np.float32))
    f=np.zeros((2,40,6),np.float32);f[...,0]=np.arange(1,41)*.8;f[...,2]=8;f[...,5]=1
    fm=np.ones((2,40),bool)
    if incomplete:fm[0,-1]=False
    meta=dict(window_id='tiny',split='train',family='synthetic',episode_key='ep',geometry_id='geo',dt_seconds=.1,
              agent_ids=['a','b'])
    window=SimpleNamespace(conditioning=c,targets=dict(future=f,future_mask=fm),input_metadata=meta,
                           label_metadata=dict(core_training_eligible=True),exact_map={})
    entry=save_window(directory/'windows'/'tiny',window)
    write_json(directory/'windows.json',[entry])
    selected={**meta,'eligible':True,'core_complete':True,'selected_agents':2,'actual_turning_agents':0}
    selection=directory/'selection.json'
    write_json(selection,dict(selected=[selected,dict(window_id='unopened-test',split='test')]))
    write_json(directory/'dataset_manifest.json',dict(schema_version='sumodiff.dataset.v1',data_id='synthetic',
        window_index=file_identity(directory/'windows.json'),curation=dict(selection=file_identity(selection))))
    cfg=deepcopy(CONFIG);cfg.update(dataset=str(directory),data_id='synthetic',expected_windows=1,expected_families={'synthetic':1})
    cfg['scale_policy']['groups']=['all'];cfg['scale_policy']['minimum_group_windows']=1
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg),encoding='utf-8',newline='\n')
    return path,c,window.targets


def test_training_only_config_and_unsupported_controls_are_explicit():
    assert validate_config(CONFIG).channels==3
    cfg=deepcopy(CONFIG);cfg['split']='test'
    with pytest.raises(ValueError,match='protocol'):validate_config(cfg)
    cfg=deepcopy(CONFIG);cfg['control']['lateral_velocity']=False
    with pytest.raises(NotImplementedError):validate_config(cfg)
    cfg=deepcopy(CONFIG);cfg['scale_policy']['candidate_scales']['yaw_rate']=[.1,.1]
    with pytest.raises(ValueError,match='strictly increasing'):validate_config(cfg)


def test_group_envelope_preserves_rare_motion_and_grid_exhaustion():
    policy=deepcopy(CONFIG['scale_policy']);policy['groups']=['all','turning'];policy['minimum_group_windows']=1
    groups={name:dict(completed_windows=10,scale_quantiles={c:x for c in ('acceleration','yaw_rate','lateral_velocity')})
            for name,x in [('all',.01),('turning',.15)]}
    result=select_scales(groups,policy)
    assert result['scales']['yaw_rate']['scale']==.2
    assert result['scales']['lateral_velocity']['scale']==.2
    groups['turning']['scale_quantiles']['yaw_rate']=2.
    assert not select_scales(groups,policy)['available']
    groups.pop('turning')
    assert select_scales(groups,policy)['reasons']==['insufficient_group:turning']


def test_distribution_empty_and_nan_are_not_zero():
    assert distribution([])['mean'] is None
    x=distribution([-2.,0.,1.])
    assert x['min']==-2 and x['max']==1 and x['abs_max']==2
    with pytest.raises(ValueError):distribution([np.nan])


@pytest.mark.parametrize('incomplete',[False,True])
def test_real_entry_retains_failures_and_never_opens_test(tmp_path,monkeypatch,incomplete):
    monkeypatch.setattr('sumodiff.experiments.recorder.collect_environment',lambda:dict(test_fixture=True))
    path,_,_=small_fixture(tmp_path,incomplete)
    result=run(path,tmp_path/'run')
    assert result['planned_windows']==1 and result['planned_agent_windows']==2
    assert result['failed_windows']==int(incomplete)
    assert result['candidate_normalization']['certified_training_statistics']==(not incomplete)
    assert result['validation_arrays_loaded']==result['test_arrays_loaded']==result['optimizer_updates']==0
    assert result['candidate_normalization']['training_integration_implemented'] is False
    assert json.loads((tmp_path/'run'/'status.json').read_text())['state']=='completed'
    with pytest.raises(FileExistsError):run(path,tmp_path/'run')


def test_padding_and_signed_controls_are_preserved(tmp_path):
    _,c,target=small_fixture(tmp_path)
    c['agent_mask'][1]=False
    for key in ('history','initial_positions','attributes'):c[key][1]=np.nan
    target['future'][1]=np.nan;target['future_mask'][1]=False
    data,row=window_controls(c,target,ControlConfig(lateral_velocity=True),CONFIG['checks'])
    assert data['controls'].shape==(1,40,3) and np.isfinite(data['lateral_change']).all()
    assert row['consistency']['numpy_control_difference']<1e-8
    target['future_mask'][0,3]=False
    with pytest.raises(ValueError,match='Incomplete'):window_controls(c,target,ControlConfig(lateral_velocity=True),CONFIG['checks'])


def test_wrong_dataset_id_stops_before_statistics(tmp_path,monkeypatch):
    monkeypatch.setattr('sumodiff.experiments.recorder.collect_environment',lambda:dict(test_fixture=True))
    path,_,_=small_fixture(tmp_path)
    cfg=yaml.safe_load(path.read_text());cfg['data_id']='wrong';path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    with pytest.raises(ValueError,match='dataset ID'):run(path,tmp_path/'out')
    assert not (tmp_path/'out').exists()
