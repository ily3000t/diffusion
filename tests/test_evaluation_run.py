import json
import subprocess
import numpy as np
import pytest
from shapely.geometry import box,mapping
from sumodiff.data.dataset import save_window,write_json
from sumodiff.data.windows import Window
from sumodiff.experiments.recorder import file_identity
from sumodiff.evaluation.config import resolve_config
from sumodiff.evaluation.cli import validate_labels


def test_resolved_options_reject_unsupported_and_invalid_settings():
    resolved = resolve_config({'road':{'geometry_repair':'bounded_make_valid'}})
    assert resolved['road']['max_repair_hausdorff_m'] == 1e-6
    for supplied in ({'guided':True},{'decoder':{'solve_dtype':'float16'}},
                     {'road':{'geometry_repair':'buffer_zero'}},{'cuda_smoke':1}):
        with pytest.raises(ValueError):
            resolve_config(supplied)


def test_label_validation_records_failure_without_dropping_or_decoding_missing_labels(tmp_path):
    repo = tmp_path/'repo';repo.mkdir()
    (repo/'.gitignore').write_text('/artifacts/\n',encoding='utf-8')
    config=repo/'config.yaml';config.write_text('schema_version: sumodiff.evaluation.config.v1\n',encoding='utf-8')
    def git(*args):
        subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
    git('init','-b','main');git('add','.');git('-c','user.name=Unit Test','-c','user.email=test@example.invalid','commit','-m','test: fixture')
    dataset=repo/'artifacts'/'dataset';entries=[]
    for name in ('valid','incomplete','invalid'):
        h=np.zeros((2,21,6),np.float32);h[...,5]=1;h[1,:,0]=10
        c=dict(history=h,history_mask=np.ones((2,21),bool),agent_mask=np.ones(2,bool),
               initial_positions=h[:,-1,:2],attributes=np.tile([4,2,1,0],(2,1)))
        f=np.zeros((2,40,6),np.float32);f[...,5]=1;fm=np.ones((2,40),bool)
        if name=='incomplete':fm[0,-1]=False
        if name=='invalid':f[0,0,0]=np.nan
        meta=dict(window_id=name,split='train',episode_key='e',geometry_id='g',dt_seconds=.1,
                  map_extent_m=[-40,-10,40,10],excluded_vehicle_count=3)
        area=mapping(box(-50,-15,50,15))
        w=Window(c,dict(future=f,future_mask=fm),meta,dict(core_training_eligible=name=='valid'),
                 dict(drivable=area,route_corridors=[area]*2))
        entries.append(save_window(dataset/'windows'/name,w))
    write_json(dataset/'windows.json',entries)
    write_json(dataset/'dataset_manifest.json',dict(schema_version='sumodiff.dataset.v1',
        window_index=file_identity(dataset/'windows.json')))
    output=repo/'artifacts'/'evaluation'
    with pytest.raises(RuntimeError,match='acceptance failed'):
        validate_labels(dataset,config,output,repo,formal=True)
    status=json.loads((output/'status.json').read_text())
    report=json.loads((output/'validation_summary.json').read_text())
    manifest=json.loads((output/'manifest.json').read_text())
    assert status['state']=='failed' and manifest['formal'] and not manifest['git']['dirty']
    assert report['windows']==3 and report['failed_windows']==1
    assert report['decoded_full_label_windows']==2 and report['incomplete_label_decode_skips']==1
    assert report['raw']['event_rates']['any']['denominator']==3
    assert report['raw']['event_rates']['any']['unknown_or_failed']==2
    with np.load(output/'trajectories'/'incomplete'/'stages.npz',allow_pickle=False) as data:
        assert 'raw_future_delta' in data.files and 'decoded_absolute_states' not in data.files
    with np.load(output/'trajectories'/'valid'/'stages.npz',allow_pickle=False) as data:
        assert 'decoded_absolute_states' in data.files and 'guided_states' not in data.files
    with pytest.raises(FileExistsError):
        validate_labels(dataset,config,output,repo)
