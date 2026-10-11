"""Synthetic trace fixtures to test the calibration/test protocol locks."""
from copy import deepcopy
import json
import yaml
import pytest
from sumodiff.models import ModelConfig
from sumodiff.diffusion.epoch_training import epoch_config
from sumodiff.diffusion.fresh_protocol import recipe_id,verify_calibration,verify_test_seal
from sumodiff.diffusion.data import dataset_identity
from sumodiff.experiments.recorder import file_identity,load_config
from sumodiff.data.dataset import write_json

def test_calibration_lock_accepts_matching_record_and_rejects_recipe_or_metrics_drift(tmp_path):
    c=epoch_config(load_config('configs/train/stage5a_epochs.yaml'))
    model=ModelConfig();data={'manifest':{'sha256':'synthetic-data'},'index':{'sha256':'synthetic-index'}}
    c.update(tail_weight=.0003,calibration_report=str(tmp_path/'calibration.json'))
    report=dict(schema_version='sumodiff.fresh.calibration.v1',passed=True,development_only=False,
        recipe_sha256=recipe_id(c,model),dataset_hashes={k:v['sha256'] for k,v in data.items()},selected_tail_weight=.0003)
    write_json(tmp_path/'calibration.json',report);write_json(tmp_path/'metrics.json',report)
    write_json(tmp_path/'manifest.json',dict(formal=True,git={'dirty':False}))
    write_json(tmp_path/'status.json',dict(state='completed'))
    before=deepcopy(c);before.update(tail_weight=None,calibration_report=None)
    (tmp_path/'resolved_config.yaml').write_text(yaml.safe_dump({'training':before}))
    assert verify_calibration(c,model,data)['sha256']
    bad=deepcopy(c);bad['raw_loss']['unified']['imitation_weight']=.001
    with pytest.raises(ValueError,match='exact fresh'):verify_calibration(bad,model,data)
    report['extra_unrecorded_change']=True;write_json(tmp_path/'calibration.json',report)
    with pytest.raises(ValueError,match='recorded metrics'):verify_calibration(c,model,data)

def test_test_lock_rejects_checkpoint_and_sampling_changes_without_loading_labels(tmp_path):
    data=tmp_path/'data';data.mkdir()
    write_json(data/'dataset_manifest.json',{'synthetic':'metadata only'})
    write_json(data/'windows.json',[])
    cp=tmp_path/'synthetic-checkpoint';cp.write_bytes(b'synthetic unit identity')
    directory=tmp_path/'seal';directory.mkdir()
    hashes={k:v['sha256'] for k,v in dataset_identity(data).items()}
    config=dict(split='test',seed=20261027,limit=600)
    doc=dict(schema_version='sumodiff.fresh.test.seal.v1',checkpoint=file_identity(cp),dataset_hashes=hashes,sampling=config)
    write_json(directory/'test_seal.json',doc)
    (directory/'resolved_config.yaml').write_text(yaml.safe_dump(doc))
    write_json(directory/'manifest.json',dict(formal=True,git={'dirty':False}))
    write_json(directory/'status.json',dict(state='completed'))
    assert verify_test_seal(directory/'test_seal.json',data,cp,config)
    with pytest.raises(ValueError,match='differ'):verify_test_seal(directory/'test_seal.json',data,cp,{**config,'seed':0})
    cp.write_bytes(b'another checkpoint')
    with pytest.raises(ValueError,match='differ'):verify_test_seal(directory/'test_seal.json',data,cp,config)
