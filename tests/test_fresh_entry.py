"""Synthetic engineering fixtures only; never scientific experiment data."""
from copy import deepcopy
import json
from pathlib import Path
import yaml
import torch
import pytest
from test_stage5a import synthetic_window
from sumodiff.data.dataset import save_window,write_json
from sumodiff.experiments.recorder import file_identity
from sumodiff.experiments.stage5a_plan import FAMILIES
from sumodiff.models import ModelConfig
from sumodiff.diffusion.epoch_training import epoch_config,verified_curated_dataset
from sumodiff.diffusion.cli import audit
from sumodiff.diffusion.training import train
from sumodiff.diffusion.checkpoint import load_checkpoint
from sumodiff.diffusion.audit import exact_tree

def dataset(tmp_path):
    base=tmp_path/'source';base.mkdir()
    window,_,_,_=synthetic_window(base)
    dest=tmp_path/'data';entries=[];rows=[]
    for split in ('train','validation','test'):
        for family in FAMILIES:
            w=deepcopy(window);wid=f'{split}-{family}'
            w.input_metadata.update(window_id=wid,split=split,family=family,geometry_id=wid,episode_key=wid)
            e=save_window(dest/'windows'/wid,w);e['family']=family;entries.append(e)
            rows.append(dict(window_id=wid,split=split,family=family,geometry_id=wid,episode_key=wid,
                eligible=True,core_complete=True,reasons=[]))
    write_json(dest/'windows.json',entries)
    write_json(dest/'selection.json',dict(schema_version='sumodiff.stage5a.selection.v2',profile='full',
        selected=rows,quotas={s:{f:1 for f in FAMILIES} for s in ('train','validation','test')},
        selection={s:{f:{'passed':True} for f in FAMILIES} for s in ('train','validation','test')}))
    write_json(dest/'dataset_manifest.json',dict(schema_version='sumodiff.dataset.v1',
        window_index=file_identity(dest/'windows.json'),
        curation=dict(label_quality_filtered=True,selection=file_identity(dest/'selection.json'))))
    return dest

def source_config():
    return dict(schema_version='sumodiff.base.epoch.config.v2',budget_kind='full',device='cpu',
        batch_size=1,epochs=20,expected_train_windows=3,expected_validation_windows=3,
        expected_test_windows=3,lr_warmup_epochs=1,auxiliary_warmup_epochs=1,train_probe_limit=3,
        validation_every_epochs=1,checkpoint_every_epochs=5,monitor_every_epochs=0,
        tail_weight=1e-6,calibration_report='synthetic-fixture-not-production')

def test_existing_training_entry_real_model_prefix_resume_is_exact_and_never_loads_test_labels(tmp_path,monkeypatch):
    import sumodiff.diffusion.fresh_protocol as protocol
    import sumodiff.diffusion.training as training
    data=dataset(tmp_path);cfg=source_config();config=tmp_path/'training.yaml';config.write_text(yaml.safe_dump(cfg))
    model=tmp_path/'model.yaml';model.write_text(yaml.safe_dump(ModelConfig(condition_dimension=16,
        unet_channels=(8,16,32),raster_channels=(8,8,8,8,8)).to_dict()))
    audit(data,config,tmp_path/'scale',Path.cwd(),False,['synthetic-scale-test'])
    monkeypatch.setattr(protocol,'verify_calibration',lambda *args:{'sha256':'synthetic-test-only','location':str(config)})
    original=training.labeled_cache;seen=[]
    def guarded(directory,split,limit):
        assert split in ('train','validation');seen.append(split)
        return original(directory,split,limit)
    monkeypatch.setattr(training,'labeled_cache',guarded)
    scale=tmp_path/'scale'/'scale_audit.json'
    def run(name,prefix,resume=None):
        return train(data,model,config,scale,tmp_path/name,Path.cwd(),['synthetic-prefix-test'],
            False,resume,False,prefix)
    first=run('first',4);second=run('resumed',3,tmp_path/'first'/'checkpoint_step_000004.pt')
    uninterrupted=run('uninterrupted',7)
    assert not first['complete_budget'] and second['end_step']==7 and uninterrupted['end_step']==7
    a=load_checkpoint(tmp_path/'resumed'/'checkpoint_step_000007.pt')
    b=load_checkpoint(tmp_path/'uninterrupted'/'checkpoint_step_000007.pt')
    for key in ('model','optimizer','generator_state','cpu_rng'):
        assert exact_tree(a[key],b[key])
    assert a['metadata']['runtime']['update_schedule']==b['metadata']['runtime']['update_schedule']
    assert seen==['train','validation']*3
    with pytest.raises(ValueError,match='signature'):
        other=deepcopy(a['metadata']['training_signature']);other['training']['epochs']=21
        load_checkpoint(tmp_path/'resumed'/'checkpoint_step_000007.pt',signature=other)

def test_three_split_curation_rejects_test_geometry_leakage(tmp_path):
    data=dataset(tmp_path);c=epoch_config(source_config())
    assert verified_curated_dataset(data,c)
    doc=json.loads((data/'selection.json').read_text())
    entries=json.loads((data/'windows.json').read_text())
    doc['selected'][6]['geometry_id']=doc['selected'][0]['geometry_id']
    entries[6]['geometry_id']=entries[0]['geometry_id']
    write_json(data/'selection.json',doc);write_json(data/'windows.json',entries)
    m=json.loads((data/'dataset_manifest.json').read_text())
    m['window_index']=file_identity(data/'windows.json');m['curation']['selection']=file_identity(data/'selection.json')
    write_json(data/'dataset_manifest.json',m)
    with pytest.raises(ValueError,match='leakage'):verified_curated_dataset(data,c)
