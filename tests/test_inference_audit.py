import json
from pathlib import Path
import pytest
from sumodiff.data.dataset import WindowDataset,write_json
from sumodiff.diffusion.inference_audit import make_projection
from sumodiff.diffusion.data import choose_indices
from sumodiff.experiments.recorder import file_identity


def test_relocated_projection_contains_no_labels_and_keeps_incomplete_task(tmp_path):
    root=tmp_path/'original';d=root/'windows'/'test';d.mkdir(parents=True)
    meta=dict(family='a',agent_ids=['a','b',None],window_id='test')
    write_json(d/'input.json',meta)
    for name in ('conditioning.npz','map.json','targets.npz','labels.json'): (d/name).write_bytes(b'synthetic fixture')
    identities={p.name:{**file_identity(p),'relative_path':str(p.relative_to(root))} for p in d.iterdir()}
    write_json(root/'windows.json',[dict(window_id='test',split='validation',core_training_eligible=False,files=identities)])
    write_json(root/'dataset_manifest.json',dict(schema_version='sumodiff.dataset.v1',window_index=file_identity(root/'windows.json')))
    ds=WindowDataset(root,'validation');indices=choose_indices(ds,1,2);assert indices==[0]
    paths=make_projection(ds,indices,tmp_path/'projection')
    assert {p.name for p in paths}=={'input.json','conditioning.npz','map.json'}
    assert not list((tmp_path/'projection').rglob('targets.npz')) and not list((tmp_path/'projection').rglob('labels.json'))
    relocated=WindowDataset(tmp_path/'projection','validation')
    assert choose_indices(relocated,1,2)==indices
    with pytest.raises(FileNotFoundError): relocated._path(relocated.entries[0],'targets.npz')
