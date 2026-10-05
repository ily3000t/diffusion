from pathlib import Path
import numpy as np
import pytest
import torch
from sumodiff.models.config import resolve_model_config
from sumodiff.models.inputs import prepare_conditioning,lane_rules,route_features


def batch():
    lanes=[dict(id='lane',edge_id='road')]
    exact=dict(lanes=lanes,connections=[],junctions=[])
    meta=dict(planned_routes=[['road'],None],route_indices_at_t0=[0,None],
        raster_row_direction='positive_y_to_negative_y',map_extent_m=[-192,-192,192,192])
    h=np.zeros((1,2,21,6),np.float32);h[0,0,:,5]=1
    raster=np.zeros((1,3,256,256),np.uint8);raster[0,0,10,10]=1
    conditioning=dict(history=h,agent_mask=np.array([[True,False]]),history_mask=np.tile([[True],[False]],(1,21))[None],
        attributes=np.array([[[4,2,1,1],[0,0,0,0]]],np.float32),initial_positions=np.zeros((1,2,2),np.float32),
        map_raster=raster,lane_polylines=np.zeros((1,1,64,8),np.float32),lane_mask=np.ones((1,1),bool),
        lane_point_mask=np.ones((1,1,64),bool),lane_adjacency=np.zeros((1,1,1),bool),route_lane_mask=np.array([[[True],[False]]]))
    return dict(conditioning=conditioning,exact_map=[exact],input_metadata=[meta],
        targets={'future':np.full((1,2,40,6),np.nan),'future_mask':np.zeros((1,2,40),bool)})


def test_adapter_never_reads_targets_and_keeps_binary_raster_and_t0_progress():
    class Unreadable:
        def __getitem__(self,key):raise AssertionError('Future label access is forbidden')
    source=batch();source['targets']=Unreadable()
    c=prepare_conditioning(source)
    assert c['map_raster'].max()==1 and c['map_raster'][0,0,10,10]==1
    assert c['route_lane_features'][0,0,0,1]==1 and not c['route_lane_features'][0,1].any()
    assert 'future_mask' not in c and c['lane_rule_relations'].dtype==torch.bool
    source['conditioning']['future_mask']=np.ones((1,2,40),bool)
    with pytest.raises(ValueError,match='labels are forbidden'):prepare_conditioning(source)


def test_adapter_rejects_unknown_orientation_and_lane_order_count():
    source=batch();source['input_metadata'][0]['raster_row_direction']='negative_y_to_positive_y'
    with pytest.raises(ValueError,match='orientation'):prepare_conditioning(source)
    source=batch();source['conditioning']['lane_mask'][0,0]=False
    with pytest.raises(ValueError,match='lane order/count'):prepare_conditioning(source)


def test_config_rejects_unsupported_switch_and_architecture_options():
    for setting in ('cfg','attack_role_embedding','partial_diffusion','rolling_generation'):
        with pytest.raises(NotImplementedError):resolve_model_config({setting:True})
    with pytest.raises(ValueError):resolve_model_config({'fusion':'unknown'})
    with pytest.raises(NotImplementedError):resolve_model_config({'future_points':50})


def test_model_input_audit_uses_relocated_current_files_without_label_files(tmp_path):
    from sumodiff.data.dataset import save_window,write_json
    from sumodiff.data.windows import Window
    from sumodiff.experiments.recorder import file_identity
    from sumodiff.models.probes import collect_inputs
    from sumodiff.models.config import ModelConfig
    original=tmp_path/'original';entries=[]
    for i,family in enumerate(('straight','ramp','intersection')):
        source=batch();c={key:value[0] for key,value in source['conditioning'].items()}
        c['lane_polylines'][...,2]=1;c['lane_polylines'][...,4]=4;c['lane_polylines'][...,5]=20
        meta=dict(source['input_metadata'][0],window_id=f'w{i}',split='validation' if i==1 else 'train',
            family=family,episode_key=f'e{i}',geometry_id=f'g{i}')
        w=Window(c,dict(future=np.zeros((2,40,6),np.float32),future_mask=np.ones((2,40),bool)),
                 meta,dict(core_training_eligible=True),source['exact_map'][0])
        entries.append(save_window(original/'windows'/meta['window_id'],w))
    write_json(original/'windows.json',entries)
    write_json(original/'dataset_manifest.json',dict(schema_version='sumodiff.dataset.v1',window_index=file_identity(original/'windows.json')))
    moved=tmp_path/'relocated';original.rename(moved)
    for entry in entries:
        for name in ('targets.npz','labels.json'):(moved/entry['files'][name]['relative_path']).unlink()
    c,audit,files=collect_inputs(moved,ModelConfig())
    assert audit['inference_windows']==3 and not audit['future_labels_accessed']
    assert c['history'].shape==(3,2,21,6)
    assert all(Path(path).is_relative_to(moved) and Path(path).is_file() for path in files)
