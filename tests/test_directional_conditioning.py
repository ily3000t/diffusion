from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from test_models import small_config,conditioning,vehicle_permutation
from test_model_inputs import batch
from sumodiff.models import ModelConfig,ConditionalDenoiser,prepare_conditioning
from sumodiff.models.inputs import lane_change_permissions,validate_conditioning
from sumodiff.diffusion.checkpoint import load_checkpoint,restore_training

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)

def config():
    return replace(small_config(),schema_version='sumodiff.model.control.config.v2',state_dimension=3)

def inputs():
    c=conditioning()
    c['lane_change_permissions']=torch.tensor([[[True,False],[False,True],[False,False]]])
    return c

@pytest.mark.parametrize('classes,expected',[(None,True),([],False),(['passenger'],True),(['all'],True),(['emergency','taxi'],False)])
def test_class_permissions_are_directional_and_absent_means_unrestricted(classes,expected):
    lane=dict(id='lane',edge_id='road')
    if classes is not None:lane['change_left_classes']=classes
    exact=dict(lanes=[lane])
    assert lane_change_permissions(exact).tolist()==[[expected,True]]
    lane['change_right_classes']=['emergency']
    assert lane_change_permissions(exact).tolist()==[[expected,False]]

def test_explicit_schema_preserves_legacy_config_and_state_keys():
    legacy=ModelConfig()
    assert not legacy.directional_lane_permissions
    assert 'directional_lane_permissions' not in legacy.to_dict()
    assert config().directional_lane_permissions
    with pytest.raises(NotImplementedError):replace(config(),state_dimension=6)
    with pytest.raises(ValueError):ModelConfig(schema_version='sumodiff.model.config.v2')
    old=ConditionalDenoiser(replace(config(),schema_version='sumodiff.model.control.config.v1'))
    new=ConditionalDenoiser(config())
    assert not any('.permissions.' in k for k in old.state_dict())
    assert len([k for k in new.state_dict() if '.permissions.' in k])==4
    with pytest.raises(RuntimeError):new.load_state_dict(old.state_dict())

def test_new_adapter_reads_t0_rules_without_touching_labels_or_legacy_inputs():
    class Unreadable:
        def __getitem__(self,key):raise AssertionError('future labels accessed')
    b=batch();b['targets']=Unreadable();b['label_metadata']=Unreadable()
    b['exact_map'][0]['lanes'][0]['change_right_classes']=['emergency']
    with pytest.raises(NotImplementedError):prepare_conditioning(b)
    c=prepare_conditioning(b,config=config())
    assert c['lane_change_permissions'].dtype==torch.bool
    assert c['lane_change_permissions'].tolist()==[[[True,False]]]
    validate_conditioning(c,config())
    with pytest.raises(ValueError,match='architecture'):validate_conditioning(c,replace(config(),schema_version='sumodiff.model.control.config.v1'))
    assert 'lane_change_permissions' not in b['conditioning']

@pytest.mark.parametrize('payload',['passenger',[False],[None],['']])
def test_invalid_class_metadata_is_rejected(payload):
    with pytest.raises(ValueError,match='classes'):
        lane_change_permissions(dict(lanes=[dict(change_left_classes=payload)]))

@pytest.mark.parametrize('fusion',['hierarchical','parallel'])
def test_permissions_reach_prediction_and_gradient_with_vehicle_and_lane_reordering(fusion):
    torch.manual_seed(13);model=ConditionalDenoiser(replace(config(),fusion=fusion)).eval()
    c=inputs();noise=torch.randn(1,4,40,3);time=torch.tensor([200])
    expected=model(noise,time,c)
    changed=deepcopy(c);changed['lane_change_permissions'][0,0,0]=False
    delta=(model(noise,time,changed)-expected).abs().max()
    assert float(delta)>1e-8
    order=torch.tensor([2,1,3,0])
    torch.testing.assert_close(model(noise[:,order],time,vehicle_permutation(c,order)),expected[:,order],atol=2e-6,rtol=2e-5)
    changed=deepcopy(c);order=torch.tensor([1,2,0])
    for key in ('lane_polylines','lane_mask','lane_point_mask','lane_change_permissions'):changed[key]=c[key][:,order]
    for key in ('lane_adjacency','lane_rule_relations'):changed[key]=c[key][:,order][:,:,order]
    for key in ('route_lane_mask','route_lane_features'):changed[key]=c[key][:,:,order]
    torch.testing.assert_close(model(noise,time,changed),expected,atol=2e-6,rtol=2e-5)
    expected.square().mean().backward()
    module=model.condition_encoder.encoders.lanes.permissions
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())

@pytest.mark.parametrize('fusion',['hierarchical','parallel'])
def test_padding_and_invalid_point_masks_cannot_carry_permissions(fusion):
    model=ConditionalDenoiser(replace(config(),fusion=fusion)).eval();c=inputs()
    noise=torch.randn(1,4,40,3);time=torch.tensor([300])
    expected=model(noise,time,c)
    changed=deepcopy(c);changed['lane_change_permissions'][:,2]=True
    torch.testing.assert_close(model(noise,time,changed),expected,atol=1e-7,rtol=1e-6)
    # An entire token with no valid polyline points is masked even when lane_mask=True.
    changed['lane_mask'][:,2]=True
    torch.testing.assert_close(model(noise,time,changed),expected,atol=1e-7,rtol=1e-6)
    wrong=deepcopy(c);wrong['lane_change_permissions']=wrong['lane_change_permissions'].float()
    with pytest.raises(ValueError,match='boolean'):model(noise,time,wrong)
    wrong=deepcopy(c);wrong['lane_change_permissions']=wrong['lane_change_permissions'][...,:1]
    with pytest.raises(ValueError,match='shape'):model(noise,time,wrong)
    wrong=deepcopy(c);wrong.pop('lane_change_permissions')
    with pytest.raises(ValueError,match='architecture'):model(noise,time,wrong)

def test_real_control_cache_uses_the_explicit_model_schema(tmp_path):
    from types import SimpleNamespace
    from sumodiff.data.dataset import save_window,write_json
    from sumodiff.experiments.recorder import file_identity
    from sumodiff.diffusion.control_training import control_cache
    b=batch();b['exact_map'][0]['lanes'][0]['change_left_classes']=['emergency']
    meta=dict(b['input_metadata'][0],window_id='fixture',split='train',family='ramp_merge',episode_key='ep',geometry_id='geo')
    c={k:v[0] for k,v in b['conditioning'].items()}
    labels=dict(future=np.zeros((2,40,6),np.float32),future_mask=np.array([[True]*40,[False]*40]))
    w=SimpleNamespace(conditioning=c,targets=labels,input_metadata=meta,label_metadata=dict(core_training_eligible=True),exact_map=b['exact_map'][0])
    data=tmp_path/'data';entry=save_window(data/'windows'/'fixture',w)
    write_json(data/'windows.json',[entry])
    write_json(data/'dataset_manifest.json',dict(schema_version='sumodiff.dataset.v1',window_index=file_identity(data/'windows.json')))
    plan=tmp_path/'tasks.json';write_json(plan,dict(schema_version='sumodiff.control_replay.tasks.v1',tasks=[dict(window_id='fixture',split='train',group='ordinary')]))
    cfg=dict(task_plan=str(plan),train_limit=1)
    with pytest.raises(NotImplementedError):control_cache(data,'train',cfg)
    cached=control_cache(data,'train',cfg,model_config=config())
    assert cached[0]['lane_change_permissions'].tolist()==[[[False,True]]]
    assert cached[2]==['fixture'] and cached[1]['future_mask'][0,0].all()
    with pytest.raises(ValueError,match='test'):control_cache('unopened','test',cfg,model_config=config())

def test_new_checkpoint_exact_continuation_and_metadata_tampering(tmp_path,monkeypatch):
    import test_control_target_training as t
    old_setup=t.setup
    def setup():
        c,d,raw,targets=old_setup();c['lane_change_permissions']=inputs()['lane_change_permissions']
        return c,d,raw,targets
    monkeypatch.setattr(t,'setup',setup);monkeypatch.setattr(t,'model_config',config)
    t.test_control_checkpoint_resume_preserves_actual_model_optimizer_rng_and_schedule(tmp_path)
    path=tmp_path/'control.pt';payload=load_checkpoint(path)
    model=ConditionalDenoiser(replace(config(),schema_version='sumodiff.model.control.config.v1'))
    before=deepcopy(model.state_dict())
    with pytest.raises(ValueError,match='architecture'):restore_training(payload,model,torch.optim.AdamW(model.parameters()),torch.Generator(),'cpu')
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    for mode in ('missing','shape','schema'):
        wrong=deepcopy(payload)
        key='condition_encoder.encoders.lanes.permissions.0.weight'
        if mode=='missing':wrong['model'].pop(key)
        elif mode=='shape':wrong['model'][key]=torch.zeros(1,2)
        else:
            wrong['metadata']['model_config']['schema_version']='sumodiff.model.control.config.v1'
            wrong['metadata']['training_signature']['model']=deepcopy(wrong['metadata']['model_config'])
        dest=tmp_path/(mode+'.pt');torch.save(wrong,dest)
        with pytest.raises(ValueError,match='checkpoint/config mismatch'):load_checkpoint(dest)

def test_new_model_resource_probes_use_three_future_channels():
    from sumodiff.models.probes import invariance_probe,profile,parameter_hash
    model=ConditionalDenoiser(config());c=inputs();before=parameter_hash(model)
    checks,arrays=invariance_probe(model,c,71)
    assert arrays['noisy_future'].shape==(1,4,40,3)
    assert checks['hierarchical']['all_parameter_gradients_finite']
    resource=profile(model,c,torch.device('cpu'),'hierarchical',71,1,1)
    assert resource['status']=='completed' and before==parameter_hash(model)


def test_gpu_protocol_rejects_scope_and_budget_changes():
    from sumodiff.experiments.recorder import load_config
    from sumodiff.models.directional_probe import resolve_probe
    from pathlib import Path
    c=load_config(Path(__file__).parents[1]/'configs/experiments/directional_conditioning.yaml')
    assert resolve_probe(c)==c
    for key,value in [('collection','test-source'),('reference_tick',600),('iterations',20),('synthetic_batch_size',16),('warmups',True)]:
        wrong=deepcopy(c);wrong[key]=value
        with pytest.raises(ValueError,match='frozen'):resolve_probe(wrong)
