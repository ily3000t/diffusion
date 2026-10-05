from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
import torch
from sumodiff.models import ModelConfig,ConditionalDenoiser,resolve_model_config
from sumodiff.models.attention import MaskedAttention,AgentAttention
from sumodiff.models.inputs import lane_rules,route_features,prepare_conditioning


@pytest.fixture(scope='module',autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def small_config():
    return ModelConfig(condition_dimension=32,unet_channels=(16,32,64),raster_channels=(8,16,16,32,32))


def conditioning(n=4):
    torch.manual_seed(101)
    a=torch.arange(n)[None,:]<2;hm=a[...,None].expand(1,n,21).clone();hm[0,0,5]=False
    h=torch.zeros(1,n,21,6);h[...,5]=1
    h[0,0,:,0]=torch.linspace(-10,0,21);h[0,1,:,0]=torch.linspace(2,12,21);h[...,2]=5
    attrs=torch.tensor([4.,2.,1.,0.]).expand(1,n,4).clone();attrs[0,0,3]=1
    initial=h[:,:,-1,:2].clone()
    lm=torch.tensor([[True,True,False]]);pm=lm[...,None].expand(1,3,64).clone();pm[0,0,10]=False
    lanes=torch.zeros(1,3,64,8);lanes[...,0]=torch.linspace(-60,60,64);lanes[0,1,:,1]=4
    lanes[...,2]=1;lanes[...,4]=4;lanes[...,5]=20;lanes[...,7]=2
    adjacency=torch.zeros(1,3,3,dtype=torch.bool);adjacency[0,0,1]=True
    rules=torch.zeros(1,3,3,2,dtype=torch.bool);rules[0,0,1]=True;rules[0,1,0,1]=True
    rm=a[...,None]&lm[:,None,:]
    rf=torch.zeros(1,n,3,2);rf[:,:,:2,1]=1;rf[:,:,1,0]=.5
    raster=torch.zeros(1,3,256,256);raster[:,:,124:132,:]=1;raster[:,2,127,:]=1
    return dict(history=h,history_mask=hm,agent_mask=a,attributes=attrs,initial_positions=initial,
        map_raster=raster,map_extent_m=torch.tensor([[-192.,-192.,192.,192.]]),
        lane_polylines=lanes,lane_mask=lm,lane_point_mask=pm,lane_adjacency=adjacency,
        lane_rule_relations=rules,route_lane_mask=rm,route_lane_features=rf)


def vehicle_permutation(c,order):
    fields={'history','history_mask','agent_mask','attributes','initial_positions','route_lane_mask','route_lane_features'}
    return {key:value[:,order] if key in fields else value.clone() for key,value in c.items()}


@pytest.mark.parametrize('fusion',['hierarchical','parallel'])
def test_vehicle_reordering_and_extra_padding_preserve_outputs(fusion):
    torch.manual_seed(4);model=ConditionalDenoiser(small_config()).eval();c=conditioning()
    noise=torch.randn(1,4,40,6);time=torch.tensor([300]);order=torch.tensor([2,1,3,0])
    expected=model(noise,time,c,fusion=fusion,return_details=True)
    actual=model(noise[:,order],time,vehicle_permutation(c,order),fusion=fusion,return_details=True)
    torch.testing.assert_close(actual['pred_noise'],expected['pred_noise'][:,order],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(actual['condition'],expected['condition'][:,order],atol=2e-6,rtol=2e-5)
    reduced=vehicle_permutation(c,torch.tensor([0,1]))
    torch.testing.assert_close(model(noise[:,:2],time,reduced,fusion=fusion),expected['pred_noise'][:,:2],atol=2e-6,rtol=2e-5)
    assert expected['condition'].shape==(1,4,32) and not expected['pred_noise'][:,2:].any()
    torch.testing.assert_close(expected['gates'][c['agent_mask']].sum(dim=-1),torch.ones(2))


@pytest.mark.parametrize('fusion',['hierarchical','parallel'])
def test_masked_nan_payload_has_no_effect_and_zero_gradient(fusion):
    torch.manual_seed(5);model=ConditionalDenoiser(small_config()).eval();c=conditioning();time=torch.tensor([120])
    noise=torch.randn(1,4,40,6)
    expected=model(noise,time,c,fusion=fusion)
    corrupted=deepcopy(c)
    a=c['agent_mask'];hm=c['history_mask']&a[...,None];pm=c['lane_point_mask']&c['lane_mask'][...,None]
    corrupted['history'][~hm]=float('nan');corrupted['lane_polylines'][~pm]=float('nan')
    for key in ('attributes','initial_positions'):corrupted[key][~a]=float('nan')
    corrupted['route_lane_features'][~c['route_lane_mask']]=float('nan')
    # Invalid lane graph edges cannot carry messages to/from real lanes.
    corrupted['lane_adjacency'][:,2,:]=True;corrupted['lane_adjacency'][:,:,2]=True
    corrupted['lane_rule_relations'][:,2,:,:]=True;corrupted['lane_rule_relations'][:,:,2,:]=True
    noise[~a]=float('nan');noise.requires_grad_(True);corrupted['history'].requires_grad_(True)
    corrupted['lane_polylines'].requires_grad_(True);corrupted['route_lane_features'].requires_grad_(True)
    actual=model(noise,time,corrupted,fusion=fusion)
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
    actual[a].square().mean().backward()
    for tensor,mask in ((noise,a),(corrupted['history'],hm),(corrupted['lane_polylines'],pm),
                        (corrupted['route_lane_features'],c['route_lane_mask'])):
        assert torch.isfinite(tensor.grad).all() and not tensor.grad[~mask].any()


def test_lane_reordering_respects_graph_and_route_axes():
    model=ConditionalDenoiser(small_config()).eval();c=conditioning();order=torch.tensor([1,2,0]);changed=deepcopy(c)
    for key in ('lane_polylines','lane_mask','lane_point_mask'):changed[key]=c[key][:,order]
    for key in ('lane_adjacency','lane_rule_relations'):changed[key]=c[key][:,order][:,:,order]
    for key in ('route_lane_mask','route_lane_features'):changed[key]=c[key][:,:,order]
    noise=torch.randn(1,4,40,6);time=torch.tensor([800])
    torch.testing.assert_close(model(noise,time,changed),model(noise,time,c),atol=2e-6,rtol=2e-5)


def test_shared_modes_have_same_parameters_but_different_dependency():
    cfg=small_config();model=ConditionalDenoiser(cfg);parallel=ConditionalDenoiser(replace(cfg,fusion='parallel'))
    parallel.load_state_dict(model.state_dict());assert list(model.state_dict())==list(parallel.state_dict())
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in parallel.parameters())
    c=conditioning();noise=torch.randn(1,4,40,6);time=torch.tensor([7])
    torch.testing.assert_close(model(noise,time,c,fusion='parallel'),parallel(noise,time,c))
    assert not torch.allclose(model(noise,time,c),parallel(noise,time,c))
    for kwargs in ({'cfg':True},{'partial_diffusion':True},{'rolling_generation':True},{'attack_role_embedding':True}):
        with pytest.raises(NotImplementedError):resolve_model_config(kwargs)
    with pytest.raises(ValueError):resolve_model_config({'fusion':'unknown'})
    with pytest.raises(ValueError):resolve_model_config({'future_mask':True})


def test_all_masked_attention_and_per_time_bottleneck_isolation():
    attention=MaskedAttention(16,4)
    q=torch.randn(2,3,16,requires_grad=True);k=torch.full((2,4,16),float('nan'),requires_grad=True)
    result=attention(q,k,torch.ones(2,3,dtype=torch.bool),torch.zeros(2,4,dtype=torch.bool))
    assert torch.isfinite(result).all() and not result.any()
    result.sum().backward();assert torch.isfinite(k.grad).all() and not k.grad.any()
    social=AgentAttention(16,4);value=torch.randn(1,3,10,16);a=torch.ones(1,3,dtype=torch.bool)
    expected=social(value,a);altered=value.clone();altered[:,:,4,:]+=torch.randn(1,3,16)
    actual=social(altered,a)
    torch.testing.assert_close(actual[:,:,:4],expected[:,:,:4]);torch.testing.assert_close(actual[:,:,5:],expected[:,:,5:])
    assert not torch.allclose(actual[:,:,4],expected[:,:,4])


def test_condition_paths_and_timestep_reach_output_with_finite_gradients():
    model=ConditionalDenoiser(small_config());c=conditioning()
    keys=('history','attributes','initial_positions','map_raster','lane_polylines','route_lane_features')
    for key in keys:c[key].requires_grad_(True)
    noise=torch.randn(1,4,40,6,requires_grad=True);time=torch.tensor([10]);result=model(noise,time,c)
    result[:,:2].square().mean().backward()
    for key in keys:
        assert c[key].grad is not None and torch.isfinite(c[key].grad).all() and c[key].grad.abs().sum()>0,key
    assert torch.isfinite(noise.grad).all()
    for name,module in (('history',model.condition_encoder.encoders.history),('raster',model.condition_encoder.encoders.raster),
        ('lanes',model.condition_encoder.encoders.lanes),('fusion',model.condition_encoder),('unet',model.unet)):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()),name
    assert not torch.allclose(model(noise.detach(),time,c),model(noise.detach(),time+1,c))
    cross=torch.autograd.grad(model(noise,time,c)[0,0].square().sum(),noise)[0]
    assert cross[0,1].abs().sum()>0  # noisy future of the other vehicle enters joint denoising.
    changed=deepcopy({key:value.detach() for key,value in c.items()})
    changed['lane_adjacency'].zero_();changed['lane_rule_relations'].zero_()
    assert not torch.allclose(model(noise,time,changed),model(noise,time,c))


def test_invalid_active_data_masks_and_timestep_are_rejected():
    model=ConditionalDenoiser(small_config());c=conditioning();noise=torch.randn(1,4,40,6);time=torch.tensor([10])
    changed=deepcopy(c);changed['history'][0,0,0,0]=float('nan')
    with pytest.raises(ValueError,match='active history'):model(noise,time,changed)
    changed=deepcopy(c);changed['future_mask']=torch.ones(1,4,40,dtype=torch.bool)
    with pytest.raises(ValueError,match='labels are forbidden'):model(noise,time,changed)
    for invalid in (torch.tensor([-.1]),torch.tensor([1000]),torch.tensor([0,1])):
        with pytest.raises(ValueError,match='Timestep'):model(noise,invalid,c)
    changed=deepcopy(c);changed['map_raster']*=255
    with pytest.raises(ValueError,match='0/255'):model(noise,time,changed)


def map_metadata():
    exact=dict(lanes=[dict(id='in',edge_id='a'),dict(id='via0',edge_id=':j0'),dict(id='via1',edge_id=':j1'),dict(id='out',edge_id='b')],
        connections=[dict(source_lane='in',target_lane='out',via_lane='via0',junction_id='j',request_index=0,sumo_raw={'from':'a','to':'b'}),
                     dict(source_lane='in',target_lane='out',via_lane='via1',junction_id='j',request_index=1,sumo_raw={'from':'a','to':'b'}),
                     dict(source_lane='via0',target_lane='out',via_lane=None,sumo_raw={'from':':j0','to':'b'}),
                     dict(source_lane='via1',target_lane='out',via_lane=None,sumo_raw={'from':':j1','to':'b'})],
        junctions=[dict(sumo_raw={'id':'j'},requests=[dict(index='0',yield_to_indices=[1],foe_indices=[1]),dict(index='1',yield_to_indices=[],foe_indices=[0])])])
    meta=dict(planned_routes=[['a','b']],route_indices_at_t0=[0])
    return exact,meta


def test_route_order_and_directional_yield_rules_use_current_metadata():
    exact,meta=map_metadata();rules=lane_rules(exact)
    assert rules[1,2,0] and not rules[2,1,0] and rules[1,2,1] and rules[2,1,1]
    features=route_features(exact,meta,np.ones((1,4),bool),np.ones(1,bool))
    np.testing.assert_array_equal(features[0,:,0],[0,.5,.5,1])
    meta['route_indices_at_t0']=[1]
    changed=route_features(exact,meta,np.ones((1,4),bool),np.ones(1,bool))
    np.testing.assert_array_equal(changed[0,:,0],[-1,-.5,-.5,0])
    np.testing.assert_array_equal(changed[0,:,1],[0,0,0,1])


def test_resource_probe_respects_embedding_range_and_does_not_update_parameters():
    from sumodiff.models.probes import parameter_hash,profile
    model=ConditionalDenoiser(replace(small_config(),diffusion_embedding_steps=3));before=parameter_hash(model)
    report=profile(model,conditioning(),torch.device('cpu'),'hierarchical',100,1,1)
    assert report['status']=='completed' and report['peak_allocated_bytes'] is None
    assert report['forward_backward_mean_seconds']>0 and parameter_hash(model)==before
