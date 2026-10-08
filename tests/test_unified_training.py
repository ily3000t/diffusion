from copy import deepcopy
import pytest
import torch
from test_controlled_training import parent_fixture
from sumodiff.diffusion.unified_training import (DEFAULTS,unified_config,unified_signature,new_optimizer,
    warm_start,optimize_batch,quality_frontier)
from sumodiff.diffusion.unified_losses import UnifiedLossConfig
from sumodiff.diffusion.schedule import UpdateSchedule,resolve_schedule
from sumodiff.diffusion.checkpoint import save_checkpoint,load_checkpoint,restore_training
from sumodiff.diffusion.audit import exact_tree
from sumodiff.models import ModelConfig
from test_physical_losses import fixture
from sumodiff.diffusion import NoiseSchedule


def requested(control=False):
    loss=UnifiedLossConfig(imitation_weight=0 if control else .01,
                          temporal_weight=0 if control else .001,road_weight=0).to_dict()
    return dict(DEFAULTS,expected_parent_sha256='1'*64,loss=loss,
                objective_name='epsilon_cosine' if control else 'unified')


def metadata(c,p):
    m=deepcopy(p['metadata'])
    m.update(training_config=c,diffusion_config=c['diffusion'],
        training_signature=unified_signature(ModelConfig(),c,m['data'],m['scale_audit'],['train'],['val']))
    m['runtime'].update(snapshot_kind='training_recovery',
        scheduler_state=UpdateSchedule(resolve_schedule(c['schedule'])).state_dict(),
        best_epsilon=dict(step=30000,mse=.01,checkpoint={'location':'unused'}),quality_history=[])
    return m


def test_new_protocol_preserves_parent_and_has_true_fifty_epoch_schedule():
    p=parent_fixture();before=deepcopy(p)
    c=unified_config(requested(),p)
    assert p==before
    assert c['max_steps']==45000 and c['epochs']==150 and c['nominal_additional_updates']==15000
    assert c['schedule']['warmup_updates']==600 and c['schedule']['auxiliary_warmup_updates']==1500
    assert c['optimizer']['betas']==[.9,.999] and c['learning_rate']==1e-6
    assert c['monitor']['limit']==20
    prefix=unified_config(requested(),p,prefix_updates=16)
    assert prefix['max_steps']==30016 and prefix['schedule']['total_updates']==15000
    assert prefix['execution_budget_kind']=='preflight_prefix'
    assert not unified_config(requested(True),p)['diffusion']['additional_losses_enabled']


@pytest.mark.parametrize('change',[dict(additional_epochs=10),dict(warmup_epochs=1),dict(auxiliary_warmup_epochs=3),
    dict(batch_size=8),dict(expected_parent_sha256='bad'),dict(monitor_tasks=6),
    dict(objective_name='epsilon_cosine'),dict(loss={'boundary_weight':.01})])
def test_new_protocol_rejects_ambiguous_or_incompatible_changes(change):
    c=requested();c.update(change)
    with pytest.raises((ValueError,NotImplementedError)):unified_config(c,parent_fixture())


def test_checkpoint_loader_validates_schedule_and_full_objective_identity(tmp_path):
    p=parent_fixture();c=unified_config(requested(),p,prefix_updates=8)
    m=metadata(c,p);net=torch.nn.Linear(6,6);opt=new_optimizer(net,c);g=torch.Generator().manual_seed(8)
    path=tmp_path/'new.pt';save_checkpoint(path,net,opt,g,30000,m)
    assert load_checkpoint(path)['metadata']['training_config']==c
    other=deepcopy(c);other['monitor']['limit']=21
    with pytest.raises(ValueError,match='signature'):
        load_checkpoint(path,signature=unified_signature(ModelConfig(),other,m['data'],m['scale_audit'],['train'],['val']))
    bad=deepcopy(m);bad['runtime']['scheduler_state']['completed_updates']=1
    path=tmp_path/'bad.pt';save_checkpoint(path,net,opt,g,30000,bad)
    with pytest.raises(ValueError):load_checkpoint(path)


def test_model_only_warm_start_and_strict_resume_reproduce_training(tmp_path):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(6,6,dtype=torch.float64)
        def forward(self,x,t,c):
            assert 'future_mask' not in c
            return self.linear(x)
    torch.manual_seed(73);source=Toy()
    oldopt=torch.optim.AdamW(source.parameters(),lr=.0003)
    source.linear.weight.sum().backward();oldopt.step()
    p=parent_fixture();p.update(model=source.state_dict(),optimizer=oldopt.state_dict(),
        generator_state=torch.Generator().manual_seed(8).get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=[])
    p['metadata']['runtime']['torch']=str(torch.__version__)
    c=unified_config(requested(),p,prefix_updates=8)
    first=Toy();opt=new_optimizer(first,c);generator=torch.Generator()
    warm_start(p,first,opt,generator,'cpu')
    assert len(opt.state)==0 and exact_tree(first.state_dict(),p['model'])
    learning=UpdateSchedule(resolve_schedule(c['schedule']))
    conditions,targets=fixture(batch=2,agents=2);schedule=NoiseSchedule()
    objective=UnifiedLossConfig(road_weight=0)
    m=metadata(c,p)
    for update in range(1,9):
        optimize_batch(first,opt,conditions,targets,schedule,generator,1.,objective,None,learning,update)
        if update==4:
            m['runtime']['scheduler_state']=learning.state_dict()
            save_checkpoint(tmp_path/'mid.pt',first,opt,generator,30004,m)
    mid=load_checkpoint(tmp_path/'mid.pt')
    second=Toy();other_opt=new_optimizer(second,c);other_g=torch.Generator()
    restore_training(mid,second,other_opt,other_g,'cpu')
    other=UpdateSchedule(resolve_schedule(c['schedule']))
    other.load_state_dict(mid['metadata']['runtime']['scheduler_state'],other_opt,expected_updates=4)
    for update in range(5,9):
        optimize_batch(second,other_opt,conditions,targets,schedule,other_g,1.,objective,None,other,update)
    assert exact_tree(first.state_dict(),second.state_dict())
    assert exact_tree(opt.state_dict(),other_opt.state_dict())
    assert torch.equal(generator.get_state(),other_g.get_state())
    assert learning.state_dict()==other.state_dict()


def test_quality_candidates_keep_tradeoffs_without_a_weighted_score():
    def item(name,good,road,jerk,unknown=0):
        return dict(checkpoint={'location':name},quality=dict(planned=20,quality_passes=good,
                    road_bad=road,route_bad=road,motion_bad=20-good,decoded_interior_jerk_p99=jerk,unknown=unknown,failures=0))
    records=[item('a',0,10,300),item('b',2,5,200),item('c',1,4,220),item('failed',20,0,0,1)]
    assert [v['location'] for v in quality_frontier(records)]==['b','c']

def test_displaced_road_activation_probe_uses_real_autograd_and_reduces_proxy(monkeypatch):
    from types import SimpleNamespace
    from shapely.geometry import box,mapping
    from sumodiff.diffusion.unified_losses import prepare_roads
    import sumodiff.diffusion.unified_diagnostics as diagnostics
    conditions,targets=fixture(batch=2,agents=2)
    conditions['attributes']=torch.tensor([4.,1.8,1.,0.],dtype=torch.float64).expand(2,2,4).clone()
    config=UnifiedLossConfig()
    roads,_=prepare_roads(dict(drivable=mapping(box(-20,-5,40,5)),
        route_corridors=[mapping(box(-20,-5,40,5))]*2),config)
    monkeypatch.setattr(diagnostics,'RoadGeometryCache',lambda *args:
        SimpleNamespace(batch=lambda ids:[roads]*len(ids)))
    cache=((conditions,targets,['a','b'],[]),None,None,None)
    c=dict(device='cpu',batch_size=2,loss=config.to_dict(),diffusion=NoiseSchedule().config.to_dict())
    out=diagnostics.road_activation_probe('unused',None,c,cache)
    assert out['descent_verified'] and out['road_proxy_before']>out['road_proxy_after']
    assert out['position_gradient_norm']>0 and out['velocity_gradient_norm']>0
    assert not out['quality_or_convergence_claim']
