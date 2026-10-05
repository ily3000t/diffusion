from types import SimpleNamespace
import numpy as np
import pytest
import torch
from sumodiff.diffusion import NoiseSchedule,ddim_sample
from sumodiff.diffusion.sampling import sample_config,failed_report,task_seed
from sumodiff.evaluation.metrics import aggregate_scenes


class Oracle:
    def __init__(self,schedule,clean):
        self.config=SimpleNamespace(diffusion_embedding_steps=1000);self.training=False;self.schedule=schedule;self.clean=clean;self.encodes=0;self.calls=0
    def condition_encoder(self,c):
        assert 'future_mask' not in c
        self.encodes+=1;return None
    def unet(self,value,timestep,condition,mask):
        self.calls+=1;return self.schedule.noise_from_clean(value,self.clean,timestep)


def test_ddim_cache_padding_and_calls_are_exact():
    schedule=NoiseSchedule();mask=torch.tensor([[True,False]]);clean=torch.randn(1,2,40,6)*3
    model=Oracle(schedule,clean);noise=torch.randn_like(clean);noise[:,1]=float('nan')
    out=ddim_sample(model,{'agent_mask':mask},noise,schedule,20)
    torch.testing.assert_close(out[:,0],clean[:,0],atol=1e-6,rtol=1e-6)
    assert not out[:,1].any() and model.calls==20 and model.encodes==1
    with pytest.raises(NotImplementedError): ddim_sample(model,{'agent_mask':mask},noise,schedule,cfg=True)


def test_sampling_flags_fail_and_seed_is_task_stable():
    for value in ({'cfg':True},{'partial_diffusion':True},{'rolling_generation':True},{'mode':'diffscene_style'}):
        with pytest.raises(NotImplementedError): sample_config(value)
    assert task_seed(1,'window')==task_seed(1,'window')!=task_seed(1,'other')


def test_failures_and_single_vehicle_tasks_keep_denominators():
    pair=failed_report({'window_id':'pair','agent_ids':['a','b',None],'excluded_vehicle_count':4},RuntimeError('failed'))
    single=failed_report({'window_id':'single','agent_ids':['a',None,None],'excluded_vehicle_count':0},RuntimeError('failed'))
    report=aggregate_scenes([pair,single])
    assert report['planned_scenes']==2 and report['failed_scenes']==2 and report['quality_pass_rate']==0
    assert report['event_rates']['any']['denominator']==1 and report['event_rates']['any']['unknown_or_failed']==1
    assert report['event_rates']['target']['confirmed_event_rate'] is None and report['effective_target_event_rate'] is None


def test_task_selection_uses_current_ids_without_future_completeness():
    from sumodiff.diffusion.data import choose_indices
    class Dataset:
        entries=[dict(core_training_eligible=False),dict(core_training_eligible=True),dict(core_training_eligible=False)]
        def _json(self,entry,name):
            assert name=='input.json'
            i=next(i for i,e in enumerate(self.entries) if e is entry)
            return dict(family=['a','b','a'][i],agent_ids=[['x','y',None],['x',None,None],['x','y','z']][i])
    assert choose_indices(Dataset(),None,minimum_agents=2)==[0,2]


def test_loss_labels_are_not_model_conditioning_or_noise_exit_rules():
    from sumodiff.diffusion.training import draw_noise
    c={'agent_mask':torch.tensor([[True,False]])}
    targets={'future':torch.zeros(1,2,40,6),'future_mask':torch.zeros(1,2,40,dtype=torch.bool)}
    changed={**targets,'future_mask':torch.ones_like(targets['future_mask'])}
    a=draw_noise(targets,c,NoiseSchedule(),torch.Generator().manual_seed(8))
    b=draw_noise(changed,c,NoiseSchedule(),torch.Generator().manual_seed(8))
    for left,right in zip(a,b): assert torch.equal(left,right)
    assert a[0][:,0].abs().sum()>0 and not a[0][:,1].any()
