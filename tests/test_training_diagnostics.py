import pytest
import torch
from sumodiff.diffusion.diagnostics import channel_mse,noise_for_ids,diagnostic_config
from sumodiff.diffusion import NoiseSchedule,DiffusionConfig,normalize,denormalize


def test_channels_have_equal_scene_weight_and_ignore_padding_nan():
    p=torch.zeros(2,2,3,6);q=torch.ones_like(p);q[0]=2
    mask=torch.ones(2,2,3,dtype=torch.bool);mask[0,1]=False;mask[0,0,1:]=False;q[~mask]=float('nan')
    mse=channel_mse(p,q,mask)
    torch.testing.assert_close(mse[0],torch.full((6,),4.,dtype=torch.float64));torch.testing.assert_close(mse[1],torch.ones(6,dtype=torch.float64))
    assert mse.mean().item()==2.5
    with pytest.raises(ValueError): channel_mse(p,q,torch.zeros_like(mask))


def test_noise_stable_under_batch_order_and_different_repeats():
    first=noise_for_ids(['a','b'],(2,40,6),20,0)
    reversed_noise=noise_for_ids(['b','a'],(2,40,6),20,0)
    assert torch.equal(first,reversed_noise.flip(0))
    assert torch.equal(first[0],noise_for_ids(['a'],(2,40,6),20,0)[0])
    assert not torch.equal(first,noise_for_ids(['a','b'],(2,40,6),20,1))


def test_known_epsilon_error_has_expected_state_channel_amplification():
    config=DiffusionConfig();schedule=NoiseSchedule(config);physical=torch.randn(2,2,40,6,dtype=torch.float64)
    clean=normalize(physical,config);noise=torch.randn_like(clean);t=torch.tensor([999,500]);bias=torch.tensor([.01,.02,.03,.04,.05,.06],dtype=torch.float64)
    noisy=schedule.add_noise(clean,noise,t);recovered=denormalize(schedule.clean_from_noise(noisy,noise+bias,t),config)
    a=schedule.alpha_bar[t][:,None,None,None];units=torch.tensor([50.,50.,20.,20.,1.,1.],dtype=torch.float64)
    expected=physical-((1-a)/a).sqrt()*bias*units
    torch.testing.assert_close(recovered,expected,atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('value',[{'timesteps':[1000]},{'sampling_steps':[1000]},{'noise_repeats':10},{'free_sampling':{'limit':None}},{'future_mask':True}])
def test_unbounded_or_unknown_diagnostics_rejected(value):
    with pytest.raises(ValueError): diagnostic_config(value)


@pytest.mark.parametrize('limit',[0,61,True,1.5,'6'])
def test_invalid_labeled_probe_budget_rejected(limit):
    with pytest.raises(ValueError): diagnostic_config({'probe_limit':limit})


def test_labeled_probe_budget_defaults_preserve_legacy_scope():
    assert diagnostic_config({})['probe_limit'] is None
    assert diagnostic_config({'probe_limit':6})['probe_limit']==6


def test_bounded_probe_checks_full_cohort_and_recorded_order(monkeypatch):
    from sumodiff.diffusion import diagnostics as d
    populations={'train':[f't{i}' for i in range(10)],'validation':[f'v{i}' for i in range(4)]}
    class Dataset:
        def __init__(self,path,split,core_only):
            assert core_only
            self.entries=[{'window_id':i} for i in populations[split]]
    def choose(ds,limit):
        return list(range(len(ds.entries) if limit is None else min(limit,len(ds.entries))))
    def cache(path,split,limit):
        return {},{},populations[split][:limit],[]
    monkeypatch.setattr(d,'WindowDataset',Dataset)
    monkeypatch.setattr(d,'choose_indices',choose)
    monkeypatch.setattr(d,'labeled_cache',cache)
    meta={'training_config':{'train_limit':None,'validation_limit':None},
          'training_signature':{'train_window_ids':populations['train'].copy(),'validation_window_ids':populations['validation'].copy()}}
    caches,counts=d.checked_labeled_caches('unused',meta,6)
    assert counts=={'train':10,'validation':4}
    assert caches['train'][2]==populations['train'][:6]
    assert caches['validation'][2]==populations['validation']
    # A changed window outside the small probe must still invalidate provenance.
    meta['training_signature']['train_window_ids'][-1]='other'
    with pytest.raises(ValueError,match='task IDs changed'): d.checked_labeled_caches('unused',meta,6)
    meta['training_signature']['train_window_ids']=populations['train'].copy()
    monkeypatch.setattr(d,'labeled_cache',lambda path,split,limit: ({},{},list(reversed(populations[split][:limit])),[]))
    with pytest.raises(ValueError,match='probe order changed'): d.checked_labeled_caches('unused',meta,6)
