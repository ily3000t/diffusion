import json
from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from sumodiff.experiments.stage5a_plan import resolve_plan,jobs,FAMILIES
from sumodiff.experiments.stage5a_data import inspect_sources
from sumodiff.data.quality import audit_window,select_balanced
from sumodiff.data.episodes import Episode,Observation
from sumodiff.data.windows import build_window
from sumodiff.geometry.maps import RoadMap
from sumodiff.simulation.config import resolve_config
from sumodiff.simulation.scenes import layout
from sumodiff.diffusion.epoch_training import epoch_config,epoch_batch,periodic_sample


def test_plan_preassigns_distinct_geometries_and_seeds_and_budgets():
    c=resolve_plan({})
    scheduled=jobs(c)
    assert len(scheduled)==78 and len(jobs(c,True))==6
    shapes={split:{json.dumps(j['scenario']['geometry'],sort_keys=True) for j in scheduled if j['split']==split}
            for split in ('train','validation')}
    assert not shapes['train']&shapes['validation']
    assert len({j['seed'] for j in scheduled})==len(scheduled)
    assert sum(c['train_quotas'].values())==1200 and sum(c['validation_quotas'].values())==200
    assert {j['scenario']['driver']['car_follow_model'] for j in scheduled}=={'IDM'}


def test_continuous_straight_has_no_artificial_internal_connections_and_legacy_is_explicit():
    config=resolve_config({'family':'three_lane_straight','geometry':{'segmented_straight':False}})
    nodes,edges,links,routes=layout(config)
    assert len(nodes)==2 and len(edges)==1 and not links and routes['main']['edges']==['main']
    assert edges[0]['numLanes']=='3'
    assert len(layout(resolve_config({'family':'three_lane_straight'}))[1])==3
    with pytest.raises(ValueError):
        resolve_config({'family':'three_lane_straight','geometry':{'segmented_straight':1}})


@pytest.mark.parametrize('bad',[{'train_geometries':1},{'pilot_demand_seconds':90},
    {'driver':{'car_follow_model':'unsupported'}},{'window':{'future_points':50}},
    {'validation_quotas':{'unknown':1}}])
def test_invalid_plan_does_not_silently_degrade(bad):
    with pytest.raises((ValueError,NotImplementedError,TypeError)): resolve_plan(bad)


def test_epoch_partial_batch_covers_population_once_and_resume_order_without_global_rng_change():
    c=epoch_config({'schema_version':'sumodiff.base.epoch.config.v1','device':'cpu'})
    assert c['steps_per_epoch']==300 and c['max_steps']==30000
    before=torch.get_rng_state().clone()
    for epoch in range(3):
        batches=[epoch_batch(epoch*3+k,10,4,17) for k in range(1,4)]
        assert [len(b[0]) for b in batches]==[4,4,2]
        assert sorted(torch.cat([b[0] for b in batches]).tolist())==list(range(10))
        assert all(b[1]==epoch+1 for b in batches)
    assert torch.equal(before,torch.get_rng_state())
    assert torch.equal(epoch_batch(5,10,4,17)[0],epoch_batch(5,10,4,17)[0])
    assert not torch.equal(epoch_batch(1,10,4,17)[0],epoch_batch(4,10,4,17)[0])
    with pytest.raises(ValueError):epoch_config({'schema_version':'sumodiff.base.epoch.config.v1','max_steps':30000})


def test_balanced_selection_keeps_diversity_and_reports_insufficiency_without_repetition():
    rows=[]
    for split in ('train','validation'):
        for family in FAMILIES:
            for geometry in range(2):
                for ep in range(2):
                    for tick in range(10):
                        rows.append(dict(window_id=f'{split}-{family}-{geometry}-{ep}-{tick}',split=split,family=family,
                            geometry_id=f'{split}-{family}-{geometry}',episode_key=f'{split}-{family}-{geometry}-{ep}',eligible=True))
    quotas={s:{f:10 for f in FAMILIES} for s in ('train','validation')}
    chosen,report=select_balanced(rows,quotas,4,2,4,.25)
    assert len(chosen)==60 and len({r['window_id'] for r in chosen})==60
    assert all(r['passed'] and max(r['episode_counts'].values())<=3 for fs in report.values() for r in fs.values())
    chosen,report=select_balanced(rows[:5],quotas,4,2,4,.25)
    assert len(chosen)<=5 and not report['train'][FAMILIES[0]]['passed']


def test_geometry_leakage_is_rejected_before_raw_frames_are_loaded(tmp_path):
    registry=[]
    for i,split in enumerate(('train','validation')):
        path=tmp_path/f'{split}.json'
        path.write_text(json.dumps(dict(schema_version='sumodiff.raw.episode.v1',state='completed',
            eligible_for_normal_training=True,diagnostic_only=False,geometry_id='same',data_id=str(i),
            family=FAMILIES[0],outputs=[],scene={'files':[]})))
        registry.append(dict(manifest=str(path),split=split))
    p=tmp_path/'sources.json';p.write_text(json.dumps(dict(schema_version='sumodiff.sources.v1',episodes=registry)))
    with pytest.raises(ValueError,match='Geometry leakage'):inspect_sources(p)


def synthetic_window(tmp_path,jump=False,jump_tick=50):
    xml=tmp_path/'net.xml'
    xml.write_text('<net><edge id="main" priority="3"><lane id="main_0" index="0" width="3.5" speed="16" length="300" shape="-100,0 200,0"/>'
                   '<lane id="main_1" index="1" width="3.5" speed="16" length="300" shape="-100,3.5 200,3.5"/></edge></net>')
    road=RoadMap.read(xml);frames=[]
    for tick in range(100):
        frame={}
        for i in range(2):
            center=np.array([tick*.5+(1. if jump and tick>=jump_tick and i==0 else 0.),i*3.5])
            raw=dict(front_position_m=(center+[2.35,0]).tolist(),navigation_angle_deg=90.,length_m=4.7,width_m=1.8,
                route_edges=['main'],route_index=0,type_id='passenger',road_id='main',lane_id=f'main_{i}',acceleration_mps2=0.)
            frame[str(i)]=Observation(center,0.,raw)
        frames.append(frame)
    ep=Episode('sha256:synthetic',dict(dt=.1,episode_id='synthetic',geometry_id='synthetic',family=FAMILIES[0]),
        tmp_path/'unused.json','train',frames,xml,[])
    plan=resolve_plan({})
    window,reason=build_window(ep,40,road,plan['window'])
    assert reason is None
    return window,ep,road,plan


def test_quality_separates_completeness_from_broken_labels_and_raw_conversion(tmp_path):
    w,e,r,p=synthetic_window(tmp_path)
    report=audit_window(w,e,r,p)
    assert report['eligible'] and report['world_local_geometry_counts_match']
    assert report['coordinate_errors']['position_m']<1e-4
    assert 'eligible' not in w.input_metadata and 'future_mask' not in w.conditioning
    w.targets['future'][0,5,0]+=1.
    report=audit_window(w,e,r,p)
    assert report['core_complete'] and not report['eligible']
    assert 'coordinate_or_precision_mismatch' in report['reasons']
    assert report['raw64']['motion']['hard_quality_pass']
    assert not report['stored32']['motion']['hard_quality_pass']


def test_source_jerk_spike_is_not_attributed_to_coordinate_code(tmp_path):
    w,e,r,p=synthetic_window(tmp_path,True)
    report=audit_window(w,e,r,p)
    assert report['coordinate_errors']['position_m']<1e-4 and not report['eligible']
    assert 'raw64:motion' in report['reasons']
    assert report['peak_source_attribution']['center_jerk_mps3']>900
    assert report['peak_source_attribution']['reported_scalar_jerk_mps3']==0.


def test_monitor_restores_python_numpy_torch_rng_even_on_sampler_failure(tmp_path,monkeypatch):
    import sumodiff.diffusion.sampling as sampling
    class R: output=tmp_path
    c=epoch_config({'schema_version':'sumodiff.base.epoch.config.v1','device':'cpu'})
    python_state=random.getstate();numpy_state=np.random.get_state();torch_state=torch.get_rng_state().clone()
    def fail(*args,**kwargs):
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        raise RuntimeError('synthetic sampler failure')
    monkeypatch.setattr(sampling,'sample',fail)
    with pytest.raises(RuntimeError,match='synthetic'):
        periodic_sample(tmp_path,tmp_path/'checkpoint.pt',R(),tmp_path,c,10,False)
    assert random.getstate()==python_state
    assert np.array_equal(np.random.get_state()[1],numpy_state[1])
    assert torch.equal(torch.get_rng_state(),torch_state)


def write_synthetic_curation_metadata(tmp_path, mutate=None, corrupt_selection=False):
    """Metadata-only test fixture; never used as generated scientific data."""
    from sumodiff.data.dataset import write_json
    from sumodiff.experiments.recorder import file_identity
    selected=[]
    for split in ('train','validation'):
        for family in FAMILIES:
            selected.append(dict(window_id=f'{split}-{family}',split=split,family=family,
                geometry_id=f'{split}-{family}',episode_key=f'{split}-{family}',eligible=True,
                core_complete=True,reasons=[]))
    entries=[{**{k:r[k] for k in ('window_id','split','family','geometry_id','episode_key')},
        'core_training_eligible':True,'files':{}} for r in selected]
    selection=dict(schema_version='sumodiff.stage5a.selection.v1',profile='pilot',selected=selected,
        quotas={s:{f:1 for f in FAMILIES} for s in ('train','validation')},
        selection={s:{f:{'passed':True} for f in FAMILIES} for s in ('train','validation')})
    if mutate: mutate(selection,entries)
    write_json(tmp_path/'selection.json',selection)
    write_json(tmp_path/'windows.json',entries)
    write_json(tmp_path/'dataset_manifest.json',dict(schema_version='sumodiff.dataset.v1',
        window_index=file_identity(tmp_path/'windows.json'),
        curation=dict(label_quality_filtered=True,selection=file_identity(tmp_path/'selection.json'))))
    if corrupt_selection:
        (tmp_path/'selection.json').write_text(json.dumps(selection)+' ')
    return epoch_config(dict(schema_version='sumodiff.base.epoch.config.v1',budget_kind='smoke',device='cpu',
        expected_train_windows=3,expected_validation_windows=3,epochs=2))


def test_curated_index_accepts_only_matching_certified_metadata(tmp_path):
    from sumodiff.diffusion.epoch_training import verified_curated_dataset
    c=write_synthetic_curation_metadata(tmp_path)
    assert verified_curated_dataset(tmp_path,c)['sha256']
    with pytest.raises(ValueError,match='Pilot'):
        verified_curated_dataset(tmp_path,{**c,'budget_kind':'full'})


@pytest.mark.parametrize('case',['selection_hash','split','geometry','uncertified','count','quota','duplicate'])
def test_curated_guards_reject_tamper_leakage_and_wrong_population(tmp_path,case):
    from sumodiff.diffusion.epoch_training import verified_curated_dataset
    def mutate(selection,entries):
        r=selection['selected']
        if case=='split': entries[0]['split']='validation'
        elif case=='geometry':
            r[3]['geometry_id']=r[0]['geometry_id'];entries[3]['geometry_id']=r[0]['geometry_id']
        elif case=='uncertified': r[0]['reasons']=['road']
        elif case=='count': r.pop();entries.pop()
        elif case=='quota': selection['quotas']['train'][FAMILIES[0]]=2
        elif case=='duplicate': r[-1]['window_id']=r[0]['window_id']
    c=write_synthetic_curation_metadata(tmp_path,mutate,case=='selection_hash')
    with pytest.raises(ValueError): verified_curated_dataset(tmp_path,c)


def test_epoch_explicit_learning_probe_option_is_honored():
    c=epoch_config(dict(schema_version='sumodiff.base.epoch.config.v1',require_probe_improvement=True))
    assert c['require_probe_improvement'] is True


def test_quality_includes_jerk_at_first_history_point(tmp_path):
    w,e,r,p=synthetic_window(tmp_path,True,jump_tick=18)
    report=audit_window(w,e,r,p)
    assert report['core_complete'] and not report['eligible']
    assert report['raw64']['motion']['violation_counts']['jerk']==1
    assert report['raw64']['motion']['boundary']['jerk_mps3']['max']<1e-8
