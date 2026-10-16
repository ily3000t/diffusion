from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from sumodiff.diffusion import control_epochs as ce
from sumodiff.diffusion.schedule import ScheduleConfig,UpdateSchedule
from sumodiff.diffusion.epoch_training import epoch_batch,periodic_sample
from sumodiff.diffusion.checkpoint import restore_training
from sumodiff.diffusion.audit import exact_tree
from sumodiff.diffusion.sampling import sample_config
from sumodiff.experiments.recorder import load_config,file_identity


def recipe():
    return ce.epoch_config(load_config('configs/train/motion_control_epochs.yaml'))


def test_full_epoch_recipe_preserves_calibrated_control_objective_and_one_schedule():
    c=recipe();s=UpdateSchedule(ScheduleConfig(**c['schedule']))
    assert c==ce.epoch_config(c)
    assert (c['steps_per_epoch'],c['max_steps'])==(375,75000)
    assert c['initialization']=='random_new_model_and_adamw'
    assert (s.learning_rate(1),s.learning_rate(3750),s.learning_rate(75000))==(1e-6,1e-4,1e-6)
    assert s.auxiliary_ramp(3750)==1
    assert [c['unified_loss'][k] for k in ('imitation_weight','temporal_weight','road_weight')]==[.003,.001,.001]
    assert c['tail_weight']==.0003 and 'raw_loss' not in c
    assert c['monitor']['task_plan'] and c['monitor']['split']=='validation'
    assert c['diffusion']['acceleration_scale_mps2']==2
    assert c['diffusion']['yaw_rate_scale_radps']==.5
    assert c['diffusion']['lateral_velocity_scale_mps']==.1


@pytest.mark.parametrize('change',[{'max_steps':16},{'schedule':{}},{'epochs':True},{'batch_size':0},
    {'stream_cache_windows':100000},{'expected_train_windows':6001},{'tail_weight':float('nan')},
    {'monitor':{'limit':6}},{'calibration_config':None},{'unknown':1}])
def test_recipe_rejects_silent_budget_or_protocol_changes(change):
    c=recipe();c.update(change)
    with pytest.raises((ValueError,TypeError)):ce.epoch_config(c)


def test_epoch_permutation_resume_and_last_partial_batch_cover_each_scene_once():
    all_rows=[]
    for step in range(1,5):
        ids,epoch,batch=epoch_batch(step,10,3,21); all_rows+=ids.tolist()
        assert (epoch,batch)==(1,step)
        assert torch.equal(ids,epoch_batch(step,10,3,21)[0])
    assert sorted(all_rows)==list(range(10)) and len(epoch_batch(4,10,3,21)[0])==1
    assert epoch_batch(5,10,3,21)[1:]==(2,1)


def payload_after_updates(tmp_path,n=3):
    c=recipe();torch.manual_seed(1);model=torch.nn.Linear(3,2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'])
    generator=torch.Generator().manual_seed(5);schedule=UpdateSchedule(ScheduleConfig(**c['schedule']))
    def step(m,o,g,s):
        update=s.completed_updates+1;s.apply(o,update);o.zero_grad()
        loss=m(torch.randn(4,3,generator=g)).square().mean();loss.backward();o.step();s.complete(update)
        return float(loss)
    for _ in range(n):step(model,optimizer,generator,schedule)
    runtime=dict(torch=str(torch.__version__),update_schedule=schedule.state_dict(),
        best_validation=dict(step=n,mse=1.,checkpoint=str(tmp_path/'best.pt')),
        pending_monitor_epoch=None,monitored_through_epoch=0)
    metadata=dict(training_config=c,training_signature={'training':c},diffusion_config=c['diffusion'],runtime=runtime)
    payload=dict(step=n,model=deepcopy(model.state_dict()),optimizer=deepcopy(optimizer.state_dict()),
        generator_state=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=[],metadata=metadata)
    return payload,model,optimizer,generator,schedule,step


def test_real_adamw_rng_and_global_schedule_restore_without_rewarmup(tmp_path):
    p,m,o,g,s,step=payload_after_updates(tmp_path);ce.validate_runtime(p)
    expected=step(m,o,g,s)
    replica=torch.nn.Linear(3,2);opt=torch.optim.AdamW(replica.parameters());gen=torch.Generator()
    restore_training(p,replica,opt,gen,'cpu');sched=UpdateSchedule(ScheduleConfig(**recipe()['schedule']))
    sched.load_state_dict(p['metadata']['runtime']['update_schedule'],opt,p['step'])
    assert step(replica,opt,gen,sched)==expected
    assert exact_tree(m.state_dict(),replica.state_dict())
    assert exact_tree(o.state_dict(),opt.state_dict())
    assert torch.equal(g.get_state(),gen.get_state()) and sched.state_dict()==s.state_dict()


@pytest.mark.parametrize('case',['optimizer_step','lr','pending','completed','best','recipe'])
def test_checkpoint_runtime_rejects_tampered_state(tmp_path,case):
    p,*_=payload_after_updates(tmp_path)
    if case=='optimizer_step':next(iter(p['optimizer']['state'].values()))['step']+=1
    elif case=='lr':p['optimizer']['param_groups'][0]['lr']=1e-4
    elif case=='pending':p['metadata']['runtime']['pending_monitor_epoch']=10
    elif case=='completed':p['metadata']['runtime']['monitored_through_epoch']=10
    elif case=='best':p['metadata']['runtime']['best_validation']['step']=4
    else:p['metadata']['training_signature']['training']=dict(recipe(),epochs=201)
    with pytest.raises(ValueError):ce.validate_runtime(p)


def test_training_stream_rejects_test_before_opening_any_file():
    with pytest.raises(ValueError,match='test'):ce.WindowStream('does-not-exist','test',None,2)


def test_stream_lru_is_bounded_and_padding_is_per_batch(monkeypatch):
    class Fake:
        def __init__(self,*a,**kw):self.entries=[dict(window_id=str(i)) for i in range(4)]
        def __getitem__(self,i):return dict(conditioning={},targets={'future_mask':np.ones((1,40),bool)},input_metadata={'window_id':str(i)})
    monkeypatch.setattr(ce,'WindowDataset',Fake)
    monkeypatch.setattr(ce,'collate_numpy',lambda samples:dict(targets={'future_mask':np.ones((len(samples),1,40),bool)}))
    monkeypatch.setattr(ce,'prepare_conditioning',lambda batch,device,config:dict(agent_mask=torch.ones((len(batch['targets']['future_mask']),1),dtype=torch.bool)))
    s=ce.WindowStream(None,'train',None,2)
    _,_,ids,_=s.batch([0,1],'cpu');assert ids==['0','1']
    s.batch([1,2],'cpu')
    assert list(s.cache)==[1,2] and len(s.loaded)==3


def test_monitor_restores_all_rng_states_even_when_sampling_fails(tmp_path,monkeypatch):
    import sumodiff.diffusion.sampling as sampling
    def fail(*a,**kw):
        random.random();np.random.random();torch.rand(3)
        raise RuntimeError('retained monitor failure')
    monkeypatch.setattr(sampling,'sample',fail)
    class Run:output=tmp_path
    c=recipe();c['device']='cpu';c['monitor']['device']='cpu'
    states=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    with pytest.raises(RuntimeError,match='monitor failure'):periodic_sample(None,None,Run(),None,c,10,False)
    assert states[0]==random.getstate() and np.array_equal(states[1][1],np.random.get_state()[1])
    assert torch.equal(states[2],torch.get_rng_state())


def test_control_monitor_and_full_validation_configs_keep_distinct_denominators():
    c=recipe()
    assert c['monitor']['limit']==60
    for split in ('validation','test'):
        s=sample_config(load_config(f'configs/experiments/motion_control_{split}.yaml'))
        assert s['limit']==600 and s['task_plan'] is None and s['minimum_agents']==1
        assert s['split']==split and s['seed']==c['monitor']['seed']
    with pytest.raises(ValueError):sample_config(dict(task_plan='tasks.json',split='test',limit=60))


def test_test_seal_required_before_task_or_dataset_access():
    from sumodiff.diffusion.fresh_protocol import verify_test_seal
    with pytest.raises(ValueError,match='Seal'):verify_test_seal(None,'not-read','not-read',{})



def fixture_selection(tmp_path):
    # Eighteen synthetic metadata records only; no numerical arrays or simulated quality claims.
    rows=[];index=[];ledger=[];summaries={}
    for split in ('train','validation','test'):
        summaries[split]={}
        for family in ce.FAMILIES:
            cohort=[]
            for policy in ce.POLICIES:
                wid=f'{split}:{family}:{policy}';ep=wid+':ep'
                source=str(tmp_path/(wid.replace(':','_')+'.json'))
                row=dict(window_id=wid,split=split,family=family,reference_policy=policy,episode_key=ep,
                    geometry_id=f'{split}:{family}',source_manifest=source,reference_tick=30,
                    eligible=True,core_complete=True,reasons=[],quality_status='assessed',
                    motion_flags=dict(lane_change=False,ramp_exit=False,turn=False),
                    motion_event_ids=dict(lane_change=[],ramp_exit=[],turn=[]))
                rows.append(row);cohort.append(row)
                index.append({**{k:row[k] for k in ('window_id','split','family','episode_key','geometry_id')},'core_training_eligible':True})
                ledger.append(dict(job_id=wid,outcome='accepted',episode_manifest=dict(location=source)))
            summaries[split][family]=dict(passed=True,checks=dict(total=True),categories={},
                requirement=dict(max_episode_windows=1,max_windows_per_event=3),
                geometry_counts=dict(Counter(r['geometry_id'] for r in cohort)),
                episode_counts=dict(Counter(r['episode_key'] for r in cohort)))
    (tmp_path/'windows.json').write_text(json.dumps(index))
    (tmp_path/'selection.json').write_text(json.dumps(dict(schema_version='sumodiff.stage5a.selection.v3',profile='full',
        selected=rows,selection=summaries,no_smoothing_or_clipping=True)))
    policy=dict(mode='synthetic-ledger-fixture')
    (tmp_path/'sources.json').write_text(json.dumps(dict(schema_version='sumodiff.sources.v2',quality_policy=policy,job_ledger=ledger)))
    population=dict(quality_policy=policy,accepted_job_ids=[r['job_id'] for r in ledger],
        summary=dict(planned_jobs=18,processed_jobs=18,accepted_jobs=18,rejected_jobs=0,pending_jobs=0))
    manifest=dict(schema_version='sumodiff.dataset.v1',data_id='synthetic-metadata',window_index=file_identity(tmp_path/'windows.json'),
        source_registry=file_identity(tmp_path/'sources.json'),curation=dict(selection=file_identity(tmp_path/'selection.json'),
            required_model_schema='sumodiff.model.control.config.v2',label_quality_filtered=True,inference_conditions_unchanged=True,
            source_population=population))
    (tmp_path/'dataset_manifest.json').write_text(json.dumps(manifest))
    c=recipe();c.update(expected_data_id='synthetic-metadata',expected_train_windows=6,expected_validation_windows=6,expected_test_windows=6)
    return c,manifest


@pytest.mark.parametrize('case',['clean','hash','quotas','leakage','reference','ledger','coverage'])
def test_motion_dataset_authenticates_selection_split_and_rejected_population_before_arrays(tmp_path,case):
    c,m=fixture_selection(tmp_path)
    if case=='clean':
        assert ce.verified_dataset(tmp_path,c)['source_population']['summary']['pending_jobs']==0
        return
    sel=json.loads((tmp_path/'selection.json').read_text())
    if case=='hash':sel['tamper']=True
    elif case=='quotas':sel['selected'][0]['reference_policy']='interaction_current'
    elif case=='leakage':sel['selected'][0]['geometry_id']=next(r['geometry_id'] for r in sel['selected'] if r['split']=='validation')
    elif case=='reference':sel['selected'][1]['episode_key']=sel['selected'][0]['episode_key']
    elif case=='ledger':m['curation']['source_population']['summary']['pending_jobs']=1
    elif case=='coverage':sel['selection']['train'][ce.FAMILIES[0]]['geometry_counts']={'other':2}
    (tmp_path/'selection.json').write_text(json.dumps(sel))
    if case!='hash':m['curation']['selection']=file_identity(tmp_path/'selection.json')
    (tmp_path/'dataset_manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):ce.verified_dataset(tmp_path,c)


def test_control_test_seal_cannot_reuse_legacy_seal_or_change_selected_model(tmp_path):
    import yaml
    from sumodiff.diffusion.data import dataset_identity
    from sumodiff.diffusion.fresh_protocol import verify_test_seal
    data=tmp_path/'data';data.mkdir()
    (data/'dataset_manifest.json').write_text('{}');(data/'windows.json').write_text('[]')
    cp=tmp_path/'checkpoint';cp.write_bytes(b'identity-only synthetic checkpoint')
    seal=tmp_path/'seal';seal.mkdir()
    config=dict(split='test',limit=600)
    doc=dict(schema_version='sumodiff.control.test.seal.v1',checkpoint=file_identity(cp),
        dataset_hashes={k:v['sha256'] for k,v in dataset_identity(data).items()},sampling=config)
    (seal/'test_seal.json').write_text(json.dumps(doc))
    (seal/'resolved_config.yaml').write_text(yaml.safe_dump(doc))
    (seal/'manifest.json').write_text(json.dumps(dict(formal=True,git=dict(dirty=False))))
    (seal/'status.json').write_text(json.dumps(dict(state='completed')))
    assert verify_test_seal(seal/'test_seal.json',data,cp,config,control=True)
    with pytest.raises(ValueError):verify_test_seal(seal/'test_seal.json',data,cp,config)
    cp.write_bytes(b'changed')
    with pytest.raises(ValueError):verify_test_seal(seal/'test_seal.json',data,cp,config,control=True)


def test_pending_monitor_is_preserved_and_cannot_be_silently_skipped(tmp_path):
    p,*_=payload_after_updates(tmp_path)
    step=3750
    c=p['metadata']['training_config'];schedule=UpdateSchedule(ScheduleConfig(**c['schedule']))
    schedule.completed_updates=step;schedule.last_lr=schedule.learning_rate(step);schedule.last_auxiliary_ramp=schedule.auxiliary_ramp(step)
    p['step']=step;p['metadata']['runtime']['update_schedule']=schedule.state_dict()
    for state in p['optimizer']['state'].values():state['step']=torch.tensor(float(step))
    p['optimizer']['param_groups'][0]['lr']=schedule.last_lr
    p['metadata']['runtime']['pending_monitor_epoch']=10
    ce.validate_runtime(p)
    p['metadata']['runtime']['pending_monitor_epoch']=None
    with pytest.raises(ValueError,match='skipped'):ce.validate_runtime(p)
    p['metadata']['runtime']['monitored_through_epoch']=10
    ce.validate_runtime(p)

def test_invalid_generated_controls_are_quality_failures_without_hiding_execution_errors():
    from sumodiff.diffusion.sampling import sampling_outcome
    from sumodiff.decoding.controls import InvalidGeneratedControl
    assert issubclass(InvalidGeneratedControl,ValueError)
    result=dict(generation_failures=60,invalid_generated_controls=60,execution_failures=0,
        raw=dict(failed_scenes=60),decoded=dict(failed_scenes=60))
    assert sampling_outcome(result,True)=='completed_with_invalid_generated_controls'
    assert sampling_outcome(result,False)=='failed_execution'
    bad=deepcopy(result);bad['execution_failures']=1
    assert sampling_outcome(bad,True)=='failed_execution'
    bad=deepcopy(result);bad['invalid_generated_controls']=59
    assert sampling_outcome(bad,True)=='failed_execution'
    clean=dict(generation_failures=0,invalid_generated_controls=0,execution_failures=0,
        raw=dict(failed_scenes=0),decoded=dict(failed_scenes=0))
    assert sampling_outcome(clean,True)=='completed'
