from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import pytest
from sumodiff.data.control_tasks import select_tasks,motion_metadata,make_motion_task_plan,verify_motion_task_plan,member
from sumodiff.decoding.control_scale_audit import scale_motion_row,groups_for
from sumodiff.diffusion.control_training import control_config
from sumodiff.diffusion.control_calibration import calibrate
from sumodiff.experiments.motion_control import pick_factor,scaled_config
from sumodiff.experiments.recorder import file_identity

def rows():
    result=[]
    for split in ('train','validation'):
        for family in ('three_lane_straight','ramp_merge','unsignalized_intersection'):
            kinds=('ordinary','lane_change') if family!='unsignalized_intersection' else ('ordinary','turning','dense_no_turn')
            for kind in kinds:
                for policy in ('first_current_id','interaction_current'):
                    for geo in range(3):
                        for n in range(10):
                            wid=f'{split}:{family}:{kind}:{policy}:{geo}:{n}'
                            flags=dict(lane_change=kind=='lane_change',ramp_exit=False,turn=kind=='turning')
                            result.append(dict(window_id=wid,split=split,family=family,reference_policy=policy,
                                geometry_id=f'{split}:{family}:g{geo}',episode_key=f'{split}:{family}:g{geo}:e{n%4}',
                                selected_agents=12 if kind in ('turning','dense_no_turn') else 3,
                                eligible=True,core_complete=True,motion_flags=flags,
                                motion_event_ids={k:[wid+':'+k] if v else [] for k,v in flags.items()}))
    return result

def test_motion_flags_are_explicit_and_legacy_is_not_silently_interpreted():
    r=rows()[0]
    assert scale_motion_row(r,'sumodiff.stage5a.selection.v3','sumodiff.control_scale_audit.protocol.v2')['actual_turning_agents']==0
    with pytest.raises(ValueError,match='Legacy'):
        scale_motion_row(r,'sumodiff.stage5a.selection.v3','sumodiff.control_scale_audit.protocol.v1')
    with pytest.raises(ValueError,match='selection v3'):
        scale_motion_row(r,'sumodiff.stage5a.selection.v2','sumodiff.control_scale_audit.protocol.v2')
    bad=deepcopy(r);bad['motion_flags']['turn']=True
    with pytest.raises(ValueError,match='flags'):motion_metadata(bad)
    bad=deepcopy(r);bad.pop('motion_event_ids')
    with pytest.raises(ValueError):motion_metadata(bad)

def test_motion_groups_use_selected_events_and_density_separately():
    r=next(r for r in rows() if r['motion_flags']['lane_change'])
    s=scale_motion_row(r,'sumodiff.stage5a.selection.v3','sumodiff.control_scale_audit.protocol.v2')
    assert 'lane_change_windows' in groups_for(r,s,r['selected_agents'])
    r=next(r for r in rows() if r['motion_flags']['turn'])
    s=scale_motion_row(r,'sumodiff.stage5a.selection.v3','sumodiff.control_scale_audit.protocol.v2')
    assert s['actual_turning_agents']==1 and 'turning_windows' in groups_for(r,s,12)
    assert 'dense_no_turn_windows' not in groups_for(r,s,12)

def test_cohorts_are_deterministic_balanced_disjoint_and_test_arrays_not_needed():
    data=rows()+[dict(split='test',window_id='test-sentinel')]
    tasks=select_tasks(data,123)
    assert tasks==select_tasks(list(reversed(data)),123)
    assert len(tasks)==120 and len({t['window_id'] for t in tasks})==120
    assert Counter(t['split'] for t in tasks)==dict(train=60,validation=60)
    for split in ('train','validation'):
        cohort=[t for t in tasks if t['split']==split]
        assert Counter(t['group'] for t in cohort)==dict(dense_no_turn=16,turning=16,lane_change=16,ordinary=12)
        for group in ('dense_no_turn','turning','lane_change','ordinary'):
            group_rows=[t for t in cohort if t['group']==group]
            refs=Counter(t['reference_policy'] for t in group_rows)
            assert len(set(refs.values()))==1 and len({t['geometry_id'] for t in group_rows})>=3
            assert all(member(t,group) for t in group_rows)

def test_insufficient_cohort_never_falls_back_or_duplicates():
    data=[r for r in rows() if r['family']!='ramp_merge' or not r['motion_flags']['lane_change']]
    with pytest.raises(ValueError,match='Insufficient'):select_tasks(data,123)

def make_fixture(tmp_path):
    s=tmp_path/'selection.json';s.write_text(json.dumps(dict(schema_version='sumodiff.stage5a.selection.v3',profile='full',selected=rows())))
    idx=tmp_path/'windows.json';idx.write_text('[]')
    (tmp_path/'dataset_manifest.json').write_text(json.dumps(dict(data_id='test-data',window_index=file_identity(idx),
        curation=dict(selection=file_identity(s)))))
    return make_motion_task_plan(tmp_path,123)

@pytest.mark.parametrize('case',['data_id','task_group','test_task','seed','selection_hash'])
def test_motion_task_identity_and_allocation_are_bound_before_arrays(tmp_path,case):
    plan=make_fixture(tmp_path);assert verify_motion_task_plan(plan,tmp_path)
    if case=='data_id':plan['data_id']='other'
    elif case=='task_group':plan['tasks'][0]['group']='ordinary'
    elif case=='test_task':plan['tasks'][0]['split']='test'
    elif case=='seed':plan['seed']+=1
    else:(tmp_path/'selection.json').write_text('{}')
    with pytest.raises(ValueError):verify_motion_task_plan(plan,tmp_path)

def test_weight_selection_uses_complete_auxiliary_norm_and_scales_every_term():
    report=dict(rows=[dict(auxiliary_to_noise_ratio=None),dict(auxiliary_to_noise_ratio=2.),
                      dict(auxiliary_to_noise_ratio=.1)])
    assert pick_factor(report,.5,[1,.3,.1])==.1
    with pytest.raises(ValueError,match='exhausted'):pick_factor(report,.01,[1,.3])
    base=control_config(dict(schema_version='sumodiff.control.train.config.v1'))
    c=scaled_config(base,.1)
    for key in ('imitation_weight','temporal_weight','road_weight'):
        assert c['unified_loss'][key]==pytest.approx(base['unified_loss'][key]*.1)
    assert c['tail_weight']==pytest.approx(base['tail_weight']*.1)
    assert c['diffusion']==base['diffusion'] and c['schedule']==base['schedule']

@pytest.mark.parametrize('times',[(400,),(),(50,50),(True,)])
def test_gradient_probe_times_reject_unsupported_values_before_model_or_data(times):
    with pytest.raises(ValueError,match='timesteps'):
        calibrate(None,None,None,None,None,None,[],probe_timesteps=times)
