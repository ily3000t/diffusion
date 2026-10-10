from copy import deepcopy
import pytest
from sumodiff.diffusion.decoded_temporal_diagnostics import balanced_indices
from sumodiff.diffusion.stage5d_report import validate_schedule_row
from sumodiff.diffusion.decoded_temporal_training import decoded_temporal_config
from sumodiff.diffusion.schedule import UpdateSchedule,resolve_schedule
from test_decoded_temporal_training import parent200,request
from sumodiff.diffusion.stage5e import setup


def descriptors():
    rows=[]
    for j,family in enumerate(['three_lane_straight','ramp_merge','unsignalized_intersection']):
        for i in range(24):
            rows.append(dict(window_id=f'{j}-{i}',family=family,geometry_id=f'g{i%2}',
                episode_key=f'e{i%4}',agents=8 if j==2 and i>=8 else 3))
    return rows


def test_calibration_selection_uses_metadata_and_balances_groups():
    rows=descriptors();dense=balanced_indices(rows,16,7,True);normal=balanced_indices(rows,16,7,False)
    assert len(set(dense))==len(set(normal))==16 and set(dense).isdisjoint(normal)
    assert all(rows[i]['agents']>=8 for i in dense)
    assert {rows[i]['family'] for i in normal}=={'three_lane_straight','ramp_merge','unsignalized_intersection'}
    for r in rows:r['future_error']=float('nan')
    assert dense==balanced_indices(rows,16,7,True)
    reversed_rows=list(reversed(rows))
    assert [rows[i]['window_id'] for i in dense]==[
        reversed_rows[i]['window_id'] for i in balanced_indices(reversed_rows,16,7,True)]


@pytest.mark.parametrize('count',[0,3,20,True])
def test_audit_selection_cannot_expand_or_silently_drop(count):
    with pytest.raises(ValueError):balanced_indices(descriptors(),count,7,True)
    with pytest.raises(ValueError,match='Insufficient'):balanced_indices(descriptors()[:24],4,7,True)


def test_report_checks_actual_reweight_and_full_inherited_raw_terms():
    p=parent200();c=decoded_temporal_config(request(p),p);schedule=UpdateSchedule(resolve_schedule(c['schedule']))
    u=17;rho=schedule.auxiliary_ramp(u);loss=c['loss'];base=loss['inherited']['unified']['temporal_weight']
    row=dict(step=60000+u,learning_rate=schedule.learning_rate(u),raw_auxiliary_ramp=1.,
        inherited_auxiliary_ramp=1.,temporal_reweight_ramp=rho,
        effective_decoded_boundary_weight=base*.1+rho*(loss['boundary_weight']-base*.1),
        effective_decoded_interior_weight=base*.9+rho*(loss['interior_weight']-base*.9))
    validate_schedule_row(row,c,schedule)
    for key,value in [('raw_auxiliary_ramp',rho),('temporal_reweight_ramp',1.),
        ('effective_decoded_boundary_weight',.001),('learning_rate',1e-3)]:
        changed=dict(row);changed[key]=value
        with pytest.raises(ValueError):validate_schedule_row(changed,c,schedule)


def test_entry_rejects_old_suite_without_reading_data(tmp_path):
    path=tmp_path/'old.yaml';path.write_text('schema_version: sumodiff.stage5d.suite.config.v1\n')
    with pytest.raises(ValueError,match='stage5E'):setup(path)


def test_timestep_exposure_is_training_only_and_not_a_gradient_percentage(tmp_path):
    import json
    from sumodiff.diffusion.decoded_temporal_diagnostics import timestep_coverage
    meta=descriptors();path=tmp_path/'steps.jsonl'
    ordinary=meta[0]['window_id'];dense=meta[-1]['window_id']
    rows=[dict(window_ids=[ordinary,dense],timesteps=[0,200]),dict(window_ids=[ordinary,dense],timesteps=[500,999])]
    path.write_text('\n'.join(json.dumps(x) for x in rows))
    result=timestep_coverage(path,meta)
    assert result['groups']['dense_intersection']['eligible_fraction']==.5
    assert result['groups']['ordinary']['t0']==1 and result['scene_visits']==4
    assert result['training_distribution_changed'] is False
    path.write_text(json.dumps(dict(window_ids=['unknown'],timesteps=[0])))
    with pytest.raises(ValueError,match='Unknown'):timestep_coverage(path,meta)
