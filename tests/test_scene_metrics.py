import numpy as np
import pytest
from shapely.geometry import box,mapping
from sumodiff.evaluation import evaluate_trajectory,aggregate_scenes


def sample(count=2):
    h = np.zeros((count,21,6)); h[...,5]=1
    h[1:,:,0]=30
    c = dict(history=h,history_mask=np.ones((count,21),bool),agent_mask=np.ones(count,bool),
        initial_positions=h[:,-1,:2],attributes=np.tile([4,2,1,0],(count,1)))
    states = np.zeros((count,40,6)); states[...,5]=1
    states[1:,:,0]=30
    geometry = mapping(box(-100,-20,100,20))
    return c,dict(dt_seconds=.1,map_extent_m=[-90,-10,90,10]),dict(drivable=geometry,route_corridors=[geometry]*count),states


def test_quality_and_rates_keep_failures_and_quality_rejections():
    c,m,g,s = sample()
    good = evaluate_trajectory(c,m,g,s,target_pair=(0,1))
    assert good['quality_pass'] and good['collisions']['new_target_event'] is False
    s[1,:,0] = -30 + np.arange(1,41)*2
    s[1,:,2] = 20
    risky = evaluate_trajectory(c,m,g,s,target_pair=(0,1))
    assert risky['collisions']['groups']['target']['collision']
    assert risky['quality_pass'] is False  # boundary jump/acceleration retained.
    assert risky['effective_target_event'] is False
    failed = dict(status='failed',quality_pass=None,task_applicability=good['task_applicability'],error='synthetic failure')
    result = aggregate_scenes([good,risky,failed])
    assert result['planned_scenes'] == 3 and result['failed_scenes'] == 1
    assert result['event_rates']['target']['denominator'] == 3
    assert result['event_rates']['target']['confirmed_event_rate'] == pytest.approx(1/3)
    assert result['event_rates']['target']['unknown_or_failed'] == 1
    assert result['effective_target_event_rate'] == 0
    assert result['event_rates']['non_target']['confirmed_event_rate'] is None


def test_single_agent_has_no_collision_metric_and_missing_pose_is_explicit():
    c,m,g,s = sample(1)
    result = evaluate_trajectory(c,m,g,s)
    assert result['collisions']['groups']['any']['collision'] is None
    assert aggregate_scenes([result])['event_rates']['any']['confirmed_event_rate'] is None
    c['history_mask'][:,-1] = False
    with pytest.raises(ValueError,match='Missing t0 heading'):
        evaluate_trajectory(c,m,g,s)
    explicit = np.tile([0.,1.],(1,1))
    result = evaluate_trajectory(c,m,g,s,initial_heading=explicit)
    assert result['motion']['boundary']['missing_boundary_agents'] >= 0


def test_malformed_active_state_is_rejected_padding_cannot_change_metrics():
    c,m,g,s = sample()
    s[0,0,0] = np.nan
    with pytest.raises(ValueError,match='Invalid active trajectory'):
        evaluate_trajectory(c,m,g,s)
    s[0,0,0] = 0
    expected = evaluate_trajectory(c,m,g,s)
    for key in ('history','history_mask','initial_positions','attributes'):
        c[key] = np.concatenate((c[key],np.zeros_like(c[key][:1])),axis=0)
    c['agent_mask'] = np.array([True,True,False])
    s = np.concatenate((s,np.full_like(s[:1],np.nan)),axis=0)
    actual = evaluate_trajectory(c,m,g,s)
    assert actual['collisions'] == expected['collisions']
    assert actual['motion'] == expected['motion']
    assert actual['roads'] == expected['roads']


def test_future_mask_cannot_broadcast_between_vehicles():
    c,m,g,s=sample()
    with pytest.raises(ValueError,match='without broadcasting'):
        evaluate_trajectory(c,m,g,s,future_mask=np.ones((1,40),bool))
