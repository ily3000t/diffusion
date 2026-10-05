import numpy as np
import pytest
from shapely.geometry import Polygon
from sumodiff.geometry.boxes import box_corners, separation_margin, interpolate_pose
from sumodiff.evaluation.collisions import CollisionConfig, check_interval, evaluate_collisions


def test_sat_against_independent_polygon_intersection():
    rng = np.random.default_rng(23005)
    for _ in range(300):
        a, b = np.r_[rng.normal(size=2)*3,rng.uniform(-np.pi,np.pi)], np.r_[rng.normal(size=2)*3,rng.uniform(-np.pi,np.pi)]
        sa,sb = rng.uniform(1,6,size=2),rng.uniform(1,6,size=2)
        oracle = Polygon(box_corners(a,sa)).intersects(Polygon(box_corners(b,sb)))
        assert (separation_margin(a,b,sa,sb) <= 0) == oracle
    assert separation_margin(np.array([0,0,0]),np.array([4,0,0]),[4,2],[4,2]) == 0
    assert separation_margin(np.array([0,0,0]),np.array([4.000001,0,0]),[4,2],[4,2]) > 0


def test_between_frame_translation_contact_not_visible_at_endpoints():
    a0,a1,b = np.array([-10.,0.,0.]),np.array([10.,0.,0.]),np.zeros(3)
    assert separation_margin(a0,b,[2,1],[2,1]) > 0 and separation_margin(a1,b,[2,1],[2,1]) > 0
    result = check_interval(a0,a1,b,b,[2,1],[2,1])
    assert result['contact'] and result['method'] == 'exact_swept_translation'
    assert result['fraction'] == pytest.approx(.4, abs=1e-8)
    poses = np.stack((np.stack((a0,a1)),np.stack((b,b))))
    report = evaluate_collisions(poses,np.array([[2,1],[2,1]]),np.ones((2,2),bool),np.ones(2,bool),target_pair=(0,1))
    assert report['new_target_event']
    assert not report['pairs'][0]['future_frame_collision']
    assert report['groups']['non_target']['collision'] is None
    assert report['groups']['background_background']['collision_pair_fraction'] is None


def test_initial_collision_cannot_be_new_target_event():
    poses = np.zeros((2,3,3))
    report = evaluate_collisions(poses,np.array([[4,2],[4,2]]),np.ones((2,3),bool),np.ones(2,bool),target_pair=(0,1))
    assert report['groups']['target']['collision'] and not report['new_target_event']
    assert report['pairs'][0]['initial_contact']


def test_rotating_budget_exhaustion_reports_unknown():
    a0,a1,b = np.zeros(3),np.array([0.,0.,1.2]),np.array([4.5,0.,0.])
    limited = check_interval(a0,a1,b,b,[4,2],[4,2],config=CollisionConfig(max_evaluations_per_interval=1))
    assert limited['contact'] is None
    resolved = check_interval(a0,a1,b,b,[4,2],[4,2])
    assert resolved['contact'] is False
    hit = check_interval(a0,a1,np.array([4.1,0.,0.]),np.array([4.1,0.,0.]),[4,2],[4,2])
    assert hit['contact'] is True
    yaw = interpolate_pose(np.array([0,0,np.deg2rad(170)]),np.array([0,0,np.deg2rad(-170)]),.5)[2]
    assert abs(abs(yaw)-np.pi) < 1e-10


def test_missing_intervals_are_not_bridged_and_padding_is_excluded():
    poses = np.zeros((3,3,3))
    poses[0,:,0] = [-10,0,10]
    poses[2] = np.nan
    mask = np.array([[True,False,True],[True,True,True],[False,False,False]])
    report = evaluate_collisions(poses,np.array([[2,1],[2,1],[0,0]]),mask,np.array([True,True,False]))
    assert report['groups']['any']['collision'] is None
    assert report['groups']['any']['observed_collision'] is False
    assert report['pairs'][0]['checked_intervals'] == 0
    singleton = evaluate_collisions(poses[:1],np.array([[2,1]]),np.ones((1,3),bool),np.ones(1,bool))
    assert singleton['groups']['any']['collision_pair_fraction'] is None


def test_non_target_roles_include_all_other_pairs():
    poses = np.zeros((4,2,3))
    poses[:, :, 0] = np.array([0,100,0,100])[:,None]
    result = evaluate_collisions(poses,np.tile([4,2],(4,1)),np.ones((4,2),bool),np.ones(4,bool),target_pair=(0,1))
    assert result['groups']['target']['collision'] is False
    assert result['groups']['non_target']['pair_count'] == 5
    assert result['groups']['attacker_background']['collision']
    assert result['groups']['target_background']['collision']
    assert result['groups']['background_background']['collision'] is False
