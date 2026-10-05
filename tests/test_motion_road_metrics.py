import numpy as np
import pytest
from shapely.geometry import Polygon, box, mapping
from sumodiff.evaluation.motion import evaluate_motion
from sumodiff.evaluation.roads import evaluate_roads, RoadConfig


def trajectory(position, yaw=None):
    times = np.arange(-20,41)*.1
    xy = position(times)
    previous = position(times-.1)
    headings = np.zeros((61,2)); headings[:,1] = 1
    if yaw is not None:
        headings = np.column_stack((np.sin(yaw(times)),np.cos(yaw(times))))
    states = np.concatenate((xy,(xy-previous)/.1,headings),axis=-1)[None]
    return states[:,:21],states[:,21:]


def motion(h,f,hm=None,fm=None):
    return evaluate_motion(h,np.ones(h.shape[:2],bool) if hm is None else hm,
        f,np.ones(f.shape[:2],bool) if fm is None else fm,h[:,-1,:2],h[:,-1,4:6],np.ones(len(h),bool))


def test_constant_acceleration_jerk_and_history_boundary_are_analytic():
    h,f = trajectory(lambda t:np.column_stack((5*t+t*t,np.zeros_like(t))))
    result = motion(h,f)
    assert result['longitudinal_accel_abs_mps2']['mean'] == pytest.approx(2.,abs=1e-10)
    assert result['jerk_norm_mps3']['max'] < 1e-9
    assert result['boundary']['acceleration_mps2']['max'] == pytest.approx(2.)
    assert result['hard_quality_pass']
    h,f = trajectory(lambda t:np.column_stack((5*t+.5*t**3,np.zeros_like(t))))
    assert motion(h,f)['jerk_norm_mps3']['mean'] == pytest.approx(3.,abs=1e-9)
    h,f = trajectory(lambda t:np.column_stack((np.zeros_like(t),np.zeros_like(t))))
    f[0,:,0] = np.arange(1,41)
    f[0,:,2] = 10
    result = motion(h,f)
    assert result['boundary']['acceleration_mps2']['max'] == pytest.approx(100.)
    assert result['boundary']['jerk_mps3']['max'] == pytest.approx(1000.)
    assert result['hard_quality_pass'] is False


def test_stopped_heading_alignment_na_and_missing_boundary():
    h,f = trajectory(lambda t:np.zeros((len(t),2)))
    result = motion(h,f)
    assert result['hard_quality_pass'] and result['observed_comfort_pass_fraction_agents'] == 1.
    assert result['heading_velocity_angle_rad']['max'] is None
    hm = np.ones(h.shape[:2],bool); hm[:,-3:] = False
    result = motion(h,f,hm=hm)
    assert result['boundary']['jerk_mps3']['max'] is None
    assert result['boundary']['missing_boundary_agents'] == 1
    assert result['hard_quality_pass'] is None
    fm = np.ones(f.shape[:2],bool); fm[:,2] = False
    result = motion(h,f,fm=fm)
    assert not result['full_future_observed'] and result['hard_quality_pass'] is None
    assert result['jerk_norm_mps3']['count'] == 36


def test_heading_movement_contradiction_and_raw_velocity_error():
    h,f = trajectory(lambda t:np.column_stack((np.zeros_like(t),5*t)))
    result = motion(h,f)
    assert result['violation_counts']['heading_velocity'] == 40
    f[...,2:4] = 0
    result = motion(h,f)
    assert result['velocity_residual_mps']['mean'] == pytest.approx(5.)
    assert result['violation_counts']['velocity_consistency'] == 40


def road_inputs(road,corridor=None,poses=None,extent=(-20,-20,20,20),size=(10,4)):
    poses = np.zeros((1,1,3)) if poses is None else poses
    return evaluate_roads(poses,np.array([size]),np.ones(poses.shape[:2],bool),np.ones(1,bool),
        dict(drivable=mapping(road),route_corridors=[mapping(road if corridor is None else corridor)]),extent)


def test_whole_body_hole_is_detected_even_when_corners_are_inside():
    road = Polygon([(-6,-6),(6,-6),(6,6),(-6,6)],holes=[[(-1,-1),(1,-1),(1,1),(-1,1)]])
    result = road_inputs(road)
    assert result['road']['violation_body_frames'] == 1
    assert result['road']['max_outside_area_m2'] == pytest.approx(4.,abs=1e-4)
    assert not result['geometry_quality_pass']


def test_coverage_and_route_are_independent_of_drivable_area():
    result = road_inputs(box(-30,-8,30,8),poses=np.array([[[10.,0.,0.]]]),extent=(-5,-5,5,5),size=(4,2))
    assert result['road']['violation_body_frames'] == 0 and result['map_coverage']['violation_body_frames'] == 1
    cross = box(-20,-2,20,2).union(box(-2,-20,2,20))
    result = road_inputs(cross,box(-20,-2,20,2),poses=np.array([[[0.,4.,np.pi/2]]]),size=(4,1.8))
    assert result['road']['violation_body_frames'] == 0 and result['route']['violation_body_frames'] == 1
    # A body straddling legal adjacent lanes is covered by their corridor union.
    lanes = box(-20,-4,20,0).union(box(-20,0,20,4))
    result = road_inputs(lanes,size=(4,2))
    assert result['geometry_quality_pass']


def test_explicit_bounded_geometry_repair_preserves_collapsed_parts_and_rejects_area_change():
    # Retraced spike collapses to a line; dropping it would move the point set.
    spike = Polygon([(-5,-5),(5,-5),(5,5),(0,5),(0,7),(0,5),(-5,5),(-5,-5)])
    assert not spike.is_valid
    args = (np.zeros((1,1,3)),np.array([[4.,2.]]),np.ones((1,1),bool),np.ones(1,bool),
        dict(drivable=mapping(spike),route_corridors=[mapping(box(-5,-5,5,5))]),(-10,-10,10,10))
    with pytest.raises(ValueError,match='Invalid drivable'):
        evaluate_roads(*args)
    report = evaluate_roads(*args,config=RoadConfig(geometry_repair='bounded_make_valid'))
    repair = report['geometry_repairs'][0]
    assert report['geometry_quality_pass'] and repair['hausdorff_m'] == 0
    assert repair['area_change_m2'] == 0 and 'LineString' in repair['component_types']
    # A positive-area rectangle cannot be covered by the zero-width spike.
    outside_args = (np.array([[[0.,6.,0.]]]),np.array([[.5,.5]]),*args[2:])
    assert evaluate_roads(*outside_args,config=RoadConfig(geometry_repair='bounded_make_valid'))['road']['violation_body_frames'] == 1
    crossed = Polygon([(0,0),(4,4),(0,4),(3,0),(0,0)])
    broken = dict(drivable=mapping(crossed),route_corridors=args[4]['route_corridors'])
    with pytest.raises(ValueError,match='exceeds numerical bounds'):
        evaluate_roads(*args[:4],broken,args[5],config=RoadConfig(geometry_repair='bounded_make_valid'))
