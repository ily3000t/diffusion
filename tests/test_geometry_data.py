import numpy as np
import pytest
from shapely.geometry import Point, shape
from sumodiff.geometry.coordinates import FixedFrame, sumo_center
from sumodiff.geometry.maps import RoadMap, rasterize


def test_center_navigation_and_fixed_frame_roundtrip():
    center, yaw = sumo_center([[10., 4.], [1., 10.]], [90., 0.], [4., 6.])
    np.testing.assert_allclose(center, [[8., 4.], [1., 7.]])
    frame = FixedFrame(center[1], yaw[1])
    points = np.array([[2., 3.], [10., -15.]])
    np.testing.assert_allclose(frame.inverse_positions(frame.positions(points)), points, atol=1e-12)
    np.testing.assert_allclose(frame.inverse_vectors(frame.vectors(points)), points, atol=1e-12)
    np.testing.assert_allclose(frame.heading(yaw)[1], [0, 1], atol=1e-12)
    with pytest.raises(ValueError):
        frame.origin[0] = 3


def test_route_internal_chain_lane_changes_and_rules(tmp_path):
    net = tmp_path / 'tiny.net.xml'
    net.write_text('''<net>
    <edge id="a"><lane id="a_0" index="0" width="3" speed="10" length="20" shape="-20,-1.5 -2,-1.5"/>
                 <lane id="a_1" index="1" width="3" speed="10" length="20" shape="-20,1.5 -2,1.5"/></edge>
    <edge id="b"><lane id="b_0" index="0" width="3" speed="10" length="20" shape="4,-1.5 20,-1.5"/></edge>
    <edge id="c"><lane id="c_0" index="0" width="3" speed="10" length="20" shape="4,7 20,7"/></edge>
    <edge id=":j_0" function="internal"><lane id=":j_0_0" index="0" width="3" speed="10" length="20" shape="-2,-1.5 0,-1.5"/></edge>
    <edge id=":j_2" function="internal"><lane id=":j_2_0" index="0" width="3" speed="10" length="20" shape="0,-1.5 4,-1.5"/></edge>
    <junction id="j" type="priority" x="0" y="0" incLanes="a_0" shape="-2,-3 4,-3 4,3 -2,3">
      <request index="0" response="0" foes="0" cont="1"/></junction>
    <connection from="a" to="b" fromLane="0" toLane="0" via=":j_0_0" dir="s" state="M"/>
    <connection from=":j_0" to="b" fromLane="0" toLane="0" via=":j_2_0" dir="s" state="m"/>
    <connection from=":j_2" to="b" fromLane="0" toLane="0" dir="s" state="M"/></net>''', encoding='utf-8')
    road = RoadMap.read(net)
    assert road.route_lane_ids(['a', 'b']) == {'a_0', 'a_1', 'b_0', ':j_0_0', ':j_2_0'}
    assert road.route_area(['a', 'b']).covers(Point(0, -1.5))
    assert not road.route_area(['a', 'b']).covers(Point(10, 7))
    frame = FixedFrame(np.array([1., 2.]), .3)
    features, adjacency, route_mask, exact = road.local(frame, [['a', 'b']])
    assert features.shape == (6, 64, 8)
    assert adjacency.sum() == 3 and route_mask.sum() == 5
    assert shape(exact['drivable']).covers(Point(*frame.positions([0, 0])))
    assert road.connections[0]['request_index'] == 0
    raster = rasterize(exact, [-30, -30, 30, 30])
    assert raster.shape == (3, 256, 256) and set(np.unique(raster)) == {0, 1}
    with pytest.raises(ValueError, match='Disconnected'):
        road.route_area(['a', 'c'])

def test_zero_length_internal_lane_is_retained_with_direction(tmp_path):
    net = tmp_path / 'zero.net.xml'
    net.write_text('''<net>
    <edge id="a"><lane id="a_0" index="0" length="20" width="4" speed="10" shape="-20,0 0,0"/></edge>
    <edge id="b"><lane id="b_0" index="0" length="20" width="4" speed="10" shape="0,0 20,0"/></edge>
    <edge id=":j_0" function="internal"><lane id=":j_0_0" index="0" length=".1" width="4" speed="10" shape="0,0 0,0"/></edge>
    <connection from="a" to="b" fromLane="0" toLane="0" via=":j_0_0" dir="s" state="M"/>
    <connection from=":j_0" to="b" fromLane="0" toLane="0" dir="s" state="M"/></net>''', encoding='utf-8')
    road = RoadMap.read(net)
    features, adjacency, routes, exact = road.local(FixedFrame(np.zeros(2), np.pi / 2), [['a', 'b']])
    assert exact['lanes'][2]['zero_geometry_length']
    np.testing.assert_allclose(features[2, :, :2], 0, atol=1e-7)
    np.testing.assert_allclose(features[2, :, 2:4], np.tile([0, -1], (64, 1)), atol=1e-7)
    assert adjacency.sum() == 2 and routes.sum() == 3
