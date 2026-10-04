from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from sumodiff.simulation.config import resolve_config
from sumodiff.simulation.lifecycle import Lifecycle
from sumodiff.simulation.scenes import layout, semantic_geometry_id


@pytest.mark.parametrize("family", ["three_lane_straight", "ramp_merge", "unsignalized_intersection"])
def test_explicit_topology_routes_and_safety_defaults(family):
    config = resolve_config({"family": family})
    nodes, edges, links, routes = layout(config)
    assert all(node["type"] != "traffic_light" for node in nodes)
    assert all(edge["allow"] == "passenger" for edge in edges)
    allowed = {(link["from"], link["to"]) for link in links}
    for route in routes.values():
        assert all(pair in allowed for pair in zip(route["edges"], route["edges"][1:]))
    assert config["simulation"]["dt"] == 0.1
    assert config["simulation"]["lane_change_duration"] > 0
    assert config["driver"]["tau"] >= 0.1
    assert not config["validation"]["diagnostic_only"]


def test_three_lane_road_and_four_lane_ramp_acceleration():
    _, edges, _, _ = layout(resolve_config({"family": "three_lane_straight"}))
    assert {edge["numLanes"] for edge in edges} == {"3"}
    _, edges, links, _ = layout(resolve_config({"family": "ramp_merge"}))
    lanes = {edge["id"]: edge["numLanes"] for edge in edges}
    assert lanes == {"main_in": "3", "ramp_in": "1", "acceleration": "4", "main_out": "3"}
    assert any(link["from"] == "ramp_in" and link["toLane"] == "0" for link in links)


def test_two_lane_intersection_connections_use_legal_turn_lanes():
    config = resolve_config({"family": "unsignalized_intersection", "geometry": {"lanes_per_direction": 2}})
    _, _, links, routes = layout(config)
    straight = [link for link in links if link["from"] == "west_in" and link["to"] == "east_out"]
    assert {(link["fromLane"],link["toLane"]) for link in straight} == {("0","0"),("1","1")}
    assert routes["west_north"]["movement"] == "left"
    assert next(link for link in links if link["from"] == "west_in" and link["to"] == "north_out")["fromLane"] == "1"


def test_demand_override_is_not_silently_augmented():
    config = resolve_config({"family": "unsignalized_intersection", "traffic": {"route_rates_per_hour": {"west_east": 100}}})
    assert config["traffic"]["route_rates_per_hour"] == {"west_east": 100}


@pytest.mark.parametrize("override", [
    {"geometry": {"nonexistent": 1}},
    {"traffic": {"route_rates_per_hour": {"not_a_route": 100}}},
    {"traffic": {"route_rates_per_hour": {"main": 0}}, "validation": {"diagnostic_only": True}},
    {"simulation": {"dt": 0.2}},
    {"driver": {"tau": 0.01}},
    {"driver": {"sigma": 1.1}},
    {"driver": {"emergencyDecel": 1}},
    {"validation": {"lane_change_probe": True}},
])
def test_invalid_or_inert_parameters_are_rejected(override):
    with pytest.raises((ValueError, TypeError)):
        resolve_config({"family": "three_lane_straight", **override})


def test_unsupported_client_source_does_not_silently_fall_back():
    with pytest.raises(NotImplementedError):
        resolve_config({"family": "three_lane_straight", "runtime": {"client_source": "auto"}})


def test_geometry_identity_ignores_generation_comments_and_tracks_lane_geometry(tmp_path):
    a, b = tmp_path/"a.xml", tmp_path/"b.xml"
    a.write_text('<net><!--time/path A--><edge id="e"><lane shape="0,0 20,0" width="3.5"/></edge></net>')
    b.write_text('<net>\n<!--time/path B--><edge id="e"><lane width="3.5" shape="0,0 20,0"/></edge>\n</net>')
    assert semantic_geometry_id(a) == semantic_geometry_id(b)
    b.write_text(b.read_text().replace('20,0','30,0'))
    assert semantic_geometry_id(a) != semantic_geometry_id(b)


def test_arrival_and_teleport_are_not_confused_with_unknown_disappearance():
    life = Lifecycle()
    life.observe(1, 0.1, {"a", "b", "c"}, {"entered": ["a", "b", "c"]})
    events = life.observe(2, 0.2, set(), {"arrived": ["a"], "teleport_started": ["b"]})
    assert next(event for event in events if event["vehicle_id"] == "a" and event["kind"] == "exited")["reason"] == "arrived"
    assert next(event for event in events if event["vehicle_id"] == "b" and event["kind"] == "exited")["reason"] == "teleport"
    assert any(event["vehicle_id"] == "c" and event["kind"] == "unexplained_disappearance" for event in events)
    assert life.arrived == {"a"}
    reentry = life.observe(3, 0.3, {"b"}, {"teleport_ended": ["b"]})
    assert not any(event["kind"] == "unexplained_appearance" for event in reentry)
    assert not life.teleporting


def test_unknown_appearance_is_preserved():
    life = Lifecycle()
    events = life.observe(1, 0.1, {"unexpected"}, {})
    assert events[0]["kind"] == "unexplained_appearance"


def test_initial_loaded_events_can_precede_first_sampling_step():
    life = Lifecycle()
    events = life.observe(0, 0.0, set(), {"loaded": ["scheduled"]})
    assert events[0]["time_seconds"] == 0
    assert events[0]["source"] == "traci.simulation.getLoadedIDList"
