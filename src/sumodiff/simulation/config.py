from __future__ import annotations

from copy import deepcopy
import math


GEOMETRIES = {
    "three_lane_straight": {"core_length": 300.0, "entry_buffer": 180.0, "exit_buffer": 180.0,
                            "lane_width": 3.5, "speed_limit": 18.0, "rotation_degrees": 0.0},
    "ramp_merge": {"approach_length": 250.0, "merge_length": 120.0, "exit_buffer": 220.0,
                   "ramp_offset": 50.0, "lane_width": 3.5, "speed_limit": 20.0, "rotation_degrees": 0.0},
    "unsignalized_intersection": {"approach_length": 200.0, "lane_width": 3.5, "speed_limit": 13.9,
                                 "lanes_per_direction": 1, "north_skew_degrees": 0.0,
                                 "rotation_degrees": 0.0, "junction_radius": 10.0, "junction_type": "priority"},
}


def route_names(family: str) -> list[str]:
    if family == "three_lane_straight":
        return ["main"]
    if family == "ramp_merge":
        return ["main", "ramp"]
    return [f"{source}_{target}" for source in ("west", "east", "north", "south")
            for target in ("west", "east", "north", "south") if source != target]


def strict_merge(defaults: dict, supplied: dict, path: str = "config") -> dict:
    if not isinstance(supplied, dict):
        raise TypeError(f"{path} must be a mapping")
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {path} keys: {sorted(unknown)}")
    result = deepcopy(defaults)
    for key, value in supplied.items():
        result[key] = strict_merge(defaults[key], value, f"{path}.{key}") if isinstance(defaults[key], dict) else value
    return result


def positive(value, name: str, *, allow_zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def resolve_config(supplied: dict) -> dict:
    family = supplied.get("family")
    if family not in GEOMETRIES:
        raise ValueError(f"Unsupported family: {family}")
    defaults = {"schema_version": "sumodiff.scenario.v1", "family": family,
                "geometry": GEOMETRIES[family],
                "traffic": {"begin_seconds": 0.0, "end_seconds": 12.0,
                            "route_rates_per_hour": {name: 900.0 if name == "main" else 120.0 for name in route_names(family)}},
                "driver": {"length": 4.7, "width": 1.8, "accel": 2.6, "decel": 4.5,
                           "emergencyDecel": 9.0, "tau": 1.2, "minGap": 2.5, "sigma": 0.3,
                           "speedFactor": 1.0, "speedDev": 0.1, "lcStrategic": 1.0,
                           "lcCooperative": 1.0, "lcSpeedGain": 1.0, "lcKeepRight": 1.0},
                "simulation": {"dt": 0.1, "max_seconds": 120.0, "lane_change_duration": 3.0,
                               "time_to_teleport": 300.0},
                "validation": {"lane_change_probe": False, "stop_probe": False, "diagnostic_only": False},
                "runtime": {"sumo_home": None, "client_source": "sumo_home_tools"}}
    # Routes are an explicit complete active-demand list, not additions to defaults.
    provided = deepcopy(supplied)
    rates = provided.get("traffic", {}).pop("route_rates_per_hour", None)
    config = strict_merge(defaults, provided)
    if rates is not None:
        if not isinstance(rates, dict) or set(rates) - set(route_names(family)):
            raise ValueError("Traffic demand contains an unknown route")
        config["traffic"]["route_rates_per_hour"] = rates
    if config["schema_version"] != "sumodiff.scenario.v1":
        raise ValueError("Unsupported scenario schema")
    for name, value in config["geometry"].items():
        if name in ("rotation_degrees", "north_skew_degrees"):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Invalid geometry.{name}")
        elif name != "junction_type":
            positive(value, f"geometry.{name}")
    if family == "unsignalized_intersection":
        geometry = config["geometry"]
        if type(geometry["lanes_per_direction"]) is not int or geometry["lanes_per_direction"] not in (1, 2):
            raise ValueError("Intersection supports one or two lanes per direction")
        if geometry["junction_type"] not in ("priority", "right_before_left") or abs(geometry["north_skew_degrees"]) > 30:
            raise ValueError("Unsupported junction rules or excessive north/south skew")
    traffic = config["traffic"]
    positive(traffic["begin_seconds"], "traffic.begin_seconds", allow_zero=True)
    positive(traffic["end_seconds"], "traffic.end_seconds")
    if traffic["end_seconds"] <= traffic["begin_seconds"]:
        raise ValueError("Traffic end must follow begin")
    for route, rate in traffic["route_rates_per_hour"].items():
        positive(rate, f"route_rate.{route}", allow_zero=True)
    for name, value in config["driver"].items():
        positive(value, f"driver.{name}", allow_zero=name.startswith("lc") or name in ("sigma", "speedDev"))
    if not 0 <= config["driver"]["sigma"] <= 1 or config["driver"]["tau"] < 0.1:
        raise ValueError("Invalid driver imperfection or reaction time")
    if config["driver"]["emergencyDecel"] < config["driver"]["decel"]:
        raise ValueError("Emergency deceleration must be at least regular deceleration")
    for name, value in config["simulation"].items():
        positive(value, f"simulation.{name}")
    if config["simulation"]["dt"] != 0.1 or config["simulation"]["max_seconds"] <= traffic["end_seconds"]:
        raise ValueError("Stage 1 requires dt=0.1 and a clearance limit after demand end")
    for name, value in config["validation"].items():
        if type(value) is not bool:
            raise ValueError(f"validation.{name} must be boolean")
    validation = config["validation"]
    if validation["lane_change_probe"] or validation["stop_probe"]:
        if family != "three_lane_straight" or not validation["diagnostic_only"]:
            raise ValueError("Straight-road probes require diagnostic_only=true")
        if config["geometry"]["entry_buffer"] <= 100:
            raise ValueError("Diagnostic stop requires entry buffer >100m")
    if not any(traffic["route_rates_per_hour"].values()) and not (validation["lane_change_probe"] or validation["stop_probe"]):
        raise ValueError("Scenario has no traffic")
    if config["runtime"]["client_source"] != "sumo_home_tools":
        raise NotImplementedError("Only explicitly version-checked SUMO installation clients are supported")
    return config
