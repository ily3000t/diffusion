from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import xml.etree.ElementTree as ET

from sumodiff.experiments.recorder import file_identity, render_command
from .lifecycle import Lifecycle, SIMULATION_EVENTS


VARIABLE_NAMES = {
    "front_position_m": "VAR_POSITION", "navigation_angle_deg": "VAR_ANGLE", "speed_mps": "VAR_SPEED",
    "acceleration_mps2": "VAR_ACCELERATION", "length_m": "VAR_LENGTH", "width_m": "VAR_WIDTH",
    "type_id": "VAR_TYPE", "vehicle_class": "VAR_VEHICLECLASS", "road_id": "VAR_ROAD_ID",
    "lane_id": "VAR_LANE_ID", "lane_index": "VAR_LANE_INDEX", "lane_position_m": "VAR_LANEPOSITION",
    "lateral_lane_position_m": "VAR_LANEPOSITION_LAT", "route_id": "VAR_ROUTE_ID",
    "route_edges": "VAR_EDGES", "route_index": "VAR_ROUTE_INDEX", "stop_state": "VAR_STOPSTATE",
    "speed_mode": "VAR_SPEEDSETMODE", "lane_change_mode": "VAR_LANECHANGE_MODE", "departure_time_seconds": "VAR_DEPARTURE",
}


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def json_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")


def capture_vehicle(connection, vehicle_id: str, variables: dict) -> dict:
    result = connection.vehicle.getSubscriptionResults(vehicle_id)
    if result is None or set(variables.values()) - set(result):
        raise RuntimeError(f"Missing subscription data for {vehicle_id}")
    state = {name: result[code] for name, code in variables.items()}
    state["front_position_m"] = list(state["front_position_m"])
    state["route_edges"] = list(state["route_edges"])
    if state["speed_mode"] != 31 or state["lane_change_mode"] != 1621:
        raise RuntimeError(f"Safety modes changed unexpectedly for {vehicle_id}: {state['speed_mode']}, {state['lane_change_mode']}")
    numeric = [value for value in state.values() if isinstance(value, (int, float))] + state["front_position_m"]
    if any(not math.isfinite(value) for value in numeric) or state["length_m"] <= 0 or state["width_m"] <= 0:
        raise RuntimeError(f"Invalid vehicle measurement: {vehicle_id}")
    return {"vehicle_id": vehicle_id, "sumo_raw": state}


def collect_episode(output: Path, config: dict, runtime: dict, traci, scene: dict, seed: int) -> dict:
    simulation = config["simulation"]
    dt, limit = simulation["dt"], round(simulation["max_seconds"] / simulation["dt"])
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    command = [runtime["sumo"]["path"], "-c", str(output/"scene"/"episode.sumocfg"),
               "--remote-port", str(port), "--step-length", str(dt), "--seed", str(seed),
               "--lanechange.duration", str(simulation["lane_change_duration"]),
               "--time-to-teleport", str(simulation["time_to_teleport"]),
               "--collision.check-junctions", "true", "--collision.action", "teleport",
               "--lanechange-output", str(output/"lanechanges.xml"), "--lanechange-output.xy", "true",
               "--collision-output", str(output/"collisions.xml"),
               "--tripinfo-output", str(output/"tripinfo.xml"), "--tripinfo-output.write-unfinished", "true",
               "--no-step-log", "true", "--duration-log.disable", "true"]
    (output/"sumo-command.txt").write_text(render_command(command)+"\n", encoding="utf-8")
    variables = {name: getattr(traci.constants, constant) for name, constant in VARIABLE_NAMES.items()}
    lifecycle = Lifecycle()
    counts, vehicles = Counter(), {}
    previous_measurements = {}
    parent_peak = sumo_peak = None
    memory_error = None
    try:
        import psutil
        parent_process = psutil.Process()
    except ImportError as exc:
        psutil = None
        memory_error = str(exc)
    process = connection = None
    close_errors = []
    complete = False
    started = time.perf_counter()
    tick = 0
    lane_change_requested = False
    actual_runtime = None
    with (output/"sumo.log").open("w", encoding="utf-8") as log, \
         (output/"frames.jsonl").open("w", encoding="utf-8") as frames, \
         (output/"events.jsonl").open("w", encoding="utf-8") as events:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            connection = traci.connect(port, host="127.0.0.1", numRetries=8, proc=process)
            protocol, server = connection.getVersion()
            actual_runtime = {"protocol": protocol, "server": server, "pid": process.pid, "port": port}
            match = re.search(r"(\d+\.\d+\.\d+)", server)
            if protocol != runtime["client_protocol"] or match is None or match.group(1) != runtime["sumo"]["version"]:
                raise RuntimeError(f"Client/server runtime mismatch: {actual_runtime}")
            if abs(connection.simulation.getDeltaT() - dt) > 1e-9:
                raise RuntimeError("SUMO step length differs from the effective configuration")
            if connection.vehicle.getIDList():
                raise RuntimeError("Collector requires an unstepped initial simulation")
            json_line(frames, {"schema_version": "sumodiff.raw.frame.v1", "tick": 0, "time_seconds": 0.0, "vehicles": []})
            counts["frames"] = 1
            initial_loaded = {"loaded": list(connection.simulation.getLoadedIDList())}
            for event in lifecycle.observe(0, 0.0, set(), initial_loaded):
                json_line(events, event)
            for tick in range(1, limit+1):
                connection.simulationStep()
                time_seconds = connection.simulation.getTime()
                if abs(time_seconds - tick*dt) > 1e-6:
                    raise RuntimeError(f"Nonuniform time grid at tick {tick}: {time_seconds}")
                active = set(connection.vehicle.getIDList())
                observed = {kind: list(getattr(connection.simulation, method)()) for kind, method in SIMULATION_EVENTS.items()}
                previous_active = lifecycle.previous.copy()
                for event in lifecycle.observe(tick, time_seconds, active, observed):
                    json_line(events, event)
                for collision in connection.simulation.getCollisions():
                    counts["collision_events"] += 1
                    json_line(events, {"kind": "collision", "tick": tick, "time_seconds": time_seconds,
                                       "source": "traci.simulation.getCollisions", "details": vars(collision)})
                states = []
                for vehicle_id in sorted(active):
                    if vehicle_id not in previous_active:
                        connection.vehicle.subscribe(vehicle_id, list(variables.values()))
                    record = capture_vehicle(connection, vehicle_id, variables)
                    state = record["sumo_raw"]
                    states.append(record)
                    counts["vehicle_records"] += 1
                    vehicle = vehicles.setdefault(vehicle_id, {"records": 0, "first_tick": tick, "first_angle": state["navigation_angle_deg"],
                                                               "route_id": state["route_id"], "lateral_offset_frames": 0, "stopped_frames": 0})
                    vehicle["records"] += 1
                    vehicle["last_tick"], vehicle["last_angle"] = tick, state["navigation_angle_deg"]
                    vehicle["lateral_offset_frames"] += abs(state["lateral_lane_position_m"]) > 1e-5
                    vehicle["stopped_frames"] += state["speed_mps"] <= 0.1
                    if vehicle_id == "probe_lane_change" and state["lane_index"] == 1:
                        counts["probe_lane_1_frames"] += 1
                    previous = previous_measurements.get(vehicle_id)
                    if previous and previous["tick"] == tick-1:
                        delta = math.dist(previous["position"], state["front_position_m"])
                        counts["max_front_displacement_m"] = max(counts["max_front_displacement_m"], delta)
                    previous_measurements[vehicle_id] = {"tick": tick, "position": state["front_position_m"]}
                json_line(frames, {"schema_version": "sumodiff.raw.frame.v1", "tick": tick, "time_seconds": time_seconds, "vehicles": states})
                counts["frames"] += 1
                counts["peak_active_vehicles"] = max(counts["peak_active_vehicles"], len(active))
                if config["validation"]["lane_change_probe"] and not lane_change_requested and time_seconds >= 3 and "probe_lane_change" in active:
                    connection.vehicle.changeLane("probe_lane_change", 1, 8.0)
                    json_line(events, {"kind": "lane_change_request", "vehicle_id": "probe_lane_change", "tick": tick,
                                       "time_seconds": time_seconds, "source": "diagnostic_traci.changeLane", "target_lane": 1,
                                       "duration_seconds": 8.0, "safety_modes_unchanged": True})
                    lane_change_requested = True
                if psutil:
                    try:
                        parent_peak = max(parent_peak or 0, parent_process.memory_info().rss)
                        sumo_peak = max(sumo_peak or 0, psutil.Process(process.pid).memory_info().rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                        memory_error = str(exc)
                if time_seconds >= config["traffic"]["end_seconds"] and connection.simulation.getMinExpectedNumber() == 0:
                    if active:
                        raise RuntimeError("SUMO reports zero expected vehicles while active IDs remain")
                    complete = True
                    break
            if not complete:
                write_json(output/"truncation.json", {"reason": "clearance_limit", "last_tick": tick,
                           "active_ids": sorted(connection.vehicle.getIDList()),
                           "pending_ids": list(connection.simulation.getPendingVehicles()),
                           "minimum_expected": connection.simulation.getMinExpectedNumber()})
                raise RuntimeError("Episode did not drain before max_seconds; retained incomplete output")
        finally:
            if connection is not None:
                try:
                    connection.close(wait=False)
                except Exception as exc:
                    close_errors.append(f"{type(exc).__name__}: {exc}")
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                        close_errors.append("SUMO required forced process termination")
            write_json(output/"collection_partial.json", {"complete_episode": complete, "last_tick": tick,
                       "dt": dt, "duration_seconds": tick*dt, "counts": dict(counts),
                       "lifecycle_counts": dict(lifecycle.counts), "entered_ids": sorted(lifecycle.entered),
                       "arrived_ids": sorted(lifecycle.arrived), "vehicles": vehicles})
            write_json(output/"process.json", {"runtime": actual_runtime, "returncode": process.returncode if process else None,
                       "complete_episode": complete, "last_tick": tick, "close_errors": close_errors,
                       "elapsed_seconds": time.perf_counter()-started})
    if process.returncode != 0 or close_errors:
        raise RuntimeError(f"SUMO process/close failed; see {output/'process.json'}")
    lane_changes = [item.attrib for item in ET.parse(output/"lanechanges.xml").getroot().findall("change")]
    turns = []
    for vehicle_id, vehicle in vehicles.items():
        delta = abs((vehicle["last_angle"]-vehicle["first_angle"]+180) % 360-180)
        vehicle["net_heading_change_degrees"] = delta
        if delta >= 30 and scene["routes"].get(vehicle["route_id"], {}).get("movement") in ("left", "right"):
            turns.append(vehicle_id)
    anomalous = {key: lifecycle.counts[key] for key in ("unexplained_disappearance", "unexplained_appearance", "teleport_started", "emergency_stop")}
    normal_quality = not any(anomalous.values()) and counts["collision_events"] == 0 and lifecycle.entered == lifecycle.arrived
    quality = {"schema_version": "sumodiff.raw.quality.v1", "complete_episode": complete, "dt": dt,
               "duration_seconds": tick*dt, "counts": dict(counts), "lifecycle_counts": dict(lifecycle.counts),
               "entered_ids": sorted(lifecycle.entered), "arrived_ids": sorted(lifecycle.arrived),
               "unarrived_entered_ids": sorted(lifecycle.entered-lifecycle.arrived), "lane_changes": lane_changes,
               "turning_vehicle_ids": turns, "vehicles": vehicles, "normal_traffic_quality_pass": normal_quality,
               "eligible_for_normal_training": normal_quality and not config["validation"]["diagnostic_only"],
               "diagnostic_only": config["validation"]["diagnostic_only"],
               "probes": {"lane_change_requested": lane_change_requested,
                          "lane_change_reached_lane_1": counts["probe_lane_1_frames"] > 0,
                          "lane_change_lateral_frames": vehicles.get("probe_lane_change", {}).get("lateral_offset_frames", 0),
                          "stop_low_speed_frames": vehicles.get("probe_stop", {}).get("stopped_frames", 0)},
               "resources": {"collection_elapsed_seconds": time.perf_counter()-started,
                             "parent_peak_sampled_rss_bytes": parent_peak, "sumo_peak_sampled_rss_bytes": sumo_peak,
                             "memory_query_error": memory_error, "gpu_used": False}}
    write_json(output/"quality.json", quality)
    return quality
