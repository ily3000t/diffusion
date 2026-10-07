from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from sumodiff.experiments.recorder import file_identity, render_command


def write_xml(root: ET.Element, path: Path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def semantic_geometry_id(path: Path) -> str:
    # Exclude netconvert comments containing time/absolute filenames. Preserve
    # compiled lane shapes, speeds, widths, links and right-of-way semantics.
    def canonical(element):
        return [element.tag, sorted(element.attrib.items()), [canonical(child) for child in element]]
    payload = json.dumps(canonical(ET.parse(path).getroot()), separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def layout(config: dict):
    family, geometry = config["family"], config["geometry"]
    nodes, edges, connections, routes = [], [], [], {}
    rotation = math.radians(geometry["rotation_degrees"])

    def point(x, y):
        return (x * math.cos(rotation) - y * math.sin(rotation),
                x * math.sin(rotation) + y * math.cos(rotation))

    def node(name, x, y, kind="priority"):
        x, y = point(x, y)
        result = {"id": name, "x": f"{x:.8f}", "y": f"{y:.8f}", "type": kind}
        if family == "unsignalized_intersection" and name == "center":
            result["radius"] = str(geometry["junction_radius"])
        nodes.append(result)

    def edge(name, start, end, lanes, priority=3, shape=None):
        value = {"id": name, "from": start, "to": end, "numLanes": str(lanes),
                 "speed": str(geometry["speed_limit"]), "width": str(geometry["lane_width"]),
                 "priority": str(priority), "allow": "passenger", "spreadType": "right"}
        if shape:
            value["shape"] = " ".join(f"{x:.8f},{y:.8f}" for x, y in [point(*p) for p in shape])
        edges.append(value)

    def connect(source, target, source_lane, target_lane):
        connections.append({"from": source, "to": target, "fromLane": str(source_lane), "toLane": str(target_lane)})

    if family == "three_lane_straight" and not geometry["segmented_straight"]:
        node("start", -geometry["entry_buffer"], 0)
        node("end", geometry["core_length"]+geometry["exit_buffer"], 0)
        edge("main", "start", "end", 3)
        routes["main"] = {"edges": ["main"], "movement": "straight"}
    elif family == "three_lane_straight":
        length, before, after = geometry["core_length"], geometry["entry_buffer"], geometry["exit_buffer"]
        for name, x in [("start", -before), ("core_start", 0), ("core_end", length), ("end", length+after)]:
            node(name, x, 0)
        for name, start, end in [("entry", "start", "core_start"), ("core", "core_start", "core_end"), ("exit", "core_end", "end")]:
            edge(name, start, end, 3)
        for lane in range(3):
            connect("entry", "core", lane, lane)
            connect("core", "exit", lane, lane)
        routes["main"] = {"edges": ["entry", "core", "exit"], "movement": "straight"}
    elif family == "ramp_merge":
        before, length, after, offset = (geometry[key] for key in ("approach_length", "merge_length", "exit_buffer", "ramp_offset"))
        node("main_start", -before, 0)
        node("ramp_start", -before, -offset)
        node("merge_start", 0, 0)
        node("merge_end", length, 0, "zipper")
        node("end", length+after, 0)
        edge("main_in", "main_start", "merge_start", 3)
        edge("ramp_in", "ramp_start", "merge_start", 1, 1,
             [(-before, -offset), (-before/2, -offset/2), (-25, -geometry["lane_width"]*4), (0, 0)])
        edge("acceleration", "merge_start", "merge_end", 4)
        edge("main_out", "merge_end", "end", 3)
        for lane in range(3):
            connect("main_in", "acceleration", lane, lane+1)
            connect("acceleration", "main_out", lane+1, lane)
        connect("ramp_in", "acceleration", 0, 0)
        connect("acceleration", "main_out", 0, 0)
        routes = {"main": {"edges": ["main_in", "acceleration", "main_out"], "movement": "straight"},
                  "ramp": {"edges": ["ramp_in", "acceleration", "main_out"], "movement": "merge"}}
    else:
        length, lanes = geometry["approach_length"], geometry["lanes_per_direction"]
        skew = math.radians(geometry["north_skew_degrees"])
        positions = {"west": (-length, 0), "east": (length, 0),
                     "north": (length*math.sin(skew), length*math.cos(skew)),
                     "south": (-length*math.sin(skew), -length*math.cos(skew))}
        node("center", 0, 0, geometry["junction_type"])
        for name, position in positions.items():
            node(name, *position)
            priority = 3 if name in ("west", "east") else 1
            edge(name+"_in", name, "center", lanes, priority)
            edge(name+"_out", "center", name, lanes, priority)
        opposites = {"west": "east", "east": "west", "north": "south", "south": "north"}
        left = {"west": "north", "east": "south", "north": "east", "south": "west"}
        for source in positions:
            for target in positions:
                if source == target:
                    continue
                if target == opposites[source]:
                    movement, pairs = "straight", [(lane, lane) for lane in range(lanes)]
                elif target == left[source]:
                    movement, pairs = "left", [(lanes-1, lanes-1)]
                else:
                    movement, pairs = "right", [(0, 0)]
                for i, j in pairs:
                    connect(source+"_in", target+"_out", i, j)
                routes[source+"_"+target] = {"edges": [source+"_in", target+"_out"], "movement": movement}
    return nodes, edges, connections, routes


def build_scene(output: Path, config: dict, runtime: dict, seed: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    nodes, edges, links, routes = layout(config)
    for name, tag, records, child_tag in [("nodes.nod.xml", "nodes", nodes, "node"),
                                          ("edges.edg.xml", "edges", edges, "edge"),
                                          ("connections.con.xml", "connections", links, "connection")]:
        document = ET.Element(tag)
        for record in records:
            ET.SubElement(document, child_tag, record)
        write_xml(document, output/name)
    command = [runtime["netconvert"]["path"], "--node-files", str(output/"nodes.nod.xml"),
               "--edge-files", str(output/"edges.edg.xml"), "--connection-files", str(output/"connections.con.xml"),
               "--output-file", str(output/"network.net.xml"), "--no-turnarounds", "true",
               "--offset.disable-normalization", "true",
               "--junctions.corner-detail", str(config["network"]["corner_detail"]),
               "--junctions.internal-link-detail", str(config["network"]["internal_link_detail"]),
               "--precision", str(config["network"]["output_precision"]),
               "--junctions.limit-turn-speed", str(config["network"]["limit_turn_lateral_accel_mps2"])]
    (output/"netconvert-command.txt").write_text(render_command(command)+"\n", encoding="utf-8")
    with (output/"netconvert.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError(f"netconvert failed ({result.returncode}); see {output/'netconvert.log'}")
    network = ET.parse(output/"network.net.xml").getroot()
    if network.findall("tlLogic"):
        raise RuntimeError("Signal control is outside the stage-1 scope")
    connected = {(link.get("from"), link.get("to")) for link in network.findall("connection")}
    for name, route in routes.items():
        if not all(pair in connected for pair in zip(route["edges"], route["edges"][1:])):
            raise RuntimeError(f"Compiled route is disconnected: {name}")
    demand = ET.Element("routes")
    vehicle_type = {"id": "passenger", "vClass": "passenger", "carFollowModel": config["driver"]["car_follow_model"], "laneChangeModel": "LC2013",
                    "maxSpeed": str(config["geometry"]["speed_limit"])}
    vehicle_type.update({key: str(value) for key, value in config["driver"].items() if key != "car_follow_model"})
    ET.SubElement(demand, "vType", vehicle_type)
    for name, route in routes.items():
        ET.SubElement(demand, "route", {"id": name, "edges": " ".join(route["edges"])})
    scheduled = []
    for name, rate in config["traffic"]["route_rates_per_hour"].items():
        if rate:
            flow = ET.Element("flow", {"id": "flow_"+name, "type": "passenger", "route": name,
                                      "begin": str(config["traffic"]["begin_seconds"]), "end": str(config["traffic"]["end_seconds"]),
                                      "vehsPerHour": str(rate), "departLane": "free", "departSpeed": "max"})
            scheduled.append((config["traffic"]["begin_seconds"], flow))
    validation = config["validation"]
    if validation["lane_change_probe"]:
        scheduled.append((0, ET.Element("vehicle", {"id": "probe_lane_change", "type": "passenger", "route": "main",
                                                     "depart": "0", "departLane": "0", "departPos": "30", "departSpeed": "max"})))
    if validation["stop_probe"]:
        vehicle = ET.Element("vehicle", {"id": "probe_stop", "type": "passenger", "route": "main",
                                         "depart": "1", "departLane": "2", "departSpeed": "max"})
        ET.SubElement(vehicle, "stop", {"lane": "entry_2", "endPos": "80", "duration": "2"})
        scheduled.append((1, vehicle))
    for _, item in sorted(scheduled, key=lambda value: value[0]):
        demand.append(item)
    write_xml(demand, output/"demand.rou.xml")
    simulation = ET.Element("configuration")
    inputs = ET.SubElement(simulation, "input")
    ET.SubElement(inputs, "net-file", {"value": "network.net.xml"})
    ET.SubElement(inputs, "route-files", {"value": "demand.rou.xml"})
    write_xml(simulation, output/"episode.sumocfg")
    identity = {"family": config["family"], "geometry_id": semantic_geometry_id(output/"network.net.xml"),
                "geometry_parameters": config["geometry"], "routes": routes,
                "files": [file_identity(output/name) for name in ("nodes.nod.xml", "edges.edg.xml", "connections.con.xml",
                                                                    "network.net.xml", "demand.rou.xml", "episode.sumocfg")],
                "netconvert_returncode": result.returncode, "netconvert_argv": command, "seed": seed}
    (output/"scene.json").write_text(json.dumps(identity, indent=2)+"\n", encoding="utf-8")
    return identity
