"""Independently inspect saved raw frames, tripinfo, identities and run status."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit(root: Path) -> dict:
    summary = json.loads((root/"validation_summary.json").read_text(encoding="utf-8"))
    cases, total = [], 0
    for name, entry in summary["results"].items():
        directory = root/name
        episode = json.loads((directory/"episode_manifest.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        for identity in episode["outputs"]+episode["scene"]["files"]:
            actual = hashlib.sha256(Path(identity["location"]).read_bytes()).hexdigest()
            require(actual == identity["sha256"], f"Output hash mismatch: {identity['location']}")
        require(manifest["git"]["commit_sha"] == summary["run_sha"] and not manifest["git"]["dirty"] and manifest["formal"], f"Unclean/inconsistent run code: {name}")
        require(hashlib.sha256((directory/"environment.json").read_bytes()).hexdigest() == manifest["environment_sha256"], f"Environment hash mismatch: {name}")
        require(hashlib.sha256((directory/"resolved_config.yaml").read_bytes()).hexdigest() == manifest["resolved_config_sha256"], f"Configuration hash mismatch: {name}")
        count, rows, all_ids = 0, 0, set()
        with (directory/"frames.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                frame = json.loads(line)
                require(frame["tick"]==count and abs(frame["time_seconds"]-count*0.1)<1e-6, f"Nonuniform time: {name} frame {count}")
                ids = [record["vehicle_id"] for record in frame["vehicles"]]
                require(len(ids)==len(set(ids)), f"Duplicate IDs: {name} frame {count}")
                for record in frame["vehicles"]:
                    state = record["sumo_raw"]
                    require(state["speed_mode"]==31 and state["lane_change_mode"]==1621 and state["vehicle_class"]=="passenger", f"Safety/class mismatch: {name}")
                    require(state["route_id"] in episode["scene"]["routes"], f"Unknown route: {name}")
                    require(state["route_edges"]==episode["scene"]["routes"][state["route_id"]]["edges"], f"Route snapshot mismatch: {name}")
                all_ids.update(ids)
                rows += len(ids)
                count += 1
        quality = entry["quality"]
        require(count==quality["counts"]["frames"] and rows==quality["counts"]["vehicle_records"], f"Raw/summary count mismatch: {name}")
        status = json.loads((directory/"status.json").read_text(encoding="utf-8"))["state"]
        if status=="completed":
            tripinfo = ET.parse(directory/"tripinfo.xml").getroot().findall("tripinfo")
            arrived = {item.get("id") for item in tripinfo if float(item.get("arrival"))>=0}
            require(arrived==set(quality["arrived_ids"])==set(quality["entered_ids"]), f"Tripinfo/lifecycle mismatch: {name}")
        total += rows
        resources = quality.get("resources", {})
        process = json.loads((directory/"process.json").read_text(encoding="utf-8"))
        cases.append({"name":name, "status":status, "vehicles":len(all_ids), "records":rows,
                      "simulated_seconds":quality["duration_seconds"], "geometry_id":episode["geometry_id"],
                      "normal_quality":quality["normal_traffic_quality_pass"],
                      "eligible_for_normal_training":quality["eligible_for_normal_training"],
                      "collection_seconds":resources.get("collection_elapsed_seconds",process["elapsed_seconds"]),
                      "parent_peak_sampled_rss_bytes":resources.get("parent_peak_sampled_rss_bytes"),
                      "sumo_peak_sampled_rss_bytes":resources.get("sumo_peak_sampled_rss_bytes")})
    return {"schema_version":"sumodiff.stage1.audit.v1", "run_sha":summary["run_sha"],
            "all_acceptance_checks_passed":summary["all_passed"], "independent_raw_file_audit_passed":True,
            "total_vehicle_records":total, "cases":cases, "checks":summary["checks"],
            "output_bytes":sum(path.stat().st_size for path in root.rglob("*") if path.is_file())}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite audit: {args.output}")
    result=audit(args.run.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(f"Audit passed: {len(result['cases'])} cases, {result['total_vehicle_records']} raw records")
