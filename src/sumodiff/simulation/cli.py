from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from sumodiff.experiments.recorder import RunRecorder, file_identity, load_config
from .collection import collect_episode, write_json
from .config import resolve_config
from .runtime import load_runtime
from .scenes import build_scene


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect one complete SUMO episode; stage 1 only")
    sub = parser.add_subparsers(dest="action", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--seed", type=int, required=True)
    collect.add_argument("--repository", type=Path, default=Path.cwd())
    collect.add_argument("--formal", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = resolve_config(load_config(args.config))
        if type(args.seed) is not int or not 0 <= args.seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        runtime, traci = load_runtime(config["runtime"]["sumo_home"])
        config["runtime"]["sumo_home"] = runtime["sumo_home"]
        effective = {"schema_version": "sumodiff.collection.config.v1", "scenario": config,
                     "runtime_identity": runtime, "recording": {"seed": args.seed, "formal": args.formal,
                     "config_source": file_identity(args.config), "output": str(args.output.resolve())}}
        command = ([sys.executable, *sys.orig_argv[1:]] if argv is None else
                   [sys.executable, "-m", "sumodiff.simulation", *argv])
        with RunRecorder(args.output, args.repository, effective, command, {"sumo": args.seed},
                         purpose="stage1_complete_episode_collection", data_files=[args.config], formal=args.formal) as run:
            episode = {"schema_version": "sumodiff.raw.episode.v1", "episode_id": run.output.name,
                       "family": config["family"], "seed": args.seed, "dt": config["simulation"]["dt"],
                       "run_sha": run.manifest["git"]["commit_sha"], "run_branch": run.manifest["git"]["branch"],
                       "runtime_identity": runtime, "state": "running", "raw_coordinates": "SUMO front bumper, navigation degrees",
                       "diagnostic_only": config["validation"]["diagnostic_only"]}
            write_json(run.output/"episode_manifest.json", episode)
            try:
                scene = build_scene(run.output/"scene", config, runtime, args.seed)
                episode["scene"] = scene
                episode["geometry_id"] = scene["geometry_id"]
                quality = collect_episode(run.output, config, runtime, traci, scene, args.seed)
                episode["state"] = "completed"
                episode["eligible_for_normal_training"] = quality["eligible_for_normal_training"]
                inputs = {"geometry_id": scene["geometry_id"],
                          "frames": file_identity(run.output/"frames.jsonl")["sha256"],
                          "events": file_identity(run.output/"events.jsonl")["sha256"]}
                episode["data_id"] = "sha256:" + hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
                run.write_metrics({"complete_episode": True, "simulated_seconds": quality["duration_seconds"],
                                   "vehicle_records": quality["counts"]["vehicle_records"],
                                   "entered_vehicles": len(quality["entered_ids"]), "arrived_vehicles": len(quality["arrived_ids"]),
                                   "sumo_collision_events": quality["counts"].get("collision_events", 0),
                                   "normal_traffic_quality_pass": quality["normal_traffic_quality_pass"],
                                   "resources": quality["resources"]})
            except BaseException as exc:
                episode["state"] = "failed"
                episode["eligible_for_normal_training"] = False
                episode["error"] = {"type": type(exc).__name__, "message": str(exc)}
                partial = run.output/"collection_partial.json"
                quality = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {}
                quality.update({"state": "failed", "error": episode["error"], "eligible_for_normal_training": False,
                                "normal_traffic_quality_pass": False})
                write_json(run.output/"quality.json", quality)
                raise
            finally:
                episode["outputs"] = [file_identity(run.output/name) for name in
                    ("frames.jsonl", "events.jsonl", "quality.json", "process.json", "collection_partial.json", "lanechanges.xml", "collisions.xml", "tripinfo.xml")
                    if (run.output/name).exists()]
                write_json(run.output/"episode_manifest.json", episode)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Collected {episode['family']}: {quality['counts']['vehicle_records']} records, {quality['duration_seconds']:.1f}s, output={run.output}")
    return 0
