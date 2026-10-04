"""Small stage-1 acceptance suite. Does not launch batch dataset collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from sumodiff.experiments.recorder import RunRecorder
from sumodiff.simulation.collection import write_json
from sumodiff.simulation.config import resolve_config
from sumodiff.simulation.runtime import load_runtime
from sumodiff.simulation.scenes import build_scene
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    cases = [("straight_a", 11), ("straight_b", 12), ("ramp_a", 21), ("ramp_b", 22),
             ("intersection_a", 31), ("intersection_b", 32)]
    inputs = [root/"configs"/"scenarios"/(name+".yaml") for name, _ in cases]
    inputs += [root/"configs/scenarios/truncation_fixture.yaml", root/"configs/scenarios/teleport_fixture.yaml", root/"configs/scenarios/straight_probe.yaml"]
    runtime, _ = load_runtime()
    effective = {"kind": "stage1_small_collection_acceptance", "cases": dict(cases),
                 "fixtures": {"teleport": 41, "truncation": 51, "lane_change_stop": 11}, "formal": args.formal,
                 "runtime_identity": runtime,
                 "scenario_configs": {path.stem: resolve_config(yaml.safe_load(path.read_text(encoding="utf-8"))) for path in inputs}}
    command = [sys.executable, *sys.orig_argv[1:]]
    with RunRecorder(args.output, root, effective, command, {"suite": 20261004},
                     purpose="stage1_short_acceptance", data_files=inputs, formal=args.formal) as run:
        results, checks = {}, {}
        def collect(name, seed, expected_failure=False):
            output = run.output/name
            command = [sys.executable, "-m", "sumodiff.simulation", "collect", "--config", str(root/"configs/scenarios"/(name+".yaml")),
                       "--output", str(output), "--seed", str(seed)]
            if args.formal:
                command.append("--formal")
            result = subprocess.run(command, check=False)
            if expected_failure:
                if result.returncode != 2:
                    raise RuntimeError("Truncation fixture did not fail as expected")
            elif result.returncode:
                raise RuntimeError(f"Collection failed: {name} ({result.returncode})")
            return json.loads((output/"quality.json").read_text()), json.loads((output/"episode_manifest.json").read_text())
        for name, seed in cases:
            quality, episode = collect(name, seed)
            results[name] = {"quality": quality, "episode_id": episode["episode_id"], "data_id": episode["data_id"],
                             "geometry_id": episode["geometry_id"], "run_sha": episode["run_sha"]}
            checks[name+"_normal_quality"] = quality["normal_traffic_quality_pass"] and quality["complete_episode"]
            checks[name+"_lifecycle"] = bool(quality["entered_ids"]) and quality["entered_ids"] == quality["arrived_ids"]
        checks["six_control_free_training_profiles"] = all(results[name]["quality"]["eligible_for_normal_training"] for name,_ in cases)
        straight, probe_episode = collect("straight_probe", 11)
        results["straight_probe"] = {"quality":straight,"run_sha":probe_episode["run_sha"]}
        checks["probe_excluded_from_normal_training"] = not straight["eligible_for_normal_training"]
        checks["continuous_lane_change"] = straight["probes"]["lane_change_reached_lane_1"] and straight["probes"]["lane_change_lateral_frames"] >= 10
        checks["stop_start_and_end"] = straight["probes"]["stop_low_speed_frames"] >= 10 and straight["lifecycle_counts"].get("stop_started",0)>0 and straight["lifecycle_counts"].get("stop_ended",0)>0
        checks["intersection_turns"] = all(results[name]["quality"]["turning_vehicle_ids"] for name in ("intersection_a", "intersection_b"))
        checks["ramp_routes_observed"] = all(any(vehicle["route_id"]=="ramp" for vehicle in results[name]["quality"]["vehicles"].values()) for name in ("ramp_a", "ramp_b"))
        checks["multiple_geometries"] = len({results[name]["geometry_id"] for name,_ in cases}) == 6
        # Recompile identical geometry with a different seed/output path: split ID must not change.
        anchor = build_scene(run.output/"geometry_identity_check", effective["scenario_configs"]["straight_b"], runtime, 99)
        checks["geometry_id_independent_of_seed_and_path"] = anchor["geometry_id"] == results["straight_b"]["geometry_id"]
        quality, episode = collect("teleport_fixture", 41)
        checks["actual_teleport_start_and_end"] = quality["lifecycle_counts"].get("teleport_started",0)>0 and quality["lifecycle_counts"].get("teleport_ended",0)>0
        checks["teleport_fixture_excluded_from_normal_training"] = not quality["eligible_for_normal_training"]
        results["teleport_fixture"] = {"quality":quality,"run_sha":episode["run_sha"]}
        quality, episode = collect("truncation_fixture", 51, expected_failure=True)
        child = run.output/"truncation_fixture"
        checks["failure_outputs_retained"] = all((child/file).exists() for file in ("frames.jsonl","events.jsonl","quality.json","episode_manifest.json","truncation.json"))
        checks["failure_status_correct"] = json.loads((child/"status.json").read_text())["state"]=="failed" and episode["state"]=="failed" and not quality["complete_episode"]
        results["truncation_fixture"] = {"quality":quality,"run_sha":episode["run_sha"]}
        summary = {"schema_version":"sumodiff.stage1.validation.v1","run_sha":run.manifest["git"]["commit_sha"],
                   "checks":checks,"all_passed":all(checks.values()),"results":results}
        write_json(run.output/"validation_summary.json",summary)
        run.write_metrics({"all_checks_passed":summary["all_passed"],"checks":checks,
                           "normal_profile_count":6,"diagnostic_fixture_count":3})
        if not summary["all_passed"]:
            raise RuntimeError(f"Acceptance checks failed: {[key for key,passed in checks.items() if not passed]}")
    print("Stage 1 acceptance passed:",run.output)


if __name__ == "__main__":
    main()
