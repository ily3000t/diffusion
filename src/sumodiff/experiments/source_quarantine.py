"""Explicit, bounded quarantine of complete abnormal Motion source episodes.

The source recipe stays fixed. A v2 registry separately records the changed
population contract, including every rejected job. No seed replacement occurs.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from sumodiff.experiments.recorder import file_identity, load_config
from sumodiff.simulation.scenes import semantic_geometry_id
from .stage5a_plan import jobs, protocol_id

POLICY = dict(schema_version="sumodiff.source.quality_policy.v1",
              mode="quarantine_complete_ineligible", source_recipe_changed=False,
              retry_rejected_seed=False, replace_seed=False,
              quality_thresholds_changed=False, planned_jobs_only=True)
ANOMALIES = ("unexplained_disappearance", "unexplained_appearance",
             "teleport_started", "emergency_stop")
EVIDENCE = ("manifest.json", "resolved_config.yaml", "status.json",
            "environment.json", "quality.json", "process.json", "events.jsonl",
            "collisions.xml")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_identity(identity, path=None):
    actual = file_identity(path or identity["location"])
    if any(actual[key] != identity[key] for key in ("sha256", "size_bytes")):
        raise ValueError("Source integrity failure: " + actual["location"])
    return Path(actual["location"])


def collection_plan(plan):
    return dict(schema_version="sumodiff.stage5a.collection.plan.v1",
                protocol_id=protocol_id(plan), plan=plan, jobs=jobs(plan),
                split_assigned_before_windowing=True)


def inspect_episode(job, path, formal):
    """Authenticate metadata; fully hash rejected outputs before quarantine.

    Accepted frame hashes are verified again by the existing load_episode during
    preparation. No raw frame arrays need to be loaded during recovery checks.
    """
    path = Path(path).resolve(strict=True)
    directory = path.parent
    ep, run = read(path), read(directory / "manifest.json")
    actual = load_config(directory / "resolved_config.yaml")
    status, quality = read(directory / "status.json"), read(directory / "quality.json")
    process = read(directory / "process.json")
    if (ep.get("schema_version") != "sumodiff.raw.episode.v1"
            or ep.get("state") != "completed" or status.get("state") != "completed"
            or quality.get("complete_episode") is not True
            or process.get("complete_episode") is not True or process.get("returncode") != 0):
        raise ValueError("Only complete successful executions can be quality-quarantined: " + str(path))
    if (run.get("purpose") != "stage1_complete_episode_collection"
            or file_identity(directory / "resolved_config.yaml")["sha256"] != run["resolved_config_sha256"]
            or file_identity(directory / "environment.json")["sha256"] != run["environment_sha256"]):
        raise ValueError("Source run/configuration/environment identity mismatch")
    expected = deepcopy(job["scenario"])
    expected["runtime"]["sumo_home"] = actual["scenario"]["runtime"]["sumo_home"]
    if (actual["scenario"] != expected or ep["family"] != expected["family"]
            or ep["seed"] != job["seed"] or actual["recording"]["seed"] != job["seed"]
            or run["seeds"].get("sumo") != job["seed"]
            or ep.get("run_sha") != run["git"]["commit_sha"]
            or ep.get("run_branch") != run["git"]["branch"]
            or ep.get("dt") != .1):
        raise ValueError("Source does not match its preassigned job/configuration/seed")
    if (ep.get("diagnostic_only") is not False or expected["validation"]["diagnostic_only"]
            or actual["recording"]["formal"] != run["formal"]):
        raise ValueError("Diagnostic or inconsistent source cannot enter a normal-data campaign")
    if formal and (run.get("formal") is not True or run["git"]["dirty"] is not False):
        raise ValueError("Formal recovery requires clean formal source episodes")

    outputs = {Path(item["location"]).name: item for item in ep["outputs"]}
    required = {"frames.jsonl", "events.jsonl", "quality.json", "process.json", "collisions.xml"}
    if len(outputs) != len(ep["outputs"]) or not required <= outputs.keys():
        raise ValueError("Missing or duplicate source output identities")
    for name in required - {"frames.jsonl"}:
        verify_identity(outputs[name], directory / name)
    network = next(item for item in ep["scene"]["files"]
                   if Path(item["location"]).name == "network.net.xml")
    network_path = verify_identity(network, directory / "scene/network.net.xml")
    if semantic_geometry_id(network_path) != ep["geometry_id"]:
        raise ValueError("Semantic geometry identity mismatch")
    content = dict(geometry_id=ep["geometry_id"], frames=outputs["frames.jsonl"]["sha256"],
                   events=outputs["events.jsonl"]["sha256"])
    expected_id = "sha256:" + hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    if ep["data_id"] != expected_id:
        raise ValueError("Raw data identity mismatch")

    observed = Counter()
    for line in (directory / "events.jsonl").open(encoding="utf-8"):
        observed[json.loads(line)["kind"]] += 1
    collision_count = quality["counts"].get("collision_events", 0)
    if observed["collision"] != collision_count:
        raise ValueError("Collision evidence and quality counts disagree")
    for name in ANOMALIES:
        if observed[name] != quality["lifecycle_counts"].get(name, 0):
            raise ValueError("Lifecycle evidence and quality counts disagree: " + name)
    reasons = []
    if collision_count:
        reasons.append("collision")
    reasons += [name for name in ANOMALIES if observed[name]]
    if set(quality["entered_ids"]) != set(quality["arrived_ids"]):
        reasons.append("unarrived_entered_vehicles")
    eligible = not reasons
    if (ep.get("eligible_for_normal_training") is not eligible
            or quality.get("eligible_for_normal_training") is not eligible
            or quality.get("normal_traffic_quality_pass") is not eligible):
        raise ValueError("Source eligibility flags disagree with retained evidence")
    if not eligible:
        for identity in ep["outputs"]:
            verify_identity(identity, directory / Path(identity["location"]).name)
        for identity in ep["scene"]["files"]:
            verify_identity(identity, directory / "scene" / Path(identity["location"]).name)
    row = dict(job_id=job["job_id"], split=job["split"],
               outcome="accepted" if eligible else "rejected",
               episode_manifest=file_identity(path), rejection_reasons=reasons,
               evidence=[file_identity(directory / name) for name in EVIDENCE])
    return row, ep


def completed_attempt(job, root, formal):
    """Return the earliest authenticated complete attempt; never retry a reject."""
    for directory in sorted((Path(root) / "episodes").glob(job["job_id"] + "-attempt-*")):
        status = directory / "status.json"
        if status.exists() and read(status).get("state") == "completed":
            path = directory / "episode_manifest.json"
            inspect_episode(job, path, formal)  # malformed complete sources are errors
            return path
    return None


def population_summary(ledger, schedule):
    families = {job["job_id"]: job["scenario"]["family"] for job in schedule}
    grouped = {}
    for row in ledger:
        key = row["split"] + "/" + families[row["job_id"]]
        group = grouped.setdefault(key, dict(accepted=0, rejected=0))
        group[row["outcome"]] += 1
    return dict(planned_jobs=len(schedule), processed_jobs=len(ledger),
                accepted_jobs=sum(row["outcome"] == "accepted" for row in ledger),
                rejected_jobs=sum(row["outcome"] == "rejected" for row in ledger),
                pending_jobs=len(schedule) - len(ledger), by_split_family=grouped)


def recovery_check(plan, campaign, formal):
    """Read-only source inspection; does not create configs or call SUMO."""
    import yaml
    ledger, files = [], []
    schedule = jobs(plan)
    for job in schedule:
        config = Path(campaign) / "configs" / (job["job_id"] + ".yaml")
        if config.exists() and config.read_text(encoding="utf-8") != yaml.safe_dump(job["scenario"], sort_keys=True):
            raise ValueError("Existing campaign scenario changed")
        path = completed_attempt(job, campaign, formal)
        if path is not None:
            row, _ = inspect_episode(job, path, formal)
            ledger.append(row)
            files += [path, *(item["location"] for item in row["evidence"])]
    return dict(quality_policy=deepcopy(POLICY), **population_summary(ledger, schedule),
                jobs=ledger, sumo_invocations=0, raw_files_modified=False), files


def make_registry(plan_file, collector_manifest, ledger):
    return dict(schema_version="sumodiff.sources.v2", quality_policy=deepcopy(POLICY),
                collection_plan=file_identity(plan_file),
                collector_manifest=file_identity(collector_manifest), job_ledger=ledger,
                episodes=[dict(manifest=row["episode_manifest"]["location"], split=row["split"])
                          for row in ledger if row["outcome"] == "accepted"])


def verify_registry(document, plan, formal):
    """Verify all planned outcomes before accepted sources can be windowed."""
    if (set(document) != {"schema_version", "quality_policy", "collection_plan",
                          "collector_manifest", "job_ledger", "episodes"}
            or document["schema_version"] != "sumodiff.sources.v2"
            or document["quality_policy"] != POLICY
            or plan["schema_version"] != "sumodiff.stage5a.config.v3"):
        raise ValueError("Invalid explicit source-quarantine registry")
    plan_path = verify_identity(document["collection_plan"])
    if read(plan_path) != collection_plan(plan):
        raise ValueError("Quarantine source plan changed")
    collector_path = verify_identity(document["collector_manifest"])
    collector = read(collector_path)
    config_path = collector_path.parent / "resolved_config.yaml"
    if (collector.get("purpose") != "stage5a_bounded_bulk_collection"
            or read(collector_path.parent / "status.json").get("state") != "completed"
            or file_identity(config_path)["sha256"] != collector["resolved_config_sha256"]):
        raise ValueError("Quarantine collection controller is not complete/authenticated")
    config = load_config(config_path)
    if config.get("quality_policy") != POLICY or config.get("campaign") != collection_plan(plan):
        raise ValueError("Collection controller did not use the declared quarantine policy")
    if formal and (not collector["formal"] or collector["git"]["dirty"]):
        raise ValueError("Formal preparation requires a clean formal collector")
    schedule = jobs(plan)
    if len(document["job_ledger"]) != len(schedule):
        raise ValueError("Missing outcomes in frozen source population")
    files = [plan_path, collector_path, config_path, collector_path.parent / "status.json"]
    geometries, accepted_jobs = {}, []
    for row, job in zip(document["job_ledger"], schedule):
        path = verify_identity(row["episode_manifest"])
        actual, episode = inspect_episode(job, path, formal)
        if actual != row:
            raise ValueError("Source ledger order/outcome/reason/evidence mismatch")
        geo = episode["geometry_id"]
        if geo in geometries and geometries[geo] != job["split"]:
            raise ValueError("Geometry leakage in source outcome ledger")
        geometries[geo] = job["split"]
        files += [path, *(item["location"] for item in row["evidence"])]
        if row["outcome"] == "accepted":
            accepted_jobs.append(job)
        else:
            files += [path.parent / Path(item["location"]).name for item in episode["outputs"]]
            files += [path.parent / "scene" / Path(item["location"]).name for item in episode["scene"]["files"]]
    expected = [dict(manifest=row["episode_manifest"]["location"], split=row["split"])
                for row in document["job_ledger"] if row["outcome"] == "accepted"]
    if document["episodes"] != expected:
        raise ValueError("Accepted sources differ from authenticated outcome ledger")
    return accepted_jobs, population_summary(document["job_ledger"], schedule), files
