"""Small synthetic metadata tests; no SUMO or scientific result fabrication."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import pytest
import yaml

from sumodiff.experiments import source_quarantine as quarantine
from sumodiff.experiments import stage5a
from sumodiff.experiments import recorder
from sumodiff.experiments.recorder import file_identity, load_config
from sumodiff.experiments.stage5a_plan import resolve_plan, jobs
from sumodiff.experiments.stage5a_data import inspect_sources
from sumodiff.experiments.motion_data import verify_sources
from sumodiff.simulation.scenes import semantic_geometry_id


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def plan():
    return resolve_plan(load_config(Path(__file__).parents[1] / "configs/experiments/motion_data.yaml"))


def episode(directory, job, rejected=False):
    directory.mkdir(parents=True)
    (directory / "scene").mkdir()
    network = directory / "scene/network.net.xml"
    network.write_text('<net><edge id="' + job["job_id"] + '"/></net>', encoding="utf-8")
    geo = semantic_geometry_id(network)
    (directory / "frames.jsonl").write_text('{"synthetic_metadata_test": true}\n', encoding="utf-8")
    events = [dict(kind="collision"), dict(kind="teleport_started")] if rejected else []
    (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    (directory / "collisions.xml").write_text("<collisions/>", encoding="utf-8")
    quality = dict(complete_episode=True, normal_traffic_quality_pass=not rejected,
                   eligible_for_normal_training=not rejected, entered_ids=[], arrived_ids=[],
                   counts=dict(collision_events=int(rejected)),
                   lifecycle_counts=dict(teleport_started=int(rejected)))
    write(directory / "quality.json", quality)
    write(directory / "process.json", dict(complete_episode=True, returncode=0))
    write(directory / "status.json", dict(state="completed"))
    write(directory / "environment.json", dict(synthetic_metadata_test=True))
    scenario = deepcopy(job["scenario"])
    scenario["runtime"]["sumo_home"] = "synthetic-runtime"
    actual = dict(scenario=scenario, recording=dict(seed=job["seed"], formal=True))
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(actual), encoding="utf-8")
    run = dict(purpose="stage1_complete_episode_collection", formal=True,
               git=dict(dirty=False, commit_sha="synthetic-test-sha", branch="test"),
               seeds=dict(sumo=job["seed"]),
               resolved_config_sha256=file_identity(directory / "resolved_config.yaml")["sha256"],
               environment_sha256=file_identity(directory / "environment.json")["sha256"])
    write(directory / "manifest.json", run)
    content = dict(geometry_id=geo, frames=file_identity(directory / "frames.jsonl")["sha256"],
                   events=file_identity(directory / "events.jsonl")["sha256"])
    ep = dict(schema_version="sumodiff.raw.episode.v1", state="completed", family=job["scenario"]["family"],
              seed=job["seed"], run_sha=run["git"]["commit_sha"], run_branch="test", dt=.1,
              eligible_for_normal_training=not rejected, diagnostic_only=False, geometry_id=geo,
              data_id="sha256:" + hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest(),
              scene=dict(files=[file_identity(network)]),
              outputs=[file_identity(directory / n) for n in
                       ("frames.jsonl", "events.jsonl", "quality.json", "process.json", "collisions.xml")])
    write(directory / "episode_manifest.json", ep)
    return directory / "episode_manifest.json"


def test_rejection_is_evidence_based_and_rejected_frames_are_authenticated(tmp_path):
    job = jobs(plan())[0]
    path = episode(tmp_path / "episode", job, True)
    row, _ = quarantine.inspect_episode(job, path, True)
    assert row["outcome"] == "rejected"
    assert row["rejection_reasons"] == ["collision", "teleport_started"]
    (path.parent / "frames.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        quarantine.inspect_episode(job, path, True)


@pytest.mark.parametrize("case", ["incomplete", "process_failure", "wrong_seed",
                                 "wrong_config", "dirty", "false_eligibility", "diagnostic"])
def test_invalid_execution_or_identity_is_never_silently_quality_rejected(tmp_path, case):
    job = jobs(plan())[0]
    path = episode(tmp_path / "episode", job, True)
    ep = quarantine.read(path)
    if case == "incomplete":
        write(path.parent / "status.json", dict(state="failed"))
    elif case == "process_failure":
        write(path.parent / "process.json", dict(complete_episode=True, returncode=1))
    elif case == "wrong_seed":
        ep["seed"] += 1
    elif case == "wrong_config":
        job = deepcopy(job)
        job["scenario"]["driver"]["tau"] += .1
    elif case == "dirty":
        run = quarantine.read(path.parent / "manifest.json")
        run["git"]["dirty"] = True
        write(path.parent / "manifest.json", run)
    elif case == "false_eligibility":
        ep["eligible_for_normal_training"] = True
    elif case == "diagnostic":
        ep["diagnostic_only"] = True
    write(path, ep)
    with pytest.raises(ValueError):
        quarantine.inspect_episode(job, path, True)


def test_reuse_complete_rejected_attempt_never_invokes_sumo(tmp_path, monkeypatch):
    job = jobs(plan())[0]
    source = episode(tmp_path / "episodes" / (job["job_id"] + "-attempt-001"), job, True)
    monkeypatch.setattr(stage5a, "collect_one", lambda args: pytest.fail("Rejected seed was retried"))
    actual, reused = stage5a.collect_job(job, tmp_path, tmp_path, True, True, "quarantine")
    assert reused and actual == source.resolve()


def test_default_strict_mode_still_stops_at_new_ineligible_source(tmp_path, monkeypatch):
    job = jobs(plan())[0]
    def collect(args):
        episode(Path(args[args.index("--output") + 1]), job, True)
        return 0
    monkeypatch.setattr(stage5a, "collect_one", collect)
    with pytest.raises(ValueError, match="not normal-training eligible"):
        stage5a.collect_job(job, tmp_path, tmp_path, False)


def campaign_fixture(tmp_path, monkeypatch):
    config = Path(__file__).parents[1] / "configs/experiments/motion_data.yaml"
    c = plan()
    all_jobs = jobs(c)
    scheduled = [all_jobs[0], all_jobs[48], all_jobs[144]]
    monkeypatch.setattr(stage5a, "jobs", lambda *args: scheduled)
    monkeypatch.setattr(quarantine, "jobs", lambda *args: scheduled)
    import sumodiff.experiments.motion_data as motion_data
    monkeypatch.setattr(motion_data, "jobs", lambda *args: scheduled)
    monkeypatch.setattr(recorder, "git_state", lambda root: dict(root=str(tmp_path),
        commit_sha="synthetic-controller-sha", branch="test", dirty=False, status_porcelain=[]))
    monkeypatch.setattr(recorder, "collect_environment", lambda: dict(synthetic_metadata_test=True))
    pilot = tmp_path / "pilot.json"
    write(pilot, {})
    monkeypatch.setattr(stage5a, "_verify_pilot", lambda *args: file_identity(pilot))
    root = tmp_path / "campaign"
    root.mkdir()
    write(root / "collection_plan.json", quarantine.collection_plan(c))
    for job, rejected in zip(scheduled[:2], [False, True]):
        episode(root / "episodes" / (job["job_id"] + "-attempt-001"), job, rejected)
    return config, c, scheduled, root, pilot


def exercise_campaign(tmp_path, monkeypatch):
    config, c, scheduled, root, pilot = campaign_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(stage5a, "collect_one", lambda args: pytest.fail("CheckOnly ran SUMO"))
    check = stage5a.collect_campaign(config, c, root, tmp_path / "check", tmp_path, ["test"],
        False, pilot, True, True, "quarantine", True)
    assert (check["accepted_jobs"], check["rejected_jobs"], check["pending_jobs"]) == (1, 1, 1)
    assert check["sumo_invocations"] == 0
    assert not (root / "sources.json").exists()
    calls = []
    def collect(args):
        calls.append(args)
        episode(Path(args[args.index("--output") + 1]), scheduled[2], False)
        return 0
    monkeypatch.setattr(stage5a, "collect_one", collect)
    result = stage5a.collect_campaign(config, c, root, tmp_path / "collect", tmp_path, ["test"],
        False, pilot, True, True, "quarantine")
    assert len(calls) == 1 and calls[0][calls[0].index("--seed") + 1] == str(scheduled[2]["seed"])
    assert result["processed_jobs"] == 3 and result["accepted_jobs"] == 2 and result["rejected_jobs"] == 1
    registry = root / "sources.json"
    sources, geos, files, population = inspect_sources(registry, plan=c, formal=False, return_population=True)
    assert len(sources) == 2 and len(geos) == 2 and population["summary"]["rejected_jobs"] == 1
    assert len(verify_sources(sources, c, False, False, population["accepted_job_ids"])) == 6
    assert any("quality.json" in f and "train-ramp_merge" in f for f in files)
    with pytest.raises(ValueError, match="population"):
        verify_sources(sources, c, False, False)  # old full-population guard still applies to v1
    with pytest.raises(FileExistsError, match="already exists"):
        stage5a.collect_campaign(config, c, root, tmp_path / "overwrite", tmp_path, ["test"],
            False, pilot, True, True, "quarantine")
    return config, c, scheduled, root, pilot


def test_recovery_check_collect_and_prepare_share_complete_outcome_ledger(tmp_path, monkeypatch):
    exercise_campaign(tmp_path, monkeypatch)


@pytest.mark.parametrize("case", ["missing_job", "duplicate_job", "changed_reasons",
                                 "readd_reject", "changed_policy", "controller_failed"])
def test_prepare_rejects_missing_tampered_or_reintroduced_rejected_sources(tmp_path, monkeypatch, case):
    _, c, _, root, _ = exercise_campaign(tmp_path, monkeypatch)
    document = quarantine.read(root / "sources.json")
    if case == "missing_job":
        document["job_ledger"].pop()
    elif case == "duplicate_job":
        document["job_ledger"][1] = document["job_ledger"][0]
    elif case == "changed_reasons":
        document["job_ledger"][1]["rejection_reasons"] = []
    elif case == "readd_reject":
        row = document["job_ledger"][1]
        document["episodes"].append(dict(manifest=row["episode_manifest"]["location"], split=row["split"]))
    elif case == "changed_policy":
        document["quality_policy"]["replace_seed"] = True
    else:
        collector = Path(document["collector_manifest"]["location"])
        write(collector.parent / "status.json", dict(state="failed"))
    write(root / "sources.json", document)
    with pytest.raises(ValueError):
        inspect_sources(root / "sources.json", plan=c, return_population=True)


def test_execution_error_stops_campaign_without_final_registry(tmp_path, monkeypatch):
    config, c, _, root, pilot = campaign_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(stage5a, "collect_one", lambda args: 2)
    with pytest.raises(RuntimeError, match="Collection failed"):
        stage5a.collect_campaign(config, c, root, tmp_path / "failed", tmp_path, ["test"],
            False, pilot, True, True, "quarantine")
    assert not (root / "sources.json").exists()
    assert quarantine.read(tmp_path / "failed/status.json")["state"] == "failed"
    ledger = quarantine.read(tmp_path / "failed/job_ledger.json")
    assert [row["outcome"] for row in ledger["jobs"]] == ["accepted", "rejected"]


def test_quarantine_and_check_are_explicit_and_legacy_is_unchanged(tmp_path):
    c = resolve_plan({})
    with pytest.raises(NotImplementedError, match="Motion"):
        stage5a.collect_campaign(None, c, tmp_path, tmp_path, tmp_path, ["test"], False,
                                allow_bulk=True, quality_policy="quarantine")
    with pytest.raises(ValueError, match="Recovery check"):
        stage5a.collect_campaign(None, c, tmp_path, tmp_path, tmp_path, ["test"], False,
                                allow_bulk=True, check_only=True)
