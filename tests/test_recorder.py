from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from sumodiff.experiments import recorder
from sumodiff.experiments.cli import main


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "-b", "test/provenance")
    git(root, "config", "user.name", "SUMODiff Test")
    git(root, "config", "user.email", "test@example.invalid")
    (root / ".gitignore").write_text("/artifacts/\n", encoding="utf-8")
    (root / "source.txt").write_text("original source", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "test(provenance): establish fixture history")
    return root


@pytest.fixture(autouse=True)
def fake_environment(monkeypatch):
    # Unit tests avoid hardware assumptions; a separate smoke queries real hardware.
    monkeypatch.setattr(recorder, "collect_environment", lambda: {"test_fixture": True})


def make_run(repository: Path, name="run", **kwargs):
    return recorder.RunRecorder(repository / "artifacts" / name, repository,
                                kwargs.pop("config", {"setting": {"value": 7}}),
                                ["python", "experiment.py", "--label", "a b"],
                                kwargs.pop("seeds", {"global": 10}),
                                purpose="unit_test", **kwargs)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_snapshot_and_original_code_sha(repository):
    config = {"setting": {"value": 7}}
    before = git(repository, "rev-parse", "HEAD")
    with make_run(repository, config=config, formal=True) as run:
        assert read(run.output / "status.json")["state"] == "running"
        run.write_metrics({"not_applicable": None, "verified_count": 3})
        config["setting"]["value"] = 99
    assert set(p.name for p in run.output.iterdir()) == {
        "manifest.json", "resolved_config.yaml", "command.txt", "environment.json", "metrics.json", "status.json"}
    assert read(run.output / "status.json")["state"] == "completed"
    assert read(run.output / "metrics.json")["not_applicable"] is None
    saved = (run.output / "resolved_config.yaml").read_bytes()
    assert yaml.safe_load(saved)["setting"]["value"] == 7
    manifest = read(run.output / "manifest.json")
    assert hashlib.sha256(saved).hexdigest() == manifest["resolved_config_sha256"]
    assert manifest["git"]["commit_sha"] == before
    assert not manifest["git"]["dirty"]
    (repository / "summary.txt").write_text("later summary", encoding="utf-8")
    git(repository, "add", "summary.txt")
    git(repository, "commit", "-m", "docs(summary): preserve original run SHA")
    assert read(run.output / "manifest.json")["git"]["commit_sha"] == before
    assert not git(repository, "status", "--porcelain")


def test_exception_is_preserved_and_rethrown(repository):
    with pytest.raises(RuntimeError, match="actual failure"):
        with make_run(repository) as run:
            run.write_metrics({"partial_count": 1})
            raise RuntimeError("actual failure")
    status = read(run.output / "status.json")
    assert status["state"] == "failed"
    assert status["error"] == {"type": "RuntimeError", "message": "actual failure"}
    assert read(run.output / "metrics.json") == {"partial_count": 1}


def test_overwrite_is_rejected_without_touching_original(repository):
    with make_run(repository) as run:
        pass
    saved = (run.output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        make_run(repository)
    assert (run.output / "manifest.json").read_bytes() == saved


def test_formal_dirty_tree_rejected_before_output(repository):
    (repository / "source.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="clean working tree"):
        make_run(repository, formal=True)
    assert not (repository / "artifacts" / "run").exists()
    with make_run(repository, formal=False) as run:
        pass
    assert run.manifest["git"]["dirty"]


def test_formal_nonignored_output_rejected(repository):
    with pytest.raises(ValueError, match="Git-ignored"):
        recorder.RunRecorder(repository / "tracked-run", repository, {"key": 1},
                             ["python"], {"global": 1}, purpose="unit_test", formal=True)


def test_wrong_repository_root_rejected(repository):
    child = repository / "nested"
    child.mkdir()
    with pytest.raises(ValueError, match="exact root"):
        recorder.git_state(child)


def test_uncommitted_repository_is_rejected(tmp_path):
    git(tmp_path, "init", "-b", "main")
    with pytest.raises(RuntimeError, match="Git query failed"):
        recorder.git_state(tmp_path)


def test_data_and_checkpoint_content_identity(repository):
    inputs = repository / "artifacts" / "inputs"
    inputs.mkdir(parents=True)
    data = inputs / "data.json"
    model = inputs / "model.bin"
    data.write_bytes(b"known input")
    model.write_bytes(b"synthetic checkpoint identity only")
    with make_run(repository, data_files=[data], data_id="fixture-v1", checkpoint=model) as run:
        pass
    assert run.manifest["data"]["id"] == "fixture-v1"
    assert run.manifest["data"]["files"][0]["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert run.manifest["checkpoint"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unhashed directory"):
        recorder.file_identity(inputs)


def test_missing_input_rejected_before_output(repository):
    with pytest.raises(FileNotFoundError):
        make_run(repository, data_files=[repository / "missing-input"])
    assert not (repository / "artifacts" / "run").exists()


@pytest.mark.parametrize("seed", [-1, 2**32, True, 0.5])
def test_invalid_seed_rejected(repository, seed):
    with pytest.raises(ValueError, match="uint32"):
        make_run(repository, seeds={"global": seed})


def test_nonfinite_metric_does_not_replace_valid_metrics(repository):
    with pytest.raises(ValueError):
        with make_run(repository) as run:
            run.write_metrics({"valid": 1})
            run.write_metrics({"invalid": float("nan")})
    assert read(run.output / "metrics.json") == {"valid": 1}
    assert read(run.output / "status.json")["state"] == "failed"


def test_finished_run_is_immutable(repository):
    with make_run(repository) as run:
        pass
    with pytest.raises(RuntimeError, match="finished"):
        run.write_metrics({"late": 1})
    with pytest.raises(RuntimeError, match="finished"):
        run.finish()


def test_initialization_failure_never_claims_completed(repository, monkeypatch):
    original = recorder._write_json
    def fail_manifest(path, value):
        if path.name == "manifest.json":
            raise OSError("injected write failure")
        original(path, value)
    monkeypatch.setattr(recorder, "_write_json", fail_manifest)
    with pytest.raises(OSError, match="injected write failure"):
        make_run(repository)
    status = read(repository / "artifacts" / "run" / "status.json")
    assert status["state"] == "failed"


def test_config_composition_explicitly_rejected(repository):
    path = repository / "artifacts" / "config.yaml"
    path.parent.mkdir()
    path.write_text("include: other.yaml\n", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="fully resolved"):
        recorder.load_config(path)


def test_cli_records_effective_settings_and_no_fabricated_metrics(repository):
    path = repository / "artifacts" / "config.yaml"
    path.parent.mkdir()
    path.write_text("experiment:\n  kind: metadata_smoke\n  formal: false\n", encoding="utf-8")
    output = repository / "artifacts" / "cli-run"
    args = ["record", "--repository", str(repository), "--config", str(path),
            "--output", str(output), "--seed", "12", "--formal"]
    assert main(args) == 0
    config = yaml.safe_load((output / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert config["experiment"]["formal"] is True
    assert config["recording"]["seeds"] == {"global": 12}
    assert read(output / "metrics.json") == {}
    assert main(args) == 2  # Existing run is preserved, no retry/overwrite.


def test_cli_rejects_unimplemented_pipeline(repository):
    path = repository / "artifacts" / "config.yaml"
    path.parent.mkdir()
    path.write_text("experiment:\n  kind: training\n", encoding="utf-8")
    output = repository / "artifacts" / "unsupported"
    assert main(["record", "--repository", str(repository), "--config", str(path),
                 "--output", str(output), "--seed", "1"]) == 2
    assert not output.exists()
