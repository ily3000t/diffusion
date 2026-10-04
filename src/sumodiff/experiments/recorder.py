"""Record the code, resolved inputs and real execution status of one run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import yaml

from .environment import collect_environment


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path, identifier: str | None = None) -> dict:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Input must be a file, not an unhashed directory: {resolved}")
    digest = sha256_file(resolved)
    return {"id": identifier or f"sha256:{digest}", "location": str(resolved),
            "sha256": digest, "size_bytes": resolved.stat().st_size}


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not config:
        raise ValueError("Resolved configuration must be a non-empty mapping")
    # No configuration composition is implemented; reject references instead of
    # silently treating a filename as an effective configuration.
    if any(key in config for key in ("defaults", "extends", "include")):
        raise NotImplementedError("Configuration composition is not implemented; supply one fully resolved YAML document")
    json.dumps(config, allow_nan=False)
    return config


def git_state(repository: str | Path) -> dict:
    root = Path(repository).resolve(strict=True)

    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", check=False)
        if result.returncode:
            raise RuntimeError(f"Git query failed: {result.stderr.strip()}")
        return result.stdout.rstrip("\r\n")

    actual = Path(git("rev-parse", "--show-toplevel")).resolve()
    if os.path.normcase(str(actual)) != os.path.normcase(str(root)):
        raise ValueError(f"Repository must be its exact root, not a child directory: {root}")
    commit = git("rev-parse", "--verify", "HEAD")
    status = git("-c", "core.quotepath=false", "status", "--porcelain=v1", "--untracked-files=all")
    return {"root": str(root), "commit_sha": commit,
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status), "status_porcelain": status.splitlines()}


def render_command(argv: Sequence[str]) -> str:
    if os.name == "nt":
        # PowerShell literal quoting, including executable paths with spaces.
        return "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in argv)
    return shlex.join(argv)


def _atomic_text(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: dict) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class RunRecorder:
    """A run context that preserves failures and never overwrites a run directory.

    Later training/sampling entry points must supply their fully resolved config,
    complete input list and appropriate checkpoint; this class records them but
    does not claim to execute a scientific pipeline.
    """

    def __init__(self, output: str | Path, repository: str | Path, config: Mapping,
                 command: Sequence[str], seeds: Mapping[str, int], *,
                 purpose: str, data_files: Sequence[str | Path] = (),
                 data_id: str | None = None, checkpoint: str | Path | None = None,
                 formal: bool = False):
        if not isinstance(config, Mapping) or not config:
            raise ValueError("A non-empty, fully resolved configuration is required")
        config_copy = deepcopy(dict(config))
        json.dumps(config_copy, allow_nan=False)
        if not purpose.strip() or not command or any(not isinstance(arg, str) for arg in command):
            raise ValueError("Purpose and a non-empty string argv are required")
        if not seeds or any(not isinstance(name, str) or not name or
                            type(seed) is not int or not 0 <= seed < 2**32
                            for name, seed in seeds.items()):
            raise ValueError("Seeds must be a non-empty mapping of names to uint32 integers")
        if type(formal) is not bool:
            raise TypeError("formal must be a boolean")
        self.output = Path(output).resolve()
        if self.output.exists():
            raise FileExistsError(f"Refusing to overwrite run: {self.output}")
        code = git_state(repository)
        if formal and code["dirty"]:
            raise ValueError("Formal runs require committed code and a clean working tree")
        if formal and self.output.is_relative_to(Path(code["root"])):
            ignore_query = subprocess.run(["git", "-C", code["root"], "check-ignore", "-q", str(self.output)], check=False)
            if ignore_query.returncode:
                raise ValueError("Formal run outputs inside the repository must be Git-ignored")
        data = [file_identity(path) for path in data_files]
        checkpoint_info = file_identity(checkpoint) if checkpoint is not None else None
        config_yaml = yaml.safe_dump(config_copy, sort_keys=True, allow_unicode=True)
        config_hash = hashlib.sha256(config_yaml.encode("utf-8")).hexdigest()
        environment = collect_environment()
        self.started_at = utc_now()
        self._state = "running"
        self.manifest = {
            "schema_version": "sumodiff.run.v1", "run_id": self.output.name,
            "purpose": purpose, "formal": formal, "started_at": self.started_at,
            "git": code, "seeds": dict(seeds),
            "data": {"id": data_id, "files": data}, "checkpoint": checkpoint_info,
            "resolved_config_sha256": config_hash,
            "command": {"argv": list(command), "rendered": render_command(command),
                        "working_directory": str(Path.cwd().resolve())},
            "files": {"config": "resolved_config.yaml", "command": "command.txt",
                      "environment": "environment.json", "metrics": "metrics.json", "status": "status.json"},
        }
        # mkdir(exist_ok=False) also rejects races with another run creator.
        self.output.mkdir(parents=True, exist_ok=False)
        try:
            _atomic_text(self.output / "resolved_config.yaml", config_yaml)
            _atomic_text(self.output / "command.txt", render_command(command) + "\n")
            _write_json(self.output / "environment.json", environment)
            self.manifest["environment_sha256"] = sha256_file(self.output / "environment.json")
            _write_json(self.output / "manifest.json", self.manifest)
            _write_json(self.output / "metrics.json", {})
            self._write_status("running")
        except Exception as exc:
            self._state = "failed"
            self._write_status("failed", exc)
            raise

    def _write_status(self, state: str, error: BaseException | None = None) -> None:
        record = {"state": state, "started_at": self.started_at, "updated_at": utc_now()}
        if state != "running":
            record["ended_at"] = record["updated_at"]
        if error is not None:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(self.output / "status.json", record)

    def write_metrics(self, metrics: Mapping) -> None:
        if self._state != "running":
            raise RuntimeError("Cannot mutate a finished run")
        if not isinstance(metrics, Mapping):
            raise TypeError("Metrics must be a mapping; use None for N/A")
        _write_json(self.output / "metrics.json", dict(metrics))

    def finish(self) -> None:
        if self._state != "running":
            raise RuntimeError("Run has already finished")
        self._write_status("completed")
        self._state = "completed"

    def __enter__(self) -> RunRecorder:
        if self._state != "running":
            raise RuntimeError("Run has already finished")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            self._write_status("failed", exc_value)
            self._state = "failed"
        elif self._state == "running":
            self.finish()
        return False
