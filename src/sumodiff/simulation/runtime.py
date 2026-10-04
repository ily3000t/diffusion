from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def binary_version(executable: Path) -> dict:
    result = subprocess.run([str(executable), "--version"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", check=True, timeout=15)
    match = re.search(r"Version\s+(\d+\.\d+\.\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not identify executable version: {executable}")
    return {"path": str(executable), "version": match.group(1), "output": result.stdout.strip()}


def load_runtime(sumo_home: str | None = None) -> tuple[dict, object]:
    # Prevent bytecode cache writes into the externally installed SUMO tools.
    # This affects only this interpreter process, not the global environment.
    sys.dont_write_bytecode = True
    location = sumo_home or os.environ.get("SUMO_HOME")
    if not location:
        raise RuntimeError("Set SUMO_HOME or runtime.sumo_home explicitly")
    home = Path(location).resolve(strict=True)
    suffix = ".exe" if os.name == "nt" else ""
    tools = home / "tools"
    for name in ("traci", "sumolib"):
        expected = (tools / name / "__init__.py").resolve(strict=True)
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve() != expected:
            raise RuntimeError(f"An incompatible {name} is already loaded: {loaded.__file__}")
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    clients = {name: importlib.import_module(name) for name in ("traci", "sumolib")}
    for name, module in clients.items():
        if Path(module.__file__).resolve() != (tools / name / "__init__.py").resolve():
            raise RuntimeError(f"Client source does not match requested installation: {name}")
    sumo = binary_version(home / "bin" / ("sumo" + suffix))
    netconvert = binary_version(home / "bin" / ("netconvert" + suffix))
    if sumo["version"] != netconvert["version"]:
        raise RuntimeError("SUMO and netconvert binary versions differ")
    bundle_revision = importlib.import_module("sumolib.version").gitDescribe()
    expected_tag = "v" + sumo["version"].replace(".", "_")
    if not bundle_revision.startswith(expected_tag):
        raise RuntimeError(f"Client bundle revision cannot be matched to binary: {bundle_revision} vs {sumo['version']}")
    identity = {"sumo_home": str(home), "sumo": sumo, "netconvert": netconvert,
                "client_source": "sumo_home_tools", "bundle_revision": bundle_revision,
                "client_protocol": clients["traci"].constants.TRACI_VERSION,
                "clients": {name: {"path": module.__file__, "source_sha256": tree_hash(tools / name)}
                            for name, module in clients.items()}}
    return identity, clients["traci"]
