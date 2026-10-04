"""Read-only, explicit environment queries without exposing environment variables."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys


def query_command(argv: list[str], timeout: float = 15) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}
    record = {"argv": argv, "returncode": result.returncode,
              "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    if result.returncode:
        record["error"] = f"Command returned {result.returncode}"
    return record


def collect_environment() -> dict:
    packages = sorted(
        [{"name": dist.metadata.get("Name", "unknown"), "version": dist.version}
         for dist in importlib.metadata.distributions()], key=lambda item: item["name"].lower()
    )
    torch_code = """
import json
try:
    import torch
    result = {'version': torch.__version__, 'cuda_build': torch.version.cuda,
              'cuda_available': torch.cuda.is_available()}
    result['devices'] = [
        {'name': torch.cuda.get_device_name(i),
         'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())]
except Exception as exc:
    result = {'error': type(exc).__name__ + ': ' + str(exc)}
print(json.dumps(result))
"""
    torch_query = query_command([sys.executable, "-c", torch_code], timeout=30)
    if "error" not in torch_query:
        try:
            torch_query["details"] = json.loads(torch_query["stdout"])
        except json.JSONDecodeError as exc:
            torch_query["error"] = f"Invalid PyTorch query output: {exc}"
    binaries = {}
    for name, args in [("git", ["--version"]), ("sumo", ["--version"]),
                       ("netconvert", ["--version"]),
                       ("nvidia-smi", ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])]:
        executable = shutil.which(name)
        binaries[name] = query_command([executable, *args]) if executable else {"error": "Executable not found on PATH"}
    return {"python": {"version": sys.version, "executable": sys.executable},
            "platform": platform.platform(), "packages": packages,
            "sumo_home": os.environ.get("SUMO_HOME"),
            "torch": torch_query, "binaries": binaries}
