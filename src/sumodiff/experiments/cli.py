"""Metadata-only CLI: record real provenance without executing a model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .recorder import RunRecorder, file_identity, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SUMODiff provenance tools (stage 0 only)")
    subcommands = parser.add_subparsers(dest="action", required=True)
    record = subcommands.add_parser("record", help="Record a metadata smoke run; does not train or sample")
    record.add_argument("--repository", type=Path, default=Path.cwd())
    record.add_argument("--config", type=Path, required=True, help="A complete, resolved YAML document")
    record.add_argument("--output", type=Path, required=True, help="New run directory; never overwritten")
    record.add_argument("--seed", type=int, required=True)
    record.add_argument("--formal", action="store_true", default=None, help="Require committed, clean code")
    record.add_argument("--data-file", type=Path, action="append", default=[])
    record.add_argument("--data-id")
    record.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        experiment = config.get("experiment")
        if not isinstance(experiment, dict) or experiment.get("kind") != "metadata_smoke":
            raise NotImplementedError("Stage 0 CLI only supports experiment.kind=metadata_smoke; no scientific pipeline is implemented")
        if "recording" in config:
            raise ValueError("The recording key is reserved for effective CLI settings")
        formal = experiment.get("formal", False) if args.formal is None else args.formal
        experiment["formal"] = formal
        config["recording"] = {
            "repository": str(args.repository.resolve()), "output": str(args.output.resolve()),
            "seeds": {"global": args.seed}, "formal": formal,
            "config_source": file_identity(args.config),
            "data_id": args.data_id, "data_files": [str(path.resolve()) for path in args.data_file],
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        }
        command = ([sys.executable, *sys.orig_argv[1:]] if argv is None else
                   [sys.executable, "-m", "sumodiff.experiments", *argv])
        with RunRecorder(args.output, args.repository, config, command,
                         {"global": args.seed}, purpose="stage0_metadata_smoke",
                         data_files=args.data_file, data_id=args.data_id,
                         checkpoint=args.checkpoint, formal=formal):
            pass  # Metadata collection itself is this command's actual work.
    except (ValueError, TypeError, RuntimeError, OSError, NotImplementedError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Recorded metadata: {args.output.resolve()}")
    return 0
