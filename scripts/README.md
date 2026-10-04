# Scripts

`validate_collection.py --output artifacts/runs/<new-name> --formal` runs the small stage-1 acceptance suite: six control-free short scene profiles plus lane-change/stop, teleport and truncation fixtures. It preserves failed child runs. Collection is also available through `python -m sumodiff.simulation collect`; no training entry point exists yet.

`audit_window_dataset.py --dataset artifacts/processed/<dataset> --sources configs/data/<sources>.json --output artifacts/runs/<new-audit> --formal` independently checks every saved state and local map against raw records, and cross-checks junction indices/conflicts using the matching SUMO SDK. It records its own code/config/environment and failures.
