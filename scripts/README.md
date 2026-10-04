# Scripts

`validate_collection.py --output artifacts/runs/<new-name> --formal` runs the small stage-1 acceptance suite: six control-free short scene profiles plus lane-change/stop, teleport and truncation fixtures. It preserves failed child runs. Collection is also available through `python -m sumodiff.simulation collect`; no training entry point exists yet.
