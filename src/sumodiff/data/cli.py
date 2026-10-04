"""Reproducible preprocessing entry point; no model training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import psutil
from sumodiff.experiments.recorder import RunRecorder, file_identity, load_config
from sumodiff.geometry.maps import RoadMap
from .config import resolve_config
from .episodes import load_registry
from .windows import candidate_ticks, build_window, audit_calibration_episodes
from .dataset import save_window, write_json


def process(output, repository, config_path, registry_path, formal=False, command=None):
    config = resolve_config(load_config(config_path))
    episodes, split_audit = load_registry(registry_path)
    source_paths = [config_path, registry_path, *[i['location'] for i in split_audit['source_manifests']],
                    *[p for episode in episodes for p in episode.input_files]]
    source_paths = sorted({str(Path(p).resolve()) for p in source_paths})
    command = command or [sys.executable, '-m', 'sumodiff.data', 'process', '--config', str(config_path), '--sources', str(registry_path), '--output', str(output)]
    start = time.perf_counter()
    with RunRecorder(output, repository, dict(schema_version='sumodiff.preprocessing.config.v1', window=config,
        split_registry=file_identity(registry_path), recording=dict(formal=formal, output=str(Path(output).resolve()))),
        command, {'preprocessing': 0}, purpose='stage2_fixed_window_preprocessing', data_files=source_paths,
        data_id=split_audit['registry']['id'], formal=formal) as run:
        write_json(run.output / 'split_audit.json', split_audit)
        calibration = audit_calibration_episodes(episodes, config)
        write_json(run.output / 'coverage_calibration.json', calibration)
        index, skipped = [], []
        by_split = {s: Counter(episodes=0, windows=0, core_windows=0, incomplete_future_windows=0,
            incomplete_history_windows=0, selected_agents=0, excluded_current_vehicles=0, valid_future_points=0,
            future_center_out_of_map=0, future_center_offroad=0) for s in ('train', 'validation', 'test')}
        missing = {'history': Counter(), 'future': Counter()}
        peaks = [psutil.Process().memory_info().rss]
        for episode in episodes:
            road = RoadMap.read(episode.network_path)
            by_split[episode.split]['episodes'] += 1
            for tick in candidate_ticks(episode, config):
                window, reason = build_window(episode, tick, road, config)
                if reason:
                    skipped.append(dict(episode_key=episode.key, reference_tick=tick, reason=reason))
                    continue
                entry = save_window(run.output / 'windows' / window.input_metadata['window_id'], window)
                index.append(entry)
                labels, inputs = window.label_metadata, window.input_metadata
                counts = by_split[episode.split]
                counts.update(windows=1, core_windows=int(labels['core_training_eligible']),
                    incomplete_future_windows=int(not labels['complete_future']),
                    incomplete_history_windows=int(not labels['complete_history']),
                    selected_agents=int(window.conditioning['agent_mask'].sum()),
                    excluded_current_vehicles=inputs['excluded_vehicle_count'],
                    **{key: labels[key] for key in ('valid_future_points', 'future_center_out_of_map', 'future_center_offroad')})
                missing['history'].update(labels['history_missing_reasons'])
                missing['future'].update(labels['future_missing_reasons'])
                peaks.append(psutil.Process().memory_info().rss)
        if not index:
            raise ValueError('No windows constructed; source/skip diagnostics retained')
        write_json(run.output / 'windows.json', index)
        report = dict(schema_version='sumodiff.window.quality.v1', by_split={k: dict(v) for k, v in by_split.items()},
            source_rejection_counts=dict(Counter(reason for e in split_audit['rejected_episodes'] for reason in e['reasons'])),
            rejected_episode_count=len(split_audit['rejected_episodes']), skipped_reference_ticks=skipped,
            missing_point_reasons={k: dict(v) for k, v in missing.items()},
            family_coverage=split_audit['family_coverage'], coverage_calibration=calibration,
            incomplete_windows_retained=True, normalization='physical_units_no_fitted_scaling',
            elapsed_seconds=time.perf_counter() - start, sampled_python_peak_rss_bytes=max(peaks))
        write_json(run.output / 'quality.json', report)
        identities = {name: file_identity(run.output / name) for name in ('windows.json', 'split_audit.json', 'coverage_calibration.json', 'quality.json')}
        dataset_id = 'sha256:' + hashlib.sha256(json.dumps(identities['windows.json']['sha256']).encode()).hexdigest()
        write_json(run.output / 'dataset_manifest.json', dict(schema_version='sumodiff.dataset.v1', data_id=dataset_id,
            run_sha=run.manifest['git']['commit_sha'], run_branch=run.manifest['git']['branch'],
            window_index=identities['windows.json'], outputs=identities, source_registry=split_audit['registry'],
            window_config=config, label_fields_excluded_from_inference=['future', 'future_mask', 'label_metadata']))
        run.write_metrics(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build episode/geometry-isolated fixed windows; stage 2')
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('process')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repository', type=Path, default=Path.cwd())
    p.add_argument('--formal', action='store_true')
    args = parser.parse_args(argv)
    command = [sys.executable, *sys.orig_argv[1:]] if argv is None else [sys.executable, '-m', 'sumodiff.data', *argv]
    try:
        report = process(args.output, args.repository, args.config, args.sources, args.formal, command)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(report['by_split'], ensure_ascii=False))
    return 0
