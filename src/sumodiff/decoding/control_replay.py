"""Recorded label/control replay. No checkpoint, denoiser, or training."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from sumodiff.data.dataset import WindowDataset, write_json
from sumodiff.evaluation.metrics import evaluate_trajectory
from sumodiff.evaluation.motion import MotionConfig, evaluate_motion, summary
from sumodiff.evaluation.roads import RoadConfig
from sumodiff.evaluation.collisions import CollisionConfig
from sumodiff.experiments.recorder import RunRecorder, load_config, file_identity
from .controls import ControlConfig, initial_control_state, inverse_label_controls, rollout_controls


def validate_protocol(config):
    required = {'schema_version', 'dataset', 'task_plan', 'seed', 'methods', 'gate', 'evaluation'}
    if set(config) != required or config['schema_version'] != 'sumodiff.control_replay.protocol.v1':
        raise ValueError('Unsupported or incomplete control replay protocol')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('Seed must be uint32')
    if not isinstance(config['methods'], dict) or not config['methods']:
        raise ValueError('Explicit replay methods are required')
    methods = {name: ControlConfig(**value) for name, value in config['methods'].items()}
    if any(not isinstance(name, str) or not name.replace('_', '').isalnum() for name in methods):
        raise ValueError('Method names must be safe identifiers')
    gate = config['gate']
    if set(gate) != {'max_position_error_m', 'max_heading_error_rad', 'max_velocity_error_mps'}:
        raise ValueError('Explicit compatibility tolerances are required')
    if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in gate.values()):
        raise ValueError('Compatibility tolerances must be finite and positive')
    evaluation = config['evaluation']
    if set(evaluation) != {'motion', 'road', 'collision'}:
        raise ValueError('Explicit independent evaluation settings are required')
    kwargs = dict(motion_config=MotionConfig(**evaluation['motion']),
                  road_config=RoadConfig(**evaluation['road']),
                  collision_config=CollisionConfig(**evaluation['collision']))
    return methods, kwargs


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get('schema_version') != 'sumodiff.control_replay.tasks.v1' or not isinstance(plan.get('tasks'), list) or not plan['tasks']:
        raise ValueError('A nonempty explicit task plan is required')
    tasks = plan['tasks']
    seen = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError('Task entries must be mappings')
        if task.get('split') not in ('train', 'validation'):
            raise ValueError('Test tasks are prohibited during representation design')
        if not all(isinstance(task.get(k), str) and task[k] for k in ('window_id', 'split', 'group')):
            raise ValueError('Each task needs a split, window ID and diagnostic group')
        key = (task['split'], task['window_id'])
        if key in seen:
            raise ValueError('Duplicate task')
        seen.add(key)
    return tasks


def comfort_by_agent(c, states, future_mask, motion_config=None):
    """Reuse the independent motion evaluator; compare identical agent slots."""
    flags = {}
    for slot in np.flatnonzero(c['agent_mask']):
        mask = np.zeros_like(c['agent_mask'], dtype=bool)
        mask[slot] = True
        result = evaluate_motion(c['history'], c['history_mask'], states, future_mask,
                                 c['initial_positions'], c['history'][:, -1, 4:6], mask, config=motion_config)
        complete = result['comfort_complete_assessed_points'] == states.shape[1]
        flags[str(int(slot))] = bool(result['comfort_pass_agents'] == 1) if complete else None
    return flags


def physical_jerk(c, states):
    """Third position difference with the last three observed positions."""
    if not c['history_mask'][c['agent_mask'], -3:].all():
        raise ValueError('Three observed points required for boundary jerk')
    past = c['history'][:, -3:, :2].astype(np.float64).copy()
    past[:, -1] = c['initial_positions']
    positions = np.concatenate((past, states[..., :2]), axis=1)
    return np.diff(positions, n=3, axis=1) / .1**3


def compatibility_gate(original, replay, original_comfort, replay_comfort, errors, limits):
    reasons = []
    for name, limit in limits.items():
        if errors[name] > limit:
            reasons.append(name)
    if original['quality_pass'] is True and replay['quality_pass'] is not True:
        reasons.append('lost_scene_quality')
    for name, count in original['motion']['violation_counts'].items():
        if replay['motion']['violation_counts'][name] > count:
            reasons.append('increased_motion_' + name)
    for name in ('road', 'route', 'map_coverage'):
        if replay['roads'][name]['violation_body_frames'] > original['roads'][name]['violation_body_frames']:
            reasons.append('increased_' + name)
    old_pairs = {tuple(p['slots']): p['collision'] for p in original['collisions']['pairs']}
    for pair in replay['collisions']['pairs']:
        if old_pairs[tuple(pair['slots'])] is False and pair['collision'] is not False:
            reasons.append('new_collision_or_unknown')
            break
    newly_uncomfortable = [s for s, passed in original_comfort.items()
                          if passed is True and replay_comfort.get(s) is not True]
    if newly_uncomfortable:
        reasons.append('new_strict_comfort_failure')
    return dict(passed=not reasons, reasons=reasons, newly_uncomfortable_slots=newly_uncomfortable)


def task_summary(records, methods):
    groups = defaultdict(list)
    for record in records:
        groups[(record['split'], record['group'])].append(record)
    result = {}
    for (split, group), tasks in sorted(groups.items()):
        entry = {}
        for name in ('label', *methods):
            good = [r['methods'][name] for r in tasks if r['status'] == 'completed' and r['methods'][name]['status'] == 'completed']
            entry[name] = dict(planned=len(tasks), completed=len(good), failed=len(tasks)-len(good),
                               scene_quality_pass=sum(r['metrics']['quality_pass'] is True for r in good),
                               motion_pass=sum(r['metrics']['motion']['hard_quality_pass'] is True for r in good),
                               comfort_pass_agents=sum(r['metrics']['motion']['comfort_pass_agents'] for r in good),
                               comfort_assessed_agents=sum(r['metrics']['motion']['comfort_assessed_agents'] for r in good),
                               gate_pass=None if name == 'label' else sum(r.get('gate', {}).get('passed') is True for r in good),
                               maximum_position_error_m=max((r['errors']['max_position_error_m'] for r in good if 'errors' in r), default=None),
                               maximum_velocity_error_mps=max((r['errors']['max_velocity_error_mps'] for r in good if 'errors' in r), default=None),
                               maximum_abs_lateral_interval_velocity_mps=max((r['lateral_velocity_mps']['max'] for r in good if 'lateral_velocity_mps' in r), default=None))
        result[split + '/' + group] = entry
    return result


def run(config_path, output, *, device='cpu', formal=False):
    config_path = Path(config_path).resolve(strict=True)
    config = load_config(config_path)
    methods, kwargs = validate_protocol(config)
    plan_path = Path(config['task_plan']).resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    tasks = validate_plan(plan)
    datasets = {split: WindowDataset(config['dataset'], split) for split in sorted({r['split'] for r in tasks})}
    indexes = {split: {e['window_id']: i for i, e in enumerate(ds.entries)} for split, ds in datasets.items()}
    sources = [config_path, plan_path, Path(config['dataset'])/'dataset_manifest.json', Path(config['dataset'])/'windows.json']
    for task in tasks:
        split = task['split']
        if task['window_id'] not in indexes[split]:
            raise ValueError('Planned window absent from declared split')
        ds = datasets[split]
        entry = ds.entries[indexes[split][task['window_id']]]
        sources.extend(ds._path(entry, name) for name in entry['files'])
    target = torch.device(device)
    if target.type not in ('cpu', 'cuda') or (target.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested replay device is unavailable')
    if target.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(target)
    manifest = json.loads((Path(config['dataset'])/'dataset_manifest.json').read_text(encoding='utf-8'))
    effective = {**config, 'evaluation': {name: asdict(kwargs[name+'_config']) for name in ('motion','road','collision')}, 'device': str(target), 'formal': formal, 'method_configs': {n: asdict(c) for n,c in methods.items()},
                 'scope': 'explicit train/validation label diagnostic; no test, checkpoint, optimization or free generation'}
    inputs = sorted({str(p.resolve(strict=True)) for p in sources})
    with RunRecorder(output, Path.cwd(), effective, [sys.executable, *sys.orig_argv[1:]],
                     {'task_protocol': config['seed']}, purpose='action_state_label_compatibility',
                     data_files=inputs, data_id=manifest.get('data_id'), formal=formal) as recorder:
        start = time.perf_counter()
        records, saved, calls = [], [], 0
        for task_index, task in enumerate(tasks):
            record = {**task, 'status': 'failed', 'methods': {}}
            stem = str(task_index).zfill(3) + '_' + task['split']
            try:
                sample = datasets[task['split']][indexes[task['split']][task['window_id']]]
                c, labels = sample['conditioning'], sample['targets']
                active = c['agent_mask'].astype(bool)
                if c['history'].shape[1:] != (21, 6) or labels['future'].shape[1:] != (40, 6) or len(active) > 12 or sample['input_metadata']['dt_seconds'] != .1:
                    raise ValueError('Unsupported window contract')
                states = labels['future'].astype(np.float64).copy()
                states[..., :2] += c['initial_positions'][:, None]
                states[~active] = 0
                fm = labels['future_mask']
                original = evaluate_trajectory(c, sample['input_metadata'], sample['exact_map'], states, future_mask=fm, **kwargs)
                original_comfort = comfort_by_agent(c, states, fm, kwargs['motion_config'])
                arrays = dict(label_states=states, agent_mask=active, history=c['history'], history_mask=c['history_mask'],
                              initial_positions=c['initial_positions'], future_mask=fm, lengths=c['attributes'][:, 0])
                record['methods']['label'] = dict(status='completed', metrics=original, comfort_by_agent=original_comfort)
                for name, cfg in methods.items():
                    try:
                        tensors = [torch.as_tensor(v, device=target) for v in (c['history'], c['history_mask'], c['initial_positions'], c['attributes'][:,0], active)]
                        initial = initial_control_state(*tensors, cfg)
                        inverse = inverse_label_controls(torch.as_tensor(states, device=target), torch.as_tensor(fm, device=target), initial)
                        trajectory = rollout_controls(inverse.controls, initial)
                        calls += 1
                        replay = trajectory.states.detach().cpu().numpy()
                        controls = inverse.controls.detach().cpu().numpy()
                        lateral = inverse.lateral_interval_velocity.detach().cpu().numpy()
                        metrics = evaluate_trajectory(c, sample['input_metadata'], sample['exact_map'], replay, future_mask=fm, **kwargs)
                        comfort = comfort_by_agent(c, replay, fm, kwargs['motion_config'])
                        heading_delta = np.arctan2(replay[..., 4], replay[..., 5]) - np.arctan2(states[...,4], states[...,5])
                        errors = dict(max_position_error_m=float(np.linalg.norm(replay[...,:2]-states[...,:2], axis=-1)[active].max()),
                                      max_velocity_error_mps=float(np.linalg.norm(replay[...,2:4]-states[...,2:4], axis=-1)[active].max()),
                                      max_heading_error_rad=float(np.abs(np.arctan2(np.sin(heading_delta), np.cos(heading_delta)))[active].max()))
                        gate = compatibility_gate(original, metrics, original_comfort, comfort, errors, config['gate'])
                        jerk = physical_jerk(c, replay)
                        record['methods'][name] = dict(status='completed', metrics=metrics, comfort_by_agent=comfort, errors=errors, gate=gate,
                             controls={f'channel_{k}': summary(controls[..., k][active]) for k in range(cfg.channels)},
                             lateral_velocity_mps=summary(np.abs(lateral[active])),
                             label_velocity_residual_mps=summary(np.linalg.norm(inverse.label_velocity_residual.detach().cpu().numpy(),axis=-1)[active]),
                             jerk_mps3={part: summary(np.linalg.norm(jerk[:, sl], axis=-1)[active]) for part, sl in
                                         [('boundary_first4', slice(0,4)), ('interior36', slice(4,None)), ('terminal4',slice(-4,None))]})
                        arrays[name+'_states'], arrays[name+'_controls'], arrays[name+'_lateral_velocity'] = replay, controls, lateral
                        arrays[name+'_jerk'] = jerk
                    except (ValueError, TypeError, RuntimeError) as exc:
                        record['methods'][name] = dict(status='failed', error=dict(type=type(exc).__name__, message=str(exc)))
                np.savez_compressed(recorder.output/(stem+'.npz'), **arrays)
                saved.append(recorder.output/(stem+'.npz'))
                record['status'] = 'completed'
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                record['error'] = dict(type=type(exc).__name__, message=str(exc))
            write_json(recorder.output/(stem+'.json'), record)
            saved.append(recorder.output/(stem+'.json'))
            records.append(record)
            if (task_index+1) % 8 == 0 or task_index+1 == len(tasks):
                print(f'Label replay {task_index+1}/{len(tasks)}', flush=True)
        if target.type == 'cuda':
            torch.cuda.synchronize(target)
        duration = time.perf_counter()-start
        results = dict(schema_version='sumodiff.control_replay.results.v1', planned=len(tasks), records=records,
                       summary=task_summary(records, methods), independent_evaluation=kwargs_to_dict(kwargs),
                       control_rollout_calls=calls, denoiser_calls=0, optimizer_updates=0, free_samples=0,
                       test_arrays_loaded=0, elapsed_replay_and_evaluation_s=duration,
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(target) if target.type == 'cuda' else None,
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(target) if target.type == 'cuda' else None,
                       gate_pass_all_tasks={name: all(r['status']=='completed' and r['methods'][name]['status']=='completed' and r['methods'][name]['gate']['passed'] for r in records) for name in methods},
                       saved_outputs=[file_identity(p) for p in saved],
                       interpretation='Label inversion with known future; compatibility evidence only, not learned generation or dynamics execution')
        write_json(recorder.output/'results.json', results)
        recorder.write_metrics({k:v for k,v in results.items() if k not in ('records','saved_outputs')})
        print(json.dumps(results['gate_pass_all_tasks']), flush=True)
    return results


def kwargs_to_dict(kwargs):
    return {k: asdict(v) for k,v in kwargs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--formal', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, device=args.device, formal=args.formal)


if __name__ == '__main__':
    main()
