"""Frozen-checkpoint attribution checks; label controls are never generation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from sumodiff.data.dataset import WindowDataset, collate_numpy, write_json
from sumodiff.decoding import DecoderConfig, decode_future
from sumodiff.evaluation.metrics import evaluate_trajectory, _heading
from sumodiff.evaluation.motion import MotionConfig, evaluate_motion, summary
from sumodiff.evaluation.roads import RoadConfig
from sumodiff.evaluation.collisions import CollisionConfig
from sumodiff.experiments.recorder import RunRecorder, load_config, file_identity
from sumodiff.models import ConditionalDenoiser, resolve_model_config, prepare_conditioning
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from .checkpoint import load_checkpoint
from .config import resolve_diffusion_config
from .data import dataset_identity
from .process import NoiseSchedule, denormalize
from .sampling import sample_config

CONTROLS = ('label_raw', 'label_decoded', 'oracle_position', 'oracle_velocity',
            'oracle_heading', 'oracle_position_velocity', 'current_constant_acceleration')


def independent_position_solve(raw, mask, config):
    """QR/SVD least squares on stacked residuals, not decoder normal equations."""
    raw = np.asarray(raw, np.float64)
    mask = np.asarray(mask, bool)
    if raw.ndim != 3 or raw.shape[-1] != 6 or mask.shape != raw.shape[:1]:
        raise ValueError('Invalid independent decoder inputs')
    if not np.isfinite(raw[mask]).all():
        raise ValueError('Nonfinite active decoder input')
    x = np.where(mask[:, None, None], raw, 0.)
    n, t = x.shape[:2]
    d = np.eye(t)
    d[np.arange(1, t), np.arange(t-1)] = -1.
    alpha = config.position_weight / config.position_unit_m**2
    beta = config.velocity_weight / config.velocity_unit_mps**2
    a = np.vstack((alpha**.5*np.eye(t), beta**.5/config.dt*d))
    rhs = np.concatenate((alpha**.5*x[:, :, :2],
                          beta**.5*x[:, :, 2:4]), axis=1).transpose(1, 0, 2).reshape(2*t, n*2)
    q = np.linalg.lstsq(a, rhs, rcond=None)[0].reshape(t, n, 2).transpose(1, 0, 2)
    return q


def current_acceleration_control(c, count=40, dt=.1):
    """Diagnostic extrapolation with backward-interval velocities; no labels."""
    a = np.asarray(c['agent_mask'], bool)
    h = np.asarray(c['history'], np.float64)
    if not np.asarray(c['history_mask'], bool)[a, -3:].all():
        raise ValueError('Control requires three currently observed positions')
    p = np.asarray(c['initial_positions'], np.float64)
    v0 = (p-h[:, -2, :2])/dt
    accel0 = (p-2*h[:, -2, :2]+h[:, -3, :2])/dt**2
    k = np.arange(1, count+1, dtype=np.float64)[None, :, None]
    q = k*dt*v0[:, None]+k*(k+1)/2*dt**2*accel0[:, None]
    v = v0[:, None]+k*dt*accel0[:, None]
    head = _heading(c)
    head = head/np.linalg.norm(head, axis=-1, keepdims=True).clip(1e-6)
    out = np.concatenate((q, v, np.broadcast_to(head[:, None], (*q.shape[:2], 2))), axis=-1)
    return np.where(a[:, None, None], out, 0.)


def hybrid_inputs(raw, labels):
    """Oracle channel substitutions for attribution; not deployable samplers."""
    out = {}
    for name, sl in [('oracle_position', slice(0, 2)), ('oracle_velocity', slice(2, 4)),
                     ('oracle_heading', slice(4, 6)), ('oracle_position_velocity', slice(0, 4))]:
        value = np.array(raw, dtype=np.float64, copy=True)
        value[..., sl] = labels[..., sl]
        out[name] = value
    return out


def physical_derivatives(c, states, dt=.1):
    """Independent complete-window differences; first future jerk uses history."""
    a = np.asarray(c['agent_mask'], bool)
    if not np.asarray(c['history_mask'], bool)[a, -3:].all():
        raise ValueError('Missing derivative context')
    history = np.asarray(c['history'], np.float64)[:, -3:, :2].copy()
    history[:, -1] = np.asarray(c['initial_positions'], np.float64)
    p = np.concatenate((history, np.asarray(states, np.float64)[..., :2]), axis=1)
    velocity = np.diff(p, axis=1)/dt
    acceleration = np.diff(velocity, axis=1)/dt
    jerk = np.diff(acceleration, axis=1)/dt
    return velocity[:, 2:], acceleration[:, 1:], jerk


def compact(c, states, metric):
    a = np.asarray(c['agent_mask'], bool)
    v, acc, jerk = physical_derivatives(c, states)
    return dict(quality_pass=metric['quality_pass'],
                motion_pass=metric['motion']['hard_quality_pass'],
                collision=metric['collisions']['groups']['any']['collision'],
                road_fraction=metric['roads']['road']['violation_fraction'],
                route_fraction=metric['roads']['route']['violation_fraction'],
                map_fraction=metric['roads']['map_coverage']['violation_fraction'],
                violations=metric['motion']['violation_counts'],
                boundary_jerk_max=metric['motion']['boundary']['jerk_mps3']['max'],
                velocity_residual_mps=summary(np.linalg.norm(states[..., 2:4]-v, axis=-1)[a]),
                interior_jerk_mps3=summary(np.linalg.norm(jerk[:, 4:], axis=-1)[a]),
                boundary_velocity_change_mps=metric['motion']['boundary']['velocity_change_mps'],
                acceleration_norm_mps2=summary(np.linalg.norm(acc, axis=-1)[a]))


def decode_array(raw, c, config):
    return decode_future(torch.as_tensor(raw, dtype=torch.float64),
                         torch.as_tensor(c['initial_positions'], dtype=torch.float64),
                         torch.as_tensor(_heading(c), dtype=torch.float64),
                         torch.as_tensor(c['agent_mask'], dtype=torch.bool), config)


@torch.no_grad()
def denoising_trace(model, tensors, initial, schedule, c, decoder, motion, steps):
    """Only inputs and saved initial noise enter the frozen-network replay."""
    mask = tensors['agent_mask'][:, :, None, None]
    value = torch.where(mask, initial, 0.)
    condition = model.condition_encoder(tensors)
    sequence = schedule.timesteps(steps)
    rows, estimates, epsilons = [], [], []
    for index, current in enumerate(sequence):
        timestep = torch.full((len(value),), current, device=value.device, dtype=torch.long)
        epsilon = model.unet(value, timestep, condition, tensors['agent_mask'])
        clean = schedule.clean_from_noise(value, epsilon, timestep)
        physical = denormalize(clean, schedule.config)[0]
        decoded = decode_array(physical.cpu().numpy(), c, decoder)
        raw_states, states = decoded.raw_states.numpy(), decoded.states.numpy()
        a = np.asarray(c['agent_mask'], bool)
        row = dict(timestep=current)
        for name, state in [('raw', raw_states), ('decoded', states)]:
            v, acc, j = physical_derivatives(c, state)
            fm = np.broadcast_to(a[:, None], state.shape[:2])
            checked = evaluate_motion(c['history'], c['history_mask'], state, fm,
                c['initial_positions'], _heading(c), a, config=motion)
            row[name] = dict(hard_motion_pass=checked['hard_quality_pass'],
                             jerk_violation_points=checked['violation_counts']['jerk'],interior_jerk_p99=float(np.quantile(np.linalg.norm(j[:, 4:], axis=-1)[a], .99)),
                             boundary_jerk_max=float(np.linalg.norm(j[:, 0], axis=-1)[a].max()),
                             velocity_residual_p99=float(np.quantile(np.linalg.norm(state[..., 2:4]-v, axis=-1)[a], .99)))
        row['position_correction_max_m'] = float(decoded.position_correction[a].norm(dim=-1).max())
        rows.append(row)
        estimates.append(clean[0].cpu().numpy())
        epsilons.append(epsilon[0].cpu().numpy())
        previous = sequence[index+1] if index+1 < len(sequence) else -1
        value = schedule.transition(value, clean, epsilon, current, previous, eta=0.)
        value = torch.where(mask, value, 0.)
    return value[0].cpu().numpy(), rows, np.stack(estimates), np.stack(epsilons)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(dataset, checkpoint, source_run, config_path, output, formal=False):
    c = load_config(config_path)
    if c != dict(schema_version='sumodiff.quality.audit.config.v1', expected_windows=200,
                 trace_limit=6, split='validation', seed=20261016):
        raise ValueError('Audit is bounded to the declared 200 validation windows and six replays')
    source = Path(source_run).resolve(strict=True)
    manifest = read(source/'manifest.json')
    if read(source/'status.json')['state'] != 'completed':
        raise ValueError('Incomplete source generation')
    if formal and (not manifest['formal'] or manifest['git']['dirty']):
        raise ValueError('Formal audit requires a clean formal source')
    resolved = load_config(source/'resolved_config.yaml')
    sc = sample_config(resolved['sampling'])
    if sc['eta'] != 0 or sc['split'] != 'validation' or sc['steps'] != 20:
        raise ValueError('Audit requires deterministic 20-step validation source')
    if file_identity(checkpoint)['sha256'] != resolved['checkpoint']['sha256']:
        raise ValueError('Source checkpoint changed')
    payload = load_checkpoint(checkpoint)
    data = dataset_identity(dataset)
    if any(data[k]['sha256'] != payload['metadata']['data'][k]['sha256'] for k in data):
        raise ValueError('Audit dataset/checkpoint mismatch')
    ds = WindowDataset(dataset, 'validation')
    catalog = read(source/'trajectory_index.json')
    lookup = {x['window_id']:i for i, x in enumerate(ds.entries)}
    if len(catalog) != 200 or len(lookup) != 200 or {x['window_id'] for x in catalog} != set(lookup):
        raise ValueError('Full validation source/population mismatch')
    if any(x['generation_status'] != 'completed' for x in catalog):
        raise ValueError('Source failures must be diagnosed explicitly, not dropped')
    files = [config_path, source/'manifest.json', source/'status.json', source/'resolved_config.yaml',
             source/'trajectory_index.json', source/'sampling_summary.json', *[x['location'] for x in data.values()]]
    for item in catalog:
        entry = ds.entries[lookup[item['window_id']]]
        files += [ds._path(entry, name) for name in ('conditioning.npz', 'input.json', 'map.json', 'targets.npz', 'labels.json')]
        for name in ('stages.npz', 'metrics.json'):
            path = source/'trajectories'/item['window_id']/name
            if file_identity(path)['sha256'] != item['files'][name]['sha256']:
                raise ValueError('Source trajectory changed')
            files.append(path)
    evaluation = sc['evaluation']
    decoder = DecoderConfig(**evaluation['decoder'])
    kwargs = dict(motion_config=MotionConfig(**evaluation['motion']),
                  road_config=RoadConfig(**evaluation['road']),
                  collision_config=CollisionConfig(**evaluation['collision']))
    effective = dict(schema_version='sumodiff.quality.audit.run.v1', audit=c, source_run=manifest,
                     data=data, checkpoint=file_identity(checkpoint), evaluation=evaluation,
                     controls=list(CONTROLS), trace_sampling=sc,
                     interpretations=['oracle controls use true future labels only after generation',
                                      'constant-acceleration extrapolation is a diagnostic control',
                                      'replay uses inputs and saved initial noise only',
                                      'thresholds and production algorithms unchanged'])
    with RunRecorder(output, Path.cwd(), effective, [sys.executable, *sys.orig_argv[1:]],
                     {'audit':c['seed']}, purpose='stage5a_frozen_quality_attribution',
                     data_files=sorted(set(map(str, files))), data_id=data['manifest']['id'],
                     checkpoint=checkpoint, formal=formal) as run:
        started = time.perf_counter()
        configure(c['seed'], sc['torch_threads'])
        groups = defaultdict(Counter)
        pooled = defaultdict(lambda:defaultdict(list))
        numerical = dict(independent_position_error_max_m=0., saved_decoder_error_max=0.,
                         label_position_correction_max_m=0.)
        with (run.output/'diagnostic_rows.jsonl').open('w', encoding='utf-8') as log:
            for ix, item in enumerate(catalog):
                sample = ds[lookup[item['window_id']]]
                cond = sample['conditioning']
                a = cond['agent_mask']
                labels = sample['targets']['future'].astype(np.float64)
                if not sample['targets']['future_mask'][a].all():
                    raise ValueError('Label attribution requires the complete declared cohort')
                directory = source/'trajectories'/item['window_id']
                with np.load(directory/'stages.npz', allow_pickle=False) as f:
                    raw = f['raw_future_delta'].copy()
                    saved = f['decoded_absolute_states'].copy()
                    raw_saved = f['raw_absolute_states'].copy()
                decoded = decode_array(raw, cond, decoder)
                independent = independent_position_solve(raw, a, decoder)
                error = float(np.abs(independent[a]-(decoded.states.numpy()[a, :, :2]-cond['initial_positions'][a, None])).max())
                replay = float(np.abs(decoded.states.numpy()[a]-saved[a]).max())
                numerical['independent_position_error_max_m'] = max(numerical['independent_position_error_max_m'], error)
                numerical['saved_decoder_error_max'] = max(numerical['saved_decoder_error_max'], replay)
                if error > 1e-8 or replay > 1e-8:
                    raise RuntimeError('Independent decoder/saved trajectory numerical mismatch')
                ld = decode_array(labels, cond, decoder)
                numerical['label_position_correction_max_m'] = max(numerical['label_position_correction_max_m'],
                    float(ld.position_correction[a].norm(dim=-1).max()))
                variants = {'raw':raw_saved, 'decoded':saved,
                            'label_raw':ld.raw_states.numpy(), 'label_decoded':ld.states.numpy()}
                for name, future in hybrid_inputs(raw, labels).items():
                    variants[name] = decode_array(future, cond, decoder).states.numpy()
                variants['current_constant_acceleration'] = decode_array(current_acceleration_control(cond), cond, decoder).states.numpy()
                records = {}
                for name, state in variants.items():
                    m = evaluate_trajectory(cond, sample['input_metadata'], sample['exact_map'], state, **kwargs)
                    row = compact(cond, state, m)
                    records[name] = row
                    key = (name, item['family'])
                    group = groups[key]
                    group['scenes'] += 1
                    group['quality_pass'] += row['quality_pass'] is True
                    group['quality_unknown'] += row['quality_pass'] is None
                    group['motion_bad'] += row['motion_pass'] is False
                    group['collision_scenes'] += row['collision'] is True
                    group['collision_unknown'] += row['collision'] is None
                    group['road_bad'] += row['road_fraction'] > 0
                    group['route_bad'] += row['route_fraction'] > 0
                    group['map_bad'] += row['map_fraction'] > 0
                    group['boundary_jerk_bad'] += row['boundary_jerk_max'] > kwargs['motion_config'].max_jerk_mps3
                    group['interior_jerk_bad'] += row['interior_jerk_mps3']['max'] > kwargs['motion_config'].max_jerk_mps3
                    v, ac, j = physical_derivatives(cond, state)
                    pooled[name]['interior_jerk'].extend(np.linalg.norm(j[:, 4:], axis=-1)[a].ravel().tolist())
                    pooled[name]['velocity_residual'].extend(np.linalg.norm(state[..., 2:4]-v, axis=-1)[a].ravel().tolist())
                log.write(json.dumps(dict(window_id=item['window_id'], family=item['family'], cases=records), allow_nan=False)+'\n')
                if (ix+1) % 25 == 0:
                    print(f'attribution={ix+1}/200', flush=True)
        write_json(run.output/'attribution_summary.json',
            {name:dict(by_family={family:dict(v) for (key, family), v in groups.items() if key==name},
                       pooled={key:summary(value) for key, value in values.items()}) for name, values in pooled.items()})
        device = torch.device(sc['device'])
        model = ConditionalDenoiser(resolve_model_config(payload['metadata']['model_config'])).to(device)
        model.load_state_dict(payload['model']); model.eval(); model.requires_grad_(False)
        weight_hash = parameter_hash(model)
        schedule = NoiseSchedule(resolve_diffusion_config(payload['metadata']['diffusion_config']), device)
        traces = []
        (run.output/'traces').mkdir()
        for item in catalog[:c['trace_limit']]:
            current = ds.inference(lookup[item['window_id']])
            tensors = prepare_conditioning(collate_numpy([current]), device)
            with np.load(source/'trajectories'/item['window_id']/'stages.npz', allow_pickle=False) as f:
                initial = torch.as_tensor(f['initial_noise'][None], device=device)
                expected = f['raw_normalized_future'].copy()
            result, rows, clean, epsilon = denoising_trace(model, tensors, initial, schedule,
                current['conditioning'], decoder, kwargs['motion_config'], sc['steps'])
            if not np.array_equal(result, expected, equal_nan=True):
                raise RuntimeError('Frozen denoising replay differs from source generation')
            np.savez_compressed(run.output/'traces'/f"{item['window_id']}.npz",
                                timesteps=[x['timestep'] for x in rows], clean_estimates=clean, pred_epsilon=epsilon)
            traces.append(dict(window_id=item['window_id'], family=item['family'], source_replay='bitwise_equal', steps=rows))
        if parameter_hash(model) != weight_hash:
            raise RuntimeError('Audit changed model parameters')
        result = dict(schema_version='sumodiff.quality.audit.metrics.v1',
                      run_sha=run.manifest['git']['commit_sha'], source_run_sha=manifest['git']['commit_sha'],
                      numerical_checks=numerical, attribution=read(run.output/'attribution_summary.json'),
                      traces=traces, training_settings=dict(learning_rate=payload['metadata']['training_config']['learning_rate'],
                          optimizer_learning_rates=[g['lr'] for g in payload['optimizer']['param_groups']],
                          objective=payload['metadata']['diffusion_config']['objective'],
                          additional_losses_enabled=payload['metadata']['diffusion_config']['additional_losses_enabled']),
                      wall_seconds=time.perf_counter()-started, denoiser_calls=len(traces)*sc['steps'],
                      actual_gradient_calls=0, weights_unchanged=True,
                      planned_windows=len(catalog), convergence_status='not_assessed')
        write_json(run.output/'audit_summary.json', result); run.write_metrics(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'checkpoint', 'source-run', 'config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--formal', action='store_true')
    a = parser.parse_args()
    try:
        result = audit(a.dataset, a.checkpoint, a.source_run, a.config, a.output, a.formal)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    print({k:result[k] for k in ('run_sha', 'wall_seconds', 'planned_windows', 'weights_unchanged')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
