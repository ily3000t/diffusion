"""Reproducible label reconstruction and independent evaluation acceptance."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import psutil
import torch
from sumodiff.data.dataset import WindowDataset,write_json
from sumodiff.decoding import decode_future,DecoderConfig
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from .config import resolve_config
from .motion import MotionConfig
from .roads import RoadConfig
from .collisions import CollisionConfig
from .metrics import evaluate_trajectory,correction_summary,aggregate_scenes,_heading,applicability
from .probes import numeric_probe


def _failure(meta,agents,error):
    return dict(schema_version='sumodiff.scene.metrics.v1',status='failed',window_id=meta['window_id'],
        planned_active_agents=int(agents.sum()),excluded_vehicle_count=meta.get('excluded_vehicle_count'),task_applicability=applicability(agents,None),
        quality_pass=None,effective_target_event=None,error=dict(type=type(error).__name__,message=str(error)))


def validate_labels(dataset_path,config_path,output,repository,formal=False,command=None):
    config = resolve_config(load_config(config_path))
    dataset = WindowDataset(dataset_path)
    dataset_path = Path(dataset_path).resolve(strict=True)
    inputs = [config_path,dataset_path/'dataset_manifest.json',dataset_path/'windows.json']
    inputs += [dataset_path/identity['relative_path'] for entry in dataset.entries for identity in entry['files'].values()]
    inputs = sorted({str(Path(p).resolve()) for p in inputs})
    effective = dict(schema_version='sumodiff.label.validation.config.v1',evaluation=config,
                    input_kind='ground_truth_labels',dataset_manifest=file_identity(dataset_path/'dataset_manifest.json'),
                    recording=dict(formal=formal,output=str(Path(output).resolve())))
    command = command or [sys.executable,'-m','sumodiff.evaluation','labels','--dataset',str(dataset_path),'--config',str(config_path),'--output',str(output)]
    with RunRecorder(output,repository,effective,command,{'probe':config['seed']},
        purpose='stage3_ground_truth_reconstruction_and_independent_metrics',data_files=inputs,
        data_id=effective['dataset_manifest']['id'],formal=formal) as run:
        started = time.perf_counter()
        torch.set_num_threads(config['torch_threads'])
        probe = numeric_probe(config['seed'],config['cuda_smoke'],DecoderConfig(**config['decoder']))
        write_json(run.output/'numeric_probe.json',probe)
        raw_records,decoded_records,rows,corrections = [],[],[],[]
        peak_rss,skipped,failures = 0,0,0
        kwargs = dict(motion_config=MotionConfig(**config['motion']),road_config=RoadConfig(**config['road']),
                      collision_config=CollisionConfig(**config['collision']))
        for index in range(len(dataset)):
            sample = dataset[index]
            c,meta,exact,targets = [sample[k] for k in ('conditioning','input_metadata','exact_map','targets')]
            agents = c['agent_mask']
            future,fm = targets['future'],targets['future_mask']
            physical_future = future.astype(np.float64)
            raw_states = np.concatenate((physical_future[...,:2]+c['initial_positions'].astype(np.float64)[:,None,:],physical_future[...,2:]),axis=-1)
            raw_states = np.where(agents[:,None,None],raw_states,0.)
            row_dir = run.output/'trajectories'/meta['window_id']
            row_dir.mkdir(parents=True,exist_ok=False)
            saved = dict(raw_future_delta=future,raw_absolute_states=raw_states,future_label_mask=fm,
                initial_positions=c['initial_positions'],agent_mask=agents)
            row = dict(window_id=meta['window_id'],split=meta['split'],input_kind='ground_truth_labels',
                guided_stage=dict(status='not_applicable',reason='stage3 has no guidance'),decoded_status='not_applicable_incomplete_ground_truth')
            try:
                heading = _heading(c)
                raw_report = evaluate_trajectory(c,meta,exact,raw_states,future_mask=fm,**kwargs)
                row['raw'] = raw_report; raw_records.append(raw_report)
                if fm[agents].all():
                    decoded = decode_future(torch.as_tensor(future),torch.as_tensor(c['initial_positions']),
                        torch.as_tensor(heading),torch.as_tensor(agents),DecoderConfig(**config['decoder']))
                    states = decoded.states.detach().cpu().numpy()
                    decoded_report = evaluate_trajectory(c,meta,exact,states,initial_heading=heading,**kwargs)
                    corrected = correction_summary(decoded)
                    row.update(decoded_status='completed',decoded=decoded_report,corrections=corrected)
                    corrections.append(corrected);decoded_records.append(decoded_report)
                    saved.update(decoded_absolute_states=states,initial_heading=heading,
                        position_correction=decoded.position_correction.detach().numpy(),
                        velocity_correction=decoded.velocity_correction.detach().numpy(),
                        heading_correction=decoded.heading_correction.detach().numpy(),
                        heading_fallback_mask=decoded.heading_fallback_mask.detach().numpy(),
                        raw_velocity_residual=decoded.raw_velocity_residual.detach().numpy())
                else:
                    row['decoded'] = None
                    row['decoder_skip_reason'] = 'incomplete ground-truth labels cannot stand in for a full generated horizon'
                    skipped += 1
            except Exception as exc:
                row['error'] = dict(type=type(exc).__name__,message=str(exc))
                if 'raw' not in row:
                    failure = _failure(meta,agents,exc);row['raw']=failure;raw_records.append(failure)
                if fm[agents].all():
                    failure = _failure(meta,agents,exc);row.update(decoded=failure,decoded_status='failed');decoded_records.append(failure)
                else:
                    skipped += 1
                failures += 1
            np.savez_compressed(row_dir/'stages.npz',**saved)
            write_json(row_dir/'metrics.json',row)
            rows.append(dict(window_id=meta['window_id'],split=meta['split'],decoded_status=row['decoded_status'],
                raw_status=row['raw']['status'],files={name:file_identity(row_dir/name) for name in ('stages.npz','metrics.json')}))
            peak_rss = max(peak_rss,psutil.Process().memory_info().rss)
        max_position = max((r['position_correction_m']['max'] for r in corrections),default=None)
        max_velocity = max((r['velocity_correction_mps']['max'] for r in corrections),default=None)
        result = dict(schema_version='sumodiff.stage3.label.report.v1',input_kind='ground_truth_labels',windows=len(dataset),
            decoded_full_label_windows=len(decoded_records),incomplete_label_decode_skips=skipped,failed_windows=failures,
            raw=aggregate_scenes(raw_records),decoded_full_label_subset=aggregate_scenes(decoded_records),
            numeric_probe=probe,max_label_position_correction_m=max_position,max_label_velocity_correction_mps=max_velocity,
            heading_fallbacks=sum(c['heading_fallback_count'] for c in corrections),
            raw_body_road_violation_frames=sum(r.get('roads',{}).get('road',{}).get('violation_body_frames',0) for r in raw_records),
            raw_geometry_repair_windows=sum(bool(r.get('roads',{}).get('geometry_repairs',[])) for r in raw_records),
            raw_geometry_repair_count=sum(len(r.get('roads',{}).get('geometry_repairs',[])) for r in raw_records),
            raw_evaluated_body_frames=sum(r.get('roads',{}).get('road',{}).get('evaluated_body_frames',0) for r in raw_records),
            selected_agent_windows=sum(r['planned_active_agents'] for r in raw_records),
            excluded_current_vehicle_windows=sum(r.get('excluded_vehicle_count',0) for r in raw_records),
            raw_map_coverage_violation_frames=sum(r.get('roads',{}).get('map_coverage',{}).get('violation_body_frames',0) for r in raw_records),
            raw_motion_violation_counts={key:sum(r.get('motion',{}).get('violation_counts',{}).get(key,0) for r in raw_records)
                for key in ('speed','longitudinal_acceleration','lateral_acceleration','jerk','yaw_rate','heading_velocity','heading_norm','velocity_consistency')},
            raw_body_route_violation_frames=sum(r.get('roads',{}).get('route',{}).get('violation_body_frames',0) for r in raw_records),
            raw_max_jerk_mps3=max((r['motion']['jerk_norm_mps3']['max'] for r in raw_records if r.get('motion',{}).get('jerk_norm_mps3',{}).get('max') is not None),default=None),
            raw_unresolved_collision_intervals=sum(r.get('collisions',{}).get('unresolved_intervals',0) for r in raw_records),
            elapsed_seconds=time.perf_counter()-started,sampled_python_peak_rss_bytes=peak_rss,
            run_sha=run.manifest['git']['commit_sha'],run_branch=run.manifest['git']['branch'],
            limitations=['label acceptance, not model performance','decoded subset uses label completeness only for numerical reconstruction',
                        'predeclared candidate quality thresholds, no behavioral calibration','selected vehicle set only'])
        write_json(run.output/'trajectory_index.json',rows)
        write_json(run.output/'validation_summary.json',result)
        run.write_metrics(result)
        if failures or not probe['gradcheck_passed'] or not probe['cpu_gradient_finite'] or not corrections or max_position > 2e-4 or max_velocity > 2e-3:
            raise RuntimeError('Stage3 numerical acceptance failed; metrics and failed cases retained')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action',required=True)
    label = sub.add_parser('labels')
    label.add_argument('--dataset',type=Path,required=True)
    label.add_argument('--config',type=Path,required=True)
    label.add_argument('--output',type=Path,required=True)
    label.add_argument('--repository',type=Path,default=Path.cwd())
    label.add_argument('--formal',action='store_true')
    args = p.parse_args(argv)
    command = [sys.executable,*sys.orig_argv[1:]] if argv is None else [sys.executable,'-m','sumodiff.evaluation',*argv]
    try:
        result = validate_labels(args.dataset,args.config,args.output,args.repository,args.formal,command)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        return 2
    print({k:v for k,v in result.items() if k not in ('raw','decoded_full_label_subset','numeric_probe','limitations')})
    return 0
