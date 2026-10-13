"""Recompute scale envelopes and quantify future motion in the observed body frame."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
from sumodiff.data.dataset import write_json
from sumodiff.experiments.recorder import RunRecorder, file_identity, load_config


def body_motion(controls, initial_speed, dt=.1):
    controls=np.asarray(controls,float);initial_speed=np.asarray(initial_speed,float)
    if controls.ndim!=3 or controls.shape[-1]!=3 or initial_speed.shape!=controls.shape[:1] or controls.shape[1]<1 or not np.isfinite(controls).all() or not np.isfinite(initial_speed).all() or dt!=.1:
        raise ValueError('Invalid body-motion audit inputs')
    yaw=np.cumsum(controls[...,1],axis=1)*dt
    basis=yaw-controls[...,1]*dt/2
    speed=initial_speed[:,None]+np.cumsum(controls[...,0],axis=1)*dt
    lateral=speed*np.sin(basis)+controls[...,2]*np.cos(basis)
    y=np.cumsum(lateral,axis=1)*dt
    zero=np.zeros((len(controls),1))
    return dict(heading_span_rad=np.ptp(np.concatenate((zero,yaw),axis=1),axis=1),
                lateral_excursion_m=np.ptp(np.concatenate((zero,y),axis=1),axis=1))


def run(source,output,formal=False):
    source=Path(source).resolve(strict=True)
    names=('manifest.json','status.json','resolved_config.yaml','results.json','tasks.json','window_results.json','candidate_normalization.json','active_controls.npz')
    paths=[source/name for name in names]
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    status=json.loads((source/'status.json').read_text(encoding='utf-8'))
    results=json.loads((source/'results.json').read_text(encoding='utf-8'))
    config=load_config(source/'resolved_config.yaml')
    if not manifest['formal'] or manifest['git']['dirty'] or status['state']!='completed' or results['failed_windows']:
        raise ValueError('Review requires completed clean formal audit with the full training cohort')
    if results['validation_arrays_loaded'] or results['test_arrays_loaded'] or results['optimizer_updates'] or results['denoiser_calls']:
        raise ValueError('Source exceeds training-only data audit scope')
    effective=dict(schema_version='sumodiff.control_scale_review.v1',source=file_identity(source/'results.json'),
                   heading_span_threshold_rad=float(np.deg2rad(5)),lateral_excursion_threshold_m=.5,
                   coordinate_frame='each observed vehicle body heading at t0, held fixed',
                   interpretation='future motion coverage; lateral excursion does not identify lane changes on curved roads')
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],manifest['seeds'],
                     purpose='verify_training_scale_envelopes_and_motion_coverage',data_files=paths,
                     data_id=manifest['data']['id'],formal=formal) as recorder:
        start=time.perf_counter()
        records=json.loads((source/'window_results.json').read_text(encoding='utf-8'))
        tasks=json.loads((source/'tasks.json').read_text(encoding='utf-8'))['tasks']
        if len(records)!=len(tasks) or len(tasks)!=results['planned_windows']:
            raise AssertionError('Training task denominator mismatch')
        cursor=0;families=[];group_rows={}
        for i,(row,task) in enumerate(zip(records,tasks)):
            if row['status']!='completed' or row['agent_offset']!=cursor or row['agent_count']!=task['selected_agents'] or any(row[k]!=v for k,v in task.items()):
                raise AssertionError('Training cache task/agent partition mismatch')
            cursor+=row['agent_count'];families.extend([row['family']]*row['agent_count'])
            for group in row['groups']:
                group_rows.setdefault(group,[]).append(row)
        with np.load(source/'active_controls.npz',allow_pickle=False) as z:
            controls=z['controls'];initial=z['initial_speed']
        if controls.shape!=(cursor,40,3) or cursor!=results['planned_agent_windows'] or cursor!=results['measured_agent_windows']:
            raise AssertionError('Active control cache does not cover the full cohort')
        group_quantiles={}
        for group,rows in group_rows.items():
            indices=np.concatenate([np.arange(row['agent_offset'],row['agent_offset']+row['agent_count']) for row in rows])
            values={name:float(np.quantile(np.abs(controls[indices,:,k]),config['scale_policy']['quantile'])) for k,name in enumerate(('acceleration','yaw_rate','lateral_velocity'))}
            for key,value in values.items():
                if not np.isclose(value,results['groups'][group]['scale_quantiles'][key],rtol=0,atol=1e-12):
                    raise AssertionError('Scale group quantile disagreement')
            group_quantiles[group]=values
        expected={}
        policy=config['scale_policy'];candidate=results['candidate_normalization']
        for channel in ('acceleration','yaw_rate','lateral_velocity'):
            envelope=max(group_quantiles[g][channel] for g in policy['groups'])
            expected[channel]=next(float(v) for v in policy['candidate_scales'][channel] if v>=envelope)
            if expected[channel]!=candidate['scales'][channel]['scale']:
                raise AssertionError('Candidate envelope/grid disagreement')
        motion=body_motion(controls,initial)
        coverage={}
        fam=np.asarray(families)
        for family in ('all',*sorted(set(families))):
            mask=np.ones(cursor,bool) if family=='all' else fam==family
            heading=motion['heading_span_rad'][mask];lateral=motion['lateral_excursion_m'][mask]
            heading_flag=heading>=effective['heading_span_threshold_rad'];lateral_flag=lateral>=effective['lateral_excursion_threshold_m']
            window_count=sum(any(motion['heading_span_rad'][r['agent_offset']:r['agent_offset']+r['agent_count']]>=effective['heading_span_threshold_rad']) for r in records if family=='all' or r['family']==family)
            coverage[family]=dict(agent_windows=int(mask.sum()),heading_span_over5deg_agents=int(heading_flag.sum()),
                                 heading_span_over5deg_windows=int(window_count),lateral_excursion_over_half_meter_agents=int(lateral_flag.sum()),
                                 max_heading_span_degrees=float(np.rad2deg(heading).max()),max_lateral_excursion_m=float(lateral.max()))
        np.savez_compressed(recorder.output/'body_motion.npz',**motion)
        summary=dict(schema_version='sumodiff.control_scale_review.v1',source_git=manifest['git'],source_results=file_identity(source/'results.json'),
                     planned_windows=len(records),agent_windows=cursor,groups_verified=len(group_quantiles),
                     independent_candidate_scales=expected,coverage=coverage,
                     elapsed_review_body_s=time.perf_counter()-start,optimizer_updates=0,test_arrays_loaded=0,validation_arrays_loaded=0,
                     caveat='No full scene evaluation; motion coverage is from training labels, not generated behavior')
        write_json(recorder.output/'review.json',summary)
        recorder.write_metrics(summary)
        print(json.dumps(coverage),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--formal',action='store_true')
    args=parser.parse_args()
    run(args.source,args.output,args.formal)


if __name__=='__main__':
    main()
