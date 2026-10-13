"""Independent NumPy step-loop audit of a recorded label/control replay."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
from sumodiff.data.dataset import WindowDataset, write_json
from sumodiff.evaluation.metrics import evaluate_trajectory
from sumodiff.evaluation.motion import MotionConfig
from sumodiff.evaluation.roads import RoadConfig
from sumodiff.evaluation.collisions import CollisionConfig
from sumodiff.experiments.recorder import RunRecorder, file_identity, load_config


def direction(angle):
    return np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def numpy_rollout(history, p0, lengths, active, controls, config):
    """Sequential interval integration; no torch or production rollout calls."""
    dt = config['dt']
    h = np.where(active[:,None,None], np.asarray(history,float), 0.)
    p0 = np.where(active[:,None], np.asarray(p0,float), 0.)
    lengths = np.where(active, np.asarray(lengths,float), 0.)
    controls = np.where(active[:,None,None], np.asarray(controls,float), 0.)
    yaw_previous = np.arctan2(h[:,-2,4], h[:,-2,5])
    yaw = np.arctan2(h[:,-1,4], h[:,-1,5])
    past_basis = yaw_previous+wrap(yaw-yaw_previous)/2 if config['integration_heading']=='midpoint' else yaw
    offset = lengths/2 if config['reference_point']=='front' else np.zeros_like(lengths)
    reference = p0 + offset[:,None]*direction(yaw)
    past_reference = h[:,-2,:2]+offset[:,None]*direction(yaw_previous)
    speed = ((reference-past_reference)*direction(past_basis)).sum(-1)/dt
    center = p0
    output = []
    for k in range(controls.shape[1]):
        previous = yaw.copy()
        yaw = yaw + dt*controls[:,k,1]
        speed = speed + dt*controls[:,k,0]
        basis = previous+(yaw-previous)/2 if config['integration_heading']=='midpoint' else yaw
        axis = direction(basis)
        displacement = dt*speed[:,None]*axis
        if config['lateral_velocity']:
            displacement += dt*controls[:,k,2,None]*np.stack((-axis[:,1],axis[:,0]),axis=-1)
        reference = reference + displacement
        new_center = reference-offset[:,None]*direction(yaw)
        velocity = (new_center-center)/dt
        state = np.concatenate((new_center,velocity,np.stack((np.sin(yaw),np.cos(yaw)),axis=-1)),axis=-1)
        output.append(np.where(active[:,None],state,0.))
        center = new_center
    return np.stack(output,axis=1)


def numpy_inverse(history, p0, lengths, active, states, config):
    """Derive controls from label poses in a NumPy step loop."""
    if config['compute_dtype'] != 'float64':
        raise NotImplementedError('Independent replay audit currently requires float64 computation')
    dt = config['dt']
    h = np.where(active[:, None, None], np.asarray(history, float), 0.)
    p0 = np.where(active[:, None], np.asarray(p0, float), 0.)
    lengths = np.where(active, np.asarray(lengths, float), 0.)
    labels = np.where(active[:, None, None], np.asarray(states, float), 0.)
    if not all(np.isfinite(v[active]).all() for v in (h[:, -2:], p0, lengths, labels)):
        raise ValueError('Non-finite active inverse audit input')
    offset = lengths/2 if config['reference_point']=='front' else np.zeros_like(lengths)
    before = np.arctan2(h[:, -2, 4], h[:, -2, 5])
    yaw = np.arctan2(h[:, -1, 4], h[:, -1, 5])
    previous_reference = h[:, -2, :2] + offset[:, None]*direction(before)
    reference = p0 + offset[:, None]*direction(yaw)
    past_basis = before + wrap(yaw-before)/2 if config['integration_heading']=='midpoint' else yaw
    speed = ((reference-previous_reference)*direction(past_basis)).sum(-1)/dt
    center = p0
    controls, residuals, velocity_errors = [], [], []
    for k in range(labels.shape[1]):
        current = labels[:, k]
        angle = np.arctan2(current[:, 4], current[:, 5])
        turn = wrap(angle-yaw)
        basis = yaw + turn/2 if config['integration_heading']=='midpoint' else angle
        axis = direction(basis)
        next_reference = current[:, :2] + offset[:, None]*direction(angle)
        interval = (next_reference-reference)/dt
        forward = (interval*axis).sum(-1)
        lateral = (interval*np.stack((-axis[:,1], axis[:,0]),-1)).sum(-1)
        values = [(forward-speed)/dt, turn/dt]
        if config['lateral_velocity']:
            values.append(lateral)
        controls.append(np.where(active[:, None], np.stack(values,-1), 0.))
        residuals.append(np.where(active, lateral, 0.))
        velocity_errors.append(np.where(active[:, None], current[:,2:4]-(current[:,:2]-center)/dt,0.))
        reference, center, speed, yaw = next_reference, current[:,:2], forward, angle
    return np.stack(controls,1), np.stack(residuals,1), np.stack(velocity_errors,1)


def numpy_errors(label, replay, active):
    position = np.sqrt(np.square(replay[...,:2]-label[...,:2]).sum(-1))
    velocity = np.sqrt(np.square(replay[...,2:4]-label[...,2:4]).sum(-1))
    angle = np.abs(wrap(np.arctan2(replay[...,4],replay[...,5])-np.arctan2(label[...,4],label[...,5])))
    return dict(max_position_error_m=float(position[active].max()),
                max_velocity_error_mps=float(velocity[active].max()),
                max_heading_error_rad=float(angle[active].max()))


def numpy_comfort(c, states, future_mask, config):
    from sumodiff.evaluation.motion import evaluate_motion
    result = {}
    for slot in np.flatnonzero(c['agent_mask']):
        active = np.zeros_like(c['agent_mask'],dtype=bool)
        active[slot] = True
        metric = evaluate_motion(c['history'],c['history_mask'],states,future_mask,
                                 c['initial_positions'],c['history'][:,-1,4:6],active,config=config)
        complete = metric['comfort_complete_assessed_points']==states.shape[1]
        result[str(int(slot))] = bool(metric['comfort_pass_agents']==1) if complete else None
    return result


def numpy_gate(original, replay, old_comfort, new_comfort, errors, limits):
    """Independent application of the recorded compatibility contract."""
    if old_comfort.keys()!=new_comfort.keys() or set(errors)!=set(limits):
        raise AssertionError('Invalid comparison identities')
    reasons=[]
    for name,value in errors.items():
        if not np.isfinite(value) or value<0:
            raise AssertionError('Invalid audit recovery error')
    if original['quality_pass'] is None or replay['quality_pass'] is None:
        reasons.append('unknown_scene_quality')
    reasons.extend(name for name,limit in limits.items() if errors[name]>limit)
    if original['quality_pass'] is True and replay['quality_pass'] is not True:
        reasons.append('lost_scene_quality')
    reasons.extend('increased_motion_'+name for name,count in original['motion']['violation_counts'].items()
                   if replay['motion']['violation_counts'][name]>count)
    reasons.extend('increased_'+name for name in ('road','route','map_coverage')
                   if replay['roads'][name]['violation_body_frames']>original['roads'][name]['violation_body_frames'])
    old_pairs={tuple(sorted(p['slots'])):p['collision'] for p in original['collisions']['pairs']}
    new_pairs={tuple(sorted(p['slots'])):p['collision'] for p in replay['collisions']['pairs']}
    if old_pairs.keys()!=new_pairs.keys():
        raise AssertionError('Audit collision pairs disagree')
    if any(old_pairs[p] is False and value is not False for p,value in new_pairs.items()):
        reasons.append('new_collision_or_unknown')
    new_bad=[slot for slot,passed in old_comfort.items() if passed is True and new_comfort[slot] is not True]
    if new_bad:
        reasons.append('new_strict_comfort_failure')
    return dict(passed=not reasons,reasons=reasons,newly_uncomfortable_slots=new_bad)


def numpy_jerk(c, states):
    past=np.where(c['agent_mask'][:,None,None],np.asarray(c['history'][:,-3:,:2],float),0.).copy()
    past[:,-1]=np.where(c['agent_mask'][:,None],c['initial_positions'],0.)
    future=np.where(c['agent_mask'][:,None,None],states[...,:2],0.)
    return np.diff(np.concatenate((past,future),axis=1),n=3,axis=1)/.1**3


def verify_source_report(results, plan, config, source_metrics):
    tasks = plan['tasks']
    records = results['records']
    if results['planned']!=len(tasks) or len(records)!=len(tasks):
        raise AssertionError('Task denominator mismatch')
    for task,record in zip(tasks,records):
        for key,value in task.items():
            if record.get(key)!=value:
                raise AssertionError('Task order/identity mismatch: '+key)
        if set(record['methods']) != {'label',*config['method_configs']}:
            raise AssertionError('Missing or unexpected replay method')
    if set(config['methods'])!=set(config['method_configs']) or 'label' in config['method_configs']:
        raise AssertionError('Method configuration identity mismatch')
    expected={k:v for k,v in results.items() if k not in ('records','saved_outputs')}
    close_json(source_metrics,expected)
    if any(results[k]!=0 for k in ('optimizer_updates','denoiser_calls','free_samples','test_arrays_loaded')):
        raise AssertionError('Source scope exceeds label replay')


def verify_group_summary(records, methods, recorded):
    """Aggregate verified rows; legacy reference gate count 0 is N/A."""
    groups={}
    for row in records:
        groups.setdefault(row['split']+'/'+row['group'],[]).append(row)
    if set(groups)!=set(recorded):
        raise AssertionError('Group identities disagree')
    for group,rows in groups.items():
        for name in ('label',*methods):
            values=[r['methods'][name] for r in rows]
            planned_agents=sum(r['planned_active_agents'] for r in rows)
            strict_pass=sum(v is True for item in values for v in item['comfort_by_agent'].values())
            strict_assessed=sum(v is not None for item in values for v in item['comfort_by_agent'].values())
            expected=dict(planned=len(rows),completed=len(rows),failed=0,
                planned_active_agents=planned_agents,unknown_planned_agent_tasks=0,
                strict_comfort_confirmed_pass_agents=strict_pass,strict_comfort_assessed_agents=strict_assessed,
                strict_comfort_confirmed_pass_fraction_all_planned=strict_pass/planned_agents if planned_agents else None,
                scene_quality_pass=sum(v['metrics']['quality_pass'] is True for v in values),
                motion_pass=sum(v['metrics']['motion']['hard_quality_pass'] is True for v in values),
                comfort_pass_agents=sum(v['metrics']['motion']['comfort_pass_agents'] for v in values),
                comfort_assessed_agents=sum(v['metrics']['motion']['comfort_assessed_agents'] for v in values),
                gate_pass=None if name=='label' else sum(v['gate']['passed'] for v in values),
                maximum_position_error_m=None if name=='label' else max(v['errors']['max_position_error_m'] for v in values),
                maximum_velocity_error_mps=None if name=='label' else max(v['errors']['max_velocity_error_mps'] for v in values),
                maximum_abs_lateral_interval_velocity_mps=None if name=='label' else max(v['lateral_velocity_mps']['max'] for v in values))
            source=dict(recorded[group][name])
            if name=='label' and source['gate_pass']==0:
                source['gate_pass']=None
            close_json(source,{key:expected[key] for key in source})


def close_json(actual, expected, path='root'):
    if isinstance(expected,dict):
        if actual.keys()!=expected.keys():
            raise AssertionError('Different keys at '+path)
        for key in expected:
            close_json(actual[key],expected[key],path+'/'+key)
    elif isinstance(expected,list):
        if len(actual)!=len(expected):
            raise AssertionError('Different list lengths at '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):
            close_json(a,b,path+'/'+str(i))
    elif type(expected) is float:
        if not np.isclose(actual,expected,rtol=1e-8,atol=1e-8):
            raise AssertionError(f'Metric mismatch at {path}: {actual} vs {expected}')
    elif actual!=expected:
        raise AssertionError('Value mismatch at '+path)


def run(source, output, formal=False, input_snapshots=None):
    source=Path(source).resolve(strict=True)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    status=json.loads((source/'status.json').read_text(encoding='utf-8'))
    results=json.loads((source/'results.json').read_text(encoding='utf-8'))
    config=load_config(source/'resolved_config.yaml')
    if not manifest['formal'] or manifest['git']['dirty'] or status['state']!='completed':
        raise ValueError('Audit requires a completed clean formal source run')
    inputs=[source/name for name in ('manifest.json','status.json','results.json','resolved_config.yaml','metrics.json','environment.json')]
    from .input_snapshots import load_snapshot_map
    aliases=load_snapshot_map(source,input_snapshots)
    if input_snapshots is not None:
        inputs.append(Path(input_snapshots))
    identities=manifest['data']['files']+results['saved_outputs']
    for identity in identities:
        actual_path=aliases.get(str(Path(identity['location']).resolve()),Path(identity['location']))
        found=file_identity(actual_path)
        if found['sha256']!=identity['sha256'] or found['size_bytes']!=identity['size_bytes']:
            raise ValueError('Source integrity failure: '+identity['location'])
        inputs.append(actual_path)
    if file_identity(source/'resolved_config.yaml')['sha256']!=manifest['resolved_config_sha256'] or file_identity(source/'environment.json')['sha256']!=manifest['environment_sha256']:
        raise ValueError('Config/environment hash mismatch')
    plan_path=aliases.get(str(Path(config['task_plan']).resolve()),Path(config['task_plan']))
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    source_metrics=json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    verify_source_report(results,plan,config,source_metrics)
    records=results['records']
    if not records or any(r['split'] not in ('train','validation') for r in records):
        raise ValueError('Invalid audit scope; test tasks are prohibited')
    if any(cfg['compute_dtype']!='float64' for cfg in config['method_configs'].values()):
        raise NotImplementedError('Independent replay audit currently requires float64 computation')
    required_outputs={source/(str(i).zfill(3)+'_'+row['split']+ext) for i,row in enumerate(records) for ext in ('.npz','.json')}
    declared_outputs={Path(item['location']).resolve(strict=True) for item in results['saved_outputs']}
    if declared_outputs!=required_outputs or len(declared_outputs)!=len(results['saved_outputs']):
        raise ValueError('Output identity coverage mismatch')
    datasets={s:WindowDataset(config['dataset'],s) for s in {r['split'] for r in records}}
    indexes={s:{e['window_id']:i for i,e in enumerate(d.entries)} for s,d in datasets.items()}
    from .control_replay import verify_task_metadata
    for row in records:
        ds=datasets[row['split']]
        entry=ds.entries[indexes[row['split']][row['window_id']]]
        verify_task_metadata(row,entry,ds._json(entry,'input.json'))
    kwargs=dict(motion_config=MotionConfig(**config['evaluation']['motion']),
                road_config=RoadConfig(**config['evaluation']['road']),
                collision_config=CollisionConfig(**config['evaluation']['collision']))
    effective=dict(schema_version='sumodiff.control_replay.audit.v2',source=file_identity(source/'results.json'),
                   independent_evaluation={k:asdict(v) for k,v in kwargs.items()},
                   allowed_splits=['train','validation'],state_tolerance=1e-8,control_tolerance=1e-8,
                   verify_inverse=True,verify_errors_and_gates=True,verify_task_and_summary=True,
                   historical_input_bundle=file_identity(input_snapshots) if input_snapshots is not None else None)
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'audit':config['seed']},
                     purpose='independent_numpy_control_replay_audit',data_files=sorted(set(inputs+[plan_path])),
                     data_id=manifest['data']['id'],formal=formal) as recorder:
        start=time.perf_counter()
        checked=0;metrics_checked=0;max_error=0.;max_control_error=0.;label_velocity_error=0.;global_counts={}
        verified_rows=[]
        for i,record in enumerate(records):
            if record['status']!='completed':
                raise AssertionError('Source task failed; cannot claim all replay checks')
            ds=datasets[record['split']]
            sample=ds[indexes[record['split']][record['window_id']]]
            c=sample['conditioning'];active=c['agent_mask'].astype(bool)
            label=sample['targets']['future'].astype(float)
            label[...,:2]+=c['initial_positions'][:,None]
            label[~active]=0
            count=int(active.sum())
            if record.get('planned_active_agents') is not None and record['planned_active_agents']!=count:
                raise AssertionError('Planned agent denominator disagrees')
            stem=str(i).zfill(3)+'_'+record['split']
            close_json(json.loads((source/(stem+'.json')).read_text(encoding='utf-8')),record)
            verified={**record,'planned_active_agents':count,'methods':{}}
            original=evaluate_trajectory(c,sample['input_metadata'],sample['exact_map'],label,future_mask=sample['targets']['future_mask'],**kwargs)
            original_comfort=numpy_comfort(c,label,sample['targets']['future_mask'],kwargs['motion_config'])
            with np.load(source/(stem+'.npz'),allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['label_states'],label)
                for field,value in dict(history=c['history'],history_mask=c['history_mask'],initial_positions=c['initial_positions'],
                                        lengths=c['attributes'][:,0],agent_mask=active,future_mask=sample['targets']['future_mask']).items():
                    np.testing.assert_array_equal(saved[field],value)
                for name,value in record['methods'].items():
                    if value['status']!='completed':
                        raise AssertionError('Source method failed')
                    state=label if name=='label' else saved[name+'_states']
                    independent=label
                    if name!='label':
                        cfg=config['method_configs'][name]
                        inverse,lateral,velocity_residual=numpy_inverse(c['history'],c['initial_positions'],c['attributes'][:,0],active,label,cfg)
                        control_error=float(np.abs(inverse-saved[name+'_controls']).max())
                        if control_error>1e-8:
                            raise AssertionError('Independent label control inversion mismatch')
                        max_control_error=max(max_control_error,control_error)
                        np.testing.assert_allclose(saved[name+'_lateral_velocity'],lateral,atol=1e-8,rtol=0)
                        independent=numpy_rollout(c['history'],c['initial_positions'],c['attributes'][:,0],active,saved[name+'_controls'],cfg)
                        error=float(np.abs(independent-state).max())
                        if error>1e-8:
                            raise AssertionError('Independent rollout mismatch')
                        max_error=max(max_error,error);checked+=1
                    metric=original if name=='label' else evaluate_trajectory(c,sample['input_metadata'],sample['exact_map'],state,future_mask=sample['targets']['future_mask'],**kwargs)
                    close_json(metric,value['metrics'])
                    comfort=original_comfort if name=='label' else numpy_comfort(c,state,sample['targets']['future_mask'],kwargs['motion_config'])
                    close_json(comfort,value['comfort_by_agent'])
                    metrics_checked+=1
                    verified_value={**value,'metrics':metric,'comfort_by_agent':comfort}
                    if name!='label':
                        from sumodiff.evaluation.motion import summary
                        errors=numpy_errors(label,independent,active)
                        close_json(errors,value['errors'])
                        gate=numpy_gate(original,metric,original_comfort,comfort,errors,config['gate'])
                        close_json(gate,value['gate'])
                        jerk=numpy_jerk(c,state)
                        np.testing.assert_allclose(saved[name+'_jerk'],jerk,atol=1e-8,rtol=0)
                        close_json({part:summary(np.linalg.norm(jerk[:,sl],axis=-1)[active]) for part,sl in
                                    [('boundary_first4',slice(0,4)),('interior36',slice(4,None)),('terminal4',slice(-4,None))]},value['jerk_mps3'])
                        close_json({f'channel_{k}':summary(inverse[...,k][active]) for k in range(inverse.shape[-1])},value['controls'])
                        residual_summary=summary(np.linalg.norm(velocity_residual,axis=-1)[active])
                        close_json(residual_summary,value['label_velocity_residual_mps'])
                        lateral_summary=summary(np.abs(lateral[active]))
                        close_json(lateral_summary,value['lateral_velocity_mps'])
                        verified_value.update(errors=errors,gate=gate,lateral_velocity_mps=lateral_summary)
                        label_velocity_error=max(label_velocity_error,residual_summary['max'])
                    verified['methods'][name]=verified_value
                    counts=global_counts.setdefault(name,dict(planned=0,planned_active_agents=0,scene_pass=0,motion_pass=0,
                        comfort_pass=0,comfort_agents=0,strict_comfort_confirmed_pass_agents=0,
                        gate_pass=None if name=='label' else 0,reasons=Counter()))
                    counts['planned']+=1;counts['planned_active_agents']+=count
                    counts['scene_pass']+=int(metric['quality_pass'] is True)
                    counts['motion_pass']+=int(metric['motion']['hard_quality_pass'] is True)
                    counts['comfort_pass']+=metric['motion']['comfort_pass_agents']
                    counts['comfort_agents']+=metric['motion']['comfort_assessed_agents']
                    counts['strict_comfort_confirmed_pass_agents']+=sum(v is True for v in comfort.values())
                    if name!='label':
                        counts['gate_pass']+=int(gate['passed']);counts['reasons'].update(gate['reasons'])
            verified_rows.append(verified)
            if (i+1)%14==0:
                print(f'Independent inverse/gate audit {i+1}/{len(records)}',flush=True)
        verify_group_summary(verified_rows,config['method_configs'],results['summary'])
        recomputed_all={name:all(row['methods'][name]['gate']['passed'] for row in verified_rows) for name in config['method_configs']}
        close_json(recomputed_all,results['gate_pass_all_tasks'])
        if results['control_rollout_calls']!=checked:
            raise AssertionError('Source rollout call count mismatch')
        totals=dict(schema_version='sumodiff.control_replay.audit.v2',source_git=manifest['git'],source_results=file_identity(source/'results.json'),
                    verified_input_and_output_identities=len(identities),independent_control_rollouts=checked,
                    independent_control_inversions=checked,independent_error_and_gate_checks=checked,
                    independent_scene_metric_replays=metrics_checked,verified_group_summaries=len(results['summary']),
                    maximum_state_disagreement=max_error,maximum_control_disagreement=max_control_error,
                    maximum_original_label_velocity_roundoff_mps=label_velocity_error,global_counts=global_counts,
                    gate_pass_all_tasks=recomputed_all,elapsed_audit_body_s=time.perf_counter()-start,
                    optimizer_updates=0,free_samples=0,test_arrays_loaded=0)
        write_json(recorder.output/'audit.json',totals)
        recorder.write_metrics(totals)
    return totals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--formal',action='store_true')
    parser.add_argument('--input-snapshots',help='Explicit byte-identical historical input bundle')
    args=parser.parse_args()
    run(args.source,args.output,args.formal,args.input_snapshots)


if __name__=='__main__':
    main()
