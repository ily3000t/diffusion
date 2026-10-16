"""Training-only audit of center-midpoint interval controls and candidate scales."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil
import torch
from sumodiff.data.dataset import WindowDataset, write_json
from sumodiff.experiments.recorder import RunRecorder, load_config, file_identity
from .controls import ControlConfig, initial_control_state, inverse_label_controls, rollout_controls
from .replay_audit import numpy_inverse, numpy_rollout, numpy_errors
from .control_replay import verify_task_metadata, physical_jerk

CHANNELS = ('acceleration', 'yaw_rate', 'lateral_velocity')
UNITS = ('m/s2', 'rad/s', 'm/s')


def validate_config(config):
    required={'schema_version','dataset','data_id','expected_windows','expected_families','control','scale_policy','checks','seed'}
    if set(config)!=required or config['schema_version'] not in ('sumodiff.control_scale_audit.protocol.v1','sumodiff.control_scale_audit.protocol.v2'):
        raise ValueError('Unsupported control scale audit protocol')
    if type(config['expected_windows']) is not int or config['expected_windows']<1:
        raise ValueError('Expected training count must be positive')
    if not config['expected_families'] or sum(config['expected_families'].values())!=config['expected_windows'] or any(type(n) is not int or n<1 for n in config['expected_families'].values()):
        raise ValueError('Explicit positive family quotas must match training count')
    if type(config['seed']) is not int or not 0<=config['seed']<2**32:
        raise ValueError('Seed must be uint32')
    control=ControlConfig(**config['control'])
    if (control.reference_point,control.integration_heading,control.lateral_velocity,control.compute_dtype)!=('center','midpoint',True,'float64'):
        raise NotImplementedError('This audit supports only center-midpoint three-control float64')
    policy=config['scale_policy']
    if set(policy)!= {'quantile','groups','minimum_group_windows','candidate_scales'} or not 0<policy['quantile']<1:
        raise ValueError('Invalid scale policy')
    if not policy['groups'] or len(set(policy['groups']))!=len(policy['groups']) or type(policy['minimum_group_windows']) is not int or policy['minimum_group_windows']<1:
        raise ValueError('Explicit unique groups and minimum count are required')
    if set(policy['candidate_scales'])!=set(CHANNELS):
        raise ValueError('Scale grids must match all three control channels')
    for grid in policy['candidate_scales'].values():
        if not grid or any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in grid) or any(b<=a for a,b in zip(grid,grid[1:])):
            raise ValueError('Scale grids must be finite, positive and strictly increasing')
    names={'max_position_roundtrip_m','max_heading_roundtrip_rad','max_numpy_state_difference','max_numpy_control_difference','max_label_velocity_roundoff_mps','two_control_position_tolerance_m','two_control_velocity_tolerance_mps'}
    if set(config['checks'])!=names or any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in config['checks'].values()):
        raise ValueError('Explicit finite positive numeric checks are required')
    return control


def distribution(values):
    x=np.asarray(values,dtype=np.float64).reshape(-1)
    if not len(x):
        return dict(count=0,mean=None,std=None,min=None,max=None,abs_quantiles=None,abs_max=None)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite active distribution')
    q=np.quantile(np.abs(x),[.5,.9,.95,.99,.999])
    return dict(count=len(x),mean=float(x.mean()),std=float(x.std()),min=float(x.min()),max=float(x.max()),
                abs_quantiles={name:float(value) for name,value in zip(('p50','p90','p95','p99','p999'),q)},
                abs_max=float(np.abs(x).max()))


def select_scales(groups, policy):
    chosen,issues={},[]
    for name in policy['groups']:
        if name not in groups or groups[name]['completed_windows']<policy['minimum_group_windows']:
            issues.append('insufficient_group:'+name)
    if issues:
        return dict(available=False,reasons=issues,scales=None)
    for channel in CHANNELS:
        envelope={name:groups[name]['scale_quantiles'][channel] for name in policy['groups']}
        target=max(envelope.values())
        candidate=next((float(v) for v in policy['candidate_scales'][channel] if v>=target),None)
        if candidate is None:
            issues.append('grid_exceeded:'+channel)
        chosen[channel]=dict(scale=candidate,envelope_abs_quantile=target,groups=envelope)
    return dict(available=not issues,reasons=issues,scales=chosen if not issues else None,
                mean_centering=False,clipping=False,selection='maximum training group abs quantile rounded up to declared grid')


def window_controls(c, targets, control, checks):
    active=np.asarray(c['agent_mask'])
    h=np.asarray(c['history']);hm=np.asarray(c['history_mask']);fm=np.asarray(targets['future_mask'])
    if active.dtype!=np.bool_ or hm.dtype!=np.bool_ or fm.dtype!=np.bool_ or h.shape!=(len(active),21,6) or targets['future'].shape!=(len(active),40,6) or len(active)>12 or not active.any():
        raise ValueError('Invalid fixed-horizon state/mask contract')
    if not hm[active].all() or not fm[active].all():
        raise ValueError('Incomplete selected training label; keep failure in audit denominator')
    args=[torch.as_tensor(v) for v in (h,hm,c['initial_positions'],c['attributes'][:,0],active)]
    initial=initial_control_state(*args,control)
    label=np.asarray(targets['future'],np.float64).copy()
    label[...,:2]+=np.asarray(c['initial_positions'],np.float64)[:,None]
    label[~active]=0.
    inverted=inverse_label_controls(torch.as_tensor(label),torch.as_tensor(fm),initial)
    controls=inverted.controls.detach().numpy()
    trajectory=rollout_controls(inverted.controls,initial)
    reconstructed=trajectory.states.detach().numpy()
    independent,_,_=numpy_inverse(h,c['initial_positions'],c['attributes'][:,0],active,label,asdict(control))
    independent_state=numpy_rollout(h,c['initial_positions'],c['attributes'][:,0],active,controls,asdict(control))
    errors=numpy_errors(label,reconstructed,active)
    consistency=dict(position_roundtrip_m=errors['max_position_error_m'],heading_roundtrip_rad=errors['max_heading_error_rad'],
                     numpy_control_difference=float(np.abs(controls-independent).max()),
                     numpy_state_difference=float(np.abs(reconstructed-independent_state).max()),
                     label_velocity_roundoff_mps=float(np.linalg.norm(inverted.label_velocity_residual.detach().numpy(),axis=-1)[active].max()))
    for name,limit in checks.items():
        if name.startswith('max_') and consistency[name[4:]]>limit:
            raise ValueError('Numeric label/control check failed: '+name)
    # Two-control comparison is a STATE reconstruction test, not the full scene gate.
    zero_lateral=controls.copy();zero_lateral[...,2]=0.
    two=numpy_rollout(h,c['initial_positions'],c['attributes'][:,0],active,zero_lateral,asdict(control))
    two_errors=numpy_errors(label,two,active)
    safe_h=np.where(active[:,None,None],h,0.).astype(float)
    safe_p0=np.where(active[:,None],c['initial_positions'],0.).astype(float)
    yaw0=np.arctan2(safe_h[:,-1,4],safe_h[:,-1,5])
    yawprev=np.arctan2(safe_h[:,-2,4],safe_h[:,-2,5])
    turn=np.arctan2(np.sin(yaw0-yawprev),np.cos(yaw0-yawprev))
    mid=yawprev+turn/2
    v0=(safe_p0-safe_h[:,-2,:2])/.1
    r0=(v0*np.stack((-np.sin(mid),np.cos(mid)),-1)).sum(-1)
    lateral=controls[...,2]
    lateral_change=np.diff(np.concatenate((r0[:,None],lateral),axis=1),axis=1)/.1
    speed=trajectory.forward_interval_speed.detach().numpy()
    jerk=physical_jerk(c,label)
    data=dict(controls=controls[active],lateral_change=lateral_change[active],
              initial_speed=initial.forward_interval_speed.detach().numpy()[active],
              forward_speed=speed[active],label_jerk=np.linalg.norm(jerk,axis=-1)[active])
    row=dict(consistency=consistency,two_control_errors=two_errors,
             two_control_state_tolerance_pass=two_errors['max_position_error_m']<=checks['two_control_position_tolerance_m'] and two_errors['max_velocity_error_mps']<=checks['two_control_velocity_tolerance_mps'],
             negative_forward_speed_points=int((speed[active]<-.001).sum()),
             heading_norm_error_max=float(inverted.label_heading_norm_error.detach().numpy()[active].max()),
             controls_abs_max={n:float(np.abs(controls[...,k][active]).max()) for k,n in enumerate(CHANNELS)},
             lateral_change_abs_max=float(np.abs(lateral_change[active]).max()))
    return data,row


def scale_motion_row(selected, selection_schema, protocol_schema):
    if protocol_schema=='sumodiff.control_scale_audit.protocol.v1':
        if 'actual_turning_agents' not in selected:
            raise ValueError('Legacy scale protocol requires legacy turning statistics')
        return selected
    if selection_schema!='sumodiff.stage5a.selection.v3':
        raise ValueError('Motion scale protocol requires authenticated Motion selection v3')
    from sumodiff.data.control_tasks import motion_metadata
    flags,events=motion_metadata(selected)
    return dict(selected,actual_turning_agents=len(events['turn']),motion_flags=flags)


def groups_for(metadata, selected, count):
    groups=['all','family:'+metadata['family']]
    if selected['actual_turning_agents']>0:
        groups.append('turning_windows')
    if metadata['family']=='unsignalized_intersection' and count>=8:
        groups.append('dense_intersection_windows')
        if selected['actual_turning_agents']==0:
            groups.append('dense_no_turn_windows')
    if 'motion_flags' in selected:
        if selected['motion_flags']['lane_change']: groups.append('lane_change_windows')
        if selected['motion_flags']['ramp_exit']: groups.append('ramp_exit_windows')
    return groups


def summarize_groups(records, controls, lateral_change, speed, policy):
    groups={}
    for row in records:
        for name in row['groups']:
            item=groups.setdefault(name,dict(planned_windows=0,completed_windows=0,failed_windows=0,agent_ranges=[]))
            item['planned_windows']+=1
            if row['status']=='completed':
                item['completed_windows']+=1
                item['agent_ranges'].append((row['agent_offset'],row['agent_count']))
            else:item['failed_windows']+=1
    result={}
    for name,item in groups.items():
        indices=np.concatenate([np.arange(start,start+count) for start,count in item['agent_ranges']]) if item['agent_ranges'] else np.array([],int)
        value={k:v for k,v in item.items() if k!='agent_ranges'}
        value.update(agent_windows=len(indices),channels={n:distribution(controls[indices,:,k]) for k,n in enumerate(CHANNELS)},
                     lateral_change_mps2=distribution(lateral_change[indices]),forward_speed_mps=distribution(speed[indices]),
                     scale_quantiles={n:float(np.quantile(np.abs(controls[indices,:,k]),policy['quantile'])) if len(indices) else None for k,n in enumerate(CHANNELS)})
        result[name]=value
    return result


def run(config_path, output, formal=False):
    config_path=Path(config_path).resolve(strict=True)
    config=load_config(config_path);control=validate_config(config)
    torch.set_num_threads(1)
    ds=WindowDataset(config['dataset'],'train')
    manifest_path=ds.directory/'dataset_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['data_id']!=config['data_id'] or len(ds)!=config['expected_windows']:
        raise ValueError('Declared training dataset ID or cohort size mismatch')
    selection_identity=manifest['curation']['selection']
    selection_path=Path(selection_identity['location']).resolve(strict=True)
    if file_identity(selection_path)['sha256']!=selection_identity['sha256']:
        raise ValueError('Frozen training selection integrity failure')
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    selected={r['window_id']:r for r in selection['selected'] if r['split']=='train'}
    if len(selected)!=len(ds) or set(selected)!={e['window_id'] for e in ds.entries}:
        raise ValueError('Selected training cohort differs from dataset index')
    sources=[config_path,manifest_path,ds.directory/'windows.json',selection_path]
    tasks=[];counts=Counter()
    for entry in ds.entries:
        row=scale_motion_row(selected[entry['window_id']],selection.get('schema_version'),config['schema_version'])
        metadata=ds._json(entry,'input.json')
        verify_task_metadata(row,entry,metadata)
        if not entry['core_training_eligible'] or not row['core_complete'] or not row['eligible']:
            raise ValueError('Declared training cohort contains an ineligible window')
        if metadata['dt_seconds']!=.1:
            raise ValueError('Invalid physical interval')
        counts[metadata['family']]+=1
        tasks.append(dict(window_id=entry['window_id'],split='train',family=metadata['family'],
                          episode_key=metadata['episode_key'],geometry_id=metadata['geometry_id'],
                          actual_turning_agents=row['actual_turning_agents'],selected_agents=row['selected_agents'],
                          **(dict(motion_flags=row['motion_flags']) if config['schema_version'].endswith('.v2') else {})))
        sources.extend(ds._path(entry,n) for n in ('conditioning.npz','targets.npz','input.json'))
    if dict(counts)!=config['expected_families']:
        raise ValueError('Training family quotas differ from protocol')
    effective={**config,'control':asdict(control),'split':'train','torch_threads':1,
               'loaded_conditioning_keys':['history','history_mask','initial_positions','attributes','agent_mask'],
               'scope':'training label/control statistics; no validation or test arrays, model or optimizer',
               'quantile_weighting':'valid agent-time elements; overlapping windows are not independent population observations'}
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],
                     {'audit_protocol':config['seed']},purpose='training_control_scale_audit',
                     data_files=sorted(set(sources)),data_id=config['data_id'],formal=formal) as recorder:
        start=time.perf_counter();rss=psutil.Process().memory_info().rss;max_rss=rss
        write_json(recorder.output/'tasks.json',dict(schema_version='sumodiff.control_scale_audit.tasks.v1',tasks=tasks))
        blocks=[];lateral_blocks=[];speed_blocks=[];initial_blocks=[];jerk_blocks=[]
        records=[];offset=0
        for i,(entry,task) in enumerate(zip(ds.entries,tasks)):
            row={**task,'status':'failed','agent_count':task['selected_agents'],
                 'groups':groups_for(task,task,task['selected_agents'])}
            try:
                with np.load(ds._path(entry,'conditioning.npz'),allow_pickle=False) as z:
                    c={name:z[name] for name in effective['loaded_conditioning_keys']}
                with np.load(ds._path(entry,'targets.npz'),allow_pickle=False) as z:
                    targets={name:z[name] for name in ('future','future_mask')}
                if int(c['agent_mask'].sum())!=task['selected_agents']:
                    raise ValueError('Planned agent count differs from conditioning')
                data,result=window_controls(c,targets,control,config['checks'])
                count=len(data['controls'])
                row.update(result,status='completed',agent_offset=offset,agent_count=count)
                blocks.append(data['controls']);lateral_blocks.append(data['lateral_change']);speed_blocks.append(data['forward_speed'])
                initial_blocks.append(data['initial_speed']);jerk_blocks.append(data['label_jerk'])
                offset+=count
            except (ValueError,TypeError,RuntimeError,OSError,KeyError) as exc:
                row['error']=dict(type=type(exc).__name__,message=str(exc))
            records.append(row)
            max_rss=max(max_rss,psutil.Process().memory_info().rss)
            if (i+1)%500==0 or i+1==len(tasks):
                print(f'Training scale audit {i+1}/{len(tasks)}, failed {sum(r["status"]=="failed" for r in records)}',flush=True)
        controls=np.concatenate(blocks) if blocks else np.zeros((0,40,3),float)
        lateral=np.concatenate(lateral_blocks) if lateral_blocks else np.zeros((0,40),float)
        speed=np.concatenate(speed_blocks) if speed_blocks else np.zeros((0,40),float)
        initial=np.concatenate(initial_blocks) if initial_blocks else np.zeros(0,float)
        jerk=np.concatenate(jerk_blocks) if jerk_blocks else np.zeros((0,40),float)
        groups=summarize_groups(records,controls,lateral,speed,config['scale_policy'])
        candidates=select_scales(groups,config['scale_policy'])
        failures=sum(r['status']!='completed' for r in records)
        candidates['certified_training_statistics']=failures==0 and candidates['available']
        candidates['training_integration_implemented']=False
        candidates['legacy_checkpoint_compatible']=False
        candidates['control_definition']=asdict(control)
        candidates['data_id']=config['data_id']
        candidates['source_run_sha']=recorder.manifest['git']['commit_sha']
        candidates['source_run_formal']=formal
        candidates['dataset_manifest_sha256']=file_identity(manifest_path)['sha256']
        candidates['window_index_sha256']=manifest['window_index']['sha256']
        candidates['quantile_weighting']=effective['quantile_weighting']
        if candidates['available']:
            scales=np.array([candidates['scales'][n]['scale'] for n in CHANNELS])
            for name,item in groups.items():
                ids=np.concatenate([np.arange(r['agent_offset'],r['agent_offset']+r['agent_count']) for r in records if r['status']=='completed' and name in r['groups']]) if item['completed_windows'] else np.array([],int)
                x=np.abs(controls[ids]/scales)
                item['normalized_tails']={n:{str(bound):float(np.mean(x[...,k]>bound)) if len(ids) else None for bound in (1,3,5)}for k,n in enumerate(CHANNELS)}
        good=[r for r in records if r['status']=='completed']
        results=dict(schema_version='sumodiff.control_scale_audit.results.v1',planned_windows=len(tasks),
                     completed_windows=len(good),failed_windows=failures,planned_agent_windows=sum(t['selected_agents'] for t in tasks),
                     measured_agent_windows=offset,measured_agent_time_elements=offset*40,groups=groups,candidate_normalization=candidates,
                     families=dict(counts),episode_count=len({t['episode_key'] for t in tasks}),geometry_count=len({t['geometry_id'] for t in tasks}),
                     numeric_maxima={k:max((r['consistency'][k] for r in good),default=None) for k in
                                     ('position_roundtrip_m','heading_roundtrip_rad','numpy_state_difference','numpy_control_difference','label_velocity_roundoff_mps')},
                     two_control_state_tolerance_pass=sum(r['two_control_state_tolerance_pass'] for r in good),
                     two_control_maximum_position_error_m=max((r['two_control_errors']['max_position_error_m'] for r in good),default=None),
                     negative_forward_speed_points=sum(r['negative_forward_speed_points'] for r in good),
                     initial_forward_speed_mps=distribution(initial),
                     label_jerk_mps3={n:distribution(jerk[:,sl]) for n,sl in [('first4',slice(0,4)),('interior36',slice(4,None)),('terminal4',slice(-4,None))]},
                     largest_lateral_cases=sorted(good,key=lambda r:r['controls_abs_max']['lateral_velocity'],reverse=True)[:12],
                     elapsed_scan_and_summary_s=time.perf_counter()-start,process_rss_start_bytes=rss,
                     process_rss_observed_peak_bytes=max(max_rss,psutil.Process().memory_info().rss),
                     cuda_compute_used=False,optimizer_updates=0,denoiser_calls=0,free_samples=0,validation_arrays_loaded=0,test_arrays_loaded=0,
                     interpretation='State recovery and training distribution audit; not full road/collision/comfort evaluation or generated model quality')
        np.savez_compressed(recorder.output/'active_controls.npz',controls=controls,lateral_change=lateral,
                            forward_speed=speed,initial_speed=initial,label_jerk=jerk)
        write_json(recorder.output/'window_results.json',records)
        write_json(recorder.output/'candidate_normalization.json',candidates)
        write_json(recorder.output/'results.json',results)
        recorder.write_metrics(results)
        print(json.dumps(dict(failed=failures,scale_candidates=candidates['scales'])),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--formal',action='store_true')
    args=parser.parse_args()
    run(args.config,args.output,args.formal)


if __name__=='__main__':
    main()
