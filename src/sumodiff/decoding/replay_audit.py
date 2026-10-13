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


def run(source, output, formal=False):
    source=Path(source).resolve(strict=True)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    status=json.loads((source/'status.json').read_text(encoding='utf-8'))
    results=json.loads((source/'results.json').read_text(encoding='utf-8'))
    config=load_config(source/'resolved_config.yaml')
    if not manifest['formal'] or manifest['git']['dirty'] or status['state']!='completed':
        raise ValueError('Audit requires a completed clean formal source run')
    inputs=[source/name for name in ('manifest.json','status.json','results.json','resolved_config.yaml','metrics.json','environment.json')]
    identities=manifest['data']['files']+results['saved_outputs']
    for identity in identities:
        found=file_identity(identity['location'])
        if found['sha256']!=identity['sha256'] or found['size_bytes']!=identity['size_bytes']:
            raise ValueError('Source integrity failure: '+identity['location'])
        inputs.append(Path(identity['location']))
    if file_identity(source/'resolved_config.yaml')['sha256']!=manifest['resolved_config_sha256'] or file_identity(source/'environment.json')['sha256']!=manifest['environment_sha256']:
        raise ValueError('Config/environment hash mismatch')
    records=results['records']
    if not records or any(r['split'] not in ('train','validation') for r in records):
        raise ValueError('Invalid audit scope; test tasks are prohibited')
    datasets={s:WindowDataset(config['dataset'],s) for s in {r['split'] for r in records}}
    indexes={s:{e['window_id']:i for i,e in enumerate(d.entries)} for s,d in datasets.items()}
    kwargs=dict(motion_config=MotionConfig(**config['evaluation']['motion']),
                road_config=RoadConfig(**config['evaluation']['road']),
                collision_config=CollisionConfig(**config['evaluation']['collision']))
    effective=dict(schema_version='sumodiff.control_replay.audit.v1',source=file_identity(source/'results.json'),
                   independent_evaluation={k:asdict(v) for k,v in kwargs.items()},
                   allowed_splits=['train','validation'],state_tolerance=1e-8)
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'audit':config['seed']},
                     purpose='independent_numpy_control_replay_audit',data_files=sorted(set(inputs)),
                     data_id=manifest['data']['id'],formal=formal) as recorder:
        start=time.perf_counter()
        checked=0;metrics_checked=0;max_error=0.;label_velocity_error=0.;global_counts={}
        for i,record in enumerate(records):
            if record['status']!='completed':
                raise AssertionError('Source task failed; cannot claim all replay checks')
            sample=datasets[record['split']][indexes[record['split']][record['window_id']]]
            c=sample['conditioning'];active=c['agent_mask'].astype(bool)
            label=sample['targets']['future'].astype(float)
            label[...,:2]+=c['initial_positions'][:,None]
            label[~active]=0
            with np.load(source/(str(i).zfill(3)+'_'+record['split']+'.npz'),allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['label_states'],label)
                for name,value in record['methods'].items():
                    if value['status']!='completed':
                        raise AssertionError('Source method failed')
                    state=label if name=='label' else saved[name+'_states']
                    if name!='label':
                        independent=numpy_rollout(c['history'],c['initial_positions'],c['attributes'][:,0],active,saved[name+'_controls'],config['method_configs'][name])
                        error=float(np.abs(independent-state).max())
                        if error>1e-8:
                            raise AssertionError('Independent rollout mismatch')
                        max_error=max(max_error,error);checked+=1
                    metric=evaluate_trajectory(c,sample['input_metadata'],sample['exact_map'],state,future_mask=sample['targets']['future_mask'],**kwargs)
                    close_json(metric,value['metrics'])
                    metrics_checked+=1
                    counts=global_counts.setdefault(name,dict(planned=0,scene_pass=0,motion_pass=0,comfort_pass=0,comfort_agents=0,gate_pass=0,reasons=Counter()))
                    counts['planned']+=1
                    counts['scene_pass']+=int(metric['quality_pass'] is True)
                    counts['motion_pass']+=int(metric['motion']['hard_quality_pass'] is True)
                    counts['comfort_pass']+=metric['motion']['comfort_pass_agents']
                    counts['comfort_agents']+=metric['motion']['comfort_assessed_agents']
                    if name!='label':
                        counts['gate_pass']+=int(value['gate']['passed'])
                        counts['reasons'].update(value['gate']['reasons'])
                        label_velocity_error=max(label_velocity_error,value['label_velocity_residual_mps']['max'])
            if (i+1)%14==0:
                print(f'Independent replay {i+1}/{len(records)}',flush=True)
        totals=dict(source_git=manifest['git'],source_results=file_identity(source/'results.json'),
                    verified_input_and_output_identities=len(identities),independent_control_rollouts=checked,
                    independent_scene_metric_replays=metrics_checked,maximum_state_disagreement=max_error,
                    maximum_original_label_velocity_roundoff_mps=label_velocity_error,global_counts=global_counts,
                    elapsed_audit_body_s=time.perf_counter()-start,optimizer_updates=0,free_samples=0,test_arrays_loaded=0)
        write_json(recorder.output/'audit.json',totals)
        recorder.write_metrics(totals)
    return totals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--formal',action='store_true')
    args=parser.parse_args()
    run(args.source,args.output,args.formal)


if __name__=='__main__':
    main()
