"""Verify saved stages with independent NumPy least squares and polygon checks."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import shapely
from shapely.geometry import Polygon,box,shape
from sumodiff.data.dataset import WindowDataset,write_json
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def assert_identity(identity):
    actual=file_identity(identity['location'])
    assert actual['sha256']==identity['sha256'] and actual['size_bytes']==identity['size_bytes']


def vector_area(serialized,config):
    geometry=shape(serialized)
    if not geometry.is_valid:
        assert config['geometry_repair']=='bounded_make_valid'
        valid=shapely.make_valid(geometry,method='linework',keep_collapsed=True)
        assert abs(valid.area-geometry.area)<=config['max_repair_area_change_m2']
        assert geometry.hausdorff_distance(valid)<=config['max_repair_hausdorff_m']
        geometry=valid
    assert geometry.is_valid and geometry.area>0
    return geometry.buffer(config['numerical_tolerance_m']) if config['numerical_tolerance_m'] else geometry


def bodies_from_states(states,sizes):
    polygons=np.empty(states.shape[:2],dtype=object)
    for a in range(len(states)):
        for t in range(states.shape[1]):
            x,y=states[a,t,:2];yaw=np.arctan2(states[a,t,4],states[a,t,5])
            l,w=sizes[a]/2
            corners=np.array([[l,w],[-l,w],[-l,-w],[l,-w]])
            rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            polygons[a,t]=Polygon(corners@rotation.T+np.array([x,y]))
    return polygons


def audit(dataset_path,run_path):
    dataset=WindowDataset(dataset_path)
    run_path=Path(run_path).resolve(strict=True)
    manifest=read(run_path/'manifest.json');summary=read(run_path/'validation_summary.json')
    assert manifest['formal'] and not manifest['git']['dirty']
    assert read(run_path/'status.json')['state']=='completed'
    assert summary['run_sha']==manifest['git']['commit_sha']
    assert summary==read(run_path/'metrics.json')
    for identity in manifest['data']['files']:assert_identity(identity)
    assert file_identity(run_path/'resolved_config.yaml')['sha256']==manifest['resolved_config_sha256']
    assert file_identity(run_path/'environment.json')['sha256']==manifest['environment_sha256']
    config=load_config(run_path/'resolved_config.yaml')['evaluation']
    rows=read(run_path/'trajectory_index.json')
    assert len(rows)==len(dataset)==summary['windows'] and len({r['window_id'] for r in rows})==len(rows)
    by_id={r['window_id']:r for r in rows}
    points=decoded_windows=skipped=repairs=0
    max_position=max_velocity=max_lstsq_error=max_consistency=0.
    road_frames=route_frames=coverage_frames=frame_contacts=0
    max_jerk=0.
    for sample in (dataset[i] for i in range(len(dataset))):
        c,targets,meta,exact=[sample[k] for k in ('conditioning','targets','input_metadata','exact_map')]
        row=by_id[meta['window_id']]
        for identity in row['files'].values():assert_identity(identity)
        saved=read(row['files']['metrics.json']['location'])
        assert saved['window_id']==meta['window_id'] and saved['split']==meta['split']
        assert saved['guided_stage']['status']=='not_applicable'
        agents=c['agent_mask'];fm=targets['future_mask'];future=targets['future'].astype(np.float64)
        with np.load(row['files']['stages.npz']['location'],allow_pickle=False) as archive:
            arrays={key:archive[key] for key in archive.files}
        np.testing.assert_array_equal(arrays['raw_future_delta'],targets['future'])
        np.testing.assert_array_equal(arrays['future_label_mask'],fm)
        np.testing.assert_array_equal(arrays['agent_mask'],agents)
        assert not any(key.startswith('guided') for key in arrays)
        raw=np.concatenate((future[...,:2]+c['initial_positions'].astype(float)[:,None],future[...,2:]),axis=-1)
        raw=np.where(agents[:,None,None],raw,0.)
        np.testing.assert_array_equal(arrays['raw_absolute_states'],raw)
        full=bool(fm[agents].all())
        if full:
            decoded_windows+=1
            assert saved['decoded_status']=='completed'
            states=arrays['decoded_absolute_states'];cfg=config['decoder'];n=states.shape[1]
            alpha=cfg['position_weight']/cfg['position_unit_m']**2
            beta=cfg['velocity_weight']/cfg['velocity_unit_mps']**2
            # Augmented least squares (SVD), independently of the torch normal solve.
            difference=np.eye(n)-np.eye(n,k=-1)
            matrix=np.concatenate((np.sqrt(alpha)*np.eye(n),np.sqrt(beta)*difference/cfg['dt']),axis=0)
            rhs=np.concatenate((np.sqrt(alpha)*future[agents,:,:2],np.sqrt(beta)*future[agents,:,2:4]),axis=1)
            for i,a in enumerate(np.flatnonzero(agents)):
                solution=np.linalg.lstsq(matrix,rhs[i],rcond=None)[0]
                expected=c['initial_positions'][a]+solution
                max_lstsq_error=max(max_lstsq_error,float(np.abs(expected-states[a,:,:2]).max()))
            expected_velocity=np.diff(np.concatenate((c['initial_positions'][:,None],states[...,:2]),axis=1),axis=1)/cfg['dt']
            max_consistency=max(max_consistency,float(np.abs(expected_velocity[agents]-states[agents,:,2:4]).max()))
            np.testing.assert_allclose(np.linalg.norm(states[agents,:,4:6],axis=-1),1.,atol=1e-12)
            pos_correction=states[...,:2]-raw[...,:2]
            velocity_correction=states[...,2:4]-raw[...,2:4]
            np.testing.assert_allclose(arrays['position_correction'][agents],pos_correction[agents],atol=2e-14)
            np.testing.assert_allclose(arrays['velocity_correction'][agents],velocity_correction[agents],atol=2e-14)
            pmax=float(np.linalg.norm(pos_correction[agents],axis=-1).max())
            vmax=float(np.linalg.norm(velocity_correction[agents],axis=-1).max())
            assert abs(pmax-saved['corrections']['position_correction_m']['max'])<2e-14
            assert abs(vmax-saved['corrections']['velocity_correction_mps']['max'])<2e-14
            max_position=max(max_position,pmax);max_velocity=max(max_velocity,vmax)
        else:
            skipped+=1
            assert saved['decoded_status']=='not_applicable_incomplete_ground_truth'
            assert 'decoded_absolute_states' not in arrays and saved['decoded'] is None
        for stage in ('raw','decoded') if full else ('raw',):
            report=saved[stage]
            assert report['status']=='completed' and report['target_pair'] is None
            assert report['effective_target_event'] is None
            states=arrays['raw_absolute_states' if stage=='raw' else 'decoded_absolute_states']
            active_states=states[agents];valid=fm[agents] if stage=='raw' else np.ones(active_states.shape[:2],bool)
            polygons=bodies_from_states(active_states,c['attributes'][agents,:2])
            road=vector_area(exact['drivable'],config['road'])
            route_areas=[vector_area(exact['route_corridors'][int(a)],config['road']) for a in np.flatnonzero(agents)]
            crop=box(*meta['map_extent_m']).buffer(config['road']['numerical_tolerance_m'])
            observed_road=int((~shapely.covers(road,polygons[valid])).sum())
            observed_route=sum(int((~shapely.covers(area,polygons[a,valid[a]])).sum()) for a,area in enumerate(route_areas))
            observed_crop=int((~shapely.covers(crop,polygons[valid])).sum())
            assert observed_road==report['roads']['road']['violation_body_frames']
            assert observed_route==report['roads']['route']['violation_body_frames']
            assert observed_crop==report['roads']['map_coverage']['violation_body_frames']
            # Direct third differences, without the evaluator's derivative helper.
            context=np.concatenate((c['history'][agents,:,:2],active_states[...,:2]),axis=1).astype(float)
            context[:,20]=c['initial_positions'][agents]
            mask=np.concatenate((c['history_mask'][agents],valid),axis=1);mask[:,20]=True
            jerk=np.diff(context,n=3,axis=1)/meta['dt_seconds']**3
            jerk_mask=mask[:,:-3]&mask[:,1:-2]&mask[:,2:-1]&mask[:,3:]
            values=np.linalg.norm(jerk[:,18:],axis=-1)[jerk_mask[:,18:]]
            actual=report['motion']['jerk_norm_mps3']
            assert actual['count']==len(values)
            if len(values):assert abs(float(values.max())-actual['max'])<1e-7
            # Positive frame intersection must be recorded even when continuous
            # checks find additional events between frames.
            contacts=0
            for a in range(len(active_states)):
                for b in range(a+1,len(active_states)):
                    both=valid[a]&valid[b]
                    if shapely.intersects(polygons[a,both],polygons[b,both]).any():contacts+=1
            if contacts:assert report['collisions']['groups']['any']['observed_collision']
            if stage=='raw':
                points+=int(valid.sum());road_frames+=observed_road;route_frames+=observed_route;coverage_frames+=observed_crop
                repairs+=len(report['roads']['geometry_repairs']);frame_contacts+=contacts
                max_jerk=max(max_jerk,float(values.max()) if len(values) else 0.)
    assert decoded_windows==summary['decoded_full_label_windows'] and skipped==summary['incomplete_label_decode_skips']
    assert summary['failed_windows']==0 and points==summary['raw_evaluated_body_frames']
    assert road_frames==summary['raw_body_road_violation_frames'] and route_frames==summary['raw_body_route_violation_frames']
    assert coverage_frames==summary['raw_map_coverage_violation_frames'] and repairs==summary['raw_geometry_repair_count']
    assert abs(max_position-summary['max_label_position_correction_m'])<2e-14
    assert abs(max_velocity-summary['max_label_velocity_correction_mps'])<2e-14
    assert abs(max_jerk-summary['raw_max_jerk_mps3'])<1e-7
    assert max_lstsq_error<1e-9 and max_consistency<1e-10
    return dict(schema_version='sumodiff.stage3.audit.v1',source_run_sha=summary['run_sha'],
        windows=len(dataset),decoded_windows=decoded_windows,incomplete_decode_skips=skipped,
        raw_body_frames=points,road_violation_frames=road_frames,route_violation_frames=route_frames,
        map_coverage_violation_frames=coverage_frames,geometry_repairs=repairs,
        observed_future_frame_contact_pairs=frame_contacts,max_raw_jerk_mps3=max_jerk,
        max_numpy_lstsq_state_difference_m=max_lstsq_error,max_saved_velocity_consistency_error_mps=max_consistency,
        limitations=['independent saved-output audit; no generated model trajectories',
                     'frame polygons supplement synthetic swept/interpolated collision tests'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--formal',action='store_true')
    args=p.parse_args()
    source=Path(args.run).resolve(strict=True)
    index=read(source/'trajectory_index.json')
    files=[source/name for name in ('manifest.json','resolved_config.yaml','environment.json','status.json','metrics.json','validation_summary.json','trajectory_index.json')]
    files += [identity['location'] for row in index for identity in row['files'].values()]
    config=dict(schema_version='sumodiff.stage3.audit.config.v1',dataset=str(args.dataset.resolve()),source_run=str(source))
    with RunRecorder(args.output,Path.cwd(),config,[sys.executable,*sys.orig_argv[1:]],{'audit':20261005},
        purpose='stage3_independent_saved_trajectory_audit',data_files=files,formal=args.formal) as recorder:
        report=audit(args.dataset,source)
        report['audit_sha']=recorder.manifest['git']['commit_sha']
        write_json(recorder.output/'audit_summary.json',report);recorder.write_metrics(report)
    print(report)


if __name__=='__main__':main()
