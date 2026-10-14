"""Bounded training-only source comparisons; no dataset curation or model updates."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys,time
import numpy as np
import yaml
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from sumodiff.simulation.config import resolve_config
from sumodiff.simulation.cli import main as collect_main
from sumodiff.data.episodes import load_episode
from sumodiff.data.windows import candidate_ticks,build_window
from sumodiff.data.selection import select_current
from sumodiff.data.quality import audit_window
from sumodiff.data.motion_coverage import raw_tracks,lane_changes,ramp_exit_events,indices_at,future_motion
from sumodiff.data.dataset import write_json
from sumodiff.evaluation.motion import summary
from .stage5a_plan import resolve_plan,scenario,FAMILIES

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def protocol(path):
    c=load_config(path)
    keys={'schema_version','base_plan','seeds','demand_seconds','max_episode_seconds','audit_reference_limit',
          'lateral_accel_mps2','lateral_speed_mps','transition_length_m','eidm_jerkmax_mps3'}
    if set(c)!=keys or c['schema_version']!='sumodiff.source.motion.pilot.config.v1':
        raise ValueError('Unsupported bounded training-only source pilot')
    if len(c['seeds'])!=2 or len(set(c['seeds']))!=2 or any(type(s) is not int or not 0<=s<2**32 for s in c['seeds']):
        raise ValueError('Pilot requires exactly two distinct uint32 seeds')
    if type(c['audit_reference_limit']) is not int or not 8<=c['audit_reference_limit']<=32:
        raise ValueError('Bounded audit limit must be 8..32')
    bounds={'demand_seconds':(1,30),'max_episode_seconds':(100,240),'lateral_accel_mps2':(.05,1),
            'lateral_speed_mps':(.1,1.5),'transition_length_m':(20,100),'eidm_jerkmax_mps3':(.5,5)}
    for k,(lo,hi) in bounds.items():
        if type(c[k]) not in (int,float) or not np.isfinite(c[k]) or not lo<=c[k]<=hi:
            raise ValueError('Out-of-protocol pilot bound: '+k)
    base=resolve_plan(load_config(c['base_plan']))
    if base['schema_version']!='sumodiff.stage5a.config.v2':
        raise ValueError('Pilot requires the frozen geometry-isolated source plan')
    return c,base


def make_cases(c,base):
    cases=[]
    for family in FAMILIES:
        index=6 if family=='unsignalized_intersection' else 0
        if index>=base['train_geometries']:raise ValueError('Source pilot geometry must be training-owned')
        original=scenario(base,family,'train',index,0,pilot=True)
        original['traffic']['end_seconds']=c['demand_seconds']
        original['simulation']['max_seconds']=c['max_episode_seconds']
        variants={'baseline':deepcopy(original)}
        if family!='unsignalized_intersection':
            sub=deepcopy(original)
            sub['driver'].update(lane_change_model='SL2015',lcAccelLat=c['lateral_accel_mps2'],
                                 maxSpeedLat=c['lateral_speed_mps'])
            sub['simulation'].update(lane_change_duration=0.,lateral_resolution=sub['geometry']['lane_width']/4)
            variants['sublane']=sub
            if family=='ramp_merge':
                smooth=deepcopy(sub);smooth['geometry']['transition_length']=c['transition_length_m']
                variants['sublane_transition']=smooth
        else:
            eidm=deepcopy(original);eidm['driver'].update(car_follow_model='EIDM',jerkmax=c['eidm_jerkmax_mps3'])
            variants['eidm']=eidm
        for name,config in variants.items():
            config=resolve_config(config)
            for seed in c['seeds']:
                cases.append(dict(case_id=f'{family}-{name}-s{seed}',family=family,variant=name,
                    seed=seed,split='train',training_geometry_index=index,scenario=config))
    if len(cases)!=14:raise ValueError('Pilot must contain exactly 14 predeclared episodes')
    return cases


def collect(config_path,output,formal):
    c,base=protocol(config_path);cases=make_cases(c,base)
    effective=dict(schema_version='sumodiff.source.motion.collection.run.v1',protocol=c,cases=cases,
                   scope='training-only source pilot; not a formal dataset',optimizer_updates=0)
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'sumo_base':c['seeds'][0]},
        purpose='source_motion_paired_collection',data_files=[config_path,c['base_plan']],formal=formal) as run:
        results=[];(run.output/'configs').mkdir()
        for i,case in enumerate(cases):
            path=run.output/'configs'/(case['case_id']+'.yaml')
            path.write_text(yaml.safe_dump(case['scenario'],sort_keys=True),encoding='utf-8',newline='\n')
            target=run.output/'episodes'/case['case_id']
            args=['collect','--config',str(path),'--output',str(target),'--seed',str(case['seed'])]
            if formal:args.append('--formal')
            start=time.perf_counter();returncode=collect_main(args)
            ep=read(target/'episode_manifest.json') if (target/'episode_manifest.json').exists() else {}
            q=read(target/'quality.json') if (target/'quality.json').exists() else {}
            results.append(dict(case_id=case['case_id'],family=case['family'],variant=case['variant'],seed=case['seed'],
                split='train',returncode=returncode,completed=returncode==0 and ep.get('state')=='completed',
                eligible_for_normal_training=ep.get('eligible_for_normal_training',False),
                episode_manifest=str(target/'episode_manifest.json'),config=file_identity(path),
                elapsed_seconds=time.perf_counter()-start,quality=q.get('normal_traffic_quality_pass')))
            write_json(run.output/'collection_results.json',dict(cases=results,planned_cases=len(cases)))
            print(f"Source pilot {i+1}/{len(cases)} {case['case_id']} code={returncode}",flush=True)
        metrics=dict(planned_cases=14,completed_cases=sum(r['completed'] for r in results),
                     normal_eligible_cases=sum(r['eligible_for_normal_training'] for r in results),optimizer_updates=0)
        run.write_metrics(metrics)
    return metrics


def selected_ticks(ep,config,limit):
    # Both policies use identical times chosen from current global occupancy.
    available=[t for t in candidate_ticks(ep,config) if len(ep.frames[t])>=2]
    if len(available)<=limit:return available
    return [available[i] for i in np.linspace(0,len(available)-1,limit,dtype=int)]


def raw_motion_summary(tracks,events,merge,road,family):
    values=defaultdict(list);event_rows=[]
    for vid,track in tracks.items():
        if len(track['ticks'])<4:continue
        ticks=track['ticks'];valid=np.all(np.diff(np.lib.stride_tricks.sliding_window_view(ticks,4),axis=1)==1,axis=1)
        jt=ticks[3:][valid];pos=track['center']
        jerk=np.linalg.norm(np.diff(pos,n=3,axis=0)/.001,axis=-1)[valid]
        values['all'].extend(jerk.tolist())
        scalar=np.abs(np.diff(track['reported_accel'])[2:]/.1)[valid]
        values['longitudinal_all'].extend(scalar.tolist())
        for kind,event_ticks,lo,hi in [('lane_change', [e['tick'] for e in events.get(vid,[])],-30,30),
                                      ('ramp_exit',merge.get(vid,[]),-80,10)]:
            for tick in event_ticks:
                selected=(jt>=tick+lo)&(jt<=tick+hi);context=jerk[selected]
                complete=len(context)==hi-lo+1
                event_rows.append(dict(vehicle_id=vid,event=kind,tick=tick,
                    context_points=len(context),context_complete=complete,
                    jerk_max_mps3=float(context.max()) if len(context) else None,
                    context_all_below15=bool(np.all(context<=15)) if complete else None))
                values[kind+'_context'].extend(context.tolist())
        if family=='unsignalized_intersection':
            center=next(j['sumo_raw'] for j in road.junctions if j['sumo_raw']['id']=='center')
            focus=np.array([float(center['x']),float(center['y'])])
            near=np.linalg.norm(pos[3:][valid]-focus,axis=-1)<=60
            values['junction60m'].extend(jerk[near].tolist())
            values['longitudinal_junction60m'].extend(scalar[near].tolist())
    return dict(distributions={k:summary(values[k]) for k in ('all','lane_change_context','ramp_exit_context','junction60m','longitudinal_all','longitudinal_junction60m')},
                event_contexts=event_rows)


def flags_for(tracks,events,merge,ids,t0):
    motions=[future_motion(tracks[v],t0) for v in ids]
    lc=[(v,e['tick']) for v in ids for e in events.get(v,[]) if t0<e['tick']<=t0+40]
    me=[(v,t) for v in ids for t in merge.get(v,[]) if t0<t<=t0+40]
    flags=dict(heading5=any(m and m['heading_span_rad']>=np.deg2rad(5) for m in motions),
               lateral_half=any(m and m['lateral_excursion_m']>=.5 for m in motions),
               lane_change=bool(lc),ramp_exit=bool(me))
    return flags,lc,me


def audit(config_path,collection,output,formal):
    c,base=protocol(config_path);cases=make_cases(c,base);collection=Path(collection).resolve(strict=True)
    parent=read(collection/'manifest.json');cfg=load_config(collection/'resolved_config.yaml')
    if read(collection/'status.json')['state']!='completed' or cfg['protocol']!=c or cfg['cases']!=cases:
        raise ValueError('Source collection does not match the frozen pilot')
    if file_identity(collection/'resolved_config.yaml')['sha256']!=parent['resolved_config_sha256']:
        raise ValueError('Collection configuration drift')
    if formal and (not parent['formal'] or parent['git']['dirty']):raise ValueError('Formal audit needs clean formal sources')
    records=read(collection/'collection_results.json')['cases']
    if [r['case_id'] for r in records]!=[r['case_id'] for r in cases]:
        raise ValueError('Case denominator/order mismatch')
    files=[config_path,c['base_plan'],*[collection/n for n in ('manifest.json','status.json','resolved_config.yaml','collection_results.json')]]
    for rec,case in zip(records,cases):
        path=Path(rec['episode_manifest']);files += [rec['config']['location']]
        current=file_identity(rec['config']['location'])
        if (current['sha256'],current['size_bytes'])!=(rec['config']['sha256'],rec['config']['size_bytes']):
            raise ValueError('Child source config identity drift')
        if load_config(rec['config']['location'])!=case['scenario']:
            raise ValueError('Child config differs from predeclared case')
        if (rec['family'],rec['variant'],rec['seed'],rec['split'])!=(case['family'],case['variant'],case['seed'],'train'):
            raise ValueError('Child task identity drift')
        if path.exists():
            m=read(path);files += [path,path.parent/'manifest.json',path.parent/'status.json',path.parent/'resolved_config.yaml']
            files += [x['location'] for x in m['outputs']]
            if 'scene' in m:files += [x['location'] for x in m['scene']['files']]
    effective=dict(schema_version='sumodiff.source.motion.audit.run.v1',protocol=c,collection=file_identity(collection/'collection_results.json'),
        policies=['first_current_id','interaction_current'],reference_times='at most 32 evenly spaced active ticks with current >=2 cars',
        quality=base['evaluation'],scope='train pilot only; unassessed quality is unknown',optimizer_updates=0)
    start=time.perf_counter()
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'audit':c['seeds'][0]},
        purpose='source_motion_fixed_quality_audit',data_files=sorted(set(map(str,files))),formal=formal) as run:
        results=[];windows=[]
        for i,rec in enumerate(records):
            if not rec['completed'] or not rec['eligible_for_normal_training']:
                results.append(dict(case_id=rec['case_id'],status='failed',reason='collection_failed_or_normal_ineligible',
                                    family=rec['family'],variant=rec['variant'],seed=rec['seed']))
                continue
            child=Path(rec['episode_manifest']).parent
            cm=read(child/'manifest.json');cc=load_config(child/'resolved_config.yaml')
            if file_identity(child/'resolved_config.yaml')['sha256']!=cm['resolved_config_sha256']:
                raise ValueError('Child resolved source configuration drift')
            if formal and (not cm['formal'] or cm['git']['dirty'] or read(child/'status.json')['state']!='completed'):
                raise ValueError('Child collection is not clean formal completed')
            expected=deepcopy(next(x['scenario'] for x in cases if x['case_id']==rec['case_id']))
            expected['runtime']['sumo_home']=cc['scenario']['runtime']['sumo_home']
            if cc['scenario']!=expected or cc['recording']['seed']!=rec['seed']:
                raise ValueError('Child effective source protocol drift')
            ep=load_episode(rec['episode_manifest'],'train')
            if ep.manifest['seed']!=rec['seed'] or ep.manifest['family']!=rec['family']:
                raise ValueError('Child raw episode identity drift')
            from sumodiff.geometry.maps import RoadMap
            road=RoadMap.read(ep.network_path)
            tracks,_=raw_tracks(Path(rec['episode_manifest']).parent/'frames.jsonl')
            events=lane_changes(Path(rec['episode_manifest']).parent/'lanechanges.xml')
            merge={v:ramp_exit_events(t) for v,t in tracks.items()}
            common=selected_ticks(ep,base['window'],c['audit_reference_limit'])
            per_policy={}
            for policy in effective['policies']:
                wc=dict(base['window'],reference_policy=policy)
                raw_counts=Counter();covered_lc=set();covered_merge=set();eligible_lc=set();eligible_merge=set()
                counts=Counter();reject=Counter()
                for t in candidate_ticks(ep,wc):
                    if not ep.frames[t]:continue
                    _,ids,_=select_current(ep.frames[t],rec['family'],road,wc)
                    flags,lc,me=flags_for(tracks,events,merge,ids,t)
                    raw_counts['windows']+=1;raw_counts.update({k+'_windows':int(v) for k,v in flags.items()})
                    covered_lc.update(lc);covered_merge.update(me)
                for t in common:
                    window,reason=build_window(ep,t,road,wc)
                    if reason:raise ValueError('Preplanned current reference disappeared: '+reason)
                    row=audit_window(window,ep,road,base)
                    ids=[v for v in window.input_metadata['agent_ids'] if v is not None]
                    flags,lc,me=flags_for(tracks,events,merge,ids,t)
                    counts['assessed_windows']+=1;counts['core_complete']+=row['core_complete'];counts['eligible']+=row['eligible']
                    for flag,value in flags.items():
                        counts[flag+'_assessed_windows']+=int(value)
                        counts[flag+'_eligible_windows']+=int(value and row['eligible'])
                    reject.update(row['reasons'])
                    if row['eligible']:eligible_lc.update(lc);eligible_merge.update(me)
                    row.update(policy=policy,case_id=rec['case_id'],seed=rec['seed'],variant=rec['variant'],
                               agent_ids=ids,motion_flags=flags)
                    windows.append(row)
                per_policy[policy]=dict(all_candidates=dict(raw_counts),assessed_quality=dict(counts),
                    rejection_reasons=dict(reject),unique_lane_changes_covered=len(covered_lc),
                    unique_ramp_exits_covered=len(covered_merge),unique_lane_changes_in_eligible_assessed=len(eligible_lc),
                    unique_ramp_exits_in_eligible_assessed=len(eligible_merge),
                    unassessed_candidate_quality='unknown')
            result=dict(case_id=rec['case_id'],family=rec['family'],variant=rec['variant'],seed=rec['seed'],
                status='completed',data_id=ep.key,geometry_id=ep.manifest['geometry_id'],reference_ticks=common,
                normal_traffic_quality_pass=True,policies=per_policy,raw=raw_motion_summary(tracks,events,merge,road,rec['family']))
            results.append(result)
            write_json(run.output/'results_partial.json',dict(cases=results))
            print(f"Source audit {i+1}/14 {rec['case_id']} assessed={len(common)*2}",flush=True)
        with (run.output/'window_quality.jsonl').open('w',encoding='utf-8',newline='\n') as f:
            for row in windows:f.write(json.dumps(row,allow_nan=False)+'\n')
        result=dict(schema_version='sumodiff.source.motion.pilot.results.v1',run_sha=run.manifest['git']['commit_sha'],
            planned_cases=14,completed_cases=sum(r['status']=='completed' for r in results),
            failed_cases=sum(r['status']!='completed' for r in results),cases=results,
            quality_assessed_windows=len(windows),elapsed_seconds=time.perf_counter()-start,
            optimizer_updates=0,validation_episode_arrays_loaded=0,test_episode_arrays_loaded=0,
            quality_thresholds_unchanged=True,formal_dataset_modified=False)
        write_json(run.output/'results.json',result);run.write_metrics(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['collect','audit'])
    p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--collection')
    p.add_argument('--formal',action='store_true');a=p.parse_args()
    if a.action=='collect':
        if a.collection:p.error('collect does not accept --collection')
        collect(a.config,a.output,a.formal)
    else:
        if not a.collection:p.error('audit requires --collection')
        audit(a.config,a.collection,a.output,a.formal)


if __name__=='__main__':main()
