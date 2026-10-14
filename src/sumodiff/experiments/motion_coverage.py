"""Trace existing training motion from raw episodes through frozen filtering."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from sumodiff.data.dataset import write_json
from sumodiff.data.motion_coverage import (read_quality_rows,lane_changes,raw_tracks,indices_at,
    future_motion,logged_future_events,ramp_exit_events,source_case,add_stage,event_opportunities)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_config(c):
    keys={'schema_version','preparation','expected_data_id','reference_scales','reference_motion','expected_train_episodes',
          'expected_train_candidates','expected_selected_windows','seed','case_limit_per_bucket'}
    if set(c)!=keys or c['schema_version']!='sumodiff.training.motion.coverage.config.v1':
        raise ValueError('Unsupported training-only coverage protocol')
    if any(type(c[k]) is not int or c[k]<1 for k in ('expected_train_episodes','expected_train_candidates','expected_selected_windows','case_limit_per_bucket')):
        raise ValueError('Explicit positive bounds are required')
    if c['case_limit_per_bucket']>3:raise ValueError('Case traces must remain bounded')


def authenticated_json(path,expected):
    content=Path(path).read_bytes()
    if len(content)!=expected['size_bytes'] or hashlib.sha256(content).hexdigest()!=expected['sha256']:
        raise ValueError('Input JSON changed: '+str(path))
    return json.loads(content)


def run(config_path,output,formal=False):
    c=load_config(config_path);validate_config(c)
    prep=Path(c['preparation']).resolve(strict=True);dataset=prep/'dataset'
    manifest=read(dataset/'dataset_manifest.json');selection=read(dataset/'selection.json')
    if manifest['data_id']!=c['expected_data_id']:raise ValueError('Frozen dataset identity mismatch')
    expected_window=dict(dt_seconds=.1,history_points=21,future_points=40,reference_stride_ticks=10,max_agents=12,selection_radius_m=80.)
    if any(manifest['window_config'].get(k)!=v for k,v in expected_window.items()):raise ValueError('Coverage protocol/window semantics mismatch')
    if file_identity(dataset/'selection.json')['sha256']!=manifest['curation']['selection']['sha256']:
        raise ValueError('Frozen selection drift')
    producer=read(prep/'manifest.json')
    if not producer['formal'] or producer['git']['dirty'] or read(prep/'status.json')['state']!='completed':
        raise ValueError('Require completed clean formal preparation')
    trusted={x['location']:x for x in producer['data']['files']}
    selected={r['window_id'] for r in selection['selected'] if r['split']=='train'}
    if len(selected)!=c['expected_selected_windows']:raise ValueError('Selected training denominator mismatch')
    print('Preflight: loading training quality rows and candidate metadata index',flush=True)
    start=time.perf_counter();rss_peak=psutil.Process().memory_info().rss
    rows_by_episode=read_quality_rows(prep/'window_quality.jsonl')
    if sum(map(len,rows_by_episode.values()))!=c['expected_train_candidates']:
        raise ValueError('Training candidate denominator mismatch')
    candidate_manifest=read(prep/'candidates/dataset_manifest.json')
    index_path=prep/'candidates/windows.json'
    if file_identity(index_path)['sha256']!=candidate_manifest['window_index']['sha256']:
        raise ValueError('Candidate index identity mismatch')
    index=read(index_path)
    rss_peak=max(rss_peak,psutil.Process().memory_info().rss)
    inputs={}
    for entry in index:
        if entry['split']=='train':
            inputs[entry['window_id']]={**entry['files']['input.json'],
                'path':str(prep/'candidates'/entry['files']['input.json']['relative_path'])}
    del index
    if set(inputs)!={r['window_id'] for rows in rows_by_episode.values() for r in rows}:
        raise ValueError('Candidate index and quality task sets differ')
    registry_path=Path(manifest['source_registry']['location']).resolve(strict=True)
    if file_identity(registry_path)['sha256']!=manifest['source_registry']['sha256']:
        raise ValueError('Source registry drift')
    sources=[];files=[config_path,prep/'manifest.json',prep/'status.json',prep/'resolved_config.yaml',
        dataset/'dataset_manifest.json',dataset/'selection.json',prep/'window_quality.jsonl',
        prep/'candidates/dataset_manifest.json',index_path,registry_path]
    files += [v['path'] for v in inputs.values()]
    for item in read(registry_path)['episodes']:
        if item['split']!='train':continue
        path=Path(item['manifest'])
        if not path.is_absolute():path=registry_path.parent/path
        path=path.resolve(strict=True);ep=read(path)
        if path.as_posix() not in trusted and str(path) not in trusted:
            raise ValueError('Training source missing from original preparation inputs')
        expected=trusted.get(str(path),trusted.get(path.as_posix()))
        authenticated_json(path,expected)
        if ep['data_id'] not in rows_by_episode or ep['state']!='completed' or not ep['eligible_for_normal_training']:
            raise ValueError('Raw training episode/source mismatch')
        parent=read(path.parent/'manifest.json')
        if not parent['formal'] or parent['git']['dirty']:
            raise ValueError('Raw source did not originate in clean formal collection')
        resolved=path.parent/'resolved_config.yaml'
        if file_identity(resolved)['sha256']!=parent['resolved_config_sha256']:
            raise ValueError('Collection scenario snapshot drift')
        scene=load_config(resolved)['scenario']
        sources.append(dict(path=path,manifest=ep,scenario=scene))
        files += [path,path.parent/'frames.jsonl',path.parent/'lanechanges.xml',
                  path.parent/'manifest.json',resolved]
    if len(sources)!=c['expected_train_episodes']:raise ValueError('Training episode denominator mismatch')
    refscale=Path(c['reference_scales']);refmotion=Path(c['reference_motion'])
    refrows={r['window_id']:r for r in read(refscale/'window_results.json')}
    with np.load(refmotion/'body_motion.npz',allow_pickle=False) as z:
        ref_heading=z['heading_span_rad'];ref_lateral=z['lateral_excursion_m']
    files += [refscale/'window_results.json',refmotion/'body_motion.npz',refmotion/'review.json']
    if set(refrows)!=selected:raise ValueError('Prior selected motion audit cohort differs')
    effective=dict(**c,scope='training raw episodes and training candidate metadata only; no validation/test arrays',
        raw_source_bytes=sum((s['path'].parent/'frames.jsonl').stat().st_size for s in sources),
        heading_threshold_degrees=5.,lateral_excursion_threshold_m=.5,
        logged_lane_change_semantics='SUMO change timestamp in (t0,t0+4s] for t0-selected IDs',
        ramp_exit_semantics='ramp-route vehicle entering main_out after acceleration; internal links skipped',
        future_motion_semantics='t0-fixed own-body-frame range, only full observed future agents',
        source_preparation_git=producer['git'],quality_thresholds_unchanged=selection['quality_config'])
    print(f"Preflight: {len(sources)} training episodes, {len(inputs)} candidate windows; binding source identities",flush=True)
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'audit':c['seed']},
                     purpose='training_motion_coverage_lineage_diagnosis',data_files=sorted(set(map(str,files))),
                     data_id=manifest['data_id'],formal=formal) as recorder:
        actual={x['location']:x for x in recorder.manifest['data']['files']}
        for source in sources:
            for identity in source['manifest']['outputs']:
                if Path(identity['location']).name not in ('frames.jsonl','lanechanges.xml'):continue
                location=str(source['path'].parent/Path(identity['location']).name)
                if actual[location]['sha256']!=identity['sha256'] or actual[location]['size_bytes']!=identity['size_bytes']:
                    raise ValueError('Raw output differs from authenticated episode: '+location)
        stages={};rejections=defaultdict(lambda:defaultdict(Counter))
        episode_summaries=[];cases=defaultdict(list);measured=0;selected_measured=0
        opportunities=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
        unique_lc=defaultdict(lambda:defaultdict(set));unique_merge=defaultdict(lambda:defaultdict(set))
        maxima=dict(reference_heading_span_difference_rad=0.,reference_lateral_span_difference_m=0.,reference_origin_difference_m=0.)
        with (recorder.output/'window_lineage.jsonl').open('w',encoding='utf-8') as out:
            for ep_index,source in enumerate(sources):
                ep=source['manifest'];family=ep['family'];path=source['path'];key=ep['data_id']
                tracks,frame_count=raw_tracks(path.parent/'frames.jsonl')
                changes=lane_changes(path.parent/'lanechanges.xml')
                merge_events={vid:ramp_exit_events(track) for vid,track in tracks.items()}
                family_stages=stages.setdefault(family,defaultdict(Counter))
                all_events=dict(lane_change=[(v,e['tick']) for v,values in changes.items() for e in values],
                                ramp_exit=[(v,t) for v,values in merge_events.items() for t in values])
                for kind,values in all_events.items():opportunities[family][kind]['all_raw_events'].update((key,*e) for e in values)
                if source['scenario']['simulation']['lane_change_duration']!=3.:
                    raise ValueError('Unexpected collection lane-change duration')
                before=sum(r['core_complete'] for r in rows_by_episode[key])
                epstats=dict(episode_key=key,family=family,geometry_id=ep['geometry_id'],frames=frame_count,
                    vehicles=len(tracks),logged_lane_changes=sum(map(len,changes.values())),
                    lane_change_reasons=dict(Counter(e['reason'] for values in changes.values() for e in values)),
                    ramp_route_vehicles=sum('ramp_in' in t['route_edges'] for t in tracks.values()),
                    ramp_exit_events=sum(map(len,merge_events.values())),
                    vehicles_lifetime_heading_span_over5deg=int(sum(float(np.ptp(np.unwrap(t['yaw'])))>=np.deg2rad(5) for t in tracks.values())),
                    core_windows=before,collection_lane_change_duration_s=3.)
                for row in rows_by_episode[key]:
                    expected=inputs[row['window_id']]
                    meta=authenticated_json(expected['path'],expected)
                    if any(meta[k]!=row[k] for k in ('window_id','family','split','episode_key','geometry_id','reference_tick')):
                        raise ValueError('Candidate metadata/quality lineage mismatch')
                    ids=[v for v in meta['agent_ids'] if v is not None];t0=row['reference_tick']
                    if len(ids)!=row['selected_agents'] or not ids or any(v not in tracks or indices_at(tracks[v],t0,t0) is None for v in ids):
                        raise ValueError('Candidate t0 vehicle identity mismatch')
                    ref=tracks[meta['reference_id']];ri=indices_at(ref,t0,t0)[0]
                    origin_error=float(np.max(np.abs(np.asarray(meta['fixed_frame']['origin_world_m'])-ref['center'][ri])))
                    maxima['reference_origin_difference_m']=max(maxima['reference_origin_difference_m'],origin_error)
                    if origin_error>1e-9:raise ValueError('Raw center and stored reference origin disagree')
                    complete=all(indices_at(tracks[v],t0-21,t0+40) is not None for v in ids)
                    if complete!=row['core_complete']:
                        raise ValueError('Raw lifecycle and stored core completeness differ')
                    for kind,values in all_events.items():
                        observed=event_opportunities(values,tracks,t0,meta['fixed_frame']['origin_world_m'],set(ids))
                        for stage,values in observed.items():opportunities[family][kind][stage].update((key,*e) for e in values)
                    metrics=[future_motion(tracks[v],t0) for v in ids]
                    lc=[(v,e) for v in ids for e in logged_future_events(changes.get(v,[]),t0)]
                    merge=[(v,t) for v in ids for t in merge_events[v] if t0<t<=t0+40]
                    chosen=row['window_id'] in selected
                    if row['eligible'] and (not complete or len(ids)<2) or chosen and not row['eligible']:
                        raise ValueError('Selected/eligible labels violate the frozen quality lineage')
                    flags=add_stage(family_stages,row,metrics,chosen,lc,merge)
                    for stage in ('candidate','core_complete','core_multiagent','eligible','selected'):
                        present=(stage=='candidate' or (stage=='core_complete' and complete) or
                                 (stage=='core_multiagent' and complete and len(ids)>=2) or
                                 (stage=='eligible' and row['eligible']) or (stage=='selected' and chosen))
                        if present:
                            unique_lc[family][stage].update((key,v,e['tick']) for v,e in lc)
                            unique_merge[family][stage].update((key,v,t) for v,t in merge)
                    if not row['eligible']:
                        for flag,value in flags.items():
                            if not value:continue
                            rejections[family][flag].update(row['reasons'])
                            for kind in ('raw64','stored32'):
                                rejections[family][flag].update(kind+':violation:'+k for k,v in row.get(kind+'_violations',{}).items() if v)
                    if chosen:
                        selected_measured+=1;refrow=refrows[row['window_id']]
                        if any(m is None for m in metrics) or refrow['agent_count']!=len(metrics):
                            raise ValueError('Selected cohort future agent partition mismatch')
                        index=refrow['agent_offset'];n=refrow['agent_count']
                        hs=np.asarray([m['heading_span_rad'] for m in metrics]);ls=np.asarray([m['lateral_excursion_m'] for m in metrics])
                        dh=float(np.max(np.abs(hs-ref_heading[index:index+n])));dl=float(np.max(np.abs(ls-ref_lateral[index:index+n])))
                        maxima['reference_heading_span_difference_rad']=max(maxima['reference_heading_span_difference_rad'],dh)
                        maxima['reference_lateral_span_difference_m']=max(maxima['reference_lateral_span_difference_m'],dl)
                        if dh>1e-6 or dl>1e-4 or not np.array_equal(hs>=np.deg2rad(5),ref_heading[index:index+n]>=np.deg2rad(5)) or not np.array_equal(ls>=.5,ref_lateral[index:index+n]>=.5):
                            raise ValueError('Raw selected motion disagrees with prior stored-label coverage')
                    trace=dict(window_id=row['window_id'],family=family,episode_key=key,reference_tick=t0,
                        core_complete=complete,eligible=row['eligible'],selected=chosen,selected_agents=len(ids),
                        full_future_agents=sum(m is not None for m in metrics),unknown_future_agents=sum(m is None for m in metrics),
                        **flags,reasons=row['reasons'],raw64_violations=row.get('raw64_violations'))
                    out.write(json.dumps(trace,allow_nan=False)+'\n')
                    if complete and 'raw64_jerk_max' in row:
                        for flag,value in flags.items():
                            if not value:continue
                            bucket=family+':'+flag+(':'+('eligible' if row['eligible'] else 'rejected'))
                            existing=cases[bucket];limit=c['case_limit_per_bucket']
                            if len(existing)<limit or row['window_id']<existing[-1]['window_id']:
                                case=source_case(row,meta,tracks,changes)
                                if case is None:raise ValueError('Complete audited source case lacks derivative context')
                                error=abs(case['raw_center_jerk_max_mps3']-row['raw64_jerk_max'])
                                if error>1e-5+abs(row['raw64_jerk_max'])*1e-8:raise ValueError('Independent raw jerk disagrees with saved quality audit')
                                existing.append(case);existing.sort(key=lambda x:x['window_id']);del existing[limit:]
                    measured+=1
                episode_summaries.append(epstats);out.flush()
                rss_peak=max(rss_peak,psutil.Process().memory_info().rss)
                print(f"Raw training episodes {ep_index+1}/{len(sources)}; windows {measured}/{c['expected_train_candidates']}",flush=True)
        if measured!=c['expected_train_candidates'] or selected_measured!=c['expected_selected_windows']:
            raise ValueError('Final planned denominator mismatch')
        for family,by_stage in stages.items():
            for stage,item in by_stage.items():
                item['unique_logged_lane_change_events']=len(unique_lc[family][stage])
                item['unique_ramp_exit_events']=len(unique_merge[family][stage])
        result=dict(schema_version='sumodiff.training.motion.coverage.results.v1',
            run_sha=recorder.manifest['git']['commit_sha'],data_id=manifest['data_id'],
            training_episodes=len(sources),planned_candidates=c['expected_train_candidates'],measured_candidates=measured,
            selected_windows=selected_measured,failed_windows=0,stages=stages,rejection_reasons_for_motion=rejections,
            raw_episode_summary=episode_summaries,reference_numeric_maxima=maxima,
            event_opportunities={f:{kind:{stage:len(values) for stage,values in stages.items()} for kind,stages in kinds.items()} for f,kinds in opportunities.items()},
            representative_cases={k:len(v) for k,v in cases.items()},
            scan_including_preflight_seconds=time.perf_counter()-start,
            process_rss_observed_peak_bytes=rss_peak,raw_frame_bytes=effective['raw_source_bytes'],
            validation_episode_arrays_loaded=0,test_episode_arrays_loaded=0,new_collection_runs=0,optimizer_updates=0,
            caveats=['window/event occurrences overlap; unique events reported separately',
                     'candidate motion only summarizes complete-future actors; unknown actors are counted',
                     'ramp exit is a declared edge transition, not a collision or closed-loop success metric',
                     'no filter threshold, existing selection, trajectory or model was modified'])
        write_json(recorder.output/'source_cases.json',dict(cases=cases))
        write_json(recorder.output/'episode_summary.json',episode_summaries)
        write_json(recorder.output/'results.json',result);recorder.write_metrics(result)
        print(json.dumps(dict(stages=stages,reference_numeric_maxima=maxima)),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--formal',action='store_true')
    args=parser.parse_args();run(args.config,args.output,args.formal)


if __name__=='__main__':main()
