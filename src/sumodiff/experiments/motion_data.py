"""Motion-balanced preparation on the original stage5a CLI, with selected-only arrays."""
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib,json,time
from pathlib import Path
import numpy as np
import psutil
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from sumodiff.data.episodes import load_episode
from sumodiff.data.windows import candidate_ticks,build_window,audit_calibration_episodes
from sumodiff.data.dataset import save_window,write_json
from sumodiff.data.quality import audit_window
from sumodiff.data.motion_selection import MotionIndex,select_motion
from sumodiff.geometry.maps import RoadMap
from .stage5a_data import inspect_sources,_dataset_manifest
from .stage5a_plan import jobs,protocol_id,FAMILIES
from .motion_data_plan import requirements,POLICIES

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def verify_sources(sources,plan,pilot,formal,accepted_job_ids=None):
    schedule=jobs(plan,pilot)
    if accepted_job_ids is not None:
        keep=set(accepted_job_ids)
        schedule=[j for j in schedule if j['job_id'] in keep]
        if [j['job_id'] for j in schedule]!=accepted_job_ids:
            raise ValueError('Accepted job population/order differs from outcome ledger')
    if len(sources)!=len(schedule):raise ValueError('Source population differs from the frozen job plan')
    extra=[]
    for source,job in zip(sources,schedule):
        p=Path(source['manifest']);em=read(p)
        run=read(p.parent/'manifest.json');actual=load_config(p.parent/'resolved_config.yaml')
        if file_identity(p.parent/'resolved_config.yaml')['sha256']!=run['resolved_config_sha256']:
            raise ValueError('Collected source configuration hash drift')
        expected=deepcopy(job['scenario'])
        expected['runtime']['sumo_home']=actual['scenario']['runtime']['sumo_home']
        if (actual['scenario']!=expected or em['seed']!=job['seed'] or source['split']!=job['split']
            or source['family']!=job['scenario']['family'] or actual['recording']['seed']!=job['seed']):
            raise ValueError('Source does not match its preassigned motion-data job')
        if formal and (not run['formal'] or run['git']['dirty'] or read(p.parent/'status.json')['state']!='completed'):
            raise ValueError('Formal data need clean completed formal source episodes')
        extra += [p.parent/n for n in ('manifest.json','resolved_config.yaml','status.json')]
    return extra

def window_signature(window):
    digest=hashlib.sha256()
    for name,arrays in [('conditioning',window.conditioning),('targets',window.targets)]:
        for key,value in sorted(arrays.items()):
            a=np.ascontiguousarray(value)
            digest.update(json.dumps([name,key,str(a.dtype),list(a.shape)]).encode());digest.update(a.tobytes())
    digest.update(json.dumps([window.input_metadata,window.label_metadata,window.exact_map],sort_keys=True,allow_nan=False,separators=(',',':')).encode())
    return digest.hexdigest()

def prepare_motion(config_path,plan,registry,output,repository,command,formal=False,pilot=False):
    sources,geos,files,population=inspect_sources(registry,plan=plan,pilot=pilot,formal=formal,return_population=True)
    files+=verify_sources(sources,plan,pilot,formal,population['accepted_job_ids'] if population else None)
    effective=dict(schema_version='sumodiff.motion.data.prepare.v1',protocol_id=protocol_id(plan),plan=plan,
        profile='pilot' if pilot else 'full',sources=sources,geometry_assignments=geos,
        label_motion_used_only_for_offline_selection=True,selected_only_array_materialization=True)
    if population: effective['source_population']=population
    with RunRecorder(output,repository,effective,command,{'selection':plan['selection_seed']},
        purpose='stage5a_motion_data_preparation',data_files=sorted(set(map(str,[config_path,*files]))),formal=formal) as run:
        start=time.perf_counter();rows=[];catalog=[];reasons=Counter();assessed=physical=0;peak=0;coordinate_failures=audit_failures=0
        with (run.output/'window_quality.jsonl').open('w',encoding='utf-8',newline='\n') as log:
            for source in sources:
                ep=load_episode(source['manifest'],source['split'])
                if source['split']!='test':audit_calibration_episodes([ep],plan['window'])
                road=RoadMap.read(ep.network_path);motion=MotionIndex(ep,road);catalog+=motion.catalog
                ticks=plan['motion']['pilot_reference_ticks'] if pilot else candidate_ticks(ep,plan['window'])
                for tick in ticks:
                    for policy in POLICIES:
                        common=dict(episode_key=ep.key,geometry_id=source['geometry_id'],source_manifest=source['manifest'],
                            reference_tick=tick,split=source['split'],family=source['family'],reference_policy=policy)
                        if tick>=len(ep.frames):window,reason=None,'outside_episode'
                        else:window,reason=build_window(ep,tick,road,dict(plan['window'],reference_policy=policy))
                        if reason:
                            row=dict(**common,window_id=f'unavailable:{ep.key}:{tick}:{policy}',eligible=False,core_complete=False,
                                reasons=[reason],quality_status='unavailable',motion_flags=None,motion_event_ids=None,selected_agents=None)
                        else:
                            assessed+=1;window.input_metadata['reference_policy']=policy
                            try:
                                row=audit_window(window,ep,road,plan)
                                ids=[v for v in window.input_metadata['agent_ids'] if v is not None]
                                flags,events=motion.labels(ids,tick,row['core_complete'])
                                row.update(common,quality_status='assessed',motion_flags=flags,motion_event_ids=events)
                                if row['eligible']:row['window_sha256']=window_signature(window)
                            except Exception as exc:
                                row=dict(**common,window_id=window.input_metadata['window_id'],eligible=False,
                                    core_complete=window.label_metadata['core_training_eligible'],reasons=['audit_failed'],
                                    quality_status='failed',motion_flags=None,motion_event_ids=None,
                                    selected_agents=int(window.conditioning['agent_mask'].sum()),
                                    error=dict(type=type(exc).__name__,message=str(exc)))
                        if 'raw64' in row:physical+=1
                        audit_failures+=int('audit_failed' in row['reasons'])
                        coordinate_failures+=int(any(k in row['reasons'] for k in ('coordinate_or_precision_mismatch','world_local_geometry_mismatch')))
                        reasons.update(row['reasons']);log.write(json.dumps(row,allow_nan=False)+'\n')
                        keep=('window_id','split','family','episode_key','geometry_id','source_manifest','reference_tick','reference_policy',
                              'eligible','core_complete','reasons','quality_status','motion_flags','motion_event_ids','selected_agents','window_sha256')
                        rows.append({k:row[k] for k in keep if k in row})
                log.flush();peak=max(peak,psutil.Process().memory_info().rss)
                print(f"motion audited {source['split']} {source['family']} rows={len(rows)}",flush=True)
                del ep,road,motion
        write_json(run.output/'event_catalog.json',catalog)
        selected=[];selection={};splits=('train',) if pilot else ('train','validation','test')
        for split in splits:
            selection[split]={}
            for family in FAMILIES:
                pool=[r for r in rows if r['split']==split and r['family']==family]
                chosen,report=select_motion(pool,requirements(plan,split,family,pilot),plan['selection_seed'])
                selection[split][family]=report;selected+=chosen
        pilot_families={}
        if pilot:
            for family in FAMILIES:
                pool=[r for r in rows if r['family']==family and r['core_complete'] and (r.get('selected_agents') or 0)>=plan['minimum_agents']]
                eligible=sum(r['eligible'] for r in pool);fraction=eligible/len(pool) if pool else None
                pilot_families[family]=dict(core_multiagent=len(pool),eligible=eligible,retention_fraction=fraction,
                    passed=eligible>=plan['pilot_minimum_eligible_per_family'] and fraction is not None and fraction>=plan['pilot_minimum_retention_fraction'])
        passed=all(s['passed'] for fs in selection.values() for s in fs.values()) and not audit_failures and not coordinate_failures and all(v['passed'] for v in pilot_families.values())
        report=dict(schema_version='sumodiff.stage5a.quality.report.v1',data_recipe_schema=plan['schema_version'],
            protocol_id=protocol_id(plan),run_sha=run.manifest['git']['commit_sha'],profile='pilot' if pilot else 'full',
            passed=passed,selection=selection,pilot_families=pilot_families,planned_source_episodes=len(jobs(plan,pilot)),completed_source_episodes=len(sources),
            windows=len(rows),assessed_windows=assessed,complete_quality_evaluations=physical,
            unavailable_windows=sum(r['quality_status']=='unavailable' for r in rows),
            eligible_windows=sum(r['eligible'] for r in rows),rejection_reasons=dict(reasons),
            audit_failures=audit_failures,conversion_failures=coordinate_failures,selected_windows=len(selected),
            elapsed_seconds=time.perf_counter()-start,sampled_peak_rss_bytes=peak,
            event_catalog_counts=dict(Counter(e['kind'] for e in catalog)),
            turn_catalog_semantics='planned maneuver IDs; selected turn labels require observed future heading change on legal internal lanes',
            retained_rejected_windows=True,no_smoothing_or_clipping=True,thresholds_unchanged=True,
            test_usage='none' if pilot else 'frozen construction-only label curation; no parameter/model tuning',
            pilot_limitations='six training episodes and sparse fixed times do not certify full population feasibility',
            candidate_arrays_saved=False,window_reconstruction='original authenticated raw episodes plus logged t0/policy')
        if population:
            report.update(source_population=population,
                accepted_source_episodes=len(sources),
                rejected_source_episodes=population['summary']['rejected_jobs'],
                completed_source_episodes=population['summary']['processed_jobs'])
        write_json(run.output/'quality_report.json',report);run.write_metrics(report)
        if not passed:raise ValueError('Motion data quality/coverage selection failed; reports retained, no quota relaxation or repetition')
        dataset=run.output/'dataset';dataset.mkdir();index=[];by_source=defaultdict(list)
        for row in selected:by_source[row['source_manifest']].append(row)
        for path,subset in sorted(by_source.items()):
            split=subset[0]['split'];ep=load_episode(path,split);road=RoadMap.read(ep.network_path)
            for row in sorted(subset,key=lambda r:r['window_id']):
                window,reason=build_window(ep,row['reference_tick'],road,dict(plan['window'],reference_policy=row['reference_policy']))
                if reason:raise ValueError('Selected window disappeared')
                window.input_metadata['reference_policy']=row['reference_policy']
                if window_signature(window)!=row['window_sha256']:raise ValueError('Selected window reconstruction drift')
                entry=save_window(dataset/'windows'/row['window_id'],window);entry['family']=row['family'];index.append(entry)
        write_json(dataset/'selection.json',dict(schema_version='sumodiff.stage5a.selection.v3',profile=report['profile'],
            protocol_id=protocol_id(plan),selected=selected,selection=selection,
            quality_report=file_identity(run.output/'quality_report.json'),quality_config=plan['evaluation'],
            motion_config=plan['motion'],no_smoothing_or_clipping=True))
        _dataset_manifest(dataset,index,plan['window'],run,registry,
            curation=dict(selection=file_identity(dataset/'selection.json'),source_quality_log=file_identity(run.output/'window_quality.jsonl'),
                label_quality_filtered=True,inference_conditions_unchanged=True,
                required_model_schema='sumodiff.model.control.config.v2',
                **(dict(source_population=population) if population else {})))
    return report
