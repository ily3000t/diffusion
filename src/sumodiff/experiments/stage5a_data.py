"""Streaming episode preprocessing, retained rejects, and a standalone curated dataset."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import psutil
import numpy as np
from sumodiff.data.dataset import save_window, write_json
from sumodiff.data.episodes import load_episode
from sumodiff.data.windows import candidate_ticks, build_window, audit_calibration_episodes
from sumodiff.data.quality import audit_window, select_balanced
from sumodiff.geometry.maps import RoadMap
from sumodiff.experiments.recorder import RunRecorder, file_identity
from .stage5a_plan import FAMILIES, protocol_id


def inspect_sources(registry):
    registry=Path(registry).resolve(strict=True)
    doc=json.loads(registry.read_text(encoding='utf-8'))
    if set(doc)!={'schema_version','episodes'} or doc['schema_version']!='sumodiff.sources.v1':
        raise ValueError('Invalid source registry')
    geos={};ids=set();sources=[];files=[registry]
    for source in doc['episodes']:
        if set(source)!={'manifest','split'} or source['split'] not in ('train','validation'):
            raise ValueError('Stage5A preparation accepts only preassigned train/validation sources')
        path=Path(source['manifest'])
        if not path.is_absolute(): path=registry.parent/path
        path=path.resolve(strict=True);m=json.loads(path.read_text(encoding='utf-8'))
        if m.get('schema_version')!='sumodiff.raw.episode.v1' or m.get('state')!='completed' or not m.get('eligible_for_normal_training') or m.get('diagnostic_only'):
            raise ValueError(f'Ineligible source episode: {path}')
        g=m['geometry_id']
        if g in geos and geos[g]!=source['split']: raise ValueError('Geometry leakage before windowing')
        if m['data_id'] in ids: raise ValueError('Duplicate episode content before windowing')
        geos[g]=source['split'];ids.add(m['data_id'])
        sources.append(dict(manifest=str(path),split=source['split'],geometry_id=g,episode_key=m['data_id'],family=m['family']))
        files.append(path)
        files.extend(path.parent/Path(v['location']).name for v in m['outputs'])
        files.extend(path.parent/'scene'/Path(v['location']).name for v in m['scene']['files'])
    if not sources: raise ValueError('Empty source registry')
    return sources,geos,sorted(set(map(str,files)))


def summarize(rows):
    groups={}
    for split in ('train','validation'):
        groups[split]={}
        for f in FAMILIES:
            subset=[r for r in rows if r['split']==split and r['family']==f]
            errors=[r['coordinate_errors'] for r in subset if r.get('coordinate_errors') is not None]
            peaks=[r['peak_source_attribution'] for r in subset if 'peak_source_attribution' in r]
            groups[split][f]=dict(windows=len(subset),core_complete=sum(r['core_complete'] for r in subset),
              eligible=sum(r['eligible'] for r in subset),rejection_reasons=dict(Counter(x for r in subset for x in r['reasons'])),
              max_coordinate_errors={k:max(e[k] for e in errors) if errors else None for k in ('position_m','velocity_mps','heading')},
              peak_source_attribution=max(peaks,key=lambda v:v['center_jerk_mps3']) if peaks else None,
              eligible_actual_turning_windows=sum(r.get('actual_turning_agents',0)>0 and r['eligible'] for r in subset),
              eligible_geometries=len({r['geometry_id'] for r in subset if r['eligible']}),
              eligible_episodes=len({r['episode_key'] for r in subset if r['eligible']}))
    return groups


def _dataset_manifest(directory,index,config,run,registry,**extra):
    write_json(directory/'windows.json',index)
    identity=file_identity(directory/'windows.json')
    write_json(directory/'dataset_manifest.json',dict(schema_version='sumodiff.dataset.v1',
        data_id='sha256:'+hashlib.sha256(identity['sha256'].encode()).hexdigest(),
        run_sha=run.manifest['git']['commit_sha'],run_branch=run.manifest['git']['branch'],
        window_index=identity,outputs={'windows.json':identity},source_registry=file_identity(registry),
        window_config=config,label_fields_excluded_from_inference=['future','future_mask','label_metadata'],**extra))


def prepare(config_path, plan, registry, output, repository, command, formal=False, pilot=False):
    sources,geo_splits,files=inspect_sources(registry)
    effective=dict(schema_version='sumodiff.stage5a.prepare.run.v1',protocol_id=protocol_id(plan),plan=plan,
        profile='pilot' if pilot else 'full',sources=sources,geometry_assignments=geo_splits)
    with RunRecorder(output,repository,effective,command,{'selection':plan['selection_seed']},
        purpose='stage5a_label_quality_preparation',data_files=[config_path,*files],formal=formal) as run:
        candidate=run.output/'candidates';candidate.mkdir()
        index=[];rows=[];skipped=[];network_diagnostics=[];start=time.perf_counter();peak_rss=0
        with (run.output/'window_quality.jsonl').open('w',encoding='utf-8') as log:
            for source in sources:
                ep=load_episode(source['manifest'],source['split'])
                audit_calibration_episodes([ep],plan['window'])
                road=RoadMap.read(ep.network_path)
                network_diagnostics.append(dict(episode_key=ep.key,family=source['family'],split=source['split'],
                    geometry_id=ep.manifest['geometry_id'],lane_length_mismatches=[
                        dict(lane_id=l['id'],declared_length_m=l['sumo_length_m'],
                             polyline_length_m=float(np.linalg.norm(np.diff(
                                 np.asarray(l['points_world_m']),axis=0),axis=-1).sum()))
                        for l in road.lanes if abs(l['sumo_length_m']-float(np.linalg.norm(
                            np.diff(np.asarray(l['points_world_m']),axis=0),axis=-1).sum()))>.001]))
                for tick in candidate_ticks(ep,plan['window']):
                    window,reason=build_window(ep,tick,road,plan['window'])
                    if reason:
                        skipped.append(dict(episode_key=ep.key,tick=tick,reason=reason));continue
                    entry=save_window(candidate/'windows'/window.input_metadata['window_id'],window)
                    entry['family']=window.input_metadata['family'];index.append(entry)
                    try: row=audit_window(window,ep,road,plan)
                    except Exception as exc:
                        row=dict(window_id=entry['window_id'],split=entry['split'],family=entry['family'],
                            geometry_id=entry['geometry_id'],episode_key=entry['episode_key'],
                            core_complete=entry['core_training_eligible'],eligible=False,reasons=['audit_failed'],
                            error=dict(type=type(exc).__name__,message=str(exc)))
                    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n')
                log.flush();peak_rss=max(peak_rss,psutil.Process().memory_info().rss)
                print(f"audited {source['split']} {source['family']} episode={ep.key[:20]} windows={len(rows)}",flush=True)
                del ep,road
        write_json(run.output/'skipped_reference_ticks.json',skipped)
        write_json(run.output/'split_audit.json',effective)
        write_json(run.output/'network_diagnostics.json',network_diagnostics)
        _dataset_manifest(candidate,index,plan['window'],run,registry)
        quotas={'train':{f:4 for f in FAMILIES},'validation':{f:2 for f in FAMILIES}} if pilot else {
            'train':plan['train_quotas'],'validation':plan['validation_quotas']}
        selected,selection=select_balanced(rows,quotas,plan['selection_seed'],
            1 if pilot else plan['minimum_geometries_per_family'],
            1 if pilot else plan['minimum_episodes_per_family'],1. if pilot else plan['max_episode_fraction'])
        groups=summarize(rows);audit_failures=sum('audit_failed' in r['reasons'] for r in rows)
        conversion_failures=sum(any(x in r['reasons'] for x in ('coordinate_or_precision_mismatch','world_local_geometry_mismatch')) for r in rows)
        pilot_families={}
        for f in FAMILIES:
            pool=[r for r in rows if r['family']==f and r['core_complete'] and r.get('selected_agents',0)>=plan['minimum_agents']]
            eligible=sum(r['eligible'] for r in pool)
            fraction=eligible/len(pool) if pool else 0.
            pilot_families[f]=dict(core_multiagent=len(pool),eligible=eligible,retention_fraction=fraction,
                passed=eligible>=plan['pilot_minimum_eligible_per_family'] and fraction>=plan['pilot_minimum_retention_fraction'])
        quota_pass=all(v['passed'] for fs in selection.values() for v in fs.values())
        passed=quota_pass and audit_failures==0 and conversion_failures==0 and (
            not pilot or all(v['passed'] for v in pilot_families.values()))
        report=dict(schema_version='sumodiff.stage5a.quality.report.v1',protocol_id=protocol_id(plan),
            run_sha=run.manifest['git']['commit_sha'],profile='pilot' if pilot else 'full',
            by_split_family=groups,selection=selection,pilot_families=pilot_families,
            audit_failures=audit_failures,conversion_failures=conversion_failures,passed=passed,
            windows=len(rows),skipped_reference_ticks=len(skipped),elapsed_seconds=time.perf_counter()-start,
            sampled_peak_rss_bytes=peak_rss,core_semantics='history+future completeness only',
            quality_semantics='raw64 and stored32 full H+F hard motion, whole-body road/route/coverage and resolved no collision',
            retained_rejected_windows=True,validation_independence='episode and geometry disjoint from train; within-episode overlap remains')
        write_json(run.output/'quality_report.json',report);run.write_metrics(report)
        if not passed:
            raise ValueError('Stage5A quality/quota/diversity checks failed; inspect quality_report.json. No repeat/padding fallback.')
        dataset=run.output/'dataset';dataset.mkdir()
        selected_ids={r['window_id'] for r in selected};new_index=[];methods=Counter()
        for entry in index:
            if entry['window_id'] not in selected_ids: continue
            dest=dataset/'windows'/entry['window_id'];dest.mkdir(parents=True)
            identities={}
            for name,identity in entry['files'].items():
                src=candidate/identity['relative_path'];target=dest/name
                try: os.link(src,target);methods['hardlink']+=1
                except OSError: shutil.copy2(src,target);methods['copy']+=1
                identities[name]={**file_identity(target),'relative_path':str(target.relative_to(dataset))}
            new_index.append({**entry,'files':identities})
        selected_document=dict(schema_version='sumodiff.stage5a.selection.v1',profile=report['profile'],
            protocol_id=protocol_id(plan),quality_report=file_identity(run.output/'quality_report.json'),
            quotas=quotas,selection=selection,selected=[{k:r[k] for k in (
                'window_id','split','family','geometry_id','episode_key','eligible','core_complete','reasons','actual_turning_agents')} for r in selected],
            file_materialization=dict(methods),quality_config=plan['evaluation'],no_smoothing_or_clipping=True)
        write_json(dataset/'selection.json',selected_document)
        _dataset_manifest(dataset,new_index,plan['window'],run,registry,
            curation=dict(selection=file_identity(dataset/'selection.json'),source_candidates_manifest=file_identity(candidate/'dataset_manifest.json'),
                          label_quality_filtered=True,inference_conditions_unchanged=True))
        print(f"curated dataset={dataset} train={sum(e['split']=='train' for e in new_index)} validation={sum(e['split']=='validation' for e in new_index)}",flush=True)
    return report
