"""Independent paired A-D validation; failures stay in planned denominators."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from sumodiff.data.dataset import WindowDataset,write_json
from sumodiff.evaluation.motion import summary
from sumodiff.experiments.recorder import RunRecorder,file_identity
from .checkpoint import load_checkpoint
from .quality_audit import physical_derivatives


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def committed_lineage(directory,source_step,end_step):
    """Retain only committed updates before a recovery checkpoint."""
    current=Path(directory);stop=end_step;segments=[];rows={}
    while True:
        if current in segments:raise ValueError('Cyclic training lineage')
        segments.append(current)
        manifest=read(current/'manifest.json')
        cfg=__import__('yaml').safe_load((current/'resolved_config.yaml').read_text(encoding='utf-8'))
        entries=[json.loads(line) for line in (current/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in entries:
            if source_step<row['step']<=stop:
                if row['step'] in rows:raise ValueError('Duplicated committed lineage step')
                rows[row['step']]=row
        parent=cfg.get('resume')
        if parent is None:
            initial=read(current/'initial_state.json');break
        identity=file_identity(parent['location'])
        if identity['sha256']!=parent['sha256']:raise ValueError('Recovery parent checkpoint changed')
        stop=load_checkpoint(parent['location'])['step']
        current=Path(parent['location']).parent
    if set(rows)!=set(range(source_step+1,end_step+1)):
        raise ValueError('Lineage does not contain the complete ten-epoch update budget')
    return [rows[k] for k in sorted(rows)],initial,segments


def paired_interval(left,right,key,seed,iterations=2000):
    if [r['window_id'] for r in left]!=[r['window_id'] for r in right]:
        raise ValueError('Paired task order differs')
    valid=[(a,b) for a,b in zip(left,right) if a.get(key) is not None and b.get(key) is not None]
    if not valid:
        return dict(paired_scenes=0,planned_scenes=len(left),mean_difference=None,ci95=None)
    strata=defaultdict(lambda:defaultdict(list))
    for a,b in valid:
        if (a['family'],a['episode_key'])!=(b['family'],b['episode_key']):
            raise ValueError('Paired clustering metadata differs')
        strata[a['family']][a['episode_key']].append(float(b[key])-float(a[key]))
    generator=np.random.default_rng(seed);samples=np.zeros(iterations)
    observed=np.mean([float(b[key])-float(a[key]) for a,b in valid])
    for family,episodes in strata.items():
        names=list(episodes);weight=sum(len(x) for x in episodes.values())/len(valid)
        for i in range(iterations):
            draw=generator.choice(len(names),len(names),replace=True)
            values=[v for index in draw for v in episodes[names[index]]]
            samples[i]+=weight*np.mean(values)
    return dict(paired_scenes=len(valid),planned_scenes=len(left),mean_difference=float(observed),
        ci95=np.quantile(samples,[.025,.975]).tolist(),
        unit='road-stratified episode clusters, conditional on this validation cohort')


def planned_catalog(catalog, planned_window_ids):
    """Default full validation, or an explicitly declared diagnostic prefix."""
    ids=[r['window_id'] for r in catalog]
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicated generation task catalog')
    if planned_window_ids is None:
        if len(ids)!=200:
            raise ValueError('Final comparison requires all 200 planned tasks')
        return catalog
    plan=list(planned_window_ids)
    if not plan or len(plan)>200 or len(set(plan))!=len(plan):
        raise ValueError('Invalid explicit diagnostic task plan')
    if len(ids) not in (len(plan),200) or ids[:len(plan)]!=plan:
        raise ValueError('Diagnostic catalog must equal the declared plan or its full-validation prefix')
    return catalog[:len(plan)]


def generation_rows(directory,conditionings,metadata,baseline_noise=None,*,planned_window_ids=None):
    root=Path(directory)
    manifest=read(root/'manifest.json');status=read(root/'status.json')['state']
    if status not in ('completed','failed') or not (root/'trajectory_index.json').exists():
        raise ValueError('Generation lacks a final full planned-task catalog')
    catalog=read(root/'trajectory_index.json')
    catalog=planned_catalog(catalog,planned_window_ids)
    rows=[];pool=defaultdict(list);files=[root/name for name in
        ('manifest.json','status.json','resolved_config.yaml','trajectory_index.json','sampling_summary.json')]
    noises={};missing_noise=0
    for item in catalog:
        wid=item['window_id']
        if wid not in metadata:raise ValueError('Generation task outside validation cohort')
        source=root/'trajectories'/wid
        for name,identity in item['files'].items():
            path=source/name
            if file_identity(path)['sha256']!=identity['sha256']:
                raise ValueError('Generation source file changed')
            files.append(path)
        metrics=read(source/'metrics.json');decoded=metrics['decoded'];raw=metrics['raw']
        row=dict(window_id=wid,family=item['family'],episode_key=metadata[wid]['episode_key'],
            quality_success=float(decoded.get('quality_pass') is True),
            quality_unknown=decoded.get('quality_pass') is None,
            generation_failed=item['generation_status']!='completed',
            metric_failed=decoded['status']!='completed',
            motion_pass=decoded.get('motion',{}).get('hard_quality_pass'),
            collision=decoded.get('collisions',{}).get('groups',{}).get('any',{}).get('collision'),
            sampling_seconds=metrics.get('sampling_decoder_seconds'),
            decoded_interior_jerk_p99=None,raw_interior_jerk_p99=None,
            boundary_jerk_max=decoded.get('motion',{}).get('boundary',{}).get('jerk_mps3',{}).get('max'),
            position_correction_max=metrics.get('corrections',{}).get('position_correction_m',{}).get('max'))
        for field in ('road','route','map_coverage'):
            frac=decoded.get('roads',{}).get(field,{}).get('violation_fraction')
            row[field+'_violation']=None if frac is None else frac>0
        with np.load(source/'stages.npz',allow_pickle=False) if (source/'stages.npz').exists() else empty_arrays() as arrays:
            if 'initial_noise' in arrays:
                from .controlled_training import tensor_hash
                import torch
                noises[wid]=tensor_hash(torch.as_tensor(arrays['initial_noise'].copy()))
                if baseline_noise is not None and noises[wid]!=baseline_noise[wid]:
                    raise ValueError('Initial sampling noise differs across arms')
            else:missing_noise+=1
            if 'decoded_absolute_states' in arrays:
                c=conditionings[wid];mask=np.asarray(c['agent_mask'],bool)
                for name in ('raw','decoded'):
                    _,_,jerk=physical_derivatives(c,arrays[name+'_absolute_states'])
                    values=np.linalg.norm(jerk[:,4:],axis=-1)[mask]
                    row[name+'_interior_jerk_p99']=float(np.quantile(values,.99))
                    row[name+'_interior_jerk_max']=float(values.max())
                    pool[name+'_interior_jerk'].extend(values.ravel().tolist())
        rows.append(row)
    counts=dict(planned_scenes=len(rows),quality_passes=sum(r['quality_success'] for r in rows),
        quality_unknown=sum(r['quality_unknown'] for r in rows),
        generation_failures=sum(r['generation_failed'] for r in rows),
        metric_failures=sum(r['metric_failed'] for r in rows),
        motion_bad=sum(r['motion_pass'] is False for r in rows),
        collision_scenes=sum(r['collision'] is True for r in rows),
        collision_unknown=sum(r['collision'] is None for r in rows),
        noise_comparisons_available=len(noises),missing_noise=missing_noise)
    for name in ('road','route','map_coverage'):
        counts[name+'_bad']=sum(r[name+'_violation'] is True for r in rows)
        counts[name+'_unknown']=sum(r[name+'_violation'] is None for r in rows)
    return rows,dict(counts=counts,pooled={k:summary(v) for k,v in pool.items()},
                     source_status=status,manifest=manifest),files,noises


class empty_arrays:
    def __enter__(self):return {}
    def __exit__(self,*args):pass


def report(config_path,arm_runs,output,formal=False):
    from .stage5b import setup,ARMS,source_files
    c,parent,candidates,sampling=setup(config_path)
    dataset=WindowDataset(c['dataset'],'validation')
    metadata={e['window_id']:e for e in dataset.entries}
    conditionings={e['window_id']:dataset.inference(i)['conditioning'] for i,e in enumerate(dataset.entries)}
    files=source_files(config_path,c)
    for i,e in enumerate(dataset.entries):
        files += [dataset._path(e,name) for name in ('conditioning.npz','input.json','map.json')]
    all_rows={};all_summaries={};training={};traces={};initials={}
    base,base_summary,base_files,noise=generation_rows(c['baseline_run'],conditionings,metadata)
    all_rows['parent']=base;all_summaries['parent']=base_summary;files+=base_files
    if len(noise)!=200:raise ValueError('Common baseline noise is incomplete')
    for arm,path in zip(ARMS,arm_runs):
        root=Path(path).resolve(strict=True)
        references=read(root/'arm_outputs.json') if (root/'arm_outputs.json').exists() else None
        train=Path(references['training_run']) if references else root/'train'
        generation=Path(references['sampling_run']) if references else root/'sample'
        if references:files.append(root/'arm_outputs.json')
        if read(train/'status.json')['state']!='completed':
            raise ValueError(f'{arm} training is not complete; use explicit recovery first')
        checkpoint=read(train/'latest_checkpoint.json')['checkpoint']
        payload=load_checkpoint(checkpoint['location'])
        if payload['step']!=33000 or payload['metadata']['training_config']!=candidates[arm]:
            raise ValueError('Only matching ten-epoch final checkpoints belong in this comparison')
        rows,initial,segments=committed_lineage(train,30000,33000)
        traces[arm]=rows;initials[arm]=initial
        for segment in segments:
            files += [segment/name for name in ('manifest.json','resolved_config.yaml','status.json','initial_state.json','steps.jsonl')]
        files += [train/'training_summary.json',checkpoint['location']]
        training[arm]=dict(final_segment=read(train/'training_summary.json'),lineage=[str(s) for s in segments])
        generated,saved,generation_files,_=generation_rows(generation,conditionings,metadata,noise)
        if [r['window_id'] for r in generated]!=[r['window_id'] for r in base]:
            raise ValueError('Sampling task order differs')
        if saved['manifest']['checkpoint']['sha256']!=checkpoint['sha256']:
            raise ValueError('Arm generation does not use its final checkpoint')
        if formal and any(not read(segment/'manifest.json')['formal'] or read(segment/'manifest.json')['git']['dirty'] for segment in segments):
            raise ValueError('Formal report needs clean formal training segments')
        all_rows[arm]=generated;all_summaries[arm]=saved;files+=generation_files
    if any(initials[arm]!=initials['A'] for arm in ARMS):
        raise ValueError('Initial model/AdamW/RNG differed')
    for arm in ARMS:
        for a,b in zip(traces['A'],traces[arm]):
            if any(a[k]!=b[k] for k in ('step','window_ids','timesteps','noise_sha256')):
                raise ValueError('Training pairing differed')
    if formal and any(not row['manifest']['formal'] or row['manifest']['git']['dirty'] for row in all_summaries.values()):
        raise ValueError('Formal report requires clean formal generation sources')
    pairs=[('parent',arm) for arm in ARMS]+[('A','B'),('B','C'),('C','D')]
    intervals={}
    for left,right in pairs:
        intervals[left+'->'+right]={key:paired_interval(all_rows[left],all_rows[right],key,c['bootstrap_seed'])
            for key in ('quality_success','decoded_interior_jerk_p99','boundary_jerk_max','position_correction_max','sampling_seconds')}
    effective=dict(schema_version='sumodiff.stage5b.report.config.v1',suite=c,arm_runs=list(map(str,arm_runs)),
        comparisons=pairs,bootstrap_iterations=2000,bootstrap_unit='road-stratified episodes',
        planned_validation_tasks=200,failures='retained in quality denominator; continuous missing pairs explicitly counted',
        thresholds=sampling['evaluation'])
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'bootstrap':c['bootstrap_seed']},
        purpose='stage5b_paired_ten_epoch_report',data_files=sorted(set(map(str,files))),
        checkpoint=c['parent_checkpoint'],formal=formal) as run:
        result=dict(schema_version='sumodiff.stage5b.report.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            generation=all_summaries,paired_differences=intervals,training=training,
            training_updates_per_arm=3000,training_pairing_equal=True,initial_states_equal=True,
            sampling_noise_paired=True,target_roles=None,
            convergence_status='not_assessed',limits='one parent, one paired sampling seed; conditional validation only')
        write_json(run.output/'paired_summary.json',result);write_json(run.output/'scene_rows.json',all_rows)
        run.write_metrics(result)
        lines=['# Stage5B paired validation','','All arms: parent epoch100 +10; 3000 updates; same AdamW/lr/noise. No test data.',
            '','| Arm | Quality pass / 200 | Decoded interior jerk P99 | Road / route violation scenes | Collision scenes | Failed generation / metrics |',
            '|---|---:|---:|---:|---:|---:|']
        for arm,s in all_summaries.items():
            v=s['counts'];j=s['pooled'].get('decoded_interior_jerk',{}).get('p99')
            lines.append(f"| {arm} | {v['quality_passes']:.0f} | {j if j is not None else 'N/A'} | {v['road_bad']} / {v['route_bad']} | {v['collision_scenes']} | {v['generation_failures']} / {v['metric_failures']} |")
        lines += ['','Paired differences are right minus left. Positive quality is better; negative jerk/correction is better.',
                  '95% intervals resample road-stratified episode clusters; unknown continuous outcomes retain explicit missing-pair counts.',
                  'No convergence declaration or automatic candidate extension. All source runtime SHAs remain in manifests.']
        (run.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
