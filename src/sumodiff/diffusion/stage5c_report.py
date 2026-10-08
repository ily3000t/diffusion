"""Stage5C parent-versus-candidate paired validation, with unchanged metrics."""
import sys
from pathlib import Path
from sumodiff.data.dataset import WindowDataset,write_json
from sumodiff.experiments.recorder import RunRecorder,file_identity
from .checkpoint import load_checkpoint
from .stage5b_report import generation_rows,paired_interval,committed_lineage,read


def report(config_path,training_run,sampling_run,output,objective='unified',formal=False):
    from .stage5c import setup,source_files
    c,parent,training,sampling=setup(config_path,objective)
    root=Path(training_run).resolve(strict=True);generated=Path(sampling_run).resolve(strict=True)
    if read(root/'status.json')['state']!='completed' or not read(root/'training_summary.json')['full_budget_complete']:
        raise ValueError('Final report requires the completed 50-epoch experiment')
    identity=read(root/'latest_checkpoint.json')['checkpoint']
    if file_identity(identity['location'])['sha256']!=identity['sha256']:
        raise ValueError('Final checkpoint changed')
    payload=load_checkpoint(identity['location'])
    if payload['step']!=45000 or payload['metadata']['training_config']!=training:
        raise ValueError('Report checkpoint differs from the full declared stage5C protocol')
    trace,initial,segments=committed_lineage(root,30000,45000)
    expected_ids=set(payload['metadata']['training_signature']['train_window_ids'])
    for epoch in range(101,151):
        ids=[wid for r in trace if r['epoch']==epoch for wid in r['window_ids']]
        if len(ids)!=1200 or len(set(ids))!=1200 or set(ids)!=expected_ids:
            raise ValueError('Full training epoch did not traverse each certified window once')
    if formal and any(not read(p/'manifest.json')['formal'] or read(p/'manifest.json')['git']['dirty'] for p in segments):
        raise ValueError('Formal comparison requires clean formal training lineage')
    ds=WindowDataset(c['dataset'],'validation')
    metadata={e['window_id']:e for e in ds.entries}
    conditionings={e['window_id']:ds.inference(i)['conditioning'] for i,e in enumerate(ds.entries)}
    left,base,base_files,noises=generation_rows(c['baseline_run'],conditionings,metadata)
    if len(noises)!=200:
        raise ValueError('Original baseline noise incomplete')
    right,new,new_files,_=generation_rows(generated,conditionings,metadata,noises)
    if new['manifest']['checkpoint']['sha256']!=identity['sha256']:
        raise ValueError('Candidate generation is from a different checkpoint')
    from sumodiff.experiments.recorder import load_config
    if load_config(generated/'resolved_config.yaml')['sampling']!=sampling:
        raise ValueError('Candidate sampling settings differ from the baseline')
    if formal and any(not s['manifest']['formal'] or s['manifest']['git']['dirty'] for s in (base,new)):
        raise ValueError('Formal report requires clean formal sampling sources')
    files=source_files(config_path,c)+base_files+new_files+[identity['location'],root/'training_summary.json']
    for segment in segments:
        files += [segment/name for name in ('manifest.json','status.json','resolved_config.yaml','steps.jsonl','initial_state.json')]
    intervals={key:paired_interval(left,right,key,c['bootstrap_seed']) for key in
               ('quality_success','decoded_interior_jerk_p99','boundary_jerk_max','position_correction_max','sampling_seconds')}
    effective=dict(schema_version='sumodiff.stage5c.report.config.v1',suite=c,objective=objective,
        training_run=str(root),sampling_run=str(generated),thresholds=sampling['evaluation'],
        comparisons='parent epoch100 vs candidate epoch150; joint objective/lr/optimizer intervention',
        bootstrap='2000 road-stratified episode-cluster samples; conditional validation only')
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'bootstrap':c['bootstrap_seed']},
        purpose='stage5c_paired_validation',data_files=sorted(set(map(str,files))),checkpoint=identity['location'],formal=formal) as run:
        result=dict(schema_version='sumodiff.stage5c.report.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            generation={'parent':base,objective:new},paired_differences=intervals,training_updates=15000,
            training_lineage=[str(p) for p in segments],training_initial_state=initial,
            training_summary=read(root/'training_summary.json'),sampling_noise_paired=True,convergence_status='not_assessed',
            attribution='Without separately running epsilon_cosine, changes cannot be attributed solely to new losses.',
            limits='200 validation windows; one training/sampling seed; no test-set claim')
        write_json(run.output/'paired_summary.json',result);write_json(run.output/'scene_rows.json',{'parent':left,objective:right})
        run.write_metrics(result)
        lines=['# Stage5C paired validation','',
               '100-epoch parent versus 50 additional epochs; unchanged 200-task evaluation and paired initial noise.','',
               '| Model | Quality / 200 | Interior jerk P99 | Road / route scenes | Collisions | Failures / unknown |',
               '|---|---:|---:|---:|---:|---:|']
        for name,s in [('parent',base),(objective,new)]:
            q=s['counts'];j=s['pooled'].get('decoded_interior_jerk',{}).get('p99')
            lines.append(f"| {name} | {q['quality_passes']} | {j if j is not None else 'N/A'} | {q['road_bad']} / {q['route_bad']} | {q['collision_scenes']} | {q['generation_failures']} / {q['quality_unknown']} |")
        lines += ['','Full paired intervals and source identities are in paired_summary.json.',
            'The schedule and optimizer also changed. A loss-specific claim requires the optional epsilon_cosine control.',
            'Failure cases remain in planned denominators. No automatic convergence, stability or stage6 declaration.']
        (run.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
