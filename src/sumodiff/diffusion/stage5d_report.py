"""Stage5D epoch150 vs epoch200 paired risk/quality/cost report."""
from pathlib import Path
import sys
import numpy as np
from sumodiff.data.dataset import WindowDataset,write_json
from sumodiff.evaluation.motion import summary
from sumodiff.experiments.recorder import RunRecorder,load_config,file_identity
from .checkpoint import load_checkpoint
from .stage5b_report import generation_rows,paired_interval,committed_lineage,read
from .quality_audit import physical_derivatives
from .schedule import UpdateSchedule,resolve_schedule
from .controlled_training import tree_hash

def extend_quality_rows(directory,rows,aggregate,conditioning):
    """Keep per-scene comparisons distinct from pooled vehicle/frame quantiles."""
    root=Path(directory);boundary=[];comfort_good=0;comfort_total=0;planned_agents=0
    for r in rows:
        source=root/'trajectories'/r['window_id'];m=read(source/'metrics.json')
        motion=m['decoded'].get('motion',{})
        good=motion.get('comfort_pass_agents',0);total=motion.get('comfort_assessed_agents',0)
        selected=int(np.asarray(conditioning[r['window_id']]['agent_mask'],bool).sum());planned_agents+=selected
        r.update(strict_comfort_pass_agents=good,strict_comfort_assessed_agents=total,
                 strict_comfort_rate=good/selected if selected else None,strict_comfort_planned_agents=selected,
                 collision_applicable=selected>=2,
                 decoded_boundary_jerk_p99=None,
                 raw_quality_success=float(m['raw'].get('quality_pass') is True),
                 velocity_correction_max=m.get('corrections',{}).get('velocity_correction_mps',{}).get('max'))
        comfort_good+=good;comfort_total+=total
        if not r['generation_failed'] and (source/'stages.npz').exists():
            with np.load(source/'stages.npz',allow_pickle=False) as a:
                c=conditioning[r['window_id']];_,_,j=physical_derivatives(c,a['decoded_absolute_states'])
                values=np.linalg.norm(j[np.asarray(c['agent_mask'],bool),:4],axis=-1)
            r['decoded_boundary_jerk_p99']=float(np.quantile(values,.99))
            boundary.extend(values.ravel().tolist())
    aggregate['strict_comfort']=dict(pass_agents=comfort_good,assessed_agents=comfort_total,
        planned_agents=planned_agents,unassessed_agents=planned_agents-comfort_total,
        pass_rate=comfort_good/planned_agents if planned_agents else None,
        assessed_pass_rate=comfort_good/comfort_total if comfort_total else None)
    aggregate['pooled']['decoded_boundary_jerk']=summary(boundary)
    applicable=[r for r in rows if r['collision_applicable']]
    unknown=sum(r.get('collision') is None for r in applicable)
    positives=sum(r.get('collision') is True for r in applicable)
    aggregate['counts']['collision_unknown']=unknown
    aggregate['counts']['collision_not_applicable_scenes']=len(rows)-len(applicable)
    aggregate['collision_applicability']=dict(applicable_scenes=len(applicable),
        not_applicable_scenes=len(rows)-len(applicable),unknown_or_failed=unknown,
        confirmed_collision_rate=positives/len(applicable) if applicable else None)
    aggregate['raw_quality_passes']=sum(r['raw_quality_success'] for r in rows)
    return rows,aggregate

def validate_schedule_row(row,training,schedule):
    update=row['step']-training['source_step']
    if row['learning_rate']!=schedule.learning_rate(update) or row['inherited_auxiliary_ramp']!=1.:
        raise ValueError('Recorded continuation schedule or inherited ramp mismatch')
    if training['schema_version']=='sumodiff.decoded.temporal.train.config.v1':
        if row['raw_auxiliary_ramp']!=1. or row['temporal_reweight_ramp']!=schedule.auxiliary_ramp(update):
            raise ValueError('Stage5E temporal reweight or raw inheritance mismatch')
        loss=training['loss'];base=loss['inherited']['unified']['temporal_weight'];rho=schedule.auxiliary_ramp(update)
        expected=(base*.1+rho*(loss['boundary_weight']-base*.1),base*.9+rho*(loss['interior_weight']-base*.9))
        if (row['effective_decoded_boundary_weight'],row['effective_decoded_interior_weight'])!=expected:
            raise ValueError('Stage5E effective temporal coefficients differ from protocol')
    elif row['raw_auxiliary_ramp']!=schedule.auxiliary_ramp(update):
        raise ValueError('Stage5D raw ramp mismatch')


def report(config_path,training_run,sampling_run,output,formal=False,*,_suite=None):
    if _suite is None:
        from . import stage5d as _suite
    setup,source_files,checked_final_training=_suite.setup,_suite.source_files,_suite.checked_final_training
    c,parent,training,sampling=setup(config_path)
    stage='stage5e' if c['schema_version']=='sumodiff.stage5e.suite.config.v1' else 'stage5d'
    parent_name=f"parent{training['source_epochs']}";candidate_name=f"candidate{training['epochs']}"
    root=Path(training_run).resolve(strict=True);generated=Path(sampling_run).resolve(strict=True)
    checkpoint=checked_final_training(c,training,root);identity=file_identity(checkpoint)
    payload=load_checkpoint(checkpoint)
    trace,initial,segments=committed_lineage(root,training['source_step'],training['max_steps'])
    expected=set(payload['metadata']['training_signature']['train_window_ids'])
    schedule=UpdateSchedule(resolve_schedule(training['schedule']))
    for epoch in range(training['source_epochs']+1,training['epochs']+1):
        ids=[wid for r in trace if r['epoch']==epoch for wid in r['window_ids']]
        if len(ids)!=1200 or len(set(ids))!=1200 or set(ids)!=expected:
            raise ValueError('Stage5D epoch must traverse each certified window once')
    for r in trace:
        validate_schedule_row(r,training,schedule)
    if initial['optimizer']!=tree_hash(parent['optimizer']) or initial['optimizer_parameter_states']!=len(parent['optimizer']['state']):
        raise ValueError('Stage5D did not inherit the original AdamW moments')
    if formal and any(not read(p/'manifest.json')['formal'] or read(p/'manifest.json')['git']['dirty'] for p in segments):
        raise ValueError('Formal report requires clean formal training lineage')
    ds=WindowDataset(c['dataset'],'validation')
    metadata={e['window_id']:e for e in ds.entries}
    cond={e['window_id']:ds.inference(i)['conditioning'] for i,e in enumerate(ds.entries)}
    left,old,oldfiles,noises=generation_rows(c['baseline_run'],cond,metadata)
    if len(noises)!=200:raise ValueError('Parent initial sampling noise incomplete')
    right,new,newfiles,_=generation_rows(generated,cond,metadata,noises)
    if [r['window_id'] for r in left]!=[r['window_id'] for r in right]:raise ValueError('Paired task order mismatch')
    if new['manifest']['checkpoint']['sha256']!=identity['sha256']:raise ValueError('Candidate checkpoint mismatch')
    if load_config(generated/'resolved_config.yaml')['sampling']!=sampling:raise ValueError('Sampling protocol changed')
    if formal and any(not s['manifest']['formal'] or s['manifest']['git']['dirty'] for s in (old,new)):
        raise ValueError('Formal report requires clean formal sampling sources')
    left,old=extend_quality_rows(c['baseline_run'],left,old,cond)
    right,new=extend_quality_rows(generated,right,new,cond)
    keys=('quality_success','decoded_interior_jerk_p99','decoded_boundary_jerk_p99','boundary_jerk_max',
          'strict_comfort_rate','road_violation','route_violation','map_coverage_violation','collision',
          'position_correction_max','velocity_correction_max','sampling_seconds')
    intervals={k:paired_interval(left,right,k,c['bootstrap_seed']) for k in keys}
    files=source_files(config_path,c)+oldfiles+newfiles+[checkpoint,root/'training_summary.json']
    for segment in segments:
        files += [segment/n for n in ('manifest.json','status.json','resolved_config.yaml','steps.jsonl','initial_state.json')]
    effective=dict(schema_version=f'sumodiff.{stage}.report.config.v1',suite=c,training=training,
        training_run=str(root),sampling_run=str(generated),thresholds=sampling['evaluation'],
        comparison=f"epoch{training['source_epochs']} to epoch{training['epochs']}; additional training, temporal supervision and new LR schedule",
        bootstrap='2000 road-stratified episode cluster replicates; paired per-scene means, not pooled quantile CIs')
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'bootstrap':c['bootstrap_seed']},
        purpose=f'{stage}_paired_quality_cost_report',data_files=sorted(set(map(str,files))),checkpoint=checkpoint,formal=formal) as run:
        result=dict(schema_version=f'sumodiff.{stage}.report.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            generation={parent_name:old,candidate_name:new},paired_differences=intervals,training_updates=training['nominal_additional_updates'],
            training_lineage=[str(p) for p in segments],training_initial_state=initial,
            training_summary=read(root/'training_summary.json'),sampling_noise_paired=True,
            convergence_status='not_assessed',attribution='Joint extra-training/loss/LR intervention; no loss-specific attribution.',
            extension_rule='Review both boundary/interior jerk, qualification, strict comfort, roads/routes/collisions and corrections together; no automatic extension.',
            limits='200 validation windows and one training/sampling seed; selected vehicles only, failures remain in planned denominators.')
        write_json(run.output/'paired_summary.json',result)
        write_json(run.output/'scene_rows.json',{parent_name:left,candidate_name:right})
        run.write_metrics(result)
        lines=[f'# {stage.upper()} paired validation','',f"Epoch{training['source_epochs']} vs epoch{training['epochs']}; same 200 tasks, initial noise and DDIM20/eta0.",'',
            '| Model | Quality / 200 | Boundary jerk P99 | Interior jerk P99 | Strict comfort agents | Road / route scenes | Collisions |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name,s in [(parent_name,old),(candidate_name,new)]:
            q=s['counts'];b=s['pooled']['decoded_boundary_jerk'].get('p99');j=s['pooled']['decoded_interior_jerk'].get('p99');co=s['strict_comfort']
            lines.append(f"| {name} | {q['quality_passes']} | {b} | {j} | {co['pass_agents']} / {co['planned_agents']} | {q['road_bad']} / {q['route_bad']} | {q['collision_scenes']} |")
        lines+=['','Paired scene means and episode-cluster uncertainty: paired_summary.json.',
            'Continuous metrics use available pairs; availability and planned counts are reported.',
            'Failed/unqualified generations remain in quality and planned comfort denominators. No score, smoothing, convergence or automatic stage6 claim.',
            'Extra epochs and LR changed alongside loss; this report does not isolate a loss contribution.']
        (run.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

def audit_baseline(config_path,output,formal=False,*,_suite=None):
    """Exercise full reporting with existing parent outputs, no new model run."""
    if _suite is None:
        from . import stage5d as _suite
    setup,source_files=_suite.setup,_suite.source_files
    c,parent,training,sampling=setup(config_path)
    stage='stage5e' if c['schema_version']=='sumodiff.stage5e.suite.config.v1' else 'stage5d'
    ds=WindowDataset(c['dataset'],'validation')
    cond={e['window_id']:ds.inference(i)['conditioning'] for i,e in enumerate(ds.entries)}
    metadata={e['window_id']:e for e in ds.entries}
    rows,aggregate,files,noises=generation_rows(c['baseline_run'],cond,metadata)
    if formal and (not aggregate['manifest']['formal'] or aggregate['manifest']['git']['dirty']):
        raise ValueError('Formal baseline audit needs clean formal sampling sources')
    rows,aggregate=extend_quality_rows(c['baseline_run'],rows,aggregate,cond)
    intervals={k:paired_interval(rows,rows,k,c['bootstrap_seed']) for k in
        ('quality_success','decoded_boundary_jerk_p99','decoded_interior_jerk_p99',
         'strict_comfort_rate','road_violation','route_violation','collision','position_correction_max',
         'velocity_correction_max')}
    if len(noises)!=200 or any(q['mean_difference'] not in (None,0.) for q in intervals.values()):
        raise RuntimeError('Baseline reporting pairing failed')
    effective=dict(schema_version=f'sumodiff.{stage}.report.audit.config.v1',suite=c,
        source_baseline=c['baseline_run'],optimizer_updates=0,new_sampling_tasks=0,
        interpretation=f"Epoch{training['source_epochs']} reporting replay; no candidate training or improvement.")
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],
        {'bootstrap':c['bootstrap_seed']},purpose=f'{stage}_existing_baseline_report_audit',
        data_files=sorted(set(map(str,source_files(config_path,c)+files))),checkpoint=c['parent_checkpoint'],formal=formal) as run:
        result=dict(schema_version=f'sumodiff.{stage}.report.audit.metrics.v1',
            run_sha=run.manifest['git']['commit_sha'],existing_baseline=aggregate,
            self_paired_intervals=intervals,noise_identities=len(noises),optimizer_updates=0,
            new_sampling_tasks=0,quality_or_convergence_claim=False)
        write_json(run.output/'baseline_report_audit.json',result)
        write_json(run.output/'baseline_rows.json',rows);run.write_metrics(result)
    return result
