"""Stage5A local-terminal campaign: short pilot, bounded collection and quality preparation."""
import argparse
import json
from pathlib import Path
import sys
import yaml
from sumodiff.data.dataset import write_json
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from sumodiff.simulation.cli import main as collect_one
from .stage5a_plan import resolve_plan,jobs,protocol_id
from .stage5a_data import prepare


def collect_job(job,root,repository,formal,reuse=False):
    config_dir=root/'configs';config_dir.mkdir(parents=True,exist_ok=True)
    config_path=config_dir/(job['job_id']+'.yaml')
    text=yaml.safe_dump(job['scenario'],sort_keys=True)
    if config_path.exists() and config_path.read_text(encoding='utf-8')!=text:
        raise ValueError('Existing campaign scenario changed; choose a new campaign')
    if not config_path.exists(): config_path.write_text(text,encoding='utf-8')
    attempts=sorted((root/'episodes').glob(job['job_id']+'-attempt-*')) if (root/'episodes').exists() else []
    for attempt in attempts:
        manifest=attempt/'episode_manifest.json';status=attempt/'status.json';resolved=attempt/'resolved_config.yaml'
        if manifest.exists() and status.exists() and resolved.exists():
            m=json.loads(manifest.read_text(encoding='utf-8'))
            if json.loads(status.read_text(encoding='utf-8'))['state']=='completed' and m.get('eligible_for_normal_training'):
                if not reuse: raise FileExistsError('Completed episode exists; use --reuse-completed with a fresh controller output')
                cfg=load_config(resolved)
                if cfg['scenario']!={**job['scenario'],'runtime':cfg['scenario']['runtime']} or m['seed']!=job['seed']:
                    raise ValueError('Completed episode does not match the campaign job')
                # Exact runtime home is resolved by the checked collector; all other parameters stay fixed.
                return manifest,True
    attempt=root/'episodes'/f"{job['job_id']}-attempt-{len(attempts)+1:03d}"
    args=['collect','--config',str(config_path),'--output',str(attempt),'--seed',str(job['seed']),'--repository',str(repository)]
    if formal: args.append('--formal')
    if collect_one(args): raise RuntimeError(f'Collection failed; retained attempt={attempt}')
    m=json.loads((attempt/'episode_manifest.json').read_text(encoding='utf-8'))
    if not m.get('eligible_for_normal_training'): raise ValueError(f'Collection is not normal-training eligible: {attempt}')
    return attempt/'episode_manifest.json',False


def _verify_pilot(pilot_path,plan):
    p=Path(pilot_path).resolve(strict=True)
    report=json.loads(p.read_text(encoding='utf-8'))
    if report.get('schema_version')!='sumodiff.stage5a.quality.report.v1' or report.get('profile')!='pilot' or not report.get('passed'):
        raise ValueError('Bulk collection requires a passed pilot label-quality report')
    if report['protocol_id']!=protocol_id(plan): raise ValueError('Pilot quality/scenario protocol changed; run a new pilot')
    manifest=json.loads((p.parent/'manifest.json').read_text(encoding='utf-8'))
    if json.loads((p.parent/'status.json').read_text(encoding='utf-8'))['state']!='completed' or manifest['git']['dirty'] or not manifest['formal']:
        raise ValueError('Bulk collection requires a completed formal clean-code pilot')
    parent=p.parent.parent
    if p.parent.name!='preparation' or not (parent/'manifest.json').exists():
        raise ValueError('Pilot must originate from the short-new-episode pilot entry point')
    parent_manifest=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    if (parent_manifest['purpose']!='stage5a_short_new_episode_pilot' or not parent_manifest['formal'] or
        parent_manifest['git']['dirty'] or
        json.loads((parent/'status.json').read_text(encoding='utf-8'))['state']!='completed' or
        json.loads((parent/'metrics.json').read_text(encoding='utf-8'))!=report):
        raise ValueError('Parent pilot collection/quality run is incomplete or inconsistent')
    return file_identity(p)


def collect_campaign(config_path,plan,campaign,output,repository,command,formal,pilot_report=None,reuse=False,allow_bulk=False):
    if not allow_bulk: raise ValueError('Bulk collection requires --allow-bulk; use the Pilot action first')
    pilot=_verify_pilot(pilot_report,plan)
    schedule=jobs(plan);campaign=Path(campaign).resolve()
    effective=dict(schema_version='sumodiff.stage5a.collection.plan.v1',protocol_id=protocol_id(plan),
        plan=plan,jobs=schedule,split_assigned_before_windowing=True)
    plan_file=campaign/'collection_plan.json'
    if campaign.exists():
        if not reuse or not plan_file.exists(): raise FileExistsError('Existing campaign requires --reuse-completed and its original plan')
        if json.loads(plan_file.read_text(encoding='utf-8'))!=effective: raise ValueError('Campaign plan changed; choose a new campaign')
    with RunRecorder(output,repository,dict(campaign=effective,pilot=pilot,reuse_completed=reuse),command,{'campaign':plan['seed']},
        purpose='stage5a_bounded_bulk_collection',data_files=[config_path,pilot_report],formal=formal) as run:
        if not campaign.exists(): campaign.mkdir(parents=True)
        if not plan_file.exists(): write_json(plan_file,effective)
        source_rows=[];geometries={};reused=0
        for index,job in enumerate(schedule):
            path,old=collect_job(job,campaign,repository,formal,reuse)
            m=json.loads(path.read_text(encoding='utf-8'));g=m['geometry_id']
            if g in geometries and geometries[g]!=job['split']: raise ValueError('Compiled geometry crosses preassigned splits')
            geometries[g]=job['split'];reused+=old
            source_rows.append(dict(manifest=str(path),split=job['split']))
            write_json(run.output/'collection_progress.json',dict(completed_jobs=index+1,total_jobs=len(schedule),reused=reused))
            print(f"collection {index+1}/{len(schedule)} {job['job_id']}",flush=True)
        write_json(campaign/'sources.json',dict(schema_version='sumodiff.sources.v1',episodes=source_rows))
        result=dict(completed_jobs=len(source_rows),reused_jobs=reused,source_registry=file_identity(campaign/'sources.json'),
            geometry_assignments=geometries,windows_not_yet_constructed=True)
        run.write_metrics(result)
    return result


def pilot(config_path,plan,output,repository,command,formal):
    with RunRecorder(output,repository,dict(schema_version='sumodiff.stage5a.pilot.run.v1',plan=plan,jobs=jobs(plan,True)),
        command,{'pilot':plan['seed']},purpose='stage5a_short_new_episode_pilot',data_files=[config_path],formal=formal) as run:
        sources=[]
        for job in jobs(plan,True):
            path,_=collect_job(job,run.output,repository,formal)
            sources.append(dict(manifest=str(path),split=job['split']))
        registry=run.output/'sources.json';write_json(registry,dict(schema_version='sumodiff.sources.v1',episodes=sources))
        result=prepare(config_path,plan,registry,run.output/'preparation',repository,
            [sys.executable,'-m','sumodiff.experiments.stage5a','prepare','--profile','pilot','--config',str(config_path),
             '--sources',str(registry),'--output',str(run.output/'preparation'),*(['--formal'] if formal else [])],formal,True)
        run.write_metrics(result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('pilot','collect','prepare'):
        a=sub.add_parser(action);a.add_argument('--config',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        a.add_argument('--repository',type=Path,default=Path.cwd());a.add_argument('--formal',action='store_true')
        if action=='collect':
            a.add_argument('--pilot-report',type=Path,required=True);a.add_argument('--campaign',type=Path,required=True)
            a.add_argument('--reuse-completed',action='store_true');a.add_argument('--allow-bulk',action='store_true')
        if action=='prepare':
            a.add_argument('--sources',type=Path,required=True);a.add_argument('--profile',choices=['pilot','full'],default='full')
    args=p.parse_args(argv)
    command=[sys.executable,*sys.orig_argv[1:]] if argv is None else [sys.executable,'-m','sumodiff.experiments.stage5a',*argv]
    try:
        plan=resolve_plan(load_config(args.config))
        if args.action=='pilot': result=pilot(args.config,plan,args.output,args.repository,command,args.formal)
        elif args.action=='collect': result=collect_campaign(args.config,plan,args.campaign,args.output,args.repository,command,args.formal,
            args.pilot_report,args.reuse_completed,args.allow_bulk)
        else: result=prepare(args.config,plan,args.sources,args.output,args.repository,command,args.formal,args.profile=='pilot')
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2
    print(json.dumps({k:v for k,v in result.items() if k in ('passed','windows','completed_jobs','elapsed_seconds')},allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
