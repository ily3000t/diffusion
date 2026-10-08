"""Direct and local-terminal entry points for stage5C unified training."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import torch
import yaml
from sumodiff.data.dataset import write_json
from sumodiff.experiments.recorder import RunRecorder,load_config,file_identity
from .checkpoint import load_checkpoint
from .data import dataset_identity
from .controlled_training import shared_cache
from .unified_training import unified_config,train_unified
from .sampling import sample,sample_config

SCHEMA='sumodiff.stage5c.suite.config.v1'
FIELDS={'schema_version','dataset','parent_checkpoint','scale_audit','sampling_config',
        'baseline_run','training_configs','preflight_updates','bootstrap_seed'}
OBJECTIVES=('unified','epsilon_cosine')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def setup(config_path,objective='unified'):
    c=load_config(config_path)
    if set(c)!=FIELDS or c['schema_version']!=SCHEMA or set(c['training_configs'])!=set(OBJECTIVES):
        raise ValueError('Invalid stage5C suite configuration')
    if objective not in OBJECTIVES:
        raise ValueError('Unsupported stage5C objective')
    if type(c['preflight_updates']) is not int or not 2<=c['preflight_updates']<=24 or c['preflight_updates']%2:
        raise ValueError('Preflight needs an even bounded update budget')
    if type(c['bootstrap_seed']) is not int or not 0<=c['bootstrap_seed']<2**32:
        raise ValueError('Invalid comparison seed')
    for name in ('dataset','parent_checkpoint','scale_audit','sampling_config','baseline_run'):
        c[name]=str(Path(c[name]).resolve(strict=True))
    c['training_configs']={k:str(Path(v).resolve(strict=True)) for k,v in c['training_configs'].items()}
    parent=load_checkpoint(c['parent_checkpoint']);identity=file_identity(c['parent_checkpoint'])
    training=unified_config(load_config(c['training_configs'][objective]),parent)
    if training['objective_name']!=objective or training['fork_source_sha256']!=identity['sha256']:
        raise ValueError('Objective mapping or original parent checksum mismatch')
    sampling=sample_config(load_config(c['sampling_config']))
    old=load_config(Path(c['baseline_run'])/'resolved_config.yaml')
    if read(Path(c['baseline_run'])/'status.json')['state']!='completed':
        raise ValueError('Original baseline generation incomplete')
    if old['sampling']!=sampling or sampling['limit']!=200 or sampling['split']!='validation' or sampling['steps']!=20 or sampling['eta']!=0:
        raise ValueError('Final evaluation must preserve the full paired baseline protocol')
    if old['checkpoint']['sha256']!=identity['sha256']:
        raise ValueError('Original baseline checkpoint differs')
    data=dataset_identity(c['dataset'])
    if any(data[k]['sha256']!=old['data'][k]['sha256'] for k in data):
        raise ValueError('Baseline/cohort identity mismatch')
    return c,parent,training,sampling


def source_files(config_path,c):
    return [config_path,c['parent_checkpoint'],c['scale_audit'],c['sampling_config'],*c['training_configs'].values(),
        *[Path(c['baseline_run'])/name for name in ('manifest.json','resolved_config.yaml','status.json',
                                                 'trajectory_index.json','sampling_summary.json')]]


def training_command(c,objective,output,formal,resume=None,prefix_updates=None):
    cmd=[sys.executable,'-m','sumodiff.diffusion.stage5c','train-direct',
         '--dataset',c['dataset'],'--parent-checkpoint',c['parent_checkpoint'],
         '--scale-audit',c['scale_audit'],'--config',c['training_configs'][objective],'--output',str(output)]
    if formal:cmd.append('--formal')
    if resume:cmd += ['--resume',str(resume)]
    if prefix_updates is not None:cmd += ['--prefix-updates',str(prefix_updates)]
    else:cmd.append('--allow-long-run')
    return cmd


def train_one(c,objective,output,formal,resume=None,prefix=None,cache=None):
    cmd=training_command(c,objective,output,formal,resume,prefix)
    return train_unified(c['dataset'],c['parent_checkpoint'],c['training_configs'][objective],c['scale_audit'],
        output,Path.cwd(),cmd,formal,resume,prefix,prefix is None,cache)


def sample_one(c,checkpoint,output,formal,config_path=None):
    config_path=config_path or c['sampling_config']
    cmd=[sys.executable,'-m','sumodiff.diffusion','sample','--dataset',c['dataset'],
         '--checkpoint',str(checkpoint),'--config',str(config_path),'--output',str(output)]
    if formal:cmd.append('--formal')
    return sample(c['dataset'],checkpoint,config_path,output,Path.cwd(),cmd,formal)


def checked_final_training(c,objective,training_run):
    root=Path(training_run).resolve(strict=True)
    summary=read(root/'training_summary.json')
    if read(root/'status.json')['state']!='completed' or not summary['full_budget_complete']:
        raise ValueError('Sampling/report require completed 50-epoch training')
    identity=read(root/'latest_checkpoint.json')['checkpoint']
    if file_identity(identity['location'])['sha256']!=identity['sha256']:
        raise ValueError('Final checkpoint changed')
    payload=load_checkpoint(identity['location'])
    if payload['step']!=45000 or payload['metadata']['training_config']!=unified_config(
        load_config(c['training_configs'][objective]),load_checkpoint(c['parent_checkpoint'])):
        raise ValueError('Wrong final training objective/budget')
    return identity['location']


def preflight(config_path,output,objective='unified',formal=False):
    from .unified_diagnostics import audit_gradients
    from .audit import audit_continuation
    c,parent,training,sampling=setup(config_path,objective)
    started=time.perf_counter()
    effective=dict(schema_version='sumodiff.stage5c.preflight.config.v1',suite=c,objective=objective,
        prefix_updates=c['preflight_updates'],schedule_budget_preserved=15000,
        gradient_audit_updates=16,gradient_audit_ramp=1.,free_generation_tasks=6)
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'audit':training['seed']},
        purpose='stage5c_bounded_preflight',data_files=source_files(config_path,c),checkpoint=c['parent_checkpoint'],formal=formal) as run:
        cache=shared_cache(c['dataset'],training)
        print(f"cache ready: {len(cache[0][2])}/{len(cache[1][2])} windows; {cache[3]:.2f}s",flush=True)
        gradient=audit_gradients(c['dataset'],parent,training,cache,run.output/'gradient_audit.json',16)
        print(f"gradient audit: {gradient['eligible_auxiliary_to_noise_norm_ratio']}",flush=True)
        uninterrupted=run.output/'uninterrupted'
        result=train_one(c,objective,uninterrupted,formal,prefix=c['preflight_updates'],cache=cache)
        mid=uninterrupted/f"checkpoint_step_{30000+c['preflight_updates']//2:06d}.pt"
        resumed=run.output/'resumed'
        second=train_one(c,objective,resumed,formal,resume=mid,prefix=c['preflight_updates'],cache=cache)
        final=read(uninterrupted/'latest_checkpoint.json')['checkpoint']['location']
        restored=read(resumed/'latest_checkpoint.json')['checkpoint']['location']
        cmd=[sys.executable,'-m','sumodiff.diffusion','audit-continuation','--uninterrupted',final,
             '--resumed',restored,'--output',str(run.output/'recovery_audit')]
        if formal:cmd.append('--formal')
        recovery=audit_continuation(final,restored,run.output/'recovery_audit',Path.cwd(),cmd,formal)
        a=load_checkpoint(final);b=load_checkpoint(restored)
        if a['metadata']['runtime']['scheduler_state']!=b['metadata']['runtime']['scheduler_state']:
            raise RuntimeError('Actual scheduler recovery mismatch')
        first=[json.loads(line) for line in (uninterrupted/'steps.jsonl').read_text().splitlines()]
        last=[json.loads(line) for line in (resumed/'steps.jsonl').read_text().splitlines()]
        keys=('step','learning_rate','auxiliary_ramp','noise_mse','imitation','temporal','road','total_loss',
              'gradient_norm_before_clip','window_ids','timesteps','noise_sha256')
        paired=[{k:r[k] for k in keys} for r in first[len(first)//2:]]
        if paired!=[{k:r[k] for k in keys} for r in last]:
            raise RuntimeError('Recovery objective/lr/ramp/noise trace mismatch')
        bounded=deepcopy(sampling);bounded['limit']=6
        path=run.output/'preflight_sampling.yaml';path.write_text(yaml.safe_dump(bounded,sort_keys=True),encoding='utf-8')
        generation=sample_one(c,final,run.output/'sample',formal,path)
        initial=read(uninterrupted/'initial_state.json')
        if initial['optimizer_parameter_states']!=0 or initial['scheduler_state']['completed_updates']!=0:
            raise RuntimeError('New experiment retained old AdamW/scheduler state')
        summary=dict(schema_version='sumodiff.stage5c.preflight.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            objective=objective,prefix_updates=c['preflight_updates'],full_epoch_training=False,
            gradient_audit=gradient,training=result,resumed=second,recovery=recovery,
            scheduler_and_loss_trace_equal=True,new_AdamW_state_empty=True,free_generation=generation,
            wall_seconds=time.perf_counter()-started,quality_or_convergence_claim=False)
        write_json(run.output/'preflight_summary.json',summary);run.write_metrics(summary)
    return summary


def run_suite(config_path,output,objective='unified',formal=False,allow_long_run=False):
    from .stage5c_report import report
    if not allow_long_run:
        raise ValueError('Full run requires explicit --allow-long-run')
    c,parent,training,sampling=setup(config_path,objective)
    with RunRecorder(output,Path.cwd(),dict(schema_version='sumodiff.stage5c.full.run.v1',suite=c,objective=objective,
        additional_epochs=50,final_validation_tasks=200),[sys.executable,*sys.orig_argv[1:]],{'training':training['seed']},
        purpose='stage5c_full_local_training',data_files=source_files(config_path,c),checkpoint=c['parent_checkpoint'],formal=formal) as run:
        trained=train_one(c,objective,run.output/'train',formal)
        checkpoint=checked_final_training(c,objective,run.output/'train')
        generated=sample_one(c,checkpoint,run.output/'sample',formal)
        result=report(config_path,run.output/'train',run.output/'sample',run.output/'report',objective,formal)
        summary=dict(schema_version='sumodiff.stage5c.full.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            training=trained,generation=generated,report=str(run.output/'report'),convergence_status='not_assessed')
        write_json(run.output/'summary.json',summary);run.write_metrics(summary)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('preflight','run','train','sample','report'):
        a=sub.add_parser(action)
        a.add_argument('--config',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        a.add_argument('--objective',choices=OBJECTIVES,default='unified');a.add_argument('--formal',action='store_true')
        if action in ('run','train'):a.add_argument('--allow-long-run',action='store_true')
        if action=='train':a.add_argument('--resume',type=Path)
        if action in ('sample','report'):a.add_argument('--training-run',type=Path,required=True)
        if action=='report':a.add_argument('--sampling-run',type=Path,required=True)
    a=sub.add_parser('train-direct')
    for name in ('dataset','parent-checkpoint','scale-audit','config','output'):
        a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--formal',action='store_true');a.add_argument('--resume',type=Path)
    a.add_argument('--allow-long-run',action='store_true');a.add_argument('--prefix-updates',type=int)
    args=p.parse_args(argv)
    command=[sys.executable,*sys.orig_argv[1:]] if argv is None else [sys.executable,'-m','sumodiff.diffusion.stage5c',*argv]
    try:
        if args.action=='train-direct':
            result=train_unified(args.dataset,args.parent_checkpoint,args.config,args.scale_audit,args.output,
                Path.cwd(),command,args.formal,args.resume,args.prefix_updates,args.allow_long_run)
        elif args.action=='preflight':
            result=preflight(args.config,args.output,args.objective,args.formal)
        elif args.action=='run':
            result=run_suite(args.config,args.output,args.objective,args.formal,args.allow_long_run)
        else:
            c,_,_,_=setup(args.config,args.objective)
            if args.action=='train':
                if not args.allow_long_run:raise ValueError('Training requires --allow-long-run')
                result=train_one(c,args.objective,args.output,args.formal,args.resume)
            elif args.action=='sample':
                checkpoint=checked_final_training(c,args.objective,args.training_run)
                with RunRecorder(args.output,Path.cwd(),dict(schema_version='sumodiff.stage5c.sample.report.run.v1',
                    suite=c,training_run=str(args.training_run),objective=args.objective),command,{'sampling':20261015},
                    purpose='stage5c_recovered_sampling_report',data_files=source_files(args.config,c),
                    checkpoint=checkpoint,formal=args.formal) as run:
                    sample_one(c,checkpoint,run.output/'sample',args.formal)
                    from .stage5c_report import report
                    result=report(args.config,args.training_run,run.output/'sample',run.output/'report',args.objective,args.formal)
                    run.write_metrics(dict(report=str(run.output/'report'),convergence_status='not_assessed'))
            else:
                from .stage5c_report import report
                result=report(args.config,args.training_run,args.sampling_run,args.output,args.objective,args.formal)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2
    print({k:v for k,v in result.items() if k in ('run_sha','objective','prefix_updates','updates','end_step',
                                               'full_budget_complete','scheduler_and_loss_trace_equal','wall_seconds')})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
