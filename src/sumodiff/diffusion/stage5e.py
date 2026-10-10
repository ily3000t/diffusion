"""Independent stage5E terminal entry: epoch200 -> epoch225."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import yaml
from sumodiff.data.dataset import write_json
from sumodiff.experiments.recorder import RunRecorder,load_config,file_identity
from .checkpoint import load_checkpoint
from .data import dataset_identity
from .controlled_training import shared_cache
from .decoded_temporal_training import decoded_temporal_config,train_decoded_temporal
from .raw_velocity_training import velocity_monitor_quality
from .sampling import sample_config
from .stage5c import sample_one,read
from .stage5d import source_files as data_source_files

SCHEMA='sumodiff.stage5e.suite.config.v1'
FIELDS={'schema_version','dataset','parent_checkpoint','scale_audit','sampling_config',
        'baseline_run','training_config','preflight_updates','gradient_batches_per_group','stress_updates','bootstrap_seed'}


def setup(config_path):
    c=load_config(config_path)
    if set(c)!=FIELDS or c['schema_version']!=SCHEMA:raise ValueError('Invalid stage5E suite')
    for k,lo,hi in [('preflight_updates',2,24),('gradient_batches_per_group',1,4),('stress_updates',1,24)]:
        if type(c[k]) is not int or not lo<=c[k]<=hi:raise ValueError(f'Unbounded {k}')
    if c['preflight_updates']%2:raise ValueError('Even preflight prefix required for recovery')
    if type(c['bootstrap_seed']) is not int or not 0<=c['bootstrap_seed']<2**32:raise ValueError('Invalid seed')
    for k in ('dataset','parent_checkpoint','scale_audit','sampling_config','baseline_run','training_config'):
        c[k]=str(Path(c[k]).resolve(strict=True))
    p=load_checkpoint(c['parent_checkpoint']);identity=file_identity(c['parent_checkpoint'])
    t=decoded_temporal_config(load_config(c['training_config']),p)
    if identity['sha256']!=t['fork_source_sha256']:raise ValueError('200-epoch parent hash mismatch')
    sampling=sample_config(load_config(c['sampling_config']))
    old=load_config(Path(c['baseline_run'])/'resolved_config.yaml')
    if (read(Path(c['baseline_run'])/'status.json')['state']!='completed' or old['checkpoint']['sha256']!=identity['sha256']
        or old['sampling']!=sampling or sampling['split']!='validation' or sampling['limit']!=200
        or sampling['steps']!=20 or sampling['eta']!=0):
        raise ValueError('Keep full paired epoch200 baseline with DDIM20/eta0')
    data=dataset_identity(c['dataset'])
    if any(data[k]['sha256']!=old['data'][k]['sha256'] for k in data):raise ValueError('Baseline dataset mismatch')
    return c,p,t,sampling


def source_files(config_path,c,*,labels=False):
    parent_run=Path(c['parent_checkpoint']).parent
    files=data_source_files(config_path,c,labels=labels)
    files += [parent_run/n for n in ('manifest.json','status.json','steps.jsonl')]
    return sorted(set(map(str,files)))


def train_one(c,output,formal,*,resume=None,prefix=None,cache=None):
    cmd=[sys.executable,'-m','sumodiff.diffusion.stage5e','train-direct','--dataset',c['dataset'],
        '--parent-checkpoint',c['parent_checkpoint'],'--scale-audit',c['scale_audit'],
        '--config',c['training_config'],'--output',str(output)]
    if formal:cmd.append('--formal')
    if resume:cmd+=['--resume',str(resume)]
    if prefix is not None:cmd+=['--prefix-updates',str(prefix)]
    else:cmd.append('--allow-long-run')
    return train_decoded_temporal(c['dataset'],c['parent_checkpoint'],c['training_config'],c['scale_audit'],
        output,Path.cwd(),cmd,formal,resume,prefix,prefix is None,cache)


def checked_final_training(c,training,root):
    root=Path(root).resolve(strict=True)
    if read(root/'status.json')['state']!='completed' or not read(root/'training_summary.json')['full_budget_complete']:
        raise ValueError('Final evaluation requires completed 200->225 training')
    identity=read(root/'latest_checkpoint.json')['checkpoint']
    if file_identity(identity['location'])!=identity:raise ValueError('Final checkpoint changed')
    p=load_checkpoint(identity['location'])
    if p['step']!=67500 or p['metadata']['training_config']!=training:raise ValueError('Wrong stage5E final protocol')
    return identity['location']


def calibrate(config_path,output,formal=False):
    from .decoded_temporal_diagnostics import joint_gradient_audit
    c,p,t,s=setup(config_path)
    with RunRecorder(output,Path.cwd(),dict(schema_version='sumodiff.stage5e.calibration.config.v1',suite=c,
        training=t,optimizer_updates=0,automatic_parameter_selection=False),[sys.executable,*sys.orig_argv[1:]],
        {'noise':t['noise_seed']},purpose='stage5e_frozen_joint_calibration',
        data_files=source_files(config_path,c,labels=True),checkpoint=c['parent_checkpoint'],formal=formal) as run:
        cache=shared_cache(c['dataset'],t)
        result=joint_gradient_audit(c['dataset'],p,t,cache,run.output/'joint_gradient_audit.json',c['gradient_batches_per_group'])
        from .decoded_temporal_diagnostics import timestep_coverage
        from .boundary_validation import descriptors
        from sumodiff.data.dataset import WindowDataset
        result.update(run_sha=run.manifest['git']['commit_sha'],parent_timestep_coverage=timestep_coverage(
            Path(c['parent_checkpoint']).parent/'steps.jsonl',descriptors(WindowDataset(c['dataset'],'train'))))
        write_json(run.output/'calibration_summary.json',result);run.write_metrics(result)
    return result


def preflight(config_path,output,formal=False):
    from .decoded_temporal_diagnostics import joint_gradient_audit,full_strength_stress
    from .audit import audit_continuation
    c,p,t,s=setup(config_path);started=time.perf_counter()
    with RunRecorder(output,Path.cwd(),dict(schema_version='sumodiff.stage5e.preflight.config.v1',suite=c,
        training=t,planned_prefix_updates=c['preflight_updates'],schedule_budget_preserved=7500,
        stress_updates=c['stress_updates'],free_generation_tasks=6,automatic_parameter_selection=False),
        [sys.executable,*sys.orig_argv[1:]],{'noise':t['noise_seed']},purpose='stage5e_bounded_joint_preflight',
        data_files=source_files(config_path,c,labels=True),checkpoint=c['parent_checkpoint'],formal=formal) as run:
        cache=shared_cache(c['dataset'],t);print(f'cache ready: {cache[3]:.2f}s',flush=True)
        gradient=joint_gradient_audit(c['dataset'],p,t,cache,run.output/'joint_gradient_audit.json',c['gradient_batches_per_group'])
        from .decoded_temporal_diagnostics import timestep_coverage
        from .boundary_validation import descriptors
        from sumodiff.data.dataset import WindowDataset
        coverage=timestep_coverage(Path(c['parent_checkpoint']).parent/'steps.jsonl',descriptors(WindowDataset(c['dataset'],'train')))
        write_json(run.output/'parent_timestep_coverage.json',coverage)
        print('ordinary/dense full-strength calibration complete',flush=True)
        stress=full_strength_stress(c['dataset'],p,t,cache,run.output/'stress.json',c['stress_updates'])
        first=run.output/'uninterrupted';result=train_one(c,first,formal,prefix=c['preflight_updates'],cache=cache)
        mid=first/f"checkpoint_step_{60000+c['preflight_updates']//2:06d}.pt"
        second=run.output/'resumed';restored=train_one(c,second,formal,resume=mid,prefix=c['preflight_updates'],cache=cache)
        left=read(first/'latest_checkpoint.json')['checkpoint']['location'];right=read(second/'latest_checkpoint.json')['checkpoint']['location']
        cmd=[sys.executable,'-m','sumodiff.diffusion','audit-continuation','--uninterrupted',left,'--resumed',right,
            '--output',str(run.output/'recovery_audit')]
        if formal:cmd.append('--formal')
        recovery=audit_continuation(left,right,run.output/'recovery_audit',Path.cwd(),cmd,formal)
        a=[json.loads(x) for x in (first/'steps.jsonl').read_text().splitlines()]
        b=[json.loads(x) for x in (second/'steps.jsonl').read_text().splitlines()]
        exclude={'seconds'}
        if [{k:v for k,v in r.items() if k not in exclude} for r in a[len(a)//2:]] != [
            {k:v for k,v in r.items() if k not in exclude} for r in b]:raise RuntimeError('Recovery objective/schedule trace differs')
        if load_checkpoint(left)['metadata']['runtime']['scheduler_state']!=load_checkpoint(right)['metadata']['runtime']['scheduler_state']:
            raise RuntimeError('Recovery scheduler differs')
        initial=read(first/'initial_state.json')
        if initial['optimizer_parameter_states']!=len(p['optimizer']['state']) or initial['scheduler_state']['completed_updates']!=0:
            raise RuntimeError('Source optimizer reset or old scheduler inherited')
        bounded=deepcopy(s);bounded['limit']=6;path=run.output/'sampling6.yaml'
        path.write_text(yaml.safe_dump(bounded,sort_keys=True),encoding='utf-8')
        generated=sample_one(c,left,run.output/'sample',formal,path);quality=velocity_monitor_quality(run.output/'sample',cache[1])
        if file_identity(c['parent_checkpoint'])['sha256']!=t['fork_source_sha256']:raise RuntimeError('Source checkpoint changed')
        summary=dict(schema_version='sumodiff.stage5e.preflight.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            joint_gradient_audit=gradient,parent_timestep_coverage=coverage,stress=stress,training=result,resumed=restored,recovery=recovery,
            scheduler_and_loss_trace_equal=True,AdamW_moments_inherited=True,source_checkpoint_unchanged=True,
            free_generation=generated,free_generation_quality=quality,wall_seconds=time.perf_counter()-started,
            full_epoch_training=False,quality_or_convergence_claim=False)
        write_json(run.output/'preflight_summary.json',summary);run.write_metrics(summary)
    return summary


def report(config_path,training_run,sampling_run,output,formal=False):
    from .stage5d_report import report as shared_report
    return shared_report(config_path,training_run,sampling_run,output,formal,_suite=sys.modules[__name__])


def audit_baseline(config_path,output,formal=False):
    from .stage5d_report import audit_baseline as shared_audit
    return shared_audit(config_path,output,formal,_suite=sys.modules[__name__])


def run_suite(config_path,output,formal=False,allow_long_run=False):
    if not allow_long_run:raise ValueError('25-epoch run needs --allow-long-run')
    c,p,t,s=setup(config_path)
    with RunRecorder(output,Path.cwd(),dict(schema_version='sumodiff.stage5e.full.run.v1',suite=c,training=t),
        [sys.executable,*sys.orig_argv[1:]],{'training':t['seed']},purpose='stage5e_full_local_training',
        data_files=source_files(config_path,c),checkpoint=c['parent_checkpoint'],formal=formal) as run:
        trained=train_one(c,run.output/'train',formal);cp=checked_final_training(c,t,run.output/'train')
        generated=sample_one(c,cp,run.output/'sample',formal)
        comparison=report(config_path,run.output/'train',run.output/'sample',run.output/'report',formal)
        result=dict(schema_version='sumodiff.stage5e.full.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            training=trained,generation=generated,report=str(run.output/'report'),
            paired_differences=comparison['paired_differences'],convergence_status='not_assessed')
        write_json(run.output/'summary.json',result);run.write_metrics(result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    for name in ('calibrate','preflight','run','train','warmup-check','sample','report','audit-baseline'):
        x=sub.add_parser(name);x.add_argument('--config',type=Path,required=True);x.add_argument('--output',type=Path,required=True)
        x.add_argument('--formal',action='store_true')
        if name in ('run','train'):x.add_argument('--allow-long-run',action='store_true')
        if name=='train':x.add_argument('--resume',type=Path)
        if name in ('sample','report'):x.add_argument('--training-run',type=Path,required=True)
        if name=='report':x.add_argument('--sampling-run',type=Path,required=True)
    x=sub.add_parser('train-direct')
    for k in ('dataset','parent-checkpoint','scale-audit','config','output'):x.add_argument('--'+k,type=Path,required=True)
    x.add_argument('--formal',action='store_true');x.add_argument('--allow-long-run',action='store_true')
    x.add_argument('--resume',type=Path);x.add_argument('--prefix-updates',type=int)
    args=parser.parse_args(argv)
    try:
        if args.action=='train-direct':
            result=train_decoded_temporal(args.dataset,args.parent_checkpoint,args.config,args.scale_audit,args.output,Path.cwd(),
                [sys.executable,*sys.orig_argv[1:]],args.formal,args.resume,args.prefix_updates,args.allow_long_run)
        elif args.action in ('calibrate','preflight','audit-baseline'):
            result={'calibrate':calibrate,'preflight':preflight,'audit-baseline':audit_baseline}[args.action](args.config,args.output,args.formal)
        elif args.action=='run':result=run_suite(args.config,args.output,args.formal,args.allow_long_run)
        else:
            c,p,t,s=setup(args.config)
            if args.action=='train':
                if not args.allow_long_run:raise ValueError('Training requires --allow-long-run')
                result=train_one(c,args.output,args.formal,resume=args.resume)
            elif args.action=='warmup-check':result=train_one(c,args.output,args.formal,prefix=300)
            elif args.action=='sample':result=sample_one(c,checked_final_training(c,t,args.training_run),args.output,args.formal)
            else:result=report(args.config,args.training_run,args.sampling_run,args.output,args.formal)
    except Exception as e:print(f'{type(e).__name__}: {e}',file=sys.stderr);return 2
    print({k:v for k,v in result.items() if k in ('run_sha','updates','end_step','full_budget_complete','wall_seconds','scheduler_and_loss_trace_equal')})
    return 0


if __name__=='__main__':raise SystemExit(main())
