"""Independent local-terminal stage5D entry; never replays stage5C."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
import yaml
from sumodiff.data.dataset import WindowDataset,write_json
from sumodiff.experiments.recorder import RunRecorder,load_config,file_identity
from .checkpoint import load_checkpoint
from .data import dataset_identity,input_files
from .controlled_training import shared_cache
from .raw_velocity_training import raw_velocity_config,train_raw_velocity,velocity_monitor_quality
from .sampling import sample_config
from .stage5c import sample_one,read

SCHEMA='sumodiff.stage5d.suite.config.v1'
FIELDS={'schema_version','dataset','parent_checkpoint','scale_audit','sampling_config',
        'baseline_run','training_config','preflight_updates','gradient_batches','peak_stress_updates','bootstrap_seed'}

def setup(config_path):
    c=load_config(config_path)
    if set(c)!=FIELDS or c['schema_version']!=SCHEMA:raise ValueError('Invalid stage5D suite')
    for key,lo,hi in (('preflight_updates',2,24),('gradient_batches',1,16),('peak_stress_updates',1,24)):
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError(f'Unbounded {key}')
    if c['preflight_updates']%2:raise ValueError('Preflight needs an even prefix for recovery')
    if type(c['bootstrap_seed']) is not int or not 0<=c['bootstrap_seed']<2**32:raise ValueError('Invalid bootstrap seed')
    for k in ('dataset','parent_checkpoint','scale_audit','sampling_config','baseline_run','training_config'):
        c[k]=str(Path(c[k]).resolve(strict=True))
    parent=load_checkpoint(c['parent_checkpoint']);identity=file_identity(c['parent_checkpoint'])
    training=raw_velocity_config(load_config(c['training_config']),parent)
    if identity['sha256']!=training['fork_source_sha256']:raise ValueError('150-epoch parent hash mismatch')
    sampling=sample_config(load_config(c['sampling_config']))
    old=load_config(Path(c['baseline_run'])/'resolved_config.yaml')
    if (read(Path(c['baseline_run'])/'status.json')['state']!='completed' or old['checkpoint']['sha256']!=identity['sha256']
        or old['sampling']!=sampling or sampling['limit']!=200 or sampling['split']!='validation'
        or sampling['steps']!=20 or sampling['eta']!=0):
        raise ValueError('Keep the full paired epoch150 baseline protocol with DDIM20/eta0')
    data=dataset_identity(c['dataset'])
    if any(data[k]['sha256']!=old['data'][k]['sha256'] for k in data):raise ValueError('Dataset/baseline mismatch')
    return c,parent,training,sampling

def source_files(config_path,c,*,labels=False):
    files=[config_path,c['training_config'],c['parent_checkpoint'],c['scale_audit'],c['sampling_config'],
           *[Path(c['baseline_run'])/n for n in ('manifest.json','resolved_config.yaml','status.json',
                                               'trajectory_index.json','sampling_summary.json')],
           *[x['location'] for x in dataset_identity(c['dataset']).values()]]
    for split in ('train','validation') if labels else ('validation',):
        ds=WindowDataset(c['dataset'],split)
        files+=input_files(ds,list(range(len(ds.entries))),labels)
    return sorted(set(map(str,files)))

def train_one(c,output,formal,*,resume=None,prefix=None,cache=None,stop=None):
    cmd=[sys.executable,'-m','sumodiff.diffusion.stage5d','train-direct',
        '--dataset',c['dataset'],'--parent-checkpoint',c['parent_checkpoint'],
        '--scale-audit',c['scale_audit'],'--config',c['training_config'],'--output',str(output)]
    if formal:cmd.append('--formal')
    if resume:cmd+=['--resume',str(resume)]
    if prefix is not None:cmd+=['--prefix-updates',str(prefix)]
    else:cmd.append('--allow-long-run')
    return train_raw_velocity(c['dataset'],c['parent_checkpoint'],c['training_config'],c['scale_audit'],output,
        Path.cwd(),cmd,formal,resume,prefix,prefix is None,cache,stop)

def checked_final_training(c,training,root):
    root=Path(root).resolve(strict=True)
    if read(root/'status.json')['state']!='completed' or not read(root/'training_summary.json')['full_budget_complete']:
        raise ValueError('Final evaluation requires completed 150->200 training')
    identity=read(root/'latest_checkpoint.json')['checkpoint']
    if file_identity(identity['location'])['sha256']!=identity['sha256']:raise ValueError('Final checkpoint changed')
    p=load_checkpoint(identity['location'])
    if p['step']!=60000 or p['metadata']['training_config']!=training:raise ValueError('Wrong stage5D final protocol')
    return identity['location']

def preflight(config_path,output,formal=False):
    from .raw_velocity_diagnostics import joint_gradient_audit,peak_stress
    from .audit import audit_continuation
    c,parent,training,sampling=setup(config_path);started=time.perf_counter()
    effective=dict(schema_version='sumodiff.stage5d.preflight.config.v1',suite=c,training=training,
        planned_prefix_updates=c['preflight_updates'],schedule_budget_preserved=15000,
        joint_audit_ramp=1.,peak_stress_updates_per_candidate=c['peak_stress_updates'],
        peak_stress_candidates=[2e-5,5e-5],free_generation_tasks=6,automatic_parameter_selection=False)
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'noise':training['noise_seed']},
        purpose='stage5d_bounded_joint_preflight',data_files=source_files(config_path,c,labels=True),
        checkpoint=c['parent_checkpoint'],formal=formal) as run:
        cache=shared_cache(c['dataset'],training);print(f'cache ready: {cache[3]:.2f}s',flush=True)
        gradient=joint_gradient_audit(c['dataset'],parent,training,cache,run.output/'joint_gradient_audit.json',c['gradient_batches'])
        print('joint full-strength audit complete',flush=True)
        stress=peak_stress(c['dataset'],parent,training,cache,run.output/'peak_stress.json',c['peak_stress_updates'])
        print('paired peak stress complete',flush=True)
        first=run.output/'uninterrupted';result=train_one(c,first,formal,prefix=c['preflight_updates'],cache=cache)
        mid=first/f"checkpoint_step_{45000+c['preflight_updates']//2:06d}.pt"
        second=run.output/'resumed';restored=train_one(c,second,formal,resume=mid,prefix=c['preflight_updates'],cache=cache)
        left=read(first/'latest_checkpoint.json')['checkpoint']['location']
        right=read(second/'latest_checkpoint.json')['checkpoint']['location']
        cmd=[sys.executable,'-m','sumodiff.diffusion','audit-continuation','--uninterrupted',left,'--resumed',right,
             '--output',str(run.output/'recovery_audit')]
        if formal:cmd.append('--formal')
        recovery=audit_continuation(left,right,run.output/'recovery_audit',Path.cwd(),cmd,formal)
        keys=('step','window_ids','timesteps','noise_sha256','learning_rate','auxiliary_ramp','raw_auxiliary_ramp',
              'inherited_auxiliary_ramp','noise_mse','imitation','temporal','road','raw_second_boundary',
              'raw_second_interior','total_loss','gradient_norm_before_clip')
        a=[{k:r[k] for k in keys} for r in map(__import__('json').loads,(first/'steps.jsonl').read_text().splitlines())]
        b=[{k:r[k] for k in keys} for r in map(__import__('json').loads,(second/'steps.jsonl').read_text().splitlines())]
        if a[len(a)//2:]!=b:raise RuntimeError('Actual recovery objective/schedule/noise differs')
        if load_checkpoint(left)['metadata']['runtime']['scheduler_state']!=load_checkpoint(right)['metadata']['runtime']['scheduler_state']:
            raise RuntimeError('Recovered scheduler mismatch')
        initial=read(first/'initial_state.json')
        if initial['optimizer_parameter_states']!=len(parent['optimizer']['state']) or initial['scheduler_state']['completed_updates']!=0:
            raise RuntimeError('Source AdamW was reset or source scheduler retained')
        bounded=deepcopy(sampling);bounded['limit']=6
        path=run.output/'sampling6.yaml';path.write_text(yaml.safe_dump(bounded,sort_keys=True),encoding='utf-8')
        generation=sample_one(c,left,run.output/'sample',formal,path)
        quality=velocity_monitor_quality(run.output/'sample',cache[1])
        if file_identity(c['parent_checkpoint'])['sha256']!=training['fork_source_sha256']:raise RuntimeError('Source checkpoint changed')
        summary=dict(schema_version='sumodiff.stage5d.preflight.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            joint_gradient_audit=gradient,peak_stress=stress,training=result,resumed=restored,recovery=recovery,
            scheduler_and_loss_trace_equal=True,AdamW_moments_inherited=True,source_checkpoint_unchanged=True,
            free_generation=generation,free_generation_quality=quality,wall_seconds=time.perf_counter()-started,
            full_epoch_training=False,quality_or_convergence_claim=False)
        write_json(run.output/'preflight_summary.json',summary);run.write_metrics(summary)
    return summary

def run_suite(config_path,output,formal=False,allow_long_run=False):
    if not allow_long_run:raise ValueError('Formal 50-epoch run needs --allow-long-run')
    from .stage5d_report import report
    c,parent,training,sampling=setup(config_path)
    with RunRecorder(output,Path.cwd(),dict(schema_version='sumodiff.stage5d.full.run.v1',suite=c,training=training),
        [sys.executable,*sys.orig_argv[1:]],{'training':training['seed']},purpose='stage5d_full_local_training',
        data_files=source_files(config_path,c),checkpoint=c['parent_checkpoint'],formal=formal) as run:
        trained=train_one(c,run.output/'train',formal)
        cp=checked_final_training(c,training,run.output/'train')
        generated=sample_one(c,cp,run.output/'sample',formal)
        comparison=report(config_path,run.output/'train',run.output/'sample',run.output/'report',formal)
        result=dict(schema_version='sumodiff.stage5d.full.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            training=trained,generation=generated,report=str(run.output/'report'),paired_differences=comparison['paired_differences'],
            convergence_status='not_assessed')
        write_json(run.output/'summary.json',result);run.write_metrics(result)
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('preflight','run','train','warmup-check','sample','report'):
        a=sub.add_parser(action);a.add_argument('--config',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        a.add_argument('--formal',action='store_true')
        if action in ('run','train'):a.add_argument('--allow-long-run',action='store_true')
        if action=='train':a.add_argument('--resume',type=Path)
        if action in ('sample','report'):a.add_argument('--training-run',type=Path,required=True)
        if action=='report':a.add_argument('--sampling-run',type=Path,required=True)
    a=sub.add_parser('train-direct')
    for key in ('dataset','parent-checkpoint','scale-audit','config','output'):a.add_argument('--'+key,type=Path,required=True)
    a.add_argument('--formal',action='store_true');a.add_argument('--allow-long-run',action='store_true')
    a.add_argument('--resume',type=Path);a.add_argument('--prefix-updates',type=int)
    args=p.parse_args(argv);command=[sys.executable,*sys.orig_argv[1:]]
    try:
        if args.action=='train-direct':
            result=train_raw_velocity(args.dataset,args.parent_checkpoint,args.config,args.scale_audit,args.output,
                Path.cwd(),command,args.formal,args.resume,args.prefix_updates,args.allow_long_run)
        elif args.action=='preflight':result=preflight(args.config,args.output,args.formal)
        elif args.action=='run':result=run_suite(args.config,args.output,args.formal,args.allow_long_run)
        else:
            c,parent,training,sampling=setup(args.config)
            if args.action=='train':
                if not args.allow_long_run:raise ValueError('Training requires --allow-long-run')
                result=train_one(c,args.output,args.formal,resume=args.resume)
            elif args.action=='warmup-check':
                result=train_one(c,args.output,args.formal,prefix=600)
            elif args.action=='sample':
                cp=checked_final_training(c,training,args.training_run)
                result=sample_one(c,cp,args.output,args.formal)
            else:
                from .stage5d_report import report
                result=report(args.config,args.training_run,args.sampling_run,args.output,args.formal)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2
    print({k:v for k,v in result.items() if k in ('run_sha','updates','end_step','full_budget_complete',
         'wall_seconds','scheduler_and_loss_trace_equal')})
    return 0

if __name__=='__main__':raise SystemExit(main())
