"""Stage5B preflight, four-arm continuation, and paired validation entry points."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml
from sumodiff.data.dataset import write_json
from sumodiff.models import ConditionalDenoiser, resolve_model_config
from sumodiff.models.cli import configure
from sumodiff.experiments.recorder import RunRecorder, file_identity, load_config
from .checkpoint import load_checkpoint, restore_training
from .config import resolve_diffusion_config
from .process import NoiseSchedule, masked_noise_mse
from .training import draw_noise
from .data import select_batch, dataset_identity
from .epoch_training import epoch_batch
from .controlled_training import (controlled_config, shared_cache, train_controlled,
    tensor_hash, tree_hash, inherited_state)
from .sampling import sample, sample_config

SCHEMA='sumodiff.stage5b.suite.config.v1'
FIELDS={'schema_version','dataset','parent_checkpoint','scale_audit','sampling_config',
        'baseline_run','arms','preflight_updates','bootstrap_seed'}
ARMS=('A','B','C','D')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def setup(config_path):
    config=load_config(config_path)
    if set(config)!=FIELDS or config['schema_version']!=SCHEMA or set(config['arms'])!=set(ARMS):
        raise ValueError('Invalid four-arm suite configuration')
    if type(config['preflight_updates']) is not int or not 1<=config['preflight_updates']<=24:
        raise ValueError('Invalid bounded preflight budget')
    if type(config['bootstrap_seed']) is not int or not 0<=config['bootstrap_seed']<2**32:
        raise ValueError('Invalid comparison seed')
    for name in ('dataset','parent_checkpoint','scale_audit','sampling_config','baseline_run'):
        config[name]=str(Path(config[name]).resolve(strict=True))
    config['arms']={k:str(Path(v).resolve(strict=True)) for k,v in config['arms'].items()}
    parent=load_checkpoint(config['parent_checkpoint'])
    identity=file_identity(config['parent_checkpoint'])
    candidates={arm:controlled_config(load_config(path),parent) for arm,path in config['arms'].items()}
    common=None;weights={}
    for arm,c in candidates.items():
        if c['arm']!=arm or c['fork_source_sha256']!=identity['sha256']:
            raise ValueError('Arm mapping/common checkpoint mismatch')
        comparison=deepcopy(c)
        for name in ('arm','physical_loss','diffusion'):
            comparison.pop(name)
        physical={k:v for k,v in c['physical_loss'].items() if not k.endswith('_weight')}
        comparison['physical_settings']=physical
        if common is not None and comparison!=common:
            raise ValueError('Arm budgets, learning rates, data or execution settings differ')
        common=comparison
        for name in ('consistency_weight','velocity_difference_weight','boundary_weight'):
            value=c['physical_loss'][name]
            if value and name in weights and weights[name]!=value:
                raise ValueError('Shared objective weights differ across arms')
            if value:weights[name]=value
    sampling=sample_config(load_config(config['sampling_config']))
    baseline=Path(config['baseline_run'])
    if read(baseline/'status.json')['state']!='completed':
        raise ValueError('Baseline sampling did not complete')
    original=load_config(baseline/'resolved_config.yaml')
    if original['sampling']!=sampling or sampling['split']!='validation' or sampling['limit']!=200 or sampling['steps']!=20 or sampling['eta']!=0:
        raise ValueError('Paired evaluation must match the original full validation protocol')
    if original['checkpoint']['sha256']!=identity['sha256']:
        raise ValueError('Baseline was generated from a different common parent')
    data=dataset_identity(config['dataset'])
    if any(data[k]['sha256']!=original['data'][k]['sha256'] for k in data):
        raise ValueError('Baseline dataset differs')
    return config,parent,candidates,sampling


def source_files(config_path,c):
    return [config_path,c['parent_checkpoint'],c['scale_audit'],c['sampling_config'],
        *c['arms'].values(),*[Path(c['baseline_run'])/name for name in
             ('manifest.json','resolved_config.yaml','status.json','trajectory_index.json','sampling_summary.json')]]


def legacy_reference(c,parent,cache,updates):
    """Literal legacy optimize sequence, using the same certified full cache."""
    device=torch.device(c['device']);configure(c['seed'],c['torch_threads'])
    model=ConditionalDenoiser(resolve_model_config(parent['metadata']['model_config'])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    generator=torch.Generator().manual_seed(c['noise_seed'])
    restore_training(parent,model,optimizer,generator,device)
    initial=inherited_state(model,optimizer,generator)
    schedule=NoiseSchedule(resolve_diffusion_config(parent['metadata']['diffusion_config']),device)
    rows=[]
    for step in range(parent['step']+1,parent['step']+updates+1):
        indices,_,_=epoch_batch(step,len(cache[0][2]),c['batch_size'],c['seed'])
        conditions,targets,ids=select_batch(cache[0],indices,device)
        model.train();optimizer.zero_grad(set_to_none=True)
        noisy,noise,timestep=draw_noise(targets,conditions,schedule,generator)
        loss=masked_noise_mse(model(noisy,timestep,conditions),noise,conditions['agent_mask'],targets['future_mask'])
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True)
        optimizer.step()
        rows.append(dict(step=step,window_ids=ids,timesteps=timestep.cpu().tolist(),
                         noise_sha256=tensor_hash(noise),noise_mse=float(loss),
                         gradient_norm_before_clip=float(norm)))
    state=dict(model=tree_hash(model.state_dict()),optimizer=tree_hash(optimizer.state_dict()),
        generator_state=tensor_hash(generator.get_state()),cpu_rng=tensor_hash(torch.get_rng_state()),
        cuda_rng=[tensor_hash(x) for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])
    return initial,state,rows


def checkpoint_state(checkpoint):
    p=load_checkpoint(checkpoint)
    return dict(model=tree_hash(p['model']),optimizer=tree_hash(p['optimizer']),
        generator_state=tensor_hash(p['generator_state']),cpu_rng=tensor_hash(p['cpu_rng']),
        cuda_rng=[tensor_hash(x) for x in p['cuda_rng']])


def trace_rows(root):
    return [json.loads(line) for line in (Path(root)/'steps.jsonl').read_text(encoding='utf-8').splitlines()]


def verify_pairing(roots,expected_updates):
    rows={arm:trace_rows(root) for arm,root in roots.items()}
    reference=rows['A']
    if len(reference)!=expected_updates:
        raise ValueError('A update budget differs from the comparison')
    for arm,values in rows.items():
        if len(values)!=len(reference):
            raise ValueError(f'{arm} update budget differs')
        for a,b in zip(reference,values):
            if any(a[k]!=b[k] for k in ('step','window_ids','timesteps','noise_sha256')):
                raise ValueError(f'{arm} training noise/batch pairing differed')
    return dict(updates_per_arm=len(reference),paired_fields=['step','window_ids','timesteps','noise_sha256'],
                all_equal=True)


def verify_sampling_pairing(roots,expected_tasks):
    reference=None
    for arm,root in roots.items():
        directory=Path(root)
        catalog=read(directory/'trajectory_index.json')
        if len(catalog)!=expected_tasks or any(item['generation_status']!='completed' for item in catalog):
            raise ValueError('Preflight sampling did not complete its paired task budget')
        values=[]
        for item in catalog:
            with np.load(directory/'trajectories'/item['window_id']/'stages.npz',allow_pickle=False) as arrays:
                values.append((item['window_id'],tensor_hash(torch.as_tensor(arrays['initial_noise'].copy()))))
        if reference is not None and values!=reference:
            raise ValueError(f'{arm} sampling task/noise pairing differed')
        reference=values
    return dict(tasks_per_arm=expected_tasks,paired_fields=['window_id','initial_noise'],all_equal=True)


def sampling_call(c,checkpoint,config_path,output,formal):
    command=[sys.executable,'-m','sumodiff.diffusion','sample','--dataset',c['dataset'],
        '--checkpoint',str(checkpoint),'--config',str(config_path),'--output',str(output)]
    if formal:command.append('--formal')
    return sample(c['dataset'],checkpoint,config_path,output,Path.cwd(),command,formal)


def training_command(c,arm,output,formal=False,allow_long_run=False,resume=None,prefix_updates=None):
    argv=[sys.executable,'-m','sumodiff.diffusion.stage5b','train-direct',
        '--dataset',c['dataset'],'--parent-checkpoint',c['parent_checkpoint'],
        '--scale-audit',c['scale_audit'],'--config',c['arms'][arm],'--output',str(output)]
    if formal:argv.append('--formal')
    if allow_long_run:argv.append('--allow-long-run')
    if resume:argv+=['--resume',str(resume)]
    if prefix_updates is not None:argv+=['--prefix-updates',str(prefix_updates)]
    return argv


def preflight(config_path,output,formal=False):
    c,parent,candidates,sampling=setup(config_path)
    shared=shared_cache(c['dataset'],candidates['A'])
    effective=dict(schema_version='sumodiff.stage5b.preflight.run.v1',suite=c,
        actual_updates_per_arm=c['preflight_updates'],nominal_full_updates_per_arm=3000,
        candidates=candidates,checks='A legacy parity; inherited states; paired noise; active physical gradients')
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'preflight':0},
        purpose='stage5b_bounded_preflight',data_files=source_files(config_path,c),
        checkpoint=c['parent_checkpoint'],data_id=dataset_identity(c['dataset'])['manifest']['id'],formal=formal) as run:
        started=time.perf_counter()
        initial,reference,reference_rows=legacy_reference(candidates['A'],parent,shared,c['preflight_updates'])
        write_json(run.output/'legacy_reference.json',dict(initial=initial,final=reference,rows=reference_rows))
        sample_cfg=deepcopy(sampling);sample_cfg['limit']=6
        sample_file=run.output/'preflight_sampling.yaml'
        sample_file.write_text(yaml.safe_dump(sample_cfg,sort_keys=True),encoding='utf-8')
        results={};roots={}
        for arm in ARMS:
            train_dir=run.output/arm/'train';roots[arm]=train_dir
            argv=training_command(c,arm,train_dir,formal,prefix_updates=c['preflight_updates'])
            results[arm]=train_controlled(c['dataset'],c['parent_checkpoint'],c['arms'][arm],c['scale_audit'],
                train_dir,Path.cwd(),argv,formal,prefix_updates=c['preflight_updates'],cache=shared)
            checkpoint=results[arm]['checkpoints'][-1]['location']
            if results[arm]['initial_state']!=initial:
                raise RuntimeError('Arm initial model/AdamW/RNG differs from the common parent')
            if arm=='A' and checkpoint_state(checkpoint)!=reference:
                raise RuntimeError('A continuation differs from the legacy optimizer trajectory')
            generation=sampling_call(c,checkpoint,sample_file,run.output/arm/'sample',formal)
            results[arm]['sampling']=generation
            print(f"preflight arm={arm} complete updates={results[arm]['updates']} physical={results[arm]['mean_components']['physical_total']:.6g}",flush=True)
        pairing=verify_pairing(roots,c['preflight_updates'])
        sampling_pairing=verify_sampling_pairing({arm:run.output/arm/'sample' for arm in ARMS},6)
        # Also compare A logs to the literal old algorithm, before final probes.
        for expected,actual in zip(reference_rows,trace_rows(roots['A'])):
            if any(expected[k]!=actual[k] for k in expected):
                raise RuntimeError('A prefix losses or gradient clipping differ from legacy training')
        record=dict(schema_version='sumodiff.stage5b.preflight.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            arms=results,legacy_A_final_state_bitwise_equal=True,legacy_A_rows_equal=True,
            all_initial_model_optimizer_rng_equal=True,training_pairing=pairing,sampling_pairing=sampling_pairing,
            cache_preparation_seconds=shared[3],wall_seconds=time.perf_counter()-started,
            estimated_optimizer_only_seconds_all_arms=sum(r['estimate']['optimizer_only_seconds_per_arm'] for r in results.values()),
            limitations=['12 update preflight is not ten epochs','no test data or trajectory smoothing',
                         'auxiliary objectives require validation; convergence not assessed'])
        write_json(run.output/'preflight_summary.json',record);run.write_metrics(record)
    return record


def train_arm(c,arm,output,formal,allow_long_run,resume=None,cache=None):
    root=Path(output)
    if root.exists():
        raise FileExistsError('Use a fresh arm root; recovery resumes into a new output')
    command=training_command(c,arm,root/'train',formal,allow_long_run,resume)
    result=train_controlled(c['dataset'],c['parent_checkpoint'],c['arms'][arm],c['scale_audit'],
        root/'train',Path.cwd(),command,formal,resume=resume,allow_long_run=allow_long_run,cache=cache)
    write_json(root/'arm_outputs.json',dict(training_run=str((root/'train').resolve()),sampling_run=str((root/'sample').resolve())))
    result['sampling']=sampling_call(c,result['checkpoints'][-1]['location'],c['sampling_config'],root/'sample',formal)
    return result


def sample_arm(config_path,arm,training_run,output,formal=False):
    c,parent,candidates,sampling=setup(config_path)
    root=Path(output)
    if root.exists():raise FileExistsError('Use a fresh sampling-recovery root')
    train=Path(training_run).resolve(strict=True)
    if read(train/'status.json')['state']!='completed':raise ValueError('Sampling recovery needs completed training')
    checkpoint=read(train/'latest_checkpoint.json')['checkpoint']
    payload=load_checkpoint(checkpoint['location'])
    if payload['step']!=33000 or payload['metadata']['training_config']!=candidates[arm]:
        raise ValueError('Sampling recovery needs this arm final ten-epoch checkpoint')
    result=sampling_call(c,checkpoint['location'],c['sampling_config'],root/'sample',formal)
    write_json(root/'arm_outputs.json',dict(training_run=str(train),sampling_run=str((root/'sample').resolve())))
    return result


def run_suite(config_path,output,formal=False,allow_long_run=False):
    if not allow_long_run:
        raise ValueError('Full A-D ten-epoch suite requires --allow-long-run')
    c,parent,candidates,sampling=setup(config_path)
    shared=shared_cache(c['dataset'],candidates['A'])
    with RunRecorder(output,Path.cwd(),dict(schema_version='sumodiff.stage5b.suite.run.v1',suite=c,candidates=candidates),
        [sys.executable,*sys.orig_argv[1:]],{'suite':0},purpose='stage5b_four_ten_epoch_continuations',
        data_files=source_files(config_path,c),checkpoint=c['parent_checkpoint'],formal=formal) as run:
        results={};errors={};begun=time.perf_counter()
        for arm in ARMS:
            try:
                results[arm]=train_arm(c,arm,run.output/arm,formal,allow_long_run,cache=shared)
            except Exception as exc:
                errors[arm]=dict(type=type(exc).__name__,message=str(exc))
                print(f'arm={arm} failed; retained outputs; remaining independent arms continue',file=sys.stderr,flush=True)
            write_json(run.output/'suite_progress.json',dict(completed=list(results),errors=errors))
        paired_report=None
        if not errors:
            from .stage5b_report import report
            report(config_path,[run.output/arm for arm in ARMS],run.output/'report',formal)
            paired_report=str(run.output/'report')
        record=dict(schema_version='sumodiff.stage5b.suite.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            common_parent=file_identity(c['parent_checkpoint']),arms=results,errors=errors,paired_report=paired_report,
            wall_seconds=time.perf_counter()-begun,cache_preparation_seconds=shared[3])
        if not errors:
            record['training_pairing']=verify_pairing({arm:run.output/arm/'train' for arm in ARMS},3000)
        write_json(run.output/'suite_summary.json',record);run.write_metrics(record)
        if errors:
            raise RuntimeError('Suite includes failed arms; recover into fresh outputs before final comparison')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','run','train-arm','train-direct','sample-arm','report'])
    p.add_argument('--config',type=Path,default=Path('configs/experiments/stage5b_ablation.yaml'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--formal',action='store_true');p.add_argument('--allow-long-run',action='store_true')
    p.add_argument('--arm',choices=ARMS);p.add_argument('--resume',type=Path)
    p.add_argument('--arm-runs',type=Path,nargs=4)
    p.add_argument('--dataset',type=Path);p.add_argument('--parent-checkpoint',type=Path)
    p.add_argument('--scale-audit',type=Path);p.add_argument('--prefix-updates',type=int)
    p.add_argument('--training-run',type=Path)
    args=p.parse_args()
    try:
        if args.training_run is not None and args.action!='sample-arm':
            raise ValueError('--training-run is only valid with sample-arm')
        if args.allow_long_run and args.action not in ('run','train-arm','train-direct'):
            raise ValueError('--allow-long-run is only valid for a training action')
        if args.arm_runs is not None and args.action!='report':
            raise ValueError('--arm-runs is only valid with report')
        if args.resume is not None and args.action not in ('train-arm','train-direct'):
            raise ValueError('--resume is only valid for an explicit training continuation')
        if args.arm is not None and args.action not in ('train-arm','sample-arm'):
            raise ValueError('--arm is only valid with train-arm or sample-arm')
        if args.action!='train-direct' and any(x is not None for x in (args.dataset,args.parent_checkpoint,args.scale_audit,args.prefix_updates)):
            raise ValueError('Direct training inputs are only valid with train-direct')
        if args.action=='train-direct':
            if args.arm or args.arm_runs or not args.dataset or not args.parent_checkpoint or not args.scale_audit:
                raise ValueError('train-direct requires dataset, parent checkpoint, scale audit and an arm config')
            result=train_controlled(args.dataset,args.parent_checkpoint,args.config,args.scale_audit,
                args.output,Path.cwd(),[sys.executable,*sys.orig_argv[1:]],args.formal,args.resume,
                args.prefix_updates,args.allow_long_run)
        elif args.action=='preflight':
            if args.arm or args.resume or args.arm_runs:raise ValueError('Preflight always verifies all four independent arms')
            result=preflight(args.config,args.output,args.formal)
        elif args.action=='run':
            if args.arm or args.resume or args.arm_runs:raise ValueError('Use train-arm for explicit recovery')
            result=run_suite(args.config,args.output,args.formal,args.allow_long_run)
        elif args.action=='sample-arm':
            if not args.arm or not args.training_run or args.resume or args.arm_runs:
                raise ValueError('sample-arm needs --arm and --training-run')
            result=sample_arm(args.config,args.arm,args.training_run,args.output,args.formal)
        elif args.action=='train-arm':
            if not args.arm:raise ValueError('train-arm needs --arm')
            c,_,_,_=setup(args.config)
            result=train_arm(c,args.arm,args.output,args.formal,args.allow_long_run,args.resume)
        else:
            if not args.arm_runs:raise ValueError('Report requires four arm roots in A B C D order')
            from .stage5b_report import report
            result=report(args.config,args.arm_runs,args.output,args.formal)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2
    print({k:result[k] for k in ('run_sha','legacy_A_final_state_bitwise_equal','estimated_optimizer_only_seconds_all_arms','wall_seconds') if k in result})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
