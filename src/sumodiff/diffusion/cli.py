"""Traceable scale audit, bounded base training, and offline base sampling."""
import argparse
from pathlib import Path
import sys
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from sumodiff.data.dataset import write_json
from .config import resolve_diffusion_config
from .data import scale_statistics,dataset_identity
from .training import train,train_config


def audit(dataset,config_path,output,repository,formal,command):
    source=load_config(config_path)
    if source.get('schema_version') in ('sumodiff.base.epoch.config.v1','sumodiff.base.epoch.config.v2'):
        from .epoch_training import epoch_config
        parsed=epoch_config(source)
    else: parsed=train_config(source)
    config=resolve_diffusion_config(parsed['diffusion'])
    report,files=scale_statistics(dataset,config)
    with RunRecorder(output,repository,dict(schema_version='sumodiff.scale.audit.run.v1',diffusion=config.to_dict(),dataset=dataset_identity(dataset)),
        command,{'audit':0},purpose='train_validation_fixed_scale_check',data_files=[config_path,*files,*[v['location'] for v in report['dataset'].values()]],
        data_id=report['dataset']['manifest']['id'],formal=formal) as run:
        report['run_sha']=run.manifest['git']['commit_sha'];report['run_dirty']=run.manifest['git']['dirty']
        write_json(run.output/'scale_audit.json',report);run.write_metrics(report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('audit-scales','train','sample'):
        a=sub.add_parser(action);a.add_argument('--dataset',type=Path,required=True);a.add_argument('--config',type=Path,required=True)
        a.add_argument('--output',type=Path,required=True);a.add_argument('--repository',type=Path,default=Path.cwd());a.add_argument('--formal',action='store_true')
        if action=='train':
            a.add_argument('--model-config',type=Path,required=True);a.add_argument('--scale-audit',type=Path,required=True)
            a.add_argument('--resume',type=Path);a.add_argument('--allow-long-run',action='store_true');a.add_argument('--prefix-updates',type=int)
        if action=='sample':
            a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--test-seal',type=Path)
    a=sub.add_parser('seal-test');a.add_argument('--dataset',type=Path,required=True);a.add_argument('--checkpoint',type=Path,required=True)
    a.add_argument('--config',type=Path,required=True);a.add_argument('--validation-run',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--repository',type=Path,default=Path.cwd());a.add_argument('--formal',action='store_true')
    a=sub.add_parser('calibrate');a.add_argument('--dataset',type=Path,required=True);a.add_argument('--model-config',type=Path,required=True)
    a.add_argument('--config',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--repository',type=Path,default=Path.cwd());a.add_argument('--formal',action='store_true');a.add_argument('--development',action='store_true')
    a.add_argument('--scale-audit',type=Path);a.add_argument('--checkpoint',type=Path)
    a.add_argument('--probe-timesteps',type=int,nargs='+')
    a=sub.add_parser('audit-continuation');a.add_argument('--uninterrupted',type=Path,required=True);a.add_argument('--resumed',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--repository',type=Path,default=Path.cwd());a.add_argument('--formal',action='store_true')
    a=sub.add_parser('audit-inference');a.add_argument('--dataset',type=Path,required=True);a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--source-run',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--repository',type=Path,default=Path.cwd());a.add_argument('--formal',action='store_true')
    a=sub.add_parser('diagnose');a.add_argument('--dataset',type=Path,required=True);a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--config',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--repository',type=Path,default=Path.cwd());a.add_argument('--formal',action='store_true')
    args=p.parse_args(argv)
    command=[sys.executable,*sys.orig_argv[1:]] if argv is None else [sys.executable,'-m','sumodiff.diffusion',*argv]
    try:
        if args.action=='seal-test':
            from .fresh_protocol import seal_test
            result=seal_test(args.dataset,args.checkpoint,args.config,args.validation_run,args.output,args.repository,command,args.formal)
        elif args.action=='calibrate':
            if load_config(args.config).get('schema_version')=='sumodiff.control.train.config.v1':
                if not args.scale_audit or args.development:raise ValueError('Control calibration requires certified --scale-audit and its explicit bounded protocol')
                from .control_calibration import calibrate
                result=calibrate(args.dataset,args.model_config,args.config,args.scale_audit,args.output,args.repository,command,args.formal,args.checkpoint,tuple(args.probe_timesteps) if args.probe_timesteps is not None else (200,))
            else:
                if args.scale_audit or args.checkpoint or args.probe_timesteps is not None:raise ValueError('These calibration options apply only to control targets')
                from .fresh_calibration import calibrate
                result=calibrate(args.dataset,args.model_config,args.config,args.output,args.repository,command,args.formal,args.development)
        elif args.action=='diagnose':
            from .diagnostics import diagnose
            result=diagnose(args.dataset,args.checkpoint,args.config,args.output,args.repository,command,args.formal)
        elif args.action=='audit-inference':
            from .inference_audit import replay
            result=replay(args.dataset,args.checkpoint,args.source_run,args.output,args.repository,command,args.formal)
        elif args.action=='audit-continuation':
            from .audit import audit_continuation
            result=audit_continuation(args.uninterrupted,args.resumed,args.output,args.repository,command,args.formal)
        elif args.action=='audit-scales': result=audit(args.dataset,args.config,args.output,args.repository,args.formal,command)
        elif args.action=='train': result=train(args.dataset,args.model_config,args.config,args.scale_audit,args.output,args.repository,command,args.formal,args.resume,args.allow_long_run,args.prefix_updates)
        else:
            from .sampling import sample
            result=sample(args.dataset,args.checkpoint,args.config,args.output,args.repository,command,args.formal,args.test_seal)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2
    print({k:v for k,v in result.items() if k in ('run_sha','updates','train_probe_relative_improvement','planned_tasks','generation_failures','displacement_norm_m','speed_norm_mps','optimizer_step_timing','peak_allocated_bytes')})
    return 0
