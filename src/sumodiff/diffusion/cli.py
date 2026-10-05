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
    config=resolve_diffusion_config(train_config(load_config(config_path))['diffusion'])
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
            a.add_argument('--resume',type=Path);a.add_argument('--allow-long-run',action='store_true')
        if action=='sample': a.add_argument('--checkpoint',type=Path,required=True)
    a=sub.add_parser('audit-continuation');a.add_argument('--uninterrupted',type=Path,required=True);a.add_argument('--resumed',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--repository',type=Path,default=Path.cwd());a.add_argument('--formal',action='store_true')
    a=sub.add_parser('audit-inference');a.add_argument('--dataset',type=Path,required=True);a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--source-run',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--repository',type=Path,default=Path.cwd());a.add_argument('--formal',action='store_true')
    args=p.parse_args(argv)
    command=[sys.executable,*sys.orig_argv[1:]] if argv is None else [sys.executable,'-m','sumodiff.diffusion',*argv]
    try:
        if args.action=='audit-inference':
            from .inference_audit import replay
            result=replay(args.dataset,args.checkpoint,args.source_run,args.output,args.repository,command,args.formal)
        elif args.action=='audit-continuation':
            from .audit import audit_continuation
            result=audit_continuation(args.uninterrupted,args.resumed,args.output,args.repository,command,args.formal)
        elif args.action=='audit-scales': result=audit(args.dataset,args.config,args.output,args.repository,args.formal,command)
        elif args.action=='train': result=train(args.dataset,args.model_config,args.config,args.scale_audit,args.output,args.repository,command,args.formal,args.resume,args.allow_long_run)
        else:
            from .sampling import sample
            result=sample(args.dataset,args.checkpoint,args.config,args.output,args.repository,command,args.formal)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2
    print({k:v for k,v in result.items() if k in ('run_sha','updates','train_probe_relative_improvement','planned_tasks','generation_failures','displacement_norm_m','speed_norm_mps','optimizer_step_timing','peak_allocated_bytes')})
    return 0
