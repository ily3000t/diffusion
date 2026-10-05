"""Traceable stage4 network smoke; no training or trajectory generation."""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from sumodiff.data.dataset import write_json
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from .config import resolve_model_config
from .denoiser import ConditionalDenoiser
from .inputs import prepare_conditioning
from .probes import collect_inputs,parameter_hash,invariance_probe,profile,full_capacity_conditioning
from sumodiff.data.dataset import WindowDataset,collate_numpy


def smoke_config(supplied):
    defaults=dict(schema_version='sumodiff.model.smoke.config.v1',seed=20261005,noise_seed=20261006,
        device='cuda:0',synthetic_batch_sizes=[1,2,4],iterations=3,warmups=1,torch_threads=1)
    if not isinstance(supplied,dict) or set(supplied)-set(defaults):raise ValueError('Unknown smoke options')
    result={**defaults,**supplied}
    if result['schema_version']!=defaults['schema_version'] or result['device'] not in ('cpu','cuda:0'):raise ValueError('Unsupported smoke schema/device')
    for key in ('seed','noise_seed'):
        if type(result[key]) is not int or not 0<=result[key]<2**32:raise ValueError('Invalid seed')
    for key in ('iterations','warmups','torch_threads'):
        if type(result[key]) is not int or not 1<=result[key]<=10:raise ValueError('Smoke only supports 1..10 iterations/threads')
    if not result['synthetic_batch_sizes'] or any(type(b) is not int or not 1<=b<=4 for b in result['synthetic_batch_sizes']):raise ValueError('Smoke batch candidates are 1..4')
    return result


def configure(seed,threads):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.manual_seed(seed);torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def smoke(dataset_path,model_config_path,probe_config_path,output,formal=False,command=None):
    architecture=resolve_model_config(load_config(model_config_path));probe=smoke_config(load_config(probe_config_path))
    effective=dict(schema_version='sumodiff.model.smoke.run.v1',model=architecture.to_dict(),probe=probe,
        dataset_manifest=file_identity(Path(dataset_path)/'dataset_manifest.json'),checkpoint=None,
        initialization='seeded untrained model; no weight update')
    # Validate current inputs and select tasks before CUDA; metadata contains no
    # future accesses. Hash the exact conditioning/map files used by this audit.
    c,input_report,files=collect_inputs(dataset_path,architecture)
    files += [model_config_path,probe_config_path,Path(dataset_path)/'dataset_manifest.json',Path(dataset_path)/'windows.json']
    command=command or [sys.executable,'-m','sumodiff.models',*sys.argv[1:]]
    with RunRecorder(output,Path.cwd(),effective,command,{'initialization':probe['seed'],'noise':probe['noise_seed']},
        purpose='stage4_untrained_network_validation',data_files=sorted(set(map(str,files))),
        data_id=effective['dataset_manifest']['id'],formal=formal) as run:
        configure(probe['seed'],probe['torch_threads']);device=torch.device(probe['device'])
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
        started=time.perf_counter();model=ConditionalDenoiser(architecture).to(device)
        initial_hash=parameter_hash(model)
        conditions={key:value.to(device) for key,value in c.items()}
        structural,arrays=invariance_probe(model,conditions,probe['noise_seed'])
        del conditions
        np.savez_compressed(run.output/'untrained_outputs.npz',**arrays)
        write_json(run.output/'input_audit.json',input_report)
        cases=[]
        workloads=[('observed_three_families',c)]+[(f'synthetic_full_capacity_b{b}',full_capacity_conditioning(c,b)) for b in probe['synthetic_batch_sizes']]
        for kind,source in workloads:
            for mode in ('hierarchical','parallel'):
                if device.type=='cuda':torch.cuda.empty_cache()
                result=profile(model,source,device,mode,probe['noise_seed'],probe['iterations'],probe['warmups'])
                result['input_kind']=kind;cases.append(result)
                write_json(run.output/'resource_cases.json',cases)
        final_hash=parameter_hash(model)
        if final_hash!=initial_hash:raise RuntimeError('Smoke must not update model parameters')
        result=dict(schema_version='sumodiff.stage4.validation.v1',run_sha=run.manifest['git']['commit_sha'],
            branch=run.manifest['git']['branch'],status='completed',input_audit=input_report,structural=structural,
            parameter_count=sum(p.numel() for p in model.parameters()),parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            parameter_sha256=initial_hash,parameters_unchanged=True,training_started=False,checkpoint=None,
            device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            device_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None,
            compute=dict(deterministic_algorithms=True,tf32=False,amp=False,torch_threads=probe['torch_threads']),
            resource_cases=cases,elapsed_seconds=time.perf_counter()-started,
            outputs={name:file_identity(run.output/name) for name in ('untrained_outputs.npz','input_audit.json','resource_cases.json')},
            limitations=['untrained noise prediction; no generation quality or innovation claim',
                'synthetic full-capacity input is a resource load, not a SUMO 12-car episode',
                'no optimizer memory or training throughput estimate','condition feature units are fixed candidates; target normalization remains stage5'])
        write_json(run.output/'validation_summary.json',result);run.write_metrics(result)
    return result


def audit(source,output,formal=False):
    import json
    source=Path(source).resolve(strict=True)
    report=json.loads((source/'validation_summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    config=load_config(source/'resolved_config.yaml')
    if json.loads((source/'status.json').read_text(encoding='utf-8'))['state']!='completed':raise ValueError('Source smoke did not complete')
    if report['run_sha']!=manifest['git']['commit_sha']:raise ValueError('Source SHA mismatch')
    if formal and (not manifest['formal'] or manifest['git']['dirty']):raise ValueError('Formal replay requires a formal clean source run')
    for identity in [*manifest['data']['files'],*report['outputs'].values()]:
        actual=file_identity(identity['location'])
        if actual['sha256']!=identity['sha256']:raise ValueError('Source input/output hash mismatch')
    for name,key in (('resolved_config.yaml','resolved_config_sha256'),('environment.json','environment_sha256')):
        if file_identity(source/name)['sha256']!=manifest[key]:raise ValueError('Source config/environment hash mismatch')
    dataset_path=Path(config['dataset_manifest']['location']).parent
    dataset=WindowDataset(dataset_path);by_id={e['window_id']:i for i,e in enumerate(dataset.entries)}
    inputs=[dataset.inference(by_id[wid]) for wid in report['input_audit']['selected_window_ids']]
    if any(s['input_metadata']['split']=='test' for s in inputs):raise ValueError('Resource selection must not use test tasks')
    files=[source/name for name in ('validation_summary.json','manifest.json','resolved_config.yaml','environment.json','status.json','untrained_outputs.npz')]
    with RunRecorder(output,Path.cwd(),dict(schema_version='sumodiff.model.replay.config.v1',source=str(source)),
        [sys.executable,*sys.orig_argv[1:]],{'initialization':config['probe']['seed']},purpose='stage4_cpu_replay',data_files=files,formal=formal) as run:
        configure(config['probe']['seed'],1);model=ConditionalDenoiser(resolve_model_config(config['model']))
        if parameter_hash(model)!=report['parameter_sha256']:raise ValueError('Initialization replay differs')
        c=prepare_conditioning(collate_numpy(inputs));errors={}
        with np.load(source/'untrained_outputs.npz',allow_pickle=False) as saved:
            noise=torch.from_numpy(saved['noisy_future']);timestep=torch.from_numpy(saved['timestep'])
            np.testing.assert_array_equal(c['agent_mask'].numpy(),saved['agent_mask'])
            for mode in ('hierarchical','parallel'):
                with torch.no_grad():actual=model(noise,timestep,c,fusion=mode,return_details=True)
                errors[mode]={}
                for name in ('pred_noise','condition','gates'):
                    expected=torch.from_numpy(saved[f'{mode}_{name}'])
                    torch.testing.assert_close(actual[name],expected,atol=2e-5,rtol=5e-5)
                    errors[mode][name]=float((actual[name]-expected).abs().max())
        result=dict(schema_version='sumodiff.stage4.replay.v1',source_run_sha=report['run_sha'],audit_sha=run.manifest['git']['commit_sha'],
            initialized_parameter_hash_matches=True,cpu_replay_max_abs_errors=errors,selected_window_ids=report['input_audit']['selected_window_ids'])
        write_json(run.output/'audit_summary.json',result);run.write_metrics(result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    check=sub.add_parser('smoke');check.add_argument('--dataset',type=Path,required=True)
    check.add_argument('--model-config',type=Path,required=True);check.add_argument('--probe-config',type=Path,required=True)
    check.add_argument('--output',type=Path,required=True);check.add_argument('--formal',action='store_true')
    replay=sub.add_parser('audit');replay.add_argument('--run',type=Path,required=True);replay.add_argument('--output',type=Path,required=True);replay.add_argument('--formal',action='store_true')
    args=p.parse_args(argv)
    try:
        result=smoke(args.dataset,args.model_config,args.probe_config,args.output,args.formal) if args.action=='smoke' else audit(args.run,args.output,args.formal)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2
    print({key:value for key,value in result.items() if key not in ('resource_cases','structural','outputs','limitations')});return 0
