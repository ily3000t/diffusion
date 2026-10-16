"""Bounded Motion control audit/calibration/preflight on the original stage5a script."""
import argparse,json,math,subprocess,sys
from copy import deepcopy
from pathlib import Path
import yaml
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from sumodiff.data.dataset import write_json
from sumodiff.data.control_tasks import make_motion_task_plan,verify_motion_task_plan
from sumodiff.diffusion.control_training import control_config,verified_control_scales
from sumodiff.diffusion.config import resolve_diffusion_config

SCHEMA='sumodiff.motion.control.preflight.config.v1'
FIELDS={'schema_version','dataset','data_id','model_config','scale_protocol','task_seed','model_seed','noise_seed',
        'validation_seed','batch_size','updates','peak_lr','minimum_lr','auxiliary_ratio_budget','weight_factors','probe_timesteps'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def resolve_suite(path,dataset):
    c=load_config(path)
    if set(c)!=FIELDS or c['schema_version']!=SCHEMA:raise ValueError('Unknown Motion control preflight suite')
    if Path(c['dataset']).resolve()!=Path(dataset).resolve():raise ValueError('Suite dataset path differs')
    dm=read(Path(dataset)/'dataset_manifest.json')
    if dm['data_id']!=c['data_id'] or dm['curation']['required_model_schema']!='sumodiff.model.control.config.v2':
        raise ValueError('Suite dataset identity/model capability differs')
    if c['batch_size']!=16 or c['updates']!=16:raise ValueError('This suite is fixed to batch16 and 16 bounded updates')
    for k in ('task_seed','model_seed','noise_seed','validation_seed'):
        if type(c[k]) is not int or not 0<=c[k]<2**32:raise ValueError('Seeds must be uint32')
    if not 0<c['minimum_lr']<c['peak_lr'] or not 0<c['auxiliary_ratio_budget']<=1:
        raise ValueError('Invalid bounded schedule/gradient budget')
    grid=c['weight_factors']
    if not grid or grid[0]!=1 or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<x<=1 for x in grid) or any(b>=a for a,b in zip(grid,grid[1:])):
        raise ValueError('Auxiliary factors must be explicit decreasing positive candidates starting at one')
    if c['probe_timesteps']!=[200,50]:raise ValueError('Motion gradient protocol fixes t200 and t50 plus natural t')
    return c

def run_child(argv):
    result=subprocess.run([sys.executable,*map(str,argv)],check=False)
    if result.returncode:raise RuntimeError(f'Child failed with exit {result.returncode}; retained command={argv}')

def verify_parent(directory,purpose,c,formal):
    directory=Path(directory).resolve(strict=True)
    m=read(directory/'manifest.json')
    if (m['purpose']!=purpose or read(directory/'status.json')['state']!='completed'
        or file_identity(directory/'resolved_config.yaml')['sha256']!=m['resolved_config_sha256']):
        raise ValueError('Required parent run is incomplete or inconsistent')
    actual=load_config(directory/'resolved_config.yaml')
    if actual['suite']!=c or m['data']['id']!=c['data_id']:
        raise ValueError('Parent suite/dataset differs')
    if formal and (not m['formal'] or m['git']['dirty']):raise ValueError('Formal preflight requires clean formal parents')
    return [directory/n for n in ('manifest.json','status.json','resolved_config.yaml','metrics.json')]

def root_inputs(path,c):
    d=Path(c['dataset'])
    return [path,c['model_config'],d/'dataset_manifest.json',d/'windows.json',d/'selection.json']

def audit(path,c,output,repository,command,formal):
    protocol=load_config(c['scale_protocol'])
    if protocol['dataset']!=c['dataset'] or protocol['data_id']!=c['data_id']:
        raise ValueError('Motion scale protocol dataset differs')
    tasks=make_motion_task_plan(c['dataset'],c['task_seed'])
    with RunRecorder(output,repository,dict(suite=c,scope='train-only scale statistics; fixed train/validation metadata cohorts'),
         command,{'tasks':c['task_seed']},purpose='motion_control_audit',data_files=root_inputs(path,c)+[c['scale_protocol']],
         data_id=c['data_id'],formal=formal) as run:
        write_json(run.output/'task_plan.json',tasks)
        run_child(['-m','sumodiff.decoding.control_scale_audit','--config',c['scale_protocol'],
                   '--output',run.output/'scale',*(['--formal'] if formal else [])])
        scale=read(run.output/'scale/candidate_normalization.json')
        if not scale.get('certified_training_statistics'):raise ValueError('Full training scales are not certified; inspect retained audit')
        run_child(['-m','sumodiff.decoding.control_scale_review','--source',run.output/'scale',
                   '--output',run.output/'review',*(['--formal'] if formal else [])])
        result=dict(passed=True,data_id=c['data_id'],task_plan=file_identity(run.output/'task_plan.json'),
            scale_audit=file_identity(run.output/'scale/candidate_normalization.json'),
            scales={k:v['scale'] for k,v in scale['scales'].items()},optimizer_updates=0,test_arrays_loaded=0)
        write_json(run.output/'results.json',result);run.write_metrics(result)
    return result

def baseline_config(c,scale_path,tasks):
    scale=read(scale_path)
    result=load_config('configs/train/control_integration.yaml')
    result.update(seed=c['model_seed'],noise_seed=c['noise_seed'],validation_seed=c['validation_seed'],
        batch_size=16,max_steps=16,train_limit=60,validation_limit=60,task_plan=str(Path(tasks).resolve()),
        validation_interval=8,checkpoint_interval=8,formal_estimate_steps=16)
    result['schedule']=dict(peak_lr=c['peak_lr'],minimum_lr=c['minimum_lr'],warmup_updates=2,total_updates=16,auxiliary_warmup_updates=4)
    for field,key in [('acceleration_scale_mps2','acceleration'),('yaw_rate_scale_radps','yaw_rate'),('lateral_velocity_scale_mps','lateral_velocity')]:
        result['diffusion'][field]=scale['scales'][key]['scale']
    return control_config(result)

def max_ratio(report):
    ratios=[r['auxiliary_to_noise_ratio'] for r in report['rows'] if r['auxiliary_to_noise_ratio'] is not None]
    if not ratios or not all(math.isfinite(v) for v in ratios):raise ValueError('No finite eligible complete auxiliary gradients')
    return max(ratios)

def pick_factor(report,budget,grid):
    worst=max_ratio(report)
    factor=next((x for x in grid if worst*x<=budget),None)
    if factor is None:raise ValueError('Declared auxiliary factor grid exhausted; no silent expansion')
    return factor

def scaled_config(base,factor):
    c=deepcopy(base)
    for key in ('imitation_weight','temporal_weight','road_weight'):c['unified_loss'][key]*=factor
    c['tail_weight']*=factor
    return control_config(c)

def calibration(path,c,scale_path,output,repository,command,formal):
    scale_path=Path(scale_path).resolve(strict=True);parent=scale_path.parent.parent
    files=verify_parent(parent,'motion_control_audit',c,formal)
    tasks=parent/'task_plan.json';verify_motion_task_plan(read(tasks),c['dataset'])
    base=baseline_config(c,scale_path,tasks)
    verified_control_scales(scale_path,c['dataset'],resolve_diffusion_config(base['diffusion']))
    with RunRecorder(output,repository,dict(suite=c,baseline=base,ratio_selection='common factor on full combined auxiliary gradient, max across fixed training probes',
        certification='numerical provisional weights; no generation or convergence certification'),
        command,{'model':c['model_seed'],'noise':c['noise_seed']},purpose='motion_control_calibration',
        data_files=root_inputs(path,c)+[scale_path,tasks,*files],data_id=c['data_id'],formal=formal) as run:
        baseline_path=run.output/'baseline_training.yaml';baseline_path.write_text(yaml.safe_dump(base,sort_keys=True),encoding='utf-8')
        def probe(config_path,directory):
            run_child(['-m','sumodiff.diffusion','calibrate','--dataset',c['dataset'],'--model-config',c['model_config'],
                '--config',config_path,'--scale-audit',scale_path,'--output',directory,'--repository',repository,
                '--probe-timesteps',*map(str,c['probe_timesteps']),*(['--formal'] if formal else [])])
            return read(directory/'gradient_audit.json')
        initial=probe(baseline_path,run.output/'baseline')
        factor=pick_factor(initial,c['auxiliary_ratio_budget'],c['weight_factors'])
        candidate=scaled_config(base,factor)
        config_path=run.output/'training_config.yaml';config_path.write_text(yaml.safe_dump(candidate,sort_keys=True),encoding='utf-8')
        if factor==1:chosen=initial;chosen_path=run.output/'baseline/gradient_audit.json'
        else:chosen=probe(config_path,run.output/'candidate');chosen_path=run.output/'candidate/gradient_audit.json'
        if max_ratio(chosen)>c['auxiliary_ratio_budget']*(1+1e-5):raise ValueError('Selected weights fail the declared combined-gradient bound')
        if any(r['combination_relative_residual']>1e-4 for r in chosen['rows']):
            raise ValueError('Combined gradient does not equal the measured noise plus auxiliary gradients')
        if not chosen['numerically_finite'] or chosen['optimizer_updates'] or chosen['test_arrays_loaded']:raise ValueError('Gradient calibration scope/finite check failed')
        result=dict(passed=True,data_id=c['data_id'],factor=factor,baseline_max_combined_ratio=max_ratio(initial),
            selected_max_combined_ratio=max_ratio(chosen),training_config=file_identity(config_path),
            scale_audit=file_identity(scale_path),task_plan=file_identity(tasks),selected_gradient=file_identity(chosen_path),
            numerically_finite=True,optimizer_updates=0,test_arrays_loaded=0,
            limitation='random-initialization numerical budget; not optimal loss weights or generated quality')
        write_json(run.output/'results.json',result);run.write_metrics(result)
    return result

def preflight(path,c,scale_path,training_path,output,repository,command,formal):
    training_path=Path(training_path).resolve(strict=True);parent=training_path.parent
    files=verify_parent(parent,'motion_control_calibration',c,formal);result=read(parent/'results.json')
    if file_identity(training_path)!=result['training_config'] or file_identity(scale_path)!=result['scale_audit']:
        raise ValueError('Preflight inputs differ from the selected calibration')
    cfg=control_config(load_config(training_path))
    if cfg['batch_size']!=16 or cfg['max_steps']!=16:raise ValueError('Only the bounded batch16 recipe is supported')
    with RunRecorder(output,repository,dict(suite=c,training=cfg,scope='16 real AdamW updates plus three train-only full-auxiliary stress updates'),
        command,{'model':c['model_seed'],'noise':c['noise_seed']},purpose='motion_control_preflight',
        data_files=root_inputs(path,c)+[scale_path,training_path,parent/'results.json',*files],
        data_id=c['data_id'],formal=formal) as run:
        run_child(['-m','sumodiff.diffusion','train','--dataset',c['dataset'],'--model-config',c['model_config'],
            '--config',training_path,'--scale-audit',scale_path,'--output',run.output/'train','--repository',repository,
            *(['--formal'] if formal else [])])
        summary=read(run.output/'train/training_summary.json')
        checkpoint=Path(summary['checkpoints'][-1]['location'])
        if file_identity(checkpoint)!=summary['checkpoints'][-1]:
            raise ValueError('Preflight checkpoint identity changed before stress')
        run_child(['-m','sumodiff.experiments.motion_control_stress','--suite',path,
            '--training-config',training_path,'--scale-audit',scale_path,'--checkpoint',checkpoint,
            '--output',run.output/'stress','--repository',repository,*(['--formal'] if formal else [])])
        stress_result=read(run.output/'stress/results.json')
        if summary['updates']!=16 or not summary['weight_update_verified'] or not stress_result['passed']:
            raise ValueError('Bounded real optimizer verification failed')
        result=dict(passed=True,data_id=c['data_id'],updates=16,stress_updates=stress_result['updates'],
            batch_size=16,training_summary=file_identity(run.output/'train/training_summary.json'),
            stress=file_identity(run.output/'stress/results.json'),
            optimizer_step_timing=summary['optimizer_step_timing'],
            peak_allocated_bytes=max(summary['peak_allocated_bytes'],stress_result['peak_allocated_bytes']),
            peak_reserved_bytes=max(summary['peak_reserved_bytes'],stress_result['peak_reserved_bytes']),
            test_arrays_loaded=0,formal_long_training_started=False,
            limitation='bounded cohorts and optimizer engineering only; no convergence/free-sampling result')
        write_json(run.output/'results.json',result);run.write_metrics(result)
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['audit','calibrate','preflight'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repository',type=Path,default=Path.cwd());p.add_argument('--formal',action='store_true')
    p.add_argument('--scale-audit',type=Path);p.add_argument('--training-config',type=Path)
    a=p.parse_args(argv);command=[sys.executable,*sys.orig_argv[1:]]
    try:
        c=resolve_suite(a.config,a.dataset)
        if a.action=='audit':
            if a.scale_audit or a.training_config:raise ValueError('Audit does not consume old scale/training inputs')
            result=audit(a.config,c,a.output,a.repository,command,a.formal)
        elif a.action=='calibrate':
            if a.scale_audit is None or a.training_config:raise ValueError('Calibrate requires only an explicit certified scale audit')
            result=calibration(a.config,c,a.scale_audit,a.output,a.repository,command,a.formal)
        else:
            if a.scale_audit is None or a.training_config is None:raise ValueError('Preflight requires selected scales and training configuration')
            result=preflight(a.config,c,a.scale_audit,a.training_config,a.output,a.repository,command,a.formal)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2
    print(json.dumps(result),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
