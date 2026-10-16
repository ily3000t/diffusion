"""Three explicit full-auxiliary AdamW stress updates; never a training checkpoint."""
from pathlib import Path
import json,time,argparse,sys
import torch
from sumodiff.models import ConditionalDenoiser,resolve_model_config
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.experiments.recorder import RunRecorder,file_identity,load_config
from sumodiff.data.dataset import write_json
from sumodiff.diffusion.control_training import control_config,control_cache,control_objective,verified_control_scales
from sumodiff.diffusion.data import select_batch,dataset_identity
from sumodiff.diffusion.process import NoiseSchedule
from sumodiff.diffusion.representation import training_target
from sumodiff.diffusion.unified_losses import RoadGeometryCache,resolve_unified_loss
from sumodiff.diffusion.checkpoint import load_checkpoint,restore_training
from sumodiff.diffusion.training import synchronize

def stress(suite,training_path,scale_path,checkpoint,output,repository,formal):
    c=control_config(load_config(training_path));architecture=resolve_model_config(load_config(suite['model_config']))
    from sumodiff.diffusion.config import resolve_diffusion_config
    diffusion=resolve_diffusion_config(c['diffusion'])
    scale_identity=verified_control_scales(scale_path,suite['dataset'],diffusion)
    cache=control_cache(suite['dataset'],'train',c,model_config=architecture)
    plan=json.loads(Path(c['task_plan']).read_text(encoding='utf-8'))['tasks']
    by_id={t['window_id']:t['group'] for t in plan if t['split']=='train'}
    groups={g:[i for i,w in enumerate(cache[2]) if by_id[w]==g] for g in ('dense_no_turn','turning','lane_change')}
    if any(len(v)!=16 for v in groups.values()):raise ValueError('Stress cases require 16 distinct real training windows each')
    payload=load_checkpoint(checkpoint)
    if (payload['step']!=16 or payload['metadata']['model_config']!=architecture.to_dict()
        or payload['metadata']['training_config']!=c or payload['metadata']['data']!=dataset_identity(suite['dataset'])
        or payload['metadata']['scale_audit']['sha256']!=scale_identity['sha256']):
        raise ValueError('Stress checkpoint differs from the selected bounded recipe')
    files=[training_path,scale_path,c['task_plan'],suite['model_config'],*cache[3]]
    with RunRecorder(output,repository,dict(suite=suite,training=c,groups=groups,timestep=200,auxiliary_ramp=1.,
            peak_lr=suite['peak_lr'],optimizer_state='clone completed 16-update model and AdamW; no checkpoint output'),
        [sys.executable,*sys.orig_argv[1:]],{'stress_noise':suite['noise_seed']+100},
        purpose='motion_control_full_auxiliary_optimizer_stress',data_files=files,checkpoint=checkpoint,
        data_id=suite['data_id'],formal=formal) as run:
        configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
        model=ConditionalDenoiser(architecture).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        generator=torch.Generator().manual_seed(c['noise_seed']+100)
        restore_training(payload,model,optimizer,torch.Generator(),device)
        for group in optimizer.param_groups:group['lr']=suite['peak_lr']
        schedule=NoiseSchedule(diffusion,device);roads=RoadGeometryCache(suite['dataset'],cache[2],resolve_unified_loss(c['unified_loss']))
        before=parameter_hash(model);rows=[]
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        for name,indices in groups.items():
            optimizer.zero_grad(set_to_none=True);model.train()
            conditions,targets,ids=select_batch(cache,torch.tensor(indices),device)
            noise=torch.randn((16,12,40,3),generator=generator).to(device)
            t=torch.full((16,),200,device=device,dtype=torch.long)
            clean=training_target(targets,conditions,diffusion)
            noisy=torch.where(conditions['agent_mask'][...,None,None],schedule.add_noise(clean,noise,t),0.)
            synchronize(device);start=time.perf_counter()
            epsilon=model(noisy,t,conditions)
            out=control_objective(epsilon,noisy,noise,t,conditions,targets,schedule,c,roads=roads.batch(ids),ramp=1.)
            if not torch.isfinite(out['total_loss']):raise FloatingPointError('Nonfinite stress loss')
            out['total_loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step();synchronize(device)
            rows.append(dict(group=name,batch_size=16,window_ids=ids,active_agents=int(conditions['agent_mask'].sum()),
                eligible_scenes=out['auxiliary_eligible_scenes'],gradient_norm_before_clip=float(norm),
                clipping_applied=float(norm)>c['gradient_clip_norm'],loss=float(out['total_loss']),seconds=time.perf_counter()-start))
        after=parameter_hash(model)
        result=dict(passed=before!=after,updates=len(rows),rows=rows,
            parameter_hash_before=before,parameter_hash_after=after,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
            validation_arrays_loaded=0,test_arrays_loaded=0,checkpoint_written=False)
        if not result['passed']:raise ValueError('Stress did not update model parameters')
        write_json(run.output/'results.json',result);run.write_metrics(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('suite','training-config','scale-audit','checkpoint','output','repository'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--formal',action='store_true');a=p.parse_args()
    from .motion_control import resolve_suite
    suite=load_config(a.suite)
    suite=resolve_suite(a.suite,suite['dataset'])
    stress(suite,a.training_config,a.scale_audit,a.checkpoint,a.output,a.repository,a.formal)
if __name__=='__main__':main()
