"""Training-only joint-gradient probe for the bounded control integration."""
import json
from pathlib import Path
import time
import torch
from sumodiff.models import ConditionalDenoiser, resolve_model_config
from sumodiff.models.cli import configure
from sumodiff.models.probes import parameter_hash
from sumodiff.experiments.recorder import RunRecorder, file_identity, load_config
from sumodiff.data.dataset import write_json
from .config import resolve_diffusion_config, validate_architecture
from .control_training import control_config, control_cache, verified_control_scales, control_objective
from .data import dataset_identity, select_batch
from .checkpoint import load_checkpoint
from .training import draw_noise, synchronize
from .process import NoiseSchedule
from .representation import training_target
from .unified_losses import RoadGeometryCache, resolve_unified_loss


def gradient_vector(loss,parameters):
    if not loss.requires_grad:
        return torch.zeros(sum(p.numel() for p in parameters),device=parameters[0].device)
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(parameters,values)])


def cosine(a,b):
    denominator=float(a.norm()*b.norm())
    return float(torch.dot(a,b)/denominator) if denominator else None


def calibrate(dataset,model_path,config_path,scale_path,output,repository,command,formal=False,checkpoint=None,probe_timesteps=(200,)):
    if (not probe_timesteps or len(set(probe_timesteps))!=len(probe_timesteps)
            or any(type(t) is not int or not 0<=t<=300 for t in probe_timesteps)):
        raise ValueError('Explicit unique low-noise gradient timesteps must lie in 0..300')
    c=control_config(load_config(config_path))
    architecture=resolve_model_config(load_config(model_path))
    diffusion=resolve_diffusion_config(c['diffusion']);validate_architecture(architecture,diffusion)
    scale=verified_control_scales(scale_path,dataset,diffusion)
    data=dataset_identity(dataset)
    cache=control_cache(dataset,'train',c,model_config=architecture)
    plan=json.loads(Path(c['task_plan']).read_text(encoding='utf-8'))['tasks']
    by_id={task['window_id']:task['group'] for task in plan if task['split']=='train'}
    groups={}
    for i,wid in enumerate(cache[2]):groups.setdefault(by_id[wid],[]).append(i)
    if any(k not in groups for k in ('dense_no_turn','turning')):
        raise ValueError('Control calibration requires explicit dense and turning training windows')
    configure(c['seed'],c['torch_threads']);device=torch.device(c['device'])
    model=ConditionalDenoiser(architecture).to(device)
    parent=load_checkpoint(checkpoint) if checkpoint else None
    if parent:
        if parent['metadata']['model_config']!=architecture.to_dict() or parent['metadata']['training_config']!=c or parent['metadata']['data']!=data or parent['metadata']['scale_audit']['sha256']!=scale['sha256'] or parent['metadata']['training_signature']['task_plan_sha256']!=file_identity(c['task_plan'])['sha256']:
            raise ValueError('Calibration checkpoint differs from the explicit control recipe/data/scales')
        model.load_state_dict(parent['model'],strict=True)
    model.eval()
    files=[model_path,config_path,scale_path,c['task_plan'],*cache[3],*[v['location'] for v in data.values()],*[v['location'] for v in scale['source_files']]]
    effective=dict(schema_version='sumodiff.control.gradient.config.v1',training=c,model=architecture.to_dict(),
        scale_audit=scale,data=data,checkpoint=file_identity(checkpoint) if checkpoint else None,
        scope='finite complete joint gradients only; candidate weights, no optimization or quality claim')
    if probe_timesteps!=(200,): effective['probe_timesteps']=list(probe_timesteps)
    with RunRecorder(output,repository,effective,command,{'model':c['seed'],'probe_noise':c['noise_seed']},
                     purpose='bounded_control_joint_gradient_calibration',data_files=sorted(set(map(str,files))),
                     data_id=data['manifest']['id'],checkpoint=checkpoint,formal=formal) as run:
        schedule=NoiseSchedule(diffusion,device)
        roads=RoadGeometryCache(dataset,cache[2],resolve_unified_loss(c['unified_loss']))
        params=[p for p in model.parameters() if p.requires_grad]
        before=parameter_hash(model)
        rows=[];calls=0
        probes=[(name,indices[:c['batch_size']],kind) for name,indices in groups.items() for kind in ('natural',*(f't{t}' for t in probe_timesteps))]
        hard=(groups['dense_no_turn']+groups['turning'])
        if len(hard)>=16:
            probes.extend(('dense_and_turning_batch16',hard[:16],f't{t}') for t in probe_timesteps)
        started=time.perf_counter()
        for index,(group,indices,kind) in enumerate(probes):
            conditions,targets,ids=select_batch(cache,torch.tensor(indices),device)
            generator=torch.Generator().manual_seed(c['noise_seed']+index)
            noisy,noise,timestep=draw_noise(targets,conditions,schedule,generator)
            if kind!='natural':
                timestep=torch.full((len(ids),),int(kind[1:]),device=device,dtype=torch.long)
                noisy=schedule.add_noise(training_target(targets,conditions,diffusion),noise,timestep)
                noisy=torch.where(conditions['agent_mask'][...,None,None],noisy,0.)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            synchronize(device);tick=time.perf_counter()
            epsilon=model(noisy,timestep,conditions)
            out=control_objective(epsilon,noisy,noise,timestep,conditions,targets,schedule,c,roads=roads.batch(ids),ramp=1.)
            if not torch.isfinite(out['total_loss']):raise FloatingPointError('Nonfinite joint control objective')
            ge=gradient_vector(out['noise_mse'],params)
            ga=gradient_vector(out['auxiliary_total'],params)
            gt=gradient_vector(out['total_loss'],params)
            calls+=sum(v.requires_grad for v in (out['noise_mse'],out['auxiliary_total'],out['total_loss']))
            components={}
            for key in ('weighted_imitation','weighted_temporal','weighted_road','weighted_tail'):
                g=gradient_vector(out[key],params);calls+=int(out[key].requires_grad)
                if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite component gradient')
                components[key]=dict(norm=float(g.norm()),ratio_to_noise=float(g.norm()/ge.norm()) if float(ge.norm()) else None,cosine_with_noise=cosine(g,ge))
            if not all(torch.isfinite(g).all() for g in (ge,ga,gt)):
                raise FloatingPointError('Nonfinite combined gradient')
            synchronize(device)
            rows.append(dict(group=group,timestep_kind=kind,window_ids=ids,batch_size=len(ids),
                timesteps=timestep.cpu().tolist(),eligible_scenes=out['auxiliary_eligible_scenes'],
                noise_norm=float(ge.norm()),combined_auxiliary_norm=float(ga.norm()),total_norm=float(gt.norm()),
                auxiliary_to_noise_ratio=float(ga.norm()/ge.norm()) if float(ge.norm()) else None,
                auxiliary_noise_cosine=cosine(ga,ge),total_noise_cosine=cosine(gt,ge),
                combination_relative_residual=float((gt-ge-ga).norm()/(gt.norm()+1e-30)),
                would_clip_total_gradient=float(gt.norm())>c['gradient_clip_norm'],
                gradient_clip_applied=False,components=components,
                losses={k:float(v.detach()) for k,v in out.items() if isinstance(v,torch.Tensor) and v.ndim==0},
                seconds=time.perf_counter()-tick,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None))
            del out,epsilon,ge,ga,gt,g
        after=parameter_hash(model)
        if before!=after or any(p.grad is not None for p in params):raise AssertionError('Gradient probe changed parameters or gradient buffers')
        result=dict(schema_version='sumodiff.control.gradient.results.v1',run_sha=run.manifest['git']['commit_sha'],
            checkpoint=file_identity(checkpoint) if checkpoint else None,rows=rows,numerically_finite=True,
            gradient_calls=calls,model_forward_calls=len(rows),optimizer_updates=0,validation_label_arrays_loaded=0,test_arrays_loaded=0,
            parameter_sha256_before=before,parameter_sha256_after=after,body_seconds=time.perf_counter()-started,
            interpretation='short numerical calibration; does not certify weights, convergence or free generation quality')
        write_json(run.output/'gradient_audit.json',result);run.write_metrics(result)
        return result
