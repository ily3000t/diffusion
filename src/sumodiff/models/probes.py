"""Short network checks and resource measurements; no optimizer or training."""
import hashlib
import time
import numpy as np
import torch
from .config import ModelConfig
from .inputs import prepare_conditioning,validate_conditioning
from sumodiff.data.dataset import WindowDataset,collate_numpy

VEHICLE_FIELDS={'history','history_mask','agent_mask','attributes','initial_positions','route_lane_mask','route_lane_features'}


def parameter_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def collect_inputs(dataset_path,config):
    dataset=WindowDataset(dataset_path)
    samples=[];selected={};sources=[];counts={'train':0,'validation':0};max_lanes=max_agents=missing_history=0
    for index,entry in enumerate(dataset.entries):
        if entry['split']=='test':continue
        sample=dataset.inference(index)
        c=prepare_conditioning(collate_numpy([sample]));a,hm,_,_,_=validate_conditioning(c,config)
        counts[entry['split']]+=1;max_lanes=max(max_lanes,c['lane_mask'].shape[1]);max_agents=max(max_agents,int(a.sum()))
        missing_history+=int((a&~hm.all(dim=-1)).sum())
        sources.extend(str(dataset.directory/entry['files'][name]['relative_path']) for name in ('conditioning.npz','input.json','map.json'))
        family=sample['input_metadata']['family']
        # Only t0 selected counts decide this resource task; no future completeness.
        rank=(int(a.sum()),sample['input_metadata']['window_id'])
        if family not in selected or rank>selected[family][0]:selected[family]=(rank,sample)
    if len(selected)!=3 or not all(counts.values()):raise ValueError('Need train/validation inputs covering three scene families')
    samples=[selected[key][1] for key in sorted(selected)]
    c=prepare_conditioning(collate_numpy(samples))
    ids=[sample['input_metadata']['window_id'] for sample in samples]
    report=dict(inference_windows=sum(counts.values()),split_counts=counts,max_map_lanes=max_lanes,
        max_observed_selected_agents=max_agents,agents_with_incomplete_history=missing_history,
        selected_window_ids=ids,selection='largest current selected count per family; ID tie-break; train/validation only',
        future_labels_accessed=False)
    return c,report,sources


def full_capacity_conditioning(source,batch_size):
    # Synthetic load uses one known map/route and 12 observed passenger slots.
    # It is not a SUMO dense-traffic sample or a new training datum.
    i=int(source['lane_mask'].sum(dim=-1).argmax());n=source['agent_mask'].shape[1]
    c={key:value[i:i+1].repeat(batch_size,*([1]*(value.ndim-1))) for key,value in source.items()}
    for key in VEHICLE_FIELDS:
        c[key]=c[key][:,:1].repeat(1,n,*([1]*(c[key].ndim-2)))
    c['agent_mask'].fill_(True)
    c['attributes'][...,3]=0;c['attributes'][:,0,3]=1
    offset=torch.arange(n,dtype=torch.float32)*8.
    c['history'][...,0]+=offset[None,:,None]
    c['initial_positions'][...,0]+=offset[None,:]
    return c


def seeded_noise(c,seed,embedding_steps):
    device=c['history'].device;generator=torch.Generator(device=device).manual_seed(seed)
    b,n=c['agent_mask'].shape
    noise=torch.randn(b,n,40,6,device=device,generator=generator)
    timestep=(torch.arange(b,device=device,dtype=torch.long)*197+100)%embedding_steps
    return noise,timestep


def invariance_probe(model,c,seed):
    noise,timestep=seeded_noise(c,seed,model.config.diffusion_embedding_steps);a=c['agent_mask'];order=torch.arange(a.shape[1]-1,-1,-1,device=a.device)
    changed={key:value[:,order] if key in VEHICLE_FIELDS else value for key,value in c.items()}
    records={};saved=dict(noisy_future=noise.cpu().numpy(),timestep=timestep.cpu().numpy(),agent_mask=a.cpu().numpy())
    for mode in ('hierarchical','parallel'):
        with torch.no_grad():
            original=model(noise,timestep,c,fusion=mode,return_details=True)
            permuted=model(noise[:,order],timestep,changed,fusion=mode,return_details=True)
        error=float((permuted['pred_noise']-original['pred_noise'][:,order]).abs().max())
        condition_error=float((permuted['condition']-original['condition'][:,order]).abs().max())
        assert error<2e-5 and condition_error<2e-5
        poison={key:value.clone() for key,value in c.items()};poison_noise=noise.clone()
        hm=poison['history_mask']&a[...,None];pm=poison['lane_point_mask']&poison['lane_mask'][...,None]
        poison['history'][~hm]=float('nan');poison['lane_polylines'][~pm]=float('nan')
        for key in ('attributes','initial_positions'):poison[key][~a]=float('nan')
        poison['route_lane_features'][~poison['route_lane_mask']]=float('nan');poison_noise[~a]=float('nan')
        poison_noise.requires_grad_(True)
        actual=model(poison_noise,timestep,poison,fusion=mode)
        payload_error=float((actual.detach()-original['pred_noise']).abs().max())
        assert payload_error<2e-5 and not actual[~a].any()
        actual[a].square().mean().backward()
        assert torch.isfinite(poison_noise.grad).all() and not poison_noise.grad[~a].any()
        nonfinite=[name for name,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not nonfinite,nonfinite
        model.zero_grad(set_to_none=True)
        records[mode]=dict(vehicle_permutation_max_abs_error=error,condition_permutation_max_abs_error=condition_error,
            masked_payload_max_abs_error=payload_error,all_parameter_gradients_finite=True,padding_gradient_zero=True,
            pred_noise_shape=list(actual.shape),condition_shape=list(original['condition'].shape))
        for name in ('pred_noise','condition','gates'):saved[f'{mode}_{name}']=original[name].cpu().numpy()
    records['same_parameter_fusion_output_difference_max']=float(np.abs(saved['hierarchical_pred_noise']-saved['parallel_pred_noise']).max())
    return records,saved


def profile(model,source,device,mode,seed,iterations,warmups):
    c={key:value.to(device) for key,value in source.items()};noise,timestep=seeded_noise(c,seed,model.config.diffusion_embedding_steps)
    cuda=device.type=='cuda'
    def synchronize():
        if cuda:torch.cuda.synchronize(device)
    for _ in range(warmups):
        prediction=model(noise,timestep,c,fusion=mode)
        prediction[c['agent_mask']].square().mean().backward();model.zero_grad(set_to_none=True)
        del prediction
    synchronize()
    if cuda:torch.cuda.reset_peak_memory_stats(device)
    forward_start=time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            prediction=model(noise,timestep,c,fusion=mode)
            # Check after the measured loop, not once per CUDA iteration.
    synchronize();forward_seconds=(time.perf_counter()-forward_start)/iterations
    assert torch.isfinite(prediction).all();del prediction
    start=time.perf_counter()
    for _ in range(iterations):
        prediction=model(noise,timestep,c,fusion=mode)
        prediction[c['agent_mask']].square().mean().backward()
        model.zero_grad(set_to_none=True);del prediction
    synchronize();backward_seconds=(time.perf_counter()-start)/iterations
    result=dict(status='completed',fusion=mode,batch_size=int(c['agent_mask'].shape[0]),agent_slots=int(c['agent_mask'].shape[1]),
        active_agents_per_scene=c['agent_mask'].sum(dim=-1).cpu().tolist(),map_lanes=c['lane_mask'].sum(dim=-1).cpu().tolist(),
        input_dtype=str(noise.dtype),forward_mean_seconds=forward_seconds,forward_backward_mean_seconds=backward_seconds,
        measured_iterations=iterations,warmup_iterations=warmups,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else None,
        scope='full network, input and parameter gradients; no optimizer states, data loader, diffusion, decoder or training')
    return result
