"""Versioned tensor/primitive-only checkpoints and strict continuation metadata."""
from pathlib import Path
import os
import torch
from sumodiff.models import resolve_model_config
from .config import resolve_diffusion_config

SCHEMA='sumodiff.base.checkpoint.v1'
CAPABILITIES=dict(base=True,cfg=False,partial_diffusion=False,rolling_generation=False,guidance=False)


def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    if value is None or type(value) in (str,int,float,bool): return value
    raise TypeError(f'Unsupported checkpoint value {type(value).__name__}')


def save_checkpoint(path,model,optimizer,generator,step,metadata):
    path=Path(path)
    if path.exists(): raise FileExistsError('Refusing to overwrite checkpoint')
    payload=dict(schema_version=SCHEMA,capabilities=CAPABILITIES.copy(),step=step,
        model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
        generator_state=generator.get_state(),cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],metadata=cpu_tree(metadata))
    temporary=path.with_suffix(path.suffix+'.tmp')
    if temporary.exists(): raise FileExistsError('Checkpoint temporary path already exists')
    torch.save(payload,temporary);os.replace(temporary,path)


def load_checkpoint(path,*,signature=None,requested=None):
    value=torch.load(path,map_location='cpu',weights_only=True)
    fields={'schema_version','capabilities','step','model','optimizer','generator_state','cpu_rng','cuda_rng','metadata'}
    if not isinstance(value,dict) or set(value)!=fields or value['schema_version']!=SCHEMA:
        raise ValueError('Unsupported checkpoint; old highway/DDPO checkpoints are incompatible')
    if value['capabilities']!=CAPABILITIES: raise ValueError('Unsupported checkpoint capabilities')
    for key,enabled in (requested or {}).items():
        if key not in CAPABILITIES or type(enabled) is not bool: raise ValueError('Unknown/invalid checkpoint request')
        if enabled and not value['capabilities'][key]: raise NotImplementedError(f'Checkpoint does not support {key}')
    if type(value['step']) is not int or value['step']<0: raise ValueError('Invalid checkpoint step')
    m=value['metadata']
    required={'model_config','diffusion_config','training_config','training_signature','data','scale_audit','run_sha','run_branch','run_dirty','parent_checkpoint','runtime'}
    if set(m)!=required: raise ValueError('Incomplete checkpoint metadata')
    model=resolve_model_config(m['model_config']);diffusion=resolve_diffusion_config(m['diffusion_config'])
    if model.diffusion_embedding_steps!=diffusion.training_steps: raise ValueError('Checkpoint schedule mismatch')
    if signature is not None and signature!=m['training_signature']: raise ValueError('Resume training signature mismatch')
    return value


def restore_training(value,model,optimizer,generator,device):
    model.load_state_dict(value['model'],strict=True);optimizer.load_state_dict(value['optimizer'])
    generator.set_state(value['generator_state']);torch.set_rng_state(value['cpu_rng'])
    if torch.device(device).type=='cuda':
        if len(value['cuda_rng'])!=torch.cuda.device_count(): raise ValueError('Resume CUDA RNG/device count mismatch')
        torch.cuda.set_rng_state_all(value['cuda_rng'])
