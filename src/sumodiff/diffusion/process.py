"""DDPM noise objective and DDIM Eq.12, with no clipping of physical states."""
import math
import torch
from .config import DiffusionConfig


def scales(value,config):
    return value.new_tensor([config.position_scale_m]*2+[config.velocity_scale_mps]*2+[1.,1.])


def normalize(value,config):
    if value.shape[-1]!=6: raise ValueError('Expected six-state trajectory')
    return value/scales(value,config)


def denormalize(value,config):
    if value.shape[-1]!=6: raise ValueError('Expected six-state trajectory')
    return value*scales(value,config)


def masked_noise_mse(prediction,noise,agent_mask,future_mask):
    if prediction.shape!=noise.shape or prediction.ndim!=4 or prediction.shape[-1]!=6: raise ValueError('Invalid noise shape')
    if agent_mask.dtype!=torch.bool or future_mask.dtype!=torch.bool or agent_mask.shape!=prediction.shape[:2] or future_mask.shape!=prediction.shape[:3]: raise ValueError('Invalid loss masks')
    valid=agent_mask[:,:,None]&future_mask
    counts=valid.sum((1,2))*prediction.shape[-1]
    if (counts==0).any(): raise ValueError('A scene has no supervised elements')
    p=torch.where(valid[...,None],prediction,0.)
    q=torch.where(valid[...,None],noise,0.)
    if not torch.isfinite(p).all() or not torch.isfinite(q).all(): raise ValueError('Nonfinite supervised noise')
    return ((p-q).square().sum((1,2,3))/counts).mean()


class NoiseSchedule:
    def __init__(self,config=None,device='cpu'):
        self.config=config or DiffusionConfig()
        beta=torch.linspace(self.config.beta_start,self.config.beta_end,self.config.training_steps,dtype=torch.float64,device=device)
        self.alpha_bar=(1.-beta).cumprod(0)

    def coefficients(self,timestep,value):
        if timestep.shape!=(value.shape[0],) or timestep.dtype!=torch.int64 or (timestep<0).any() or (timestep>=len(self.alpha_bar)).any(): raise ValueError('Invalid training timestep')
        a=self.alpha_bar[timestep].to(value.dtype)
        return a.reshape(-1,*([1]*(value.ndim-1)))

    def add_noise(self,clean,noise,timestep):
        if clean.shape!=noise.shape: raise ValueError('Noise and state shapes differ')
        a=self.coefficients(timestep,clean)
        return a.sqrt()*clean+(1-a).sqrt()*noise

    def clean_from_noise(self,noisy,epsilon,timestep):
        a=self.coefficients(timestep,noisy)
        return (noisy-(1-a).sqrt()*epsilon)/a.sqrt()

    def noise_from_clean(self,noisy,clean,timestep):
        a=self.coefficients(timestep,noisy)
        return (noisy-a.sqrt()*clean)/(1-a).sqrt()

    def timesteps(self,steps):
        if type(steps) is not int or not 1<=steps<=len(self.alpha_bar): raise ValueError('Invalid DDIM step count')
        # Include the noisiest training timestep and, for >1 steps, timestep 0.
        return torch.linspace(len(self.alpha_bar)-1,0,steps,dtype=torch.float64).round().long().tolist()

    def transition(self,noisy,clean,epsilon,current,previous,eta=0.,noise=None):
        if type(current) is not int or type(previous) is not int or not -1<=previous<current<len(self.alpha_bar): raise ValueError('Invalid DDIM timestep transition')
        if not isinstance(eta,(float,int)) or isinstance(eta,bool) or not math.isfinite(eta) or not 0<=eta<=1: raise ValueError('eta must be in [0,1]')
        if noisy.shape!=clean.shape or noisy.shape!=epsilon.shape: raise ValueError('DDIM state shape mismatch')
        a=self.alpha_bar[current].to(noisy.dtype)
        ap=self.alpha_bar[previous].to(noisy.dtype) if previous>=0 else noisy.new_tensor(1.)
        variance=((1-ap)/(1-a)*(1-a/ap)).clamp_min(0)
        sigma=eta*variance.sqrt()
        result=ap.sqrt()*clean+(1-ap-sigma.square()).clamp_min(0).sqrt()*epsilon
        if eta and previous>=0:
            if noise is None or noise.shape!=noisy.shape: raise ValueError('Stochastic DDIM requires explicit matching noise')
            result=result+sigma*noise
        return result


@torch.no_grad()
def ddim_sample(model,conditioning,initial_noise,schedule,steps=20,eta=0.,generator=None,**options):
    if options: raise NotImplementedError(f'Unsupported sampling options: {sorted(options)}')
    if model.training: raise ValueError('Sampling requires eval mode')
    if model.config.diffusion_embedding_steps!=schedule.config.training_steps: raise ValueError('Embedding/schedule mismatch')
    if initial_noise.shape!=(conditioning['agent_mask'].shape[0],conditioning['agent_mask'].shape[1],40,6): raise ValueError('Invalid initial noise shape')
    mask=conditioning['agent_mask'][:,:,None,None]
    value=torch.where(mask,initial_noise,0.)
    if not torch.isfinite(value).all(): raise ValueError('Nonfinite initial noise')
    condition=model.condition_encoder(conditioning)
    sequence=schedule.timesteps(steps)
    for i,current in enumerate(sequence):
        time=torch.full((len(value),),current,dtype=torch.long,device=value.device)
        epsilon=model.unet(value,time,condition,conditioning['agent_mask'])
        clean=schedule.clean_from_noise(value,epsilon,time)
        previous=sequence[i+1] if i+1<len(sequence) else -1
        noise=torch.randn(value.shape,device=value.device,dtype=value.dtype,generator=generator) if eta and previous>=0 else None
        value=schedule.transition(value,clean,epsilon,current,previous,eta,noise)
        value=torch.where(mask,value,0.)
        if not torch.isfinite(value).all(): raise FloatingPointError('Nonfinite DDIM output')
    return value
