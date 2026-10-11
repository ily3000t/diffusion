"""Physical all-horizon jerk tail; shared with fixed fresh-training objectives."""
from dataclasses import asdict, dataclass
import math
import torch
import torch.nn.functional as F
from sumodiff.decoding import DecoderConfig, decode_future, observed_heading
from .process import denormalize, masked_noise_mse
from .raw_velocity_losses import resolve_raw_velocity_loss, raw_velocity_objective

@dataclass(frozen=True)
class JerkTailConfig:
    threshold_mps3: float = 15.
    top_fraction: float = .1
    top_multiplier: float = 1.
    huber_beta: float = 1.
    dt: float = .1

    def __post_init__(self):
        for k in ('threshold_mps3','top_fraction','top_multiplier','huber_beta','dt'):
            v = getattr(self,k)
            if type(v) not in (float,int) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'Invalid tail {k}')
        if self.top_fraction > 1 or self.dt != .1:
            raise ValueError('Tail requires fraction<=1 and dt=0.1')

    def to_dict(self):
        return asdict(self)

def physical_jerk(states, conditioning, future_mask):
    """Backward third position difference; t0 is exactly the fixed observation."""
    a = conditioning['agent_mask']
    mask = future_mask & a[...,None]
    if states.shape[-2:] != (40,6) or mask.shape != states.shape[:-1]:
        raise ValueError('Jerk supervision requires 40 future frames')
    if not conditioning['history_mask'][a][...,-3:].all() or not mask[a].all():
        raise ValueError('Tail supervision requires complete active H+F')
    h = torch.where(a[...,None,None],conditioning['history'][...,-3:,:2],0.).to(states.dtype).clone()
    h[...,-1,:] = torch.where(a[...,None],conditioning['initial_positions'],0.).to(states.dtype)
    p = torch.where(mask[...,None],states[...,:2],0.)
    if not torch.isfinite(p).all() or not torch.isfinite(h).all():
        raise ValueError('Nonfinite valid physical position')
    # Same operation order as independent physical_derivatives evaluation.
    j = torch.diff(torch.diff(torch.diff(torch.cat((h,p),-2),dim=-2)/.1,dim=-2)/.1,dim=-2)/.1
    return j,mask

def jerk_tail_scenes(states, conditioning, future_mask, config):
    jerk,mask = physical_jerk(states,conditioning,future_mask)
    excess = torch.relu(torch.linalg.vector_norm(jerk,dim=-1)/config.threshold_mps3-1)
    e = F.smooth_l1_loss(excess,torch.zeros_like(excess),reduction='none',beta=config.huber_beta)
    means,tops,counts = [],[],[]
    for i in range(len(states)):
        valid = e[i][mask[i]]
        if not valid.numel():
            raise ValueError('No valid tail elements in scene')
        k = max(1,math.ceil(config.top_fraction*valid.numel()))
        means.append(valid.mean());tops.append(valid.topk(k).values.mean());counts.append(k)
    mean,top = torch.stack(means),torch.stack(tops)
    return mean+config.top_multiplier*top,mean,top,counts

def fresh_objective(epsilon,noisy,noise,timestep,conditioning,targets,schedule,
                    raw_config,tail_config,tail_weight,*,roads=None,ramp=1.):
    if type(tail_weight) not in (float,int) or not math.isfinite(tail_weight) or tail_weight <= 0:
        raise ValueError('Tail weight must be calibrated and positive')
    if type(ramp) not in (float,int) or not math.isfinite(ramp) or not 0<=ramp<=1:
        raise ValueError('Invalid unified auxiliary warmup')
    out = raw_velocity_objective(epsilon,noisy,noise,timestep,conditioning,targets,schedule,
                                 raw_config,roads=roads,ramp=1.)
    baseline = out['auxiliary_total']
    zero = out['noise_mse'].new_zeros(())
    out.update(tail=zero,tail_mean=zero,tail_top=zero,tail_top_counts=[],
               baseline_auxiliary=baseline,auxiliary_ramp=float(ramp))
    eligible = torch.nonzero(timestep<=raw_config.unified.max_auxiliary_timestep).flatten()
    if len(eligible):
        c = {k:v[eligible] for k,v in conditioning.items()}
        a = c['agent_mask']
        clean = schedule.clean_from_noise(noisy[eligible],epsilon[eligible],timestep[eligible])
        raw = denormalize(torch.where(a[...,None,None],clean,0.),schedule.config)
        decoded = decode_future(raw,c['initial_positions'],
            observed_heading(c['history'],c['history_mask'],a),a,DecoderConfig())
        loss,mean,top,counts = jerk_tail_scenes(decoded.states,c,targets['future_mask'][eligible],tail_config)
        alpha = schedule.alpha_bar[timestep[eligible]].to(loss.dtype).detach()
        for key,term in (('tail',loss),('tail_mean',mean),('tail_top',top)):
            out[key] = (term*alpha).sum()/len(epsilon)
        out['tail_top_counts'] = counts
    for key in ('weighted_imitation','weighted_temporal','weighted_road',
                'weighted_raw_boundary','weighted_raw_interior'):
        out[key] = ramp*out[key]
    out['raw_auxiliary_total'] = ramp*out['raw_auxiliary_total']
    out['inherited_auxiliary_total'] = ramp*out['inherited_auxiliary_total']
    out['raw_auxiliary_ramp'] = float(ramp)
    out['inherited_auxiliary_ramp'] = float(ramp)
    out['weighted_tail'] = ramp*tail_weight*out['tail']
    out['auxiliary_total'] = ramp*(baseline+tail_weight*out['tail'])
    out['total_loss'] = out['noise_mse']+out['auxiliary_total']
    return out
