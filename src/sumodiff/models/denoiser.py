"""Shared temporal U-Net with per-time bottleneck vehicle attention."""
import math
import torch
from torch import nn
from .attention import AgentAttention
from .config import ModelConfig
from .fusion import ConditionEncoder


def sinusoidal(value,dimension):
    half=dimension//2
    frequencies=torch.exp(-math.log(10000.)*torch.arange(half,device=value.device,dtype=torch.float32)/max(1,half-1))
    phases=value.float()[...,None]*frequencies
    return torch.cat((phases.sin(),phases.cos()),dim=-1)


class TemporalBlock(nn.Module):
    def __init__(self,in_channels,out_channels,embedding_dimension):
        super().__init__()
        self.first=nn.Sequential(nn.GroupNorm(8,in_channels),nn.SiLU(),nn.Conv1d(in_channels,out_channels,3,padding=1))
        self.norm=nn.GroupNorm(8,out_channels)
        self.film=nn.Sequential(nn.SiLU(),nn.Linear(embedding_dimension,out_channels*2))
        self.second=nn.Sequential(nn.SiLU(),nn.Conv1d(out_channels,out_channels,3,padding=1))
        self.skip=nn.Identity() if in_channels==out_channels else nn.Conv1d(in_channels,out_channels,1)

    def forward(self,value,embedding):
        update=self.first(value);scale,shift=self.film(embedding).chunk(2,dim=-1)
        update=self.norm(update)*(1+scale[...,None])+shift[...,None]
        return self.skip(value)+self.second(update)


class TemporalUNet(nn.Module):
    def __init__(self,config):
        super().__init__();d=config.condition_dimension;c0,c1,c2=config.unet_channels;self.config=config
        self.time=nn.Sequential(nn.Linear(d,d*2),nn.SiLU(),nn.Linear(d*2,d))
        self.future_position=nn.Linear(d,c0)
        self.input=nn.Conv1d(6,c0,3,padding=1)
        self.level0=TemporalBlock(c0,c0,d)
        self.down0=nn.Conv1d(c0,c1,4,stride=2,padding=1);self.level1=TemporalBlock(c1,c1,d)
        self.down1=nn.Conv1d(c1,c2,4,stride=2,padding=1);self.bottleneck=TemporalBlock(c2,c2,d)
        self.agents=AgentAttention(c2,config.attention_heads);self.middle=TemporalBlock(c2,c2,d)
        self.up1=nn.Conv1d(c2,c1,3,padding=1);self.decode1=TemporalBlock(c1*2,c1,d)
        self.up0=nn.Conv1d(c1,c0,3,padding=1);self.decode0=TemporalBlock(c0*2,c0,d)
        self.output=nn.Sequential(nn.GroupNorm(8,c0),nn.SiLU(),nn.Conv1d(c0,6,3,padding=1))

    def forward(self,noisy_future,timestep,condition,agent_mask):
        b,n,t,s=noisy_future.shape
        if t!=40 or s!=6 or condition.shape!=(b,n,self.config.condition_dimension) or agent_mask.shape!=(b,n):raise ValueError('Invalid denoiser shapes')
        if timestep.shape!=(b,) or timestep.dtype not in (torch.int32,torch.int64) or (timestep<0).any() or (timestep>=self.config.diffusion_embedding_steps).any():raise ValueError('Timestep must be an integer [B] in the embedding range')
        if not noisy_future.is_floating_point() or not torch.isfinite(noisy_future[agent_mask]).all():raise ValueError('Invalid active noisy future')
        if not torch.isfinite(condition[agent_mask]).all():raise ValueError('Invalid active condition')
        clean=torch.where(agent_mask[:,:,None,None],noisy_future,0.)
        condition=torch.where(agent_mask[...,None],condition,0.)
        embedding=(condition+self.time(sinusoidal(timestep,self.config.condition_dimension))[:,None]).reshape(b*n,-1)
        time_index=torch.arange(1,t+1,device=clean.device)
        position=self.future_position(sinusoidal(time_index,self.config.condition_dimension)).transpose(0,1)
        value=self.input(clean.reshape(b*n,t,6).transpose(1,2))+position[None]
        first=self.level0(value,embedding)
        second=self.level1(self.down0(first),embedding)
        value=self.bottleneck(self.down1(second),embedding)
        width,length=value.shape[1:]
        value=self.agents(value.reshape(b,n,width,length).transpose(2,3),agent_mask).transpose(2,3).reshape(b*n,width,length)
        value=self.middle(value,embedding)
        value=self.up1(nn.functional.interpolate(value,size=second.shape[-1],mode='linear',align_corners=False))
        value=self.decode1(torch.cat((value,second),dim=1),embedding)
        value=self.up0(nn.functional.interpolate(value,size=first.shape[-1],mode='linear',align_corners=False))
        value=self.decode0(torch.cat((value,first),dim=1),embedding)
        result=self.output(value).transpose(1,2).reshape(b,n,t,6)
        return torch.where(agent_mask[:,:,None,None],result,0.)


class ConditionalDenoiser(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or ModelConfig()
        self.condition_encoder=ConditionEncoder(self.config);self.unet=TemporalUNet(self.config)

    def forward(self,noisy_future,timestep,conditioning,*,fusion=None,return_details=False):
        encoded=self.condition_encoder(conditioning,fusion,return_details=return_details)
        condition=encoded['condition'] if return_details else encoded
        prediction=self.unet(noisy_future,timestep,condition,conditioning['agent_mask'])
        if return_details:return dict(pred_noise=prediction,**encoded)
        return prediction
