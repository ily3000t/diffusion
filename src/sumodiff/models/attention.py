"""Masked attention with an explicit zero result for all-blocked keys."""
import math
import torch
from torch import nn


class MaskedAttention(nn.Module):
    def __init__(self,dimension,heads):
        super().__init__()
        if dimension%heads:raise ValueError('Attention dimension must divide into heads')
        self.dimension,self.heads=dimension,heads
        self.query=nn.Linear(dimension,dimension,bias=False)
        self.key=nn.Linear(dimension,dimension,bias=False)
        self.value=nn.Linear(dimension,dimension,bias=False)
        self.output=nn.Linear(dimension,dimension,bias=False)

    def forward(self,query,keys,query_mask,key_mask):
        if query.ndim!=3 or keys.ndim!=3 or query.shape[0]!=keys.shape[0] or query.shape[-1]!=self.dimension or keys.shape[-1]!=self.dimension:
            raise ValueError('Invalid attention tensor shapes')
        if query_mask.shape!=query.shape[:2] or key_mask.shape!=keys.shape[:2] or query_mask.dtype!=torch.bool or key_mask.dtype!=torch.bool or keys.shape[1]<1:
            raise ValueError('Invalid attention masks')
        query=torch.where(query_mask[...,None],query,0.)
        keys=torch.where(key_mask[...,None],keys,0.)
        def heads(value):
            return value.reshape(value.shape[0],value.shape[1],self.heads,-1).transpose(1,2)
        q,k,v=heads(self.query(query)),heads(self.key(keys)),heads(self.value(keys))
        # Float32 logits/softmax for AMP; mask before and after softmax so an
        # all-blocked row never evaluates softmax(-inf,...,-inf).
        with torch.autocast(device_type=q.device.type,enabled=False):
            logits=q.float()@k.float().transpose(-1,-2)/math.sqrt(self.dimension//self.heads)
            valid=key_mask[:,None,None,:]
            weights=torch.softmax(logits.masked_fill(~valid,torch.finfo(logits.dtype).min),dim=-1)
            weights=torch.where(valid,weights,0.)
        update=(weights.to(v.dtype)@v).transpose(1,2).reshape(query.shape)
        result=self.output(update)
        return torch.where(query_mask[...,None],result,0.)


class AgentAttention(nn.Module):
    """Social attention separately at every supplied temporal index."""
    def __init__(self,dimension,heads):
        super().__init__()
        self.norm=nn.LayerNorm(dimension)
        self.attention=MaskedAttention(dimension,heads)
        self.ff=nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,dimension*2),nn.SiLU(),nn.Linear(dimension*2,dimension))

    def forward(self,value,agent_mask):
        if value.ndim!=4 or agent_mask.shape!=value.shape[:2]:raise ValueError('Expected [B,N,T,D] and [B,N]')
        b,n,t,d=value.shape
        mask=agent_mask[:,None,:].expand(b,t,n).reshape(b*t,n)
        x=torch.where(agent_mask[:,:,None,None],value,0.).permute(0,2,1,3).reshape(b*t,n,d)
        y=self.norm(x)
        x=x+self.attention(y,y,mask,mask)
        x=x+self.ff(x)
        x=torch.where(mask[...,None],x,0.)
        return x.reshape(b,t,n,d).permute(0,2,1,3)
