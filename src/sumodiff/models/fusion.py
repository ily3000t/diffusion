"""Equal-parameter hierarchical and parallel conditional fusion."""
import torch
from torch import nn
from .attention import MaskedAttention
from .encoders import SharedConditionEncoders


class ConditionEncoder(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;d=config.condition_dimension
        self.encoders=SharedConditionEncoders(config)
        self.road_query_norm=nn.LayerNorm(d);self.road_attention=MaskedAttention(d,config.attention_heads)
        self.road_norm=nn.LayerNorm(d)
        self.social_norm=nn.LayerNorm(d);self.social_attention=MaskedAttention(d,config.attention_heads)
        self.social_ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d*2),nn.SiLU(),nn.Linear(d*2,d))
        self.gate=nn.Sequential(nn.Linear(d*3,d),nn.SiLU(),nn.Linear(d,3))
        self.output=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.LayerNorm(d))

    def forward(self,c,fusion=None,return_details=False):
        fusion=fusion or self.config.fusion
        if fusion not in ('hierarchical','parallel'):raise ValueError('Unsupported fusion')
        encoded=self.encoders(c);h,route,a=encoded['history'],encoded['route'],encoded['agent_mask']
        query=self.road_query_norm(h+route)
        road=self.road_norm(route+self.road_attention(query,encoded['road_tokens'],a,encoded['road_token_mask']))
        road=torch.where(a[...,None],road,0.)
        # Only the dependency order changes: modules, keys and parameter count
        # are identical. Hierarchical social features see road context first.
        social_input=h+road if fusion=='hierarchical' else h
        normalized=self.social_norm(social_input)
        social=social_input+self.social_attention(normalized,normalized,a,a)
        social=social+self.social_ff(social)
        social=torch.where(a[...,None],social,0.)
        weights=torch.softmax(self.gate(torch.cat((h,road,social),dim=-1)),dim=-1)
        weights=torch.where(a[...,None],weights,0.)
        mixed=(torch.stack((h,road,social),dim=-2)*weights[...,None]).sum(dim=-2)
        result=torch.where(a[...,None],self.output(mixed),0.)
        if return_details:return dict(condition=result,gates=weights,history=h,road=road,social=social)
        return result
