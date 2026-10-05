"""Shared physical-history, spatial-raster, lane-graph and planned-route encoders."""
import torch
from torch import nn
from .inputs import validate_conditioning


def masked_mean(value,mask,dim):
    clean=torch.where(mask[...,None],value,0.)
    return clean.sum(dim=dim)/mask.sum(dim=dim).clamp_min(1)[...,None]


class HistoryEncoder(nn.Module):
    def __init__(self,config):
        super().__init__();d=config.condition_dimension;self.config=config
        self.layers=nn.ModuleList([nn.Conv1d(8,d,3,padding=1),nn.Conv1d(d,d,3,padding=2,dilation=2),nn.Conv1d(d,d,3,padding=4,dilation=4)])
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in self.layers])
        self.attributes=nn.Sequential(nn.Linear(7,d),nn.SiLU(),nn.Linear(d,d))
        self.fuse=nn.Sequential(nn.Linear(d*3,d),nn.SiLU(),nn.LayerNorm(d))

    def forward(self,c,a,hm):
        config=self.config;b,n,t,_=c['history'].shape
        scale=c['history'].new_tensor([config.condition_position_unit_m]*2+[config.condition_velocity_unit_mps]*2+[1.,1.])
        values=torch.where(hm[...,None],c['history'],0.)/scale
        clock=torch.linspace(-1.,0.,t,device=values.device,dtype=values.dtype).expand(b,n,t)
        values=torch.cat((values,hm[...,None].to(values.dtype),torch.where(hm,clock,0.)[...,None]),dim=-1).reshape(b*n,t,8)
        mask=hm.reshape(b*n,t)
        for conv,norm in zip(self.layers,self.norms):
            values=nn.functional.silu(norm(conv(values.transpose(1,2)).transpose(1,2)))
            values=torch.where(mask[...,None],values,0.)
        pooled=masked_mean(values,mask,dim=1)
        latest=(torch.arange(t,device=values.device)[None,:]*mask).max(dim=1).values
        last=values[torch.arange(b*n,device=values.device),latest]
        last=torch.where(mask.any(dim=-1)[:,None],last,0.)
        attrs=torch.where(a[...,None],c['attributes'],0.)/values.new_tensor([config.condition_length_unit_m,config.condition_width_unit_m,1.,1.])
        position=torch.where(a[...,None],c['initial_positions'],0.)/config.condition_position_unit_m
        vehicle=self.attributes(torch.cat((attrs,position,hm.float().mean(dim=-1,keepdim=True)),dim=-1)).reshape(b*n,-1)
        result=self.fuse(torch.cat((pooled,last,vehicle),dim=-1)).reshape(b,n,-1)
        return torch.where(a[...,None],result,0.)


class RasterEncoder(nn.Module):
    def __init__(self,config):
        super().__init__();layers=[];previous=3;d=config.condition_dimension;self.config=config
        for width in config.raster_channels:
            layers.extend([nn.Conv2d(previous,width,4,stride=2,padding=1),nn.GroupNorm(8,width),nn.SiLU()]);previous=width
        self.cnn=nn.Sequential(*layers);self.project=nn.Conv2d(previous,d,1)
        self.position=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,d))
        self.norm=nn.LayerNorm(d)

    def forward(self,raster,extent):
        value=self.project(self.cnn(raster.float()));b,d,h,w=value.shape
        fy=(torch.arange(h,device=value.device,dtype=torch.float32)+.5)/h
        fx=(torch.arange(w,device=value.device,dtype=torch.float32)+.5)/w
        yy,xx=torch.meshgrid(fy,fx,indexing='ij')
        x=extent[:,0,None]+xx.reshape(1,-1)*(extent[:,2]-extent[:,0])[:,None]
        y=extent[:,3,None]-yy.reshape(1,-1)*(extent[:,3]-extent[:,1])[:,None]
        coordinates=torch.stack((x,y),dim=-1)/self.config.condition_position_unit_m
        tokens=value.flatten(2).transpose(1,2)
        return self.norm(tokens+self.position(coordinates))


class LaneEncoder(nn.Module):
    def __init__(self,config):
        super().__init__();d=config.condition_dimension;self.config=config
        self.points=nn.Sequential(nn.Linear(9,d),nn.SiLU(),nn.Linear(d,d))
        self.graph=nn.Sequential(nn.Linear(d*5,d*2),nn.SiLU(),nn.Linear(d*2,d))
        self.norm=nn.LayerNorm(d)
        self.route_order=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,d))
        self.route=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.LayerNorm(d))

    def forward(self,c,lm,pm,rm):
        config=self.config
        scale=c['lane_polylines'].new_tensor([config.condition_position_unit_m]*2+[1.,1.,config.condition_width_unit_m,config.condition_velocity_unit_mps,1.,config.condition_priority_unit])
        clean=torch.where(pm[...,None],c['lane_polylines'],0.)/scale
        order=torch.linspace(0.,1.,64,device=clean.device,dtype=clean.dtype).expand(pm.shape)
        points=self.points(torch.cat((clean,torch.where(pm,order,0.)[...,None]),dim=-1))
        lane=masked_mean(points,pm,dim=2)
        pair_mask=lm[:,:,None]&lm[:,None,:]
        outgoing=c['lane_adjacency']&pair_mask
        incoming=outgoing.transpose(1,2)
        rules=c['lane_rule_relations']&pair_mask[...,None]
        messages=[]
        for relation in (outgoing,incoming,rules[...,0],rules[...,1]):
            weights=relation.to(lane.dtype)/relation.sum(dim=-1,keepdim=True).clamp_min(1)
            messages.append(weights@lane)
        lane=self.norm(lane+self.graph(torch.cat([lane,*messages],dim=-1)))
        lane=torch.where(lm[...,None],lane,0.)
        route_features=torch.where(rm[...,None],c['route_lane_features'],0.)
        route_tokens=lane[:,None]+self.route_order(route_features)
        route=self.route(masked_mean(route_tokens,rm,dim=2))
        route=torch.where(rm.any(dim=-1)[...,None],route,0.)
        return lane,route


class SharedConditionEncoders(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        self.history=HistoryEncoder(config);self.raster=RasterEncoder(config);self.lanes=LaneEncoder(config)

    def forward(self,c):
        a,hm,lm,pm,rm=validate_conditioning(c,self.config)
        history=self.history(c,a,hm)
        raster=self.raster(c['map_raster'],c['map_extent_m'])
        lanes,route=self.lanes(c,lm,pm,rm)
        tokens=torch.cat((lanes,raster),dim=1)
        token_mask=torch.cat((lm,torch.ones(raster.shape[:2],dtype=torch.bool,device=lm.device)),dim=1)
        return dict(history=history,route=route,road_tokens=tokens,road_token_mask=token_mask,agent_mask=a)
