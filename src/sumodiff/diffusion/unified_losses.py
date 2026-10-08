"""Low-noise paired reconstruction supervision, with the shared decoder.

The road proxy checks finite oriented-body samples against vector geometry.
It is not a substitute for independent whole-body coverage evaluation.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
import shapely
import torch
from sumodiff.data.dataset import WindowDataset
from sumodiff.decoding import DecoderConfig, decode_future, observed_heading
from sumodiff.evaluation.roads import RoadConfig, _checked_geometry
from .physical_losses import scene_huber
from .process import masked_noise_mse, denormalize


@dataclass(frozen=True)
class UnifiedLossConfig:
    schema_version: str = 'sumodiff.unified.loss.config.v1'
    imitation_weight: float = .01
    temporal_weight: float = .001
    road_weight: float = .001
    max_auxiliary_timestep: int = 300
    huber_beta: float = 1.
    longitudinal_unit_m: float = 2.
    lateral_unit_m: float = .5
    velocity_unit_mps: float = 5.
    velocity_difference_unit_mps: float = 1.
    heading_factor: float = .1
    second_difference_factor: float = .1
    road_distance_unit_m: float = 1.
    road_numerical_tolerance_m: float = 1e-6
    body_grid_size: int = 5

    def __post_init__(self):
        if self.schema_version != 'sumodiff.unified.loss.config.v1':
            raise ValueError('Unsupported unified loss schema')
        for name in ('imitation_weight','temporal_weight','road_weight','heading_factor','second_difference_factor',
                     'road_numerical_tolerance_m'):
            x = getattr(self,name)
            if type(x) not in (int,float) or not math.isfinite(x) or x < 0:
                raise ValueError(f'Invalid {name}')
        for name in ('huber_beta','longitudinal_unit_m','lateral_unit_m','velocity_unit_mps',
                     'velocity_difference_unit_mps','road_distance_unit_m'):
            x = getattr(self,name)
            if type(x) not in (int,float) or not math.isfinite(x) or x <= 0:
                raise ValueError(f'Invalid {name}')
        if type(self.max_auxiliary_timestep) is not int or not 0 <= self.max_auxiliary_timestep < 1000:
            raise ValueError('Invalid auxiliary timestep')
        if self.body_grid_size != 5 or type(self.body_grid_size) is not int:
            raise NotImplementedError('V1 road proxy uses a fixed 5 by 5 body grid')

    @property
    def enabled(self):
        return bool(self.imitation_weight or self.temporal_weight or self.road_weight)

    def to_dict(self):
        return asdict(self)


def resolve_unified_loss(value):
    if not isinstance(value,dict) or set(value)-set(UnifiedLossConfig.__dataclass_fields__):
        raise ValueError('Unknown unified loss options')
    return UnifiedLossConfig(**value)


class RoadGeometryCache:
    """Static known maps only, without reading future labels or exit times."""
    def __init__(self, dataset, ids, config):
        ds = WindowDataset(dataset,'train')
        by_id = {e['window_id']:e for e in ds.entries}
        if len(ids) != len(set(ids)) or any(wid not in by_id for wid in ids):
            raise ValueError('Invalid training geometry identities')
        self.maps = {}
        self.repairs = []
        for wid in ids:
            exact = ds._json(by_id[wid],'map.json')
            geometries, repairs = prepare_roads(exact,config)
            self.maps[wid] = geometries
            self.repairs.extend(dict(window_id=wid,**r) for r in repairs)

    def batch(self, ids):
        return [self.maps[wid] for wid in ids]


def prepare_roads(exact,config):
    check = RoadConfig(numerical_tolerance_m=config.road_numerical_tolerance_m,
                       geometry_repair='bounded_make_valid')
    repairs = []
    def checked(value,name):
        if value is None:
            return None  # Inactive slot; active slots are checked at use.
        geometry,repair = _checked_geometry(value,name,check)
        if repair is not None:
            repairs.append(repair)
        if config.road_numerical_tolerance_m:
            geometry = geometry.buffer(config.road_numerical_tolerance_m)
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError('Road proxy geometry is invalid after bounded repair/buffer')
        shapely.prepare(geometry)
        return geometry
    road = checked(exact['drivable'],'drivable')
    if road is None:
        raise ValueError('Missing drivable geometry')
    return dict(road=road,routes=[checked(v,f'route:{i}') for i,v in enumerate(exact['route_corridors'])]),repairs


def body_points(states,sizes,count=5):
    """Corners, edges and interior, in the fixed shared local coordinates."""
    if states.shape[-1] != 6 or sizes.shape != (*states.shape[:-2],2):
        raise ValueError('Invalid body sample shapes')
    axis = torch.linspace(-.5,.5,count,device=states.device,dtype=states.dtype)
    u,v = torch.meshgrid(axis,axis,indexing='ij')
    grid = torch.stack((u.flatten(),v.flatten()),-1)
    offsets = sizes[...,None,None,:].to(states.dtype)*grid
    dx,dy = offsets.unbind(-1)
    sin,cos = states[...,4:6].unbind(-1)
    x = states[...,0,None]+cos[...,None]*dx-sin[...,None]*dy
    y = states[...,1,None]+sin[...,None]*dx+cos[...,None]*dy
    return torch.stack((x,y),-1)


def outside_distance(points,geometry):
    """Exact distance values and piecewise exact position gradients.

    Projection is a detached static-geometry query. For d(p,A), the gradient
    outside A is (p-proj_A(p))/d; the envelope theorem permits a detached
    closest point. Ties/corners need not have a unique gradient. No grid clamp.
    """
    values = points.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite active body sample')
    flat = values.reshape(-1,2)
    query = shapely.points(flat)
    outside = ~shapely.covers(geometry,query)
    projected = flat.copy()
    if outside.any():
        lines = shapely.shortest_line(geometry,query[outside])
        nearest = shapely.get_point(lines,0)
        projected[outside] = np.column_stack((shapely.get_x(nearest),shapely.get_y(nearest)))
        if not np.isfinite(projected[outside]).all():
            raise ValueError('Road nearest-point query failed')
    nearest = torch.as_tensor(projected.reshape(values.shape),device=points.device,dtype=points.dtype)
    mask = torch.as_tensor(outside.reshape(values.shape[:-1]),device=points.device)
    delta = torch.where(mask[...,None],points-nearest,0.)
    return torch.linalg.vector_norm(delta,dim=-1),mask


def road_proxy(states,conditioning,roads,config):
    if roads is None or len(roads) != len(states):
        raise ValueError('Road supervision requires known vector maps for every eligible scene')
    active = conditioning['agent_mask']
    sizes = torch.where(active[...,None],conditioning['attributes'][...,:2],1.)
    if not torch.isfinite(sizes).all() or (sizes<=0).any():
        raise ValueError('Invalid active vehicle sizes')
    points = body_points(states,sizes,config.body_grid_size)
    per_scene, outside_counts = [], []
    for i in range(len(states)):
        distances, masks = [], []
        for slot in torch.nonzero(active[i]).flatten().tolist():
            m = roads[i]
            if slot >= len(m['routes']) or m['routes'][slot] is None:
                raise ValueError('Missing active route corridor')
            for geometry in (m['road'],m['routes'][slot]):
                distance,mask = outside_distance(points[i,slot],geometry)
                distances.append(distance.flatten())
                masks.append(mask.flatten())
        if not distances:
            raise ValueError('Empty eligible road scene')
        residual = torch.cat(distances)/config.road_distance_unit_m
        huber = torch.nn.functional.smooth_l1_loss(residual,torch.zeros_like(residual),
                    reduction='mean',beta=config.huber_beta)
        per_scene.append(huber)
        outside_counts.append(torch.cat(masks).to(states.dtype).mean())
    return torch.stack(per_scene),torch.stack(outside_counts)


def unified_objective(epsilon,noisy,noise,timestep,conditioning,targets,schedule,config,*,roads=None,ramp=1.):
    if not 0 <= ramp <= 1 or not math.isfinite(ramp):
        raise ValueError('Auxiliary ramp must be in [0,1]')
    mse = masked_noise_mse(epsilon,noise,conditioning['agent_mask'],targets['future_mask'])
    zero = mse.new_zeros(())
    out = dict(noise_mse=mse,longitudinal=zero,lateral=zero,velocity=zero,heading=zero,
               imitation=zero,first_difference=zero,second_difference=zero,temporal=zero,road=zero,
               weighted_imitation=zero,weighted_temporal=zero,weighted_road=zero,
               auxiliary_total=zero,total_loss=mse,auxiliary_eligible_scenes=0,
               road_proxy_outside_fraction=zero,heading_fallback_frames=0,auxiliary_ramp=float(ramp))
    if not config.enabled:
        return out
    if config.max_auxiliary_timestep >= schedule.config.training_steps:
        raise ValueError('Auxiliary timestep outside schedule')
    eligible = torch.nonzero(timestep<=config.max_auxiliary_timestep).flatten()
    out['auxiliary_eligible_scenes'] = len(eligible)
    if not len(eligible) or ramp == 0:
        return out
    c = {k:v[eligible] for k,v in conditioning.items()}
    a = c['agent_mask']
    fm = targets['future_mask'][eligible] & a[...,None]
    if not fm[a].all() or not c['history_mask'][a][...,-2:].all():
        raise ValueError('Unified decoder supervision requires complete curated future and last two history points')
    clean = schedule.clean_from_noise(noisy[eligible],epsilon[eligible],timestep[eligible])
    raw = denormalize(torch.where(a[...,None,None],clean,0.),schedule.config)
    truth = torch.where(a[...,None,None],targets['future'][eligible],0.).to(raw.dtype)
    h0 = observed_heading(c['history'],c['history_mask'],a)
    prediction = decode_future(raw,c['initial_positions'],h0,a,DecoderConfig())
    with torch.no_grad():
        label = decode_future(truth,c['initial_positions'],h0,a,DecoderConfig())
    alpha = schedule.alpha_bar[timestep[eligible]].to(prediction.states.dtype).detach()
    def aggregate(scene):
        return (scene*alpha).sum()/len(epsilon)
    def huber(x,mask=fm):
        return aggregate(scene_huber(x,mask,config.huber_beta))
    if config.imitation_weight:
        heading = label.states[...,4:6]
        tangent = torch.stack((heading[...,1],heading[...,0]),-1)
        normal = torch.stack((-heading[...,0],heading[...,1]),-1)
        delta = prediction.states[...,:2]-label.states[...,:2]
        out['longitudinal'] = huber((delta*tangent).sum(-1,keepdim=True)/config.longitudinal_unit_m)
        out['lateral'] = huber((delta*normal).sum(-1,keepdim=True)/config.lateral_unit_m)
        out['velocity'] = huber((prediction.states[...,2:4]-label.states[...,2:4])/config.velocity_unit_mps)
        out['heading'] = huber(raw[...,4:6]-heading)
        out['imitation'] = out['longitudinal']+out['lateral']+out['velocity']+config.heading_factor*out['heading']
    if config.temporal_weight:
        observed = torch.where(a[...,None,None],c['history'][...,-2:,2:4],0.).to(prediction.states.dtype)
        p = torch.cat((observed,prediction.states[...,2:4]),-2)
        q = torch.cat((observed,label.states[...,2:4]),-2)
        first = (torch.diff(p,dim=-2)-torch.diff(q,dim=-2))[...,1:,:]
        second = torch.diff(p,n=2,dim=-2)-torch.diff(q,n=2,dim=-2)
        first_mask = a[...,None].expand(*a.shape,first.shape[-2])
        second_mask = a[...,None].expand(*a.shape,second.shape[-2])
        out['first_difference'] = huber(first/config.velocity_difference_unit_mps,first_mask)
        out['second_difference'] = huber(second/config.velocity_difference_unit_mps,second_mask)
        out['temporal'] = out['first_difference']+config.second_difference_factor*out['second_difference']
    if config.road_weight:
        selected = [roads[i] for i in eligible.tolist()] if roads is not None else None
        scene,outside = road_proxy(prediction.states,c,selected,config)
        out['road'] = aggregate(scene)
        out['road_proxy_outside_fraction'] = outside.mean().detach()
    out['heading_fallback_frames'] = int(prediction.heading_fallback_mask.sum())
    out['weighted_imitation'] = ramp*config.imitation_weight*out['imitation']
    out['weighted_temporal'] = ramp*config.temporal_weight*out['temporal']
    out['weighted_road'] = ramp*config.road_weight*out['road']
    out['auxiliary_total'] = out['weighted_imitation']+out['weighted_temporal']+out['weighted_road']
    out['total_loss'] = mse+out['auxiliary_total']
    return out
