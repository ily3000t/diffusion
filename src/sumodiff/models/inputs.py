"""Conditioning-only adapter: routes/rules from t0, never future labels."""
import numpy as np
import torch

BASE_FIELDS={'history','agent_mask','history_mask','attributes','initial_positions','map_raster',
    'lane_polylines','lane_mask','lane_point_mask','lane_adjacency','route_lane_mask'}
MODEL_FIELDS=BASE_FIELDS|{'map_extent_m','lane_rule_relations','route_lane_features'}


def lane_rules(exact):
    """Movement anchors: first via lane, or source when SUMO has no via."""
    ids={lane['id']:i for i,lane in enumerate(exact['lanes'])}
    if len(ids)!=len(exact['lanes']):raise ValueError('Duplicate lane IDs')
    links={}
    for c in exact['connections']:
        if 'junction_id' in c and 'request_index' in c:
            key=(c['junction_id'],c['request_index'])
            if key in links:raise ValueError('Duplicate junction request index')
            links[key]=ids[c['via_lane'] or c['source_lane']]
    result=np.zeros((len(ids),len(ids),2),bool)
    for junction in exact['junctions']:
        jid=junction['sumo_raw']['id']
        for request in junction['requests']:
            key=(jid,int(request['index']))
            if key not in links:raise ValueError('Junction request has no movement anchor')
            a=links[key]
            for channel,name in enumerate(('yield_to_indices','foe_indices')):
                for index in request[name]:
                    other=(jid,int(index))
                    if other not in links:raise ValueError('Missing referenced request')
                    result[a,links[other],channel]=True
    return result


def route_features(exact,meta,route_mask,agents):
    """Known route order/progress on allowed lanes; no exit-time information."""
    lanes=exact['lanes'];ids={lane['id']:i for i,lane in enumerate(lanes)}
    successors={}
    for connection in exact['connections']:successors.setdefault(connection['source_lane'],[]).append(connection)
    result=np.zeros((*route_mask.shape,2),np.float32)
    for slot in np.flatnonzero(agents):
        route=meta['planned_routes'][slot];index=meta['route_indices_at_t0'][slot]
        if not route or type(index) is not int or not 0<=index<len(route):raise ValueError('Invalid t0 planned route/progress')
        if len(set(route))!=len(route):raise NotImplementedError('Repeated-edge planned routes require a richer route interface')
        order={edge:float(i) for i,edge in enumerate(route)}
        lane_order={lane['id']:order[lane['edge_id']] for lane in lanes if lane['edge_id'] in order}
        for start,end in zip(route[:-1],route[1:]):
            movements=[c for c in exact['connections'] if c['sumo_raw']['from']==start and c['sumo_raw']['to']==end]
            if not movements:raise ValueError('Disconnected planned route')
            for movement in movements:
                cursor=movement;visited=set()
                while cursor['via_lane']:
                    via=cursor['via_lane']
                    if via in visited:raise ValueError('Internal route cycle')
                    visited.add(via);lane_order[via]=(order[start]+order[end])/2
                    following=[c for c in successors.get(via,[]) if c['target_lane']==movement['target_lane']]
                    if len(following)!=1:raise ValueError('Ambiguous internal route continuation')
                    cursor=following[0]
        for lane in np.flatnonzero(route_mask[slot]):
            lid=lanes[int(lane)]['id']
            if lid not in lane_order:raise ValueError('Allowed lane lacks planned order')
            rank=lane_order[lid]
            result[slot,lane]=[(rank-index)/max(1,len(route)-1),float(rank>=index)]
    return result


def prepare_conditioning(batch,device='cpu'):
    """Read collated conditioning and current metadata; targets are untouched."""
    source=batch['conditioning']
    if set(source)!=BASE_FIELDS:raise ValueError('Unexpected/missing conditioning fields; labels are forbidden')
    agents=np.asarray(source['agent_mask'],bool)
    b,n=agents.shape;l=source['lane_mask'].shape[1]
    if len(batch['exact_map'])!=b or len(batch['input_metadata'])!=b:raise ValueError('Invalid metadata batch size')
    relations=np.zeros((b,l,l,2),bool);features=np.zeros((b,n,l,2),np.float32);extents=[]
    for i,(exact,meta) in enumerate(zip(batch['exact_map'],batch['input_metadata'])):
        count=len(exact['lanes'])
        if count>l or not np.asarray(source['lane_mask'][i,:count],bool).all() or np.asarray(source['lane_mask'][i,count:],bool).any():
            raise ValueError('Map lane order/count differs from collated tensors')
        if meta['raster_row_direction']!='positive_y_to_negative_y':raise ValueError('Unsupported raster orientation')
        relations[i,:count,:count]=lane_rules(exact)
        features[i,:,:count]=route_features(exact,meta,np.asarray(source['route_lane_mask'][i,:,:count],bool),agents[i])
        extents.append(meta['map_extent_m'])
    values={**source,'lane_rule_relations':relations,'route_lane_features':features,'map_extent_m':np.asarray(extents,np.float32)}
    # stage2 raster is binary uint8 0/1, not 0/255. No guessed rescaling.
    return {key:torch.as_tensor(np.array(value,copy=True),device=device,
        dtype=torch.bool if np.asarray(value).dtype==bool else torch.float32) for key,value in values.items()}


def validate_conditioning(c,config):
    if set(c)!=MODEL_FIELDS:raise ValueError('Unexpected/missing model fields; future masks and labels are forbidden')
    h=c['history'];b,n=h.shape[:2]
    if n<1 or n>config.max_agents or h.shape!=(b,n,21,6):raise ValueError('Invalid history/agent count')
    l=c['lane_mask'].shape[1]
    expected=dict(agent_mask=(b,n),history_mask=(b,n,21),attributes=(b,n,4),initial_positions=(b,n,2),
        map_raster=(b,3,256,256),lane_polylines=(b,l,64,8),lane_mask=(b,l),lane_point_mask=(b,l,64),
        lane_adjacency=(b,l,l),route_lane_mask=(b,n,l),lane_rule_relations=(b,l,l,2),route_lane_features=(b,n,l,2),map_extent_m=(b,4))
    if l<1 or b<1:raise ValueError('Empty batch/map')
    for name,shape in expected.items():
        if c[name].shape!=shape or c[name].device!=h.device:raise ValueError(f'Invalid {name} shape/device')
    bool_fields={'agent_mask','history_mask','lane_mask','lane_point_mask','lane_adjacency','route_lane_mask','lane_rule_relations'}
    for name in bool_fields:
        if c[name].dtype!=torch.bool:raise ValueError(f'{name} must be boolean')
    a=c['agent_mask'];hm=c['history_mask']&a[...,None]
    lm=c['lane_mask']&c['lane_point_mask'].any(dim=-1)
    pm=c['lane_point_mask']&lm[...,None]
    if not a.any(dim=1).all():raise ValueError('Empty selected vehicle task')
    applicable={'history':hm,'attributes':a,'initial_positions':a,'lane_polylines':pm,
        'route_lane_features':c['route_lane_mask']&a[...,None]&lm[:,None,:]}
    for name,mask in applicable.items():
        if not c[name].is_floating_point() or not torch.isfinite(c[name][mask]).all():raise ValueError(f'Invalid active {name}')
    if not torch.isfinite(c['map_raster']).all() or (c['map_raster']<0).any() or (c['map_raster']>1).any():raise ValueError('Raster must be binary/unit-scaled, not 0/255')
    extent=c['map_extent_m']
    if not torch.isfinite(extent).all() or (extent[:,2:]<=extent[:,:2]).any():raise ValueError('Invalid map extent')
    attrs=c['attributes'][a]
    if (attrs[:,:2]<=0).any() or (attrs[:,2]!=1).any() or ((attrs[:,3]!=0)&(attrs[:,3]!=1)).any():raise ValueError('Invalid passenger attributes')
    if not (torch.where(a,c['attributes'][...,3],0.).sum(dim=1)==1).all():raise ValueError('Exactly one reference marker per scene required')
    rm=c['route_lane_mask']&lm[:,None,:]&a[...,None]
    if not rm.any(dim=-1)[a].all():raise ValueError('Active vehicle has no valid planned-route tokens')
    return a,hm,lm,pm,rm
