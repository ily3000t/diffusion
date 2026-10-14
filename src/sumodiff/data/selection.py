"""Reference and neighbors use current observations and known route/map only."""
import numpy as np


def select_current(current, family, road, config):
    if not current:
        return None
    policy = config['reference_policy']
    if policy == 'first_current_id':
        reference = sorted(current)[0]
    elif policy == 'interaction_current':
        candidates = []
        focus_id = None
        if family == 'ramp_merge':
            focus_id = 'merge_end'
            for vid,obs in current.items():
                route = obs.raw['route_edges']
                if 'ramp_in' in route and 'main_out' in route and obs.raw['route_index'] < route.index('main_out'):
                    candidates.append(vid)
        elif family == 'unsignalized_intersection':
            focus_id = 'center'
            turning_routes = {(c['sumo_raw']['from'],c['sumo_raw']['to']) for c in road.connections
                              if c['sumo_raw'].get('dir','').lower() in ('l','r')}
            for vid,obs in current.items():
                route = obs.raw['route_edges'];i=obs.raw['route_index']
                if i+1<len(route) and (route[i],route[i+1]) in turning_routes:
                    candidates.append(vid)
        if candidates:
            nodes = [j['sumo_raw'] for j in road.junctions if j['sumo_raw']['id']==focus_id]
            if len(nodes)!=1:
                raise ValueError('Interaction reference requires its declared map junction')
            focus = np.array([float(nodes[0]['x']),float(nodes[0]['y'])])
            reference = min(candidates,key=lambda v:(float(np.linalg.norm(current[v].center-focus)),v))
        else:
            # Straight roads and times without an approaching planned interaction:
            # maximize observed neighborhood, then minimize mean neighbor distance.
            def density(vid):
                d=[float(np.linalg.norm(obs.center-current[vid].center)) for obs in current.values()]
                near=[x for x in d if x<=config['selection_radius_m']]
                return (-len(near),sum(near)/len(near),vid)
            reference = min(current,key=density)
    else:
        raise NotImplementedError('Unsupported current reference policy')
    origin=current[reference].center
    ranked=sorted((float(np.linalg.norm(obs.center-origin)),vid) for vid,obs in current.items() if vid!=reference)
    near=[(d,vid) for d,vid in ranked if d<=config['selection_radius_m']]
    ids=[reference]+[vid for _,vid in near[:config['max_agents']-1]]
    return reference,ids,dict(excluded_by_radius=len(ranked)-len(near),
        excluded_by_capacity=max(0,len(near)-(config['max_agents']-1)))
