"""Offline motion labels and deterministic capped selection, never inference conditions."""
from collections import Counter,defaultdict,deque
import hashlib,json
import numpy as np
from .motion_coverage import raw_tracks,lane_changes,ramp_exit_events,indices_at,future_motion

KINDS=('lane_change','ramp_exit','turn')

def event_id(episode,kind,vehicle,tick):
    return hashlib.sha256(json.dumps([episode,kind,vehicle,tick],separators=(',',':')).encode()).hexdigest()

class MotionIndex:
    def __init__(self,episode,road):
        self.episode=episode
        self.tracks,_=raw_tracks(episode.manifest_path.parent/'frames.jsonl')
        self.changes=lane_changes(episode.manifest_path.parent/'lanechanges.xml')
        self.exits={v:ramp_exit_events(t) for v,t in self.tracks.items()}
        self.turn_lanes={};self.catalog=[]
        for vid,track in self.tracks.items():
            mainline='ramp_in' not in track['route_edges']
            if episode.manifest['family']=='three_lane_straight' or (episode.manifest['family']=='ramp_merge' and mainline):
                for e in self.changes.get(vid,[]):
                    self.catalog.append(dict(event_id=event_id(episode.key,'lane_change',vid,e['tick']),
                        episode_key=episode.key,kind='lane_change',vehicle_id=vid,tick=e['tick']))
            for tick in self.exits[vid]:
                self.catalog.append(dict(event_id=event_id(episode.key,'ramp_exit',vid,tick),
                    episode_key=episode.key,kind='ramp_exit',vehicle_id=vid,tick=tick))
            route=track['route_edges']
            if episode.manifest['family']=='unsignalized_intersection' and len(route)==2:
                links=[c for c in road.connections if c['sumo_raw'].get('from')==route[0] and c['sumo_raw'].get('to')==route[1]
                    and c['sumo_raw'].get('dir') in ('l','L','r','R')]
                if links:
                    self.turn_lanes[vid]={lane for lane in road.route_lane_ids(route) if lane.startswith(':')}
                    self.catalog.append(dict(event_id=event_id(episode.key,'turn',vid,None),episode_key=episode.key,
                        kind='turn',vehicle_id=vid,tick=None,definition='one planned-turn vehicle maneuver; future observed heading and internal-lane evidence required'))

    def labels(self,ids,t0,complete):
        if not complete:return None,None
        result={k:[] for k in KINDS}
        family=self.episode.manifest['family']
        for vid in ids:
            track=self.tracks[vid]
            if family=='three_lane_straight' or (family=='ramp_merge' and 'ramp_in' not in track['route_edges']):
                result['lane_change'] += [event_id(self.episode.key,'lane_change',vid,e['tick'])
                    for e in self.changes.get(vid,[]) if t0<e['tick']<=t0+40]
            result['ramp_exit'] += [event_id(self.episode.key,'ramp_exit',vid,t) for t in self.exits[vid] if t0<t<=t0+40]
            if vid in self.turn_lanes:
                idx=indices_at(track,t0,t0+40);motion=future_motion(track,t0)
                if (idx is not None and motion is not None and motion['heading_span_rad']>=np.deg2rad(5)
                    and any(track['lane'][i] in self.turn_lanes[vid] for i in idx[1:])):
                    result['turn'].append(event_id(self.episode.key,'turn',vid,None))
        result={k:sorted(set(v)) for k,v in result.items()}
        return {k:bool(v) for k,v in result.items()},result

def select_motion(rows,requirement,seed):
    """Bounded deterministic heuristic, not a proof that an unsatisfied pool is infeasible."""
    pool=[r for r in rows if r['eligible']]
    if len({r['window_id'] for r in pool})!=len(pool):raise ValueError('Duplicate candidate window identity')
    policies=requirement['policy_quotas'];kinds=list(requirement['minimum_windows'])
    best=None;best_report=None
    for attempt in range(requirement['attempts']):
        ranked=sorted(pool,key=lambda r:hashlib.sha256(f"{seed+attempt}:{r['window_id']}".encode()).digest())
        selected=[];ids=set();times=set();eps=Counter();geos=Counter();pcs=Counter();events=Counter()
        wc=Counter();unique={k:set() for k in kinds};cat_geo={k:set() for k in kinds};cat_ep={k:set() for k in kinds}
        def feasible(row):
            all_events={e for v in row['motion_event_ids'].values() for e in v}
            return (row['window_id'] not in ids and (row['episode_key'],row['reference_tick']) not in times
                and pcs[row['reference_policy']]<policies[row['reference_policy']]
                and eps[row['episode_key']]<requirement['max_episode_windows']
                and all(events[e]<requirement['max_windows_per_event'] for e in all_events))
        def add(row):
            selected.append(row);ids.add(row['window_id']);times.add((row['episode_key'],row['reference_tick']))
            eps[row['episode_key']]+=1;geos[row['geometry_id']]+=1;pcs[row['reference_policy']]+=1
            for e in {e for v in row['motion_event_ids'].values() for e in v}:events[e]+=1
            for kind in kinds:
                tokens=row['motion_event_ids'][kind]
                if tokens:
                    wc[kind]+=1;unique[kind].update(tokens);cat_geo[kind].add(row['geometry_id']);cat_ep[kind].add(row['episode_key'])
        def deficits(kind):
            return (wc[kind]<requirement['minimum_windows'][kind] or len(unique[kind])<requirement['minimum_unique'][kind]
                or len(cat_geo[kind])<requirement['minimum_motion_geometries'] or len(cat_ep[kind])<requirement['minimum_motion_episodes'])
        def improves(row,kind):
            return (wc[kind]<requirement['minimum_windows'][kind] or
                (len(unique[kind])<requirement['minimum_unique'][kind] and bool(set(row['motion_event_ids'][kind])-unique[kind])) or
                (len(cat_geo[kind])<requirement['minimum_motion_geometries'] and row['geometry_id'] not in cat_geo[kind]) or
                (len(cat_ep[kind])<requirement['minimum_motion_episodes'] and row['episode_key'] not in cat_ep[kind]))
        def draw(candidates,need,accept):
            grouped=defaultdict(deque)
            for row in candidates:grouped[(row['geometry_id'],row['episode_key'],row['reference_policy'])].append(row)
            while need() and len(selected)<requirement['quota']:
                progress=False
                order=sorted(grouped,key=lambda k:(geos[k[0]],eps[k[1]],pcs[k[2]],k))
                for key in order:
                    queue=grouped[key]
                    while queue:
                        row=queue.popleft()
                        if feasible(row) and accept(row):add(row);progress=True;break
                    if not need() or len(selected)>=requirement['quota']:break
                if not progress:break
        # Scarce categories first; a chosen row may satisfy several minima.
        for kind in sorted(kinds,key=lambda k:sum(bool(r['motion_event_ids'][k]) for r in pool)):
            draw([r for r in ranked if r['motion_event_ids'][kind]],lambda k=kind:deficits(k),lambda r,k=kind:improves(r,k))
        policy_order=list(policies)
        if attempt%2:policy_order.reverse()
        for policy in policy_order:
            draw([r for r in ranked if r['reference_policy']==policy],lambda p=policy:pcs[p]<policies[p],lambda r:True)
        checks=dict(total=len(selected)==requirement['quota'],policy_counts=dict(pcs)==policies,
            geometries=len(geos)>=requirement['minimum_geometries'],episodes=len(eps)>=requirement['minimum_episodes'])
        categories={k:dict(minimum_windows=requirement['minimum_windows'][k],selected_windows=wc[k],
            minimum_unique=requirement['minimum_unique'][k],selected_unique=len(unique[k]),
            eligible_windows=sum(bool(r['motion_event_ids'][k]) for r in pool),
            eligible_unique=len({e for r in pool for e in r['motion_event_ids'][k]}),
            geometry_count=len(cat_geo[k]),episode_count=len(cat_ep[k]),passed=not deficits(k)) for k in kinds}
        passed=all(checks.values()) and all(v['passed'] for v in categories.values())
        report=dict(passed=passed,checks=checks,requirement=requirement,eligible_pool=len(pool),selected=len(selected),
            reference_counts=dict(pcs),geometry_counts=dict(geos),episode_counts=dict(eps),categories=categories,
            maximum_event_reuse=max(events.values(),default=0),attempt=attempt,
            selection_semantics='deterministic bounded heuristic; failure is not a proof of pool infeasibility')
        score=(sum(checks.values())+sum(v['passed'] for v in categories.values()),len(selected))
        if best_report is None or score>best_report[0]:best=selected;best_report=(score,report)
        if passed:return selected,report
    return best,best_report[1]
