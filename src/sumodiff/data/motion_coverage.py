"""Training-only raw motion lineage; original labels and filters are unchanged."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def read_quality_rows(path):
    rows=defaultdict(list)
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            source=json.loads(line)
            if source['split']!='train':
                continue
            row={k:source[k] for k in ('window_id','family','split','episode_key','geometry_id','reference_tick',
                                      'selected_agents','core_complete','eligible','reasons')}
            row['old_turning_agents']=source.get('actual_turning_agents',0)
            for kind in ('raw64','stored32'):
                if kind in source:
                    row[kind+'_violations']=source[kind]['motion']['violation_counts']
                    row[kind+'_jerk_max']=source[kind]['motion']['jerk_norm_mps3']['max']
            rows[row['episode_key']].append(row)
    return dict(rows)


def lane_changes(path,dt=.1):
    events=defaultdict(list)
    for item in ET.parse(path).getroot():
        if item.tag!='change':
            continue
        time=float(item.attrib['time']);tick=round(time/dt)
        if abs(tick*dt-time)>1e-6:
            raise ValueError('Lane-change time is outside the recorded sample grid')
        events[item.attrib['id']].append(dict(tick=tick,time_s=time,reason=item.attrib.get('reason'),
                                             from_lane=item.attrib['from'],to_lane=item.attrib['to']))
    return dict(events)


def raw_tracks(path,dt=.1):
    tracks={};frame_count=0
    with Path(path).open(encoding='utf-8') as stream:
        for tick,line in enumerate(stream):
            row=json.loads(line)
            if row['schema_version']!='sumodiff.raw.frame.v1' or row['tick']!=tick or abs(row['time_seconds']-tick*dt)>1e-7:
                raise ValueError('Raw time grid/schema mismatch')
            seen=set()
            for vehicle in row['vehicles']:
                vid=vehicle['vehicle_id'];raw=vehicle['sumo_raw']
                if vid in seen:raise ValueError('Duplicate raw vehicle')
                seen.add(vid)
                front=np.asarray(raw['front_position_m'],float)
                angle=math.radians(raw['navigation_angle_deg'])
                direction=np.array([math.sin(angle),math.cos(angle)])
                center=front-.5*raw['length_m']*direction
                yaw=math.atan2(direction[1],direction[0])
                if not np.isfinite(center).all() or not np.isfinite(yaw) or raw['length_m']<=0:
                    raise ValueError('Invalid raw pose')
                track=tracks.setdefault(vid,dict(ticks=[],center=[],front=[],yaw=[],road=[],lane=[],reported_accel=[],
                                                route_edges=list(raw['route_edges'])))
                if track['route_edges']!=raw['route_edges']:raise ValueError('Unexpected route change during coverage diagnosis')
                for key,value in [('ticks',tick),('center',center),('front',front),('yaw',yaw),
                                  ('road',raw['road_id']),('lane',raw['lane_id']),('reported_accel',raw['acceleration_mps2'])]:
                    track[key].append(value)
            frame_count+=1
    for track in tracks.values():
        for key in ('ticks','center','front','yaw','reported_accel'):
            track[key]=np.asarray(track[key])
    return tracks,frame_count


def indices_at(track,start,end):
    wanted=np.arange(start,end+1)
    ticks=track['ticks'];index=np.searchsorted(ticks,wanted)
    if np.any(index>=len(ticks)) or not np.array_equal(ticks[index],wanted):
        return None
    return index


def future_motion(track,t0,horizon=40):
    index=indices_at(track,t0,t0+horizon)
    if index is None:return None
    yaw=np.unwrap(track['yaw'][index])
    delta=track['center'][index]-track['center'][index[0]]
    lateral=-math.sin(yaw[0])*delta[:,0]+math.cos(yaw[0])*delta[:,1]
    return dict(heading_span_rad=float(np.ptp(yaw)),lateral_excursion_m=float(np.ptp(lateral)))


def logged_future_events(events,t0,horizon=40):
    return [e for e in events if t0<e['tick']<=t0+horizon]


def ramp_exit_events(track):
    if 'ramp_in' not in track['route_edges']:return []
    last=None;events=[]
    for tick,road in zip(track['ticks'],track['road']):
        if not road or road.startswith(':'):continue
        if last=='acceleration' and road=='main_out':events.append(int(tick))
        last=road
    return events


def source_case(row,meta,tracks,events):
    t0=row['reference_tick'];best=None
    for vid in [v for v in meta['agent_ids'] if v is not None]:
        track=tracks[vid];index=indices_at(track,t0-23,t0+40)
        if index is None:continue
        center=track['center'][index];front=track['front'][index]
        jerk=np.diff(np.diff(np.diff(center,axis=0)/.1,axis=0)/.1,axis=0)/.1
        front_jerk=np.diff(np.diff(np.diff(front,axis=0)/.1,axis=0)/.1,axis=0)/.1
        norm=np.linalg.norm(jerk,axis=-1)
        k=int(np.argmax(norm));peak=float(norm[k])
        if best is not None and peak<=best['raw_center_jerk_max_mps3']:continue
        peak_index=index[k+3];peak_tick=int(track['ticks'][peak_index])
        near=sorted(events.get(vid,[]),key=lambda e:abs(e['tick']-peak_tick))[:2]
        heading=track['yaw'][index[0]]
        delta=center-center[0]
        lateral=-math.sin(heading)*delta[:,0]+math.cos(heading)*delta[:,1]
        best=dict(window_id=row['window_id'],family=row['family'],episode_key=row['episode_key'],
            reference_tick=t0,eligible=row['eligible'],reasons=row['reasons'],peak_vehicle_id=vid,
            raw_center_jerk_max_mps3=peak,raw_front_jerk_max_mps3=float(np.linalg.norm(front_jerk,axis=-1).max()),
            stored_raw64_jerk_max_mps3=row.get('raw64_jerk_max'),
            peak_tick=peak_tick,peak_road=track['road'][peak_index],peak_lane=track['lane'][peak_index],
            peak_reported_longitudinal_acceleration_mps2=float(track['reported_accel'][peak_index]),
            nearest_logged_lane_changes=near,peak_vehicle_route=track['route_edges'],
            times_relative_t0_s=((track['ticks'][index]-t0)*.1).tolist(),
            lateral_position_from_context_start_m=lateral.tolist(),
            world_positions_m=center.tolist(),navigation_yaw_rad=track['yaw'][index].tolist(),
            physical_jerk_norm_mps3=norm.tolist(),raw_front_jerk_norm_mps3=np.linalg.norm(front_jerk,axis=-1).tolist())
    return best


def add_stage(counts,row,metrics,selected,events,merge_ticks):
    flags=dict(heading5=any(m['heading_span_rad']>=math.radians(5) for m in metrics if m is not None),
               lateral_half=any(m['lateral_excursion_m']>=.5 for m in metrics if m is not None),
               logged_lane_change=bool(events),ramp_exit=bool(merge_ticks))
    stages=['candidate']
    if row['core_complete']:stages.append('core_complete')
    if row['core_complete'] and row['selected_agents']>=2:stages.append('core_multiagent')
    if row['eligible']:stages.append('eligible')
    if selected:stages.append('selected')
    for stage in stages:
        c=counts[stage];c['windows']+=1
        c['selected_agent_windows']+=row['selected_agents']
        c['full_future_agent_windows']+=sum(m is not None for m in metrics)
        c['unknown_future_agent_windows']+=sum(m is None for m in metrics)
        c['windows_without_any_complete_future_agent']+=all(m is None for m in metrics)
        c['heading5_agent_windows']+=sum(m is not None and m['heading_span_rad']>=math.radians(5) for m in metrics)
        c['lateral_half_agent_windows']+=sum(m is not None and m['lateral_excursion_m']>=.5 for m in metrics)
        c['logged_lane_change_occurrences']+=len(events)
        c['ramp_exit_occurrences']+=len(merge_ticks)
        for key,flag in flags.items():c[key+'_windows']+=flag
    return flags


def event_opportunities(events,tracks,t0,origin,selected_ids,radius=80.):
    result={k:set() for k in ('actor_present','inside_radius','selected')}
    for vid,tick in events:
        if not t0<tick<=t0+40:continue
        index=indices_at(tracks[vid],t0,t0)
        if index is None:continue
        event=(vid,tick)
        result['actor_present'].add(event)
        distance=float(np.linalg.norm(tracks[vid]['center'][index[0]]-np.asarray(origin)))
        if distance<=radius+1e-9:result['inside_radius'].add(event)
        if vid in selected_ids:result['selected'].add(event)
    return result
