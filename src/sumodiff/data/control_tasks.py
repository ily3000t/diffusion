"""Frozen train/validation-only cohorts for the Motion control diagnostics."""
from collections import Counter
from copy import deepcopy
import hashlib,json
from pathlib import Path
from sumodiff.experiments.recorder import file_identity

SCHEMA='sumodiff.control_replay.tasks.v2'
POLICIES=('first_current_id','interaction_current')
FAMILIES=('three_lane_straight','ramp_merge','unsignalized_intersection')
GROUPS=('dense_no_turn','turning','lane_change','ordinary')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def motion_metadata(row):
    flags=row.get('motion_flags');events=row.get('motion_event_ids')
    names={'lane_change','ramp_exit','turn'}
    if (not isinstance(flags,dict) or set(flags)!=names or not isinstance(events,dict) or set(events)!=names
        or any(type(flags[k]) is not bool or not isinstance(events[k],list)
               or len(events[k])!=len(set(events[k])) or any(not isinstance(v,str) or not v for v in events[k])
               or flags[k]!=bool(events[k]) for k in names)):
        raise ValueError('Complete Motion flags and unique event IDs are required')
    return flags,events

def member(row,group):
    flags,_=motion_metadata(row)
    if group=='dense_no_turn':
        return row['family']=='unsignalized_intersection' and row['selected_agents']>=8 and not flags['turn']
    if group=='turning':return flags['turn']
    if group=='lane_change':return flags['lane_change'] and row['family'] in FAMILIES[:2]
    if group=='ordinary':return not any(flags.values())
    raise ValueError('Unknown diagnostic group')

def choose(rows,count,used,seed,dense):
    pool=[r for r in rows if r['window_id'] not in used]
    selected=[];geos=Counter();episodes=Counter()
    def key(row):
        digest=hashlib.sha256(f"{seed}:{row['window_id']}".encode()).hexdigest()
        return (geos[row['geometry_id']],episodes[row['episode_key']],
                -row['selected_agents'] if dense else 0,digest)
    if len(pool)<count:raise ValueError('Insufficient fixed diagnostic cohort; no fallback/duplication')
    for _ in range(count):
        row=min(pool,key=key);pool.remove(row);selected.append(row)
        used.add(row['window_id']);geos[row['geometry_id']]+=1;episodes[row['episode_key']]+=1
    return selected

def select_tasks(rows,seed):
    tasks=[]
    for split in ('train','validation'):
        cohort=[r for r in rows if r['split']==split]
        if any(not r['eligible'] or not r['core_complete'] for r in cohort):
            raise ValueError('Diagnostic selection requires certified complete selected windows')
        used=set()
        for group in GROUPS:
            families=FAMILIES if group=='ordinary' else FAMILIES[:2] if group=='lane_change' else FAMILIES[2:]
            per_policy=2 if group=='ordinary' else 4 if group=='lane_change' else 8
            chosen=[]
            for family in families:
                for policy in POLICIES:
                    pool=[r for r in cohort if r['family']==family and r['reference_policy']==policy and member(r,group)]
                    chosen+=choose(pool,per_policy,used,seed,dense=group in ('dense_no_turn','turning'))
            if len({r['geometry_id'] for r in chosen})<3:raise ValueError('Diagnostic group needs at least three geometries')
            for row in chosen:
                flags,_=motion_metadata(row)
                tasks.append({**{k:row[k] for k in ('window_id','split','family','episode_key','geometry_id','reference_policy','selected_agents')},
                              'group':group,'motion_flags':flags,'motion_event_ids':deepcopy(row['motion_event_ids'])})
    assert len(tasks)==120
    return tasks

def make_motion_task_plan(directory,seed):
    directory=Path(directory).resolve(strict=True);manifest=read(directory/'dataset_manifest.json')
    si=manifest['curation']['selection'];selection_path=directory/'selection.json'
    if file_identity(selection_path)['sha256']!=si['sha256']:
        raise ValueError('Selection hash mismatch')
    selection=read(selection_path)
    if selection.get('schema_version')!='sumodiff.stage5a.selection.v3' or selection['profile']!='full':
        raise ValueError('Motion control tasks require full Motion selection v3')
    if file_identity(directory/'windows.json')['sha256']!=manifest['window_index']['sha256']:
        raise ValueError('Dataset index hash mismatch')
    return dict(schema_version=SCHEMA,data_id=manifest['data_id'],seed=seed,
        dataset_manifest=file_identity(directory/'dataset_manifest.json'),
        window_index=file_identity(directory/'windows.json'),selection=file_identity(selection_path),
        selection_rule='fixed buckets; geometry/episode round robin; dense preference then seeded hash',
        tasks=select_tasks(selection['selected'],seed))

def verify_motion_task_plan(document,directory):
    if document.get('schema_version')!=SCHEMA:raise ValueError('Expected explicit Motion task schema')
    expected=make_motion_task_plan(directory,document['seed'])
    if document!=expected:
        raise ValueError('Motion task plan differs from its frozen data/buckets/seed')
    return True
