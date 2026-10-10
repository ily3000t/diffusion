"""Bounded paired screening of existing Stage5E recovery checkpoints."""
import argparse,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from sumodiff.data.dataset import WindowDataset,write_json
from sumodiff.diffusion.data import choose_indices,input_files
from sumodiff.diffusion.sampling import sample,sample_config
from sumodiff.diffusion.stage5b_report import generation_rows,paired_interval
from sumodiff.diffusion.stage5d_report import extend_quality_rows
from sumodiff.diffusion.quality_audit import physical_derivatives
from sumodiff.diffusion.checkpoint import load_checkpoint
from sumodiff.evaluation.motion import evaluate_motion,summary,MotionConfig
from sumodiff.experiments.recorder import RunRecorder,load_config,file_identity

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def details(directory,rows,conditions,cfg):
    counts=Counter();pools=defaultdict(list);times=[[] for _ in range(40)];scenes=[];families=defaultdict(Counter)
    for row in rows:
        wid=row['window_id'];c=conditions[wid];mask=np.asarray(c['agent_mask'],bool)
        metric=read(Path(directory)/'trajectories'/wid/'metrics.json')
        dm=metric['decoded'].get('motion',{});m=dm.get('violation_counts',{})
        n=int(mask.sum());f=families[row['family']];f['planned']+=1;f['quality']+=int(row['quality_success'])
        f['agents']+=n;f['comfort']+=dm.get('comfort_pass_agents',0)
        for k,v in m.items():counts[k+'_bad_scenes']+=v>0;counts[k+'_bad_points']+=v
        values=dict(window_id=wid,family=row['family'],agents=n,quality=bool(row['quality_success']))
        if row['generation_failed'] or row['metric_failed']:
            counts['unassessed_scenes']+=1;scenes.append(values);continue
        with np.load(Path(directory)/'trajectories'/wid/'stages.npz',allow_pickle=False) as z:
            state=z['decoded_absolute_states'];_,_,j=physical_derivatives(c,state);ja=np.linalg.norm(j[mask],axis=-1)
            fm=np.broadcast_to(mask[:,None],(len(mask),40))
            replay=evaluate_motion(c['history'],c['history_mask'],state,fm,c['initial_positions'],c['history'][:,-1,4:6],
                mask,config=MotionConfig(**cfg['evaluation']['motion']))
            assert replay['violation_counts']==m and replay['hard_quality_pass']==dm['hard_quality_pass']
            pools['position_correction'].extend(np.linalg.norm(z['position_correction'][mask],axis=-1).ravel())
            pools['velocity_correction'].extend(np.linalg.norm(z['velocity_correction'][mask],axis=-1).ravel())
        pools['all_jerk'].extend(ja.ravel())
        for t in range(40):times[t].extend(ja[:,t])
        bad=ja>cfg['evaluation']['motion']['max_jerk_mps3']
        assert int(bad.sum())==m['jerk']
        for name,slice_ in [('boundary',slice(0,4)),('interior',slice(4,40)),('endpoint',slice(36,40))]:
            counts[name+'_bad_scenes']+=bool(bad[:,slice_].any());counts[name+'_bad_points']+=int(bad[:,slice_].sum())
            values[name+'_jerk_max']=float(ja[:,slice_].max())
        scenes.append(values)
    return dict(counts=dict(counts),pooled={k:summary(v) for k,v in pools.items()},families=dict(families),
        jerk_by_time=[dict(time_s=(t+1)/10,values=summary(v),bad_points=int(sum(y>15 for y in v))) for t,v in enumerate(times)],
        scenes=scenes,motion_replayed=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/experiments/stage5e_checkpoint_screen.yaml')
    p.add_argument('--output',required=True);p.add_argument('--reuse-samples',type=Path);p.add_argument('--formal',action='store_true');args=p.parse_args()
    root=Path.cwd();suite=load_config(args.config);cfg=sample_config(load_config(suite['sampling_config']))
    assert suite['schema_version']=='sumodiff.checkpoint.screen.config.v1'
    assert suite['planned_tasks']==cfg['limit']==60 and cfg['seed']==20261015 and cfg['steps']==20 and cfg['eta']==0.
    torch.set_num_threads(1)
    ds=WindowDataset(suite['dataset'],'validation');inds=choose_indices(ds,60,cfg['minimum_agents'])
    ids=[ds.entries[i]['window_id'] for i in inds];assert len(ids)==60
    meta={ds.entries[i]['window_id']:ds.entries[i] for i in inds}
    conditions={ds.entries[i]['window_id']:ds.inference(i)['conditioning'] for i in inds}
    assert Counter(e['family'] for e in meta.values())==dict(ramp_merge=20,three_lane_straight=20,unsignalized_intersection=20)
    train=Path(suite['source_training']);ts=read(train/'training_summary.json')
    cpids={Path(x['location']).name:x for x in ts['checkpoints']}
    expected=[name for name in cpids if name!='checkpoint_step_067500.pt']
    assert suite['checkpoints']==expected,'Screen must include every saved intermediate recovery checkpoint'
    files=[Path(__file__),args.config,suite['sampling_config'],train/'training_summary.json',train/'manifest.json',train/'status.json',
        *input_files(ds,inds,False)]
    for name in expected:
        ident=cpids[name];assert file_identity(ident['location'])==ident
        files.append(ident['location'])
    existing={}
    for name,key in [('epoch200','baseline_run'),('epoch225','endpoint_run')]:
        path=Path(suite[key]);m=read(path/'manifest.json')
        assert m['formal'] and not m['git']['dirty'] and read(path/'status.json')['state']=='completed'
        actual=load_config(path/'resolved_config.yaml')['sampling'].copy();actual['limit']=60
        assert actual==cfg,'Paired source sampling protocol changed'
        existing[name]=path
        for n in ['manifest.json','status.json','resolved_config.yaml','sampling_summary.json','trajectory_index.json']:files.append(path/n)
        catalog=read(path/'trajectory_index.json');assert [r['window_id'] for r in catalog[:60]]==ids
        for item in catalog[:60]:
            for ident in item['files'].values():assert file_identity(ident['location'])==ident;files.append(ident['location'])
    if args.reuse_samples:
        reuse=args.reuse_samples.resolve(strict=True)
        files.extend(reuse/n for n in ('manifest.json','status.json','resolved_config.yaml'))
        for name in expected:
            cp=cpids[name];epoch=load_checkpoint(cp['location'])['step']//300
            directory=reuse/f'epoch{epoch}';m=read(directory/'manifest.json')
            assert read(directory/'status.json')['state']=='completed'
            assert m['checkpoint']==cp and load_config(directory/'resolved_config.yaml')['sampling']==cfg
            assert not args.formal or (m['formal'] and not m['git']['dirty'])
            catalog=read(directory/'trajectory_index.json');assert [r['window_id'] for r in catalog]==ids
            files.extend(directory/n for n in ('manifest.json','status.json','resolved_config.yaml','trajectory_index.json','sampling_summary.json'))
            for item in catalog:
                for ident in item['files'].values():
                    assert file_identity(ident['location'])==ident;files.append(ident['location'])
            existing[f'epoch{epoch}']=directory
    configs=dict(suite=suite,sampling=cfg,planned_window_ids=ids,checkpoint_identities=[cpids[n] for n in expected],
        selection='fixed first60 of original full validation ordering; not selected by future quality',
        new_sampling_budget=0 if args.reuse_samples else 360,reuse_samples=str(args.reuse_samples) if args.reuse_samples else None,optimizer_updates=0,multiple_comparisons='exploratory; no multiplicity-adjusted superiority claim')
    output=Path(args.output).resolve();begun=time.perf_counter()
    with RunRecorder(output,root,configs,[sys.executable,*sys.argv],{'initial_noise':cfg['seed'],'bootstrap':suite['bootstrap_seed']},
        purpose='stage5e_intermediate_checkpoint_screen',data_files=sorted(set(map(str,files))),formal=args.formal) as run:
        write_json(output/'tasks.json',[dict(window_id=w,family=meta[w]['family'],agents=int(conditions[w]['agent_mask'].sum())) for w in ids])
        results={};allrows={};details_all={};baseline_noise=None
        for name in ([] if args.reuse_samples else expected):
            cp=cpids[name]['location'];step=load_checkpoint(cp)['step'];epoch=step//300
            directory=output/f'epoch{epoch}'
            sample(suite['dataset'],cp,suite['sampling_config'],directory,root,
                [sys.executable,'-m','sumodiff.diffusion.cli','sample','--dataset',str(suite['dataset']),'--checkpoint',str(cp),'--config',str(suite['sampling_config']),'--output',str(directory),'--repository',str(root)]+(['--formal'] if args.formal else []),args.formal)
            existing[f'epoch{epoch}']=directory
        for name,directory in sorted(existing.items(),key=lambda p:int(p[0][5:])):
            rows,agg,sourcefiles,noises=generation_rows(directory,conditions,meta,baseline_noise,planned_window_ids=ids)
            if name=='epoch200':baseline_noise=noises;assert len(noises)==60
            assert noises==baseline_noise,'Noise pairing incomplete'
            rows,agg=extend_quality_rows(directory,rows,agg,conditions)
            agg={k:v for k,v in agg.items() if k!='manifest'}
            agg['sampling']=read(directory/'sampling_summary.json')['sampling_decoder_seconds']
            agg['checkpoint']=read(directory/'manifest.json')['checkpoint']
            allrows[name]=rows;results[name]=agg;details_all[name]=details(directory,rows,conditions,cfg)
        keys=['quality_success','decoded_boundary_jerk_p99','decoded_interior_jerk_p99','strict_comfort_rate','position_correction_max','velocity_correction_max']
        intervals={name:{k:paired_interval(allrows['epoch200'],rows,k,suite['bootstrap_seed']) for k in keys} for name,rows in allrows.items() if name!='epoch200'}
        transitions={}
        for name,rows in allrows.items():
            transitions[name]=dict(Counter(('pass' if a['quality_success'] else 'fail')+'->'+('pass' if b['quality_success'] else 'fail') for a,b in zip(allrows['epoch200'],rows)))
        result=dict(schema_version='sumodiff.checkpoint.screen.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            planned_tasks=60,new_sampling_tasks=0 if args.reuse_samples else 360,source_sampling_tasks=360,reused_sampling_tasks=360 if args.reuse_samples else 0,optimizer_updates=0,initial_noise_paired=True,
            generation=results,details=details_all,paired_intervals=intervals,transitions=transitions,
            wall_seconds=time.perf_counter()-begun,limitations=['60 validation tasks, one noise seed; four dense intersection tasks only.',
            'Multiple saved candidates screened on validation; intervals are conditional and not multiplicity adjusted.',
            'Source epsilon MSE is not generation quality; no training or loss-specific attribution.',
            'Failures retained; whole-scene qualification and per-agent comfort have different denominators.'])
        write_json(output/'summary.json',result);write_json(output/'scene_rows.json',allrows);run.write_metrics(result)
        print(json.dumps({n:dict(quality=g['counts']['quality_passes'],comfort=g['strict_comfort'],
            boundary_p99=g['pooled']['decoded_boundary_jerk']['p99'],interior_p99=g['pooled']['decoded_interior_jerk']['p99'],
            failures=g['counts']['generation_failures']) for n,g in results.items()},indent=2))
if __name__=='__main__':main()
