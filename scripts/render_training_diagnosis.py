"""Render audited learning/reconstruction/free-generation trends and full trajectory extents."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from sumodiff.experiments.recorder import RunRecorder,file_identity
from sumodiff.data.dataset import WindowDataset,write_json

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def report(diagnoses,training_runs,dataset_path,output,formal=False):
    roots=[Path(p).resolve(strict=True) for p in diagnoses]
    rows=[read(p/'diagnosis.json') for p in roots]
    order=np.argsort([r['checkpoint_step'] for r in rows])
    roots=[roots[i] for i in order];rows=[rows[i] for i in order]
    steps=[r['checkpoint_step'] for r in rows]
    if len(set(steps))!=len(steps): raise ValueError('Duplicate checkpoint diagnosis')
    traces={}
    files=[]
    expected=None
    for root,row in zip(roots,rows):
        if read(root/'status.json')['state']!='completed': raise ValueError('Incomplete diagnostic source')
        config=__import__('yaml').safe_load((root/'resolved_config.yaml').read_text(encoding='utf-8'))
        signature={k:config[k] for k in ('model','diffusion','data','labeled_task_ids','free_task_ids')}
        signature['noise_seed']=config['diagnostic']['noise_seed']
        signature['noise_repeats']=config['diagnostic']['noise_repeats']
        signature['timesteps']=config['diagnostic']['timesteps']
        signature['free_sampling']=config['diagnostic']['free_sampling']
        if expected is not None and signature!=expected: raise ValueError('Diagnostic tasks/model/units/noise changed')
        expected=signature
        files += [root/name for name in ('diagnosis.json','manifest.json','resolved_config.yaml','status.json')]
    for root in map(Path,training_runs):
        if read(root/'status.json')['state']!='completed': raise ValueError('Incomplete training source')
        metrics=read(root/'metrics.json')
        for row in metrics['validation']: traces[row['step']]=row
        files += [root/name for name in ('metrics.json','manifest.json','status.json','steps.jsonl')]
    last=rows[-1];last_root=roots[-1];catalog=read(last_root/'free_validation_20'/'trajectory_index.json')
    dataset=WindowDataset(dataset_path,'validation')
    lookup={e['window_id']:i for i,e in enumerate(dataset.entries)}
    selected={}
    for item in catalog:
        if item['family'] not in selected: selected[item['family']]=item
    overlays={}
    for family,item in selected.items():
        i=lookup[item['window_id']];overlays[family]=dataset[i]
        directory=last_root/'free_validation_20'/'trajectories'/item['window_id']
        for name in ('stages.npz','metrics.json'):
            p=directory/name
            if file_identity(p)['sha256']!=item['files'][name]['sha256']: raise ValueError('Trajectory source changed')
            files.append(p)
        files += [dataset._path(dataset.entries[i],name) for name in ('conditioning.npz','targets.npz','input.json','labels.json','map.json')]
    script=Path(__file__).resolve();files.append(script)
    effective=dict(schema_version='sumodiff.diagnosis.report.config.v1',diagnoses=[str(p) for p in roots],training_runs=[str(Path(p).resolve()) for p in training_runs],
        source_run_shas=[r['run_sha'] for r in rows],checkpoint_steps=steps,plotting='full observed arrays; no clipping, smoothing or threshold changes',
        convergence_status='not_assessed',dataset=str(Path(dataset_path).resolve()),renderer=file_identity(script))
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],{'report':0},
        purpose='bounded_training_diagnostic_report',data_files=sorted(set(map(str,files))),formal=formal) as run:
        os.environ['MPLCONFIGDIR']=str(Path('artifacts/cache/matplotlib-diagnosis').resolve())
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        records=[]
        for r in rows:
            v=r['free_generation']['validation_20'];stat=v['details']['statistics']
            records.append(dict(step=r['checkpoint_step'],run_sha=r['run_sha'],
                train_probe_mse=traces[r['checkpoint_step']]['train_probe_mse'],validation_probe_mse=traces[r['checkpoint_step']]['validation_probe_mse'],
                validation_t999_position_rmse_m=next(x['position_vector_rmse_m'] for x in r['reconstruction']['validation'] if x['timestep']==999),
                raw_displacement_max_m=stat['raw_displacement_m']['max'],raw_displacement_p99_m=stat['raw_displacement_m']['p99'],
                decoded_displacement_max_m=stat['decoded_displacement_m']['max'],correction_max_m=stat['position_correction_m']['max'],
                correction_p99_m=stat['position_correction_m']['p99'],quality_pass_rate=v['metrics']['decoded']['quality_pass_rate'],
                failure_count=v['metrics']['generation_failures'],parameter_sha256=r['parameter_sha256']))
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(1,3,figsize=(14,4.3))
        points=[traces[k] for k in sorted(traces)]
        axes[0].semilogy([r['step'] for r in points],[r['train_probe_mse'] for r in points],label='train fixed probe')
        axes[0].semilogy([r['step'] for r in points],[r['validation_probe_mse'] for r in points],label='validation fixed probe')
        axes[0].set_ylabel('epsilon MSE');axes[0].set_title('Fixed-noise learning probes')
        for split in ('train','validation'):
            axes[1].plot(steps,[next(x['position_vector_rmse_m'] for x in r['reconstruction'][split] if x['timestep']==999) for r in rows],'-o',label=split)
        axes[1].set_ylabel('position vector RMSE (m)');axes[1].set_title('Known noisy input, t=999')
        axes[2].plot(steps,[r['raw_displacement_p99_m'] for r in records],'-o',label='raw displacement P99')
        axes[2].plot(steps,[r['correction_p99_m'] for r in records],'-o',label='decode correction P99')
        axes[2].set_ylabel('metres');axes[2].set_title('Free generation, fixed validation tasks')
        for ax in axes: ax.set_xlabel('total optimizer updates');ax.grid(alpha=.25);ax.legend(fontsize=9)
        fig.suptitle('Bounded training diagnosis: fixed data/model/noise; convergence not assessed',fontsize=12)
        fig.tight_layout();fig.savefig(run.output/'diagnostic_trends.png',dpi=160);fig.savefig(run.output/'diagnostic_trends.svg');plt.close(fig)
        families=sorted(selected);fig,axes=plt.subplots(2,len(families),figsize=(14,8.2),squeeze=False)
        for col,family in enumerate(families):
            item=selected[family];sample=overlays[family];c=sample['conditioning'];targets=sample['targets']
            with np.load(last_root/'free_validation_20'/'trajectories'/item['window_id']/'stages.npz',allow_pickle=False) as d:
                raw=d['raw_absolute_states'].copy();decoded=d['decoded_absolute_states'].copy();mask=d['agent_mask'].copy()
            label=targets['future'][...,:2]+c['initial_positions'][:,None,:]
            all_points=[raw[mask,:,:2].reshape(-1,2),decoded[mask,:,:2].reshape(-1,2),c['initial_positions'][mask]]
            for i in np.where(mask)[0]: all_points.append(label[i,targets['future_mask'][i]])
            xy=np.concatenate(all_points);lo=xy.min(0);hi=xy.max(0);padding=np.maximum((hi-lo)*.08,5.)
            for row,(stage,states,color) in enumerate((('raw',raw,'#d95f02'),('decoded',decoded,'#1f77b4'))):
                ax=axes[row,col]
                for lane in sample['exact_map']['lanes']:
                    points=np.asarray(lane['points_local_m'])
                    ax.plot(points[:,0],points[:,1],color='.85',linewidth=.65,zorder=0)
                for i in np.where(mask)[0]:
                    hm=c['history_mask'][i];ax.plot(c['history'][i,hm,0],c['history'][i,hm,1],color='.25',linewidth=1)
                    valid=targets['future_mask'][i]
                    ax.plot(label[i,valid,0],label[i,valid,1],':',color='#2ca02c',linewidth=1.8,label='observed future' if i==np.where(mask)[0][0] else None)
                    path=np.vstack((c['initial_positions'][i],states[i,:,:2]))
                    ax.plot(path[:,0],path[:,1],color=color,linewidth=1.3,label=stage if i==np.where(mask)[0][0] else None)
                    ax.scatter(*c['initial_positions'][i],color='black',marker='x',s=22)
                ax.set_xlim(lo[0]-padding[0],hi[0]+padding[0]);ax.set_ylim(lo[1]-padding[1],hi[1]+padding[1]);ax.set_aspect('equal',adjustable='box')
                ax.set_title(f'{family}\n{stage}, checkpoint {steps[-1]}');ax.set_xlabel('local x (m)');ax.set_ylabel('local y (m)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Fixed validation scenes: all generated points retained; labels overlaid after sampling',fontsize=12)
        fig.tight_layout();fig.savefig(run.output/'final_trajectories.png',dpi=160);plt.close(fig)
        sweep={}
        for name,value in last['free_generation'].items():
            sweep[name]=dict(quality_pass_rate=value['metrics']['decoded']['quality_pass_rate'],statistics=value['details']['statistics'],
                sampling_decoder_seconds=value['metrics']['sampling_decoder_seconds'],failures=value['metrics']['generation_failures'])
        result=dict(schema_version='sumodiff.bounded.diagnosis.summary.v1',run_sha=run.manifest['git']['commit_sha'],
            checkpoints=records,sampling_step_comparison=sweep,oracle_checks={str(r['checkpoint_step']):r['oracle'] for r in rows},
            training_resources=[read(Path(p)/'metrics.json') for p in training_runs[1:]],convergence_status='not_assessed',
            limits='diagnostic budget ended at 3000 total updates; no architecture change, new data, guidance or convergence claim',
            figures={name:file_identity(run.output/name) for name in ('diagnostic_trends.png','diagnostic_trends.svg','final_trajectories.png')})
        write_json(run.output/'summary.json',result);run.write_metrics(result)
        seen_rows=last['free_generation']['train_20']['details']['rows']
        seen_count=sum(r['seen_in_training'] for r in seen_rows)
        lines=['# 补充训练诊断（2026-10-07）','','保持12/6窗口、原架构/尺度/optimizer/种子，从240恢复到总3000步；预算到达后停止。收敛未判定。','',
            '| 总步数 | 固定验证 ε MSE | t999位置重构RMSE（m） | 原始最大位移（m） | 最大协调修正（m） | 解码质量通过 |',
            '|---:|---:|---:|---:|---:|---:|']
        for r in records:
            lines.append(f"| {r['step']} | {r['validation_probe_mse']:.6f} | {r['validation_t999_position_rmse_m']:.2f} | {r['raw_displacement_max_m']:.2f} | {r['correction_max_m']:.2f} | {r['quality_pass_rate']:.1%} |")
        lines += ['',f'全部checkpoint使用相同任务和噪声。oracle数值检查通过不代表模型生成合格；任务范围为被选车辆。固定train-split自由任务中{seen_count}/{len(seen_rows)}被训练见过；独立重构train探针覆盖全部实际训练窗。',
            '',f"报告运行SHA：{result['run_sha']}。训练/诊断来源SHA分别保留在JSON及原始清单中，不回填成报告或summary提交SHA。"]
        (run.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnoses',nargs='+',required=True);p.add_argument('--training-runs',nargs='+',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--formal',action='store_true')
    a=p.parse_args();report(a.diagnoses,a.training_runs,a.dataset,a.output,a.formal)
