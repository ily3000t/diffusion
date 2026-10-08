import json
from copy import deepcopy
import pytest
from sumodiff.diffusion.stage5b_report import paired_interval
from sumodiff.diffusion.stage5b import verify_pairing,training_command


def rows(values):
    return [dict(window_id=str(i),family='road'+str(i%2),episode_key='episode'+str(i//2),value=v)
            for i,v in enumerate(values)]


def test_paired_cluster_interval_constant_effect_and_missing_counts():
    left=rows([1.,2.,3.,4.,5.,6.]);right=deepcopy(left)
    for r in right:r['value']+=2
    out=paired_interval(left,right,'value',17,iterations=200)
    assert out['mean_difference']==2 and out['ci95']==[2,2]
    right[0]['value']=None
    out=paired_interval(left,right,'value',17,iterations=200)
    assert out['paired_scenes']==5 and out['planned_scenes']==6
    assert out['mean_difference']==2
    right=rows([None]*6)
    out=paired_interval(left,right,'value',17,iterations=200)
    assert out['mean_difference'] is None and out['paired_scenes']==0


def test_paired_tasks_are_not_silently_reordered_or_reclustered():
    left=rows([1.,2.]);right=deepcopy(left)
    with pytest.raises(ValueError,match='order'):paired_interval(left,right[::-1],'value',1,20)
    right[0]['episode_key']='other'
    with pytest.raises(ValueError,match='clustering'):paired_interval(left,right,'value',1,20)


def test_training_pairing_detects_noise_and_budget_changes(tmp_path):
    roots={}
    base=[dict(step=k,window_ids=['train'],timesteps=[100],noise_sha256='paired') for k in (30001,30002)]
    for arm in 'ABCD':
        directory=tmp_path/arm;directory.mkdir()
        (directory/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in base))
        roots[arm]=directory
    assert verify_pairing(roots,2)['all_equal']
    changed=deepcopy(base);changed[0]['noise_sha256']='different'
    (roots['D']/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in changed))
    with pytest.raises(ValueError,match='pairing'):verify_pairing(roots,2)


def test_saved_training_command_has_replayable_direct_inputs():
    c=dict(dataset='data',parent_checkpoint='parent.pt',scale_audit='scales.json',arms={'A':'A.yaml'})
    cmd=training_command(c,'A','output/train',True,prefix_updates=12)
    assert 'train-direct' in cmd and '--prefix-updates' in cmd and '12' in cmd
    for flag,value in [('--dataset','data'),('--parent-checkpoint','parent.pt'),('--scale-audit','scales.json'),
                       ('--config','A.yaml'),('--output','output/train')]:
        assert cmd[cmd.index(flag)+1]==value
    assert '--formal' in cmd

def test_failed_generations_remain_in_all_200_planned_denominators(tmp_path):
    from sumodiff.diffusion.stage5b_report import generation_rows
    from sumodiff.data.dataset import write_json
    from sumodiff.experiments.recorder import file_identity
    write_json(tmp_path/'manifest.json',{})
    write_json(tmp_path/'status.json',{'state':'failed'})
    catalog=[];metadata={}
    for i in range(200):
        wid=str(i);folder=tmp_path/'trajectories'/wid;folder.mkdir(parents=True)
        failed=dict(status='failed',quality_pass=None)
        write_json(folder/'metrics.json',dict(raw=failed,decoded=failed))
        catalog.append(dict(window_id=wid,family='road',generation_status='failed',
                            files={'metrics.json':file_identity(folder/'metrics.json')}))
        metadata[wid]=dict(episode_key='episode'+str(i//10))
    write_json(tmp_path/'trajectory_index.json',catalog)
    result,summary,_,noise=generation_rows(tmp_path,{},metadata)
    assert len(result)==200 and summary['counts']['planned_scenes']==200
    assert summary['counts']['quality_passes']==0 and summary['counts']['quality_unknown']==200
    assert summary['counts']['generation_failures']==200 and summary['counts']['collision_unknown']==200
    assert summary['pooled']=={} and noise=={}


def test_recovery_lineage_ignores_uncommitted_updates_after_parent_checkpoint(tmp_path):
    import torch
    from test_controlled_training import parent_fixture,requested
    from sumodiff.diffusion.controlled_training import controlled_config
    from sumodiff.diffusion.training import training_signature
    from sumodiff.diffusion.checkpoint import save_checkpoint
    from sumodiff.models import ModelConfig
    from sumodiff.data.dataset import write_json
    from sumodiff.experiments.recorder import file_identity
    from sumodiff.diffusion.stage5b_report import committed_lineage
    import yaml
    first=tmp_path/'first';last=tmp_path/'last';first.mkdir();last.mkdir()
    parent=parent_fixture();c=controlled_config(requested(),parent)
    meta=deepcopy(parent['metadata']);meta['training_config']=c
    meta['training_signature']=training_signature(ModelConfig(),c,meta['data'],meta['scale_audit'],['train'],['val'])
    net=torch.nn.Linear(6,6);opt=torch.optim.AdamW(net.parameters());g=torch.Generator().manual_seed(2)
    checkpoint=first/'checkpoint.pt';save_checkpoint(checkpoint,net,opt,g,30002,meta)
    for root in (first,last):write_json(root/'manifest.json',{})
    write_json(first/'initial_state.json',{'model':'original'})
    (first/'resolved_config.yaml').write_text(yaml.safe_dump({'resume':None}))
    (last/'resolved_config.yaml').write_text(yaml.safe_dump({'resume':file_identity(checkpoint)}))
    for root,steps in [(first,[30001,30002,30003]),(last,[30003,30004])]:
        (root/'steps.jsonl').write_text(''.join(json.dumps({'step':n})+'\n' for n in steps))
    values,initial,segments=committed_lineage(last,30000,30004)
    assert [v['step'] for v in values]==[30001,30002,30003,30004]
    assert initial['model']=='original' and segments==[last,first]


def test_sampling_pairing_rejects_initial_noise_changes(tmp_path):
    import numpy as np
    from sumodiff.diffusion.stage5b import verify_sampling_pairing
    roots={}
    for arm in 'ABCD':
        root=tmp_path/arm;folder=root/'trajectories'/'task';folder.mkdir(parents=True)
        np.savez(folder/'stages.npz',initial_noise=np.ones((2,40,6),dtype=np.float32))
        (root/'trajectory_index.json').write_text(json.dumps([{'window_id':'task','generation_status':'completed'}]))
        roots[arm]=root
    assert verify_sampling_pairing(roots,1)['all_equal']
    np.savez(roots['D']/'trajectories'/'task'/'stages.npz',initial_noise=np.zeros((2,40,6),dtype=np.float32))
    with pytest.raises(ValueError,match='pairing'):
        verify_sampling_pairing(roots,1)
