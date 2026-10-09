from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from test_physical_losses import fixture
from sumodiff.diffusion.raw_velocity_diagnostics import vector_statistics,summarize_rows
from sumodiff.diffusion.velocity_validation import gradient_vector
from sumodiff.diffusion.raw_velocity_training import velocity_frontier
from sumodiff.diffusion.stage5d import setup
from sumodiff.diffusion.stage5d_report import extend_quality_rows

def test_joint_gradient_vector_detects_cancellation_and_correct_clip():
    p=torch.nn.Parameter(torch.tensor([0.,0.],dtype=torch.float64))
    noise=p[0];one=3*p[0]+p[1];two=-3*p[0]+p[1]
    reference=gradient_vector(noise,[p]);joint=gradient_vector(one+two,[p])
    a=vector_statistics(gradient_vector(one,[p]),reference,1.)
    b=vector_statistics(gradient_vector(two,[p]),reference,1.)
    q=vector_statistics(joint,reference,1.)
    assert q['norm_ratio_to_epsilon']==2
    assert q['norm_ratio_to_epsilon'] < a['norm_ratio_to_epsilon']+b['norm_ratio_to_epsilon']
    assert q['cosine_with_epsilon']==0 and q['angle_degrees']==90
    assert q['would_clip'] and q['clip_scale']==pytest.approx(1/(2+1e-6))
    assert p.grad is None

def test_zero_gradient_and_empty_eligible_population_are_na():
    r=vector_statistics(torch.zeros(2),torch.ones(2),1.)
    assert r['cosine_with_epsilon'] is None and r['angle_degrees'] is None
    assert r['clip_scale']==1 and not r['would_clip']
    result=summarize_rows([])
    assert result['components']['auxiliary_total']['norm_ratio_to_epsilon']['median'] is None

def test_new_suite_rejects_old_stage5c_configuration(tmp_path):
    path=tmp_path/'old.yaml';path.write_text('schema_version: sumodiff.stage5c.suite.config.v1\n')
    with pytest.raises(ValueError,match='stage5D'):setup(path)

def test_candidate_frontier_keeps_boundary_and_interior_tradeoff_and_rejects_unknown():
    def record(name,boundary,interior,unknown=0):
        return dict(checkpoint={'location':name},quality=dict(planned=20,unknown=unknown,failures=0,
            collision_unknown=0,quality_passes=10,road_bad=1,route_bad=1,collision_scenes=0,
            decoded_boundary_jerk_p99=boundary,decoded_interior_jerk_p99=interior,
            strict_comfort_pass_agents=10,strict_comfort_assessed_agents=40,strict_comfort_planned_agents=40))
    records=[record('a',5,20),record('b',10,10),record('dominated',15,30),record('unknown',0,0,1)]
    assert [r['location'] for r in velocity_frontier(records)]==['a','b']

def test_report_recovers_boundary_jerk_from_history_and_records_unassessed_failures(tmp_path):
    c,t=fixture(count=40);cond={k:v[0].numpy() for k,v in c.items()}
    states=t['future'][0].numpy().copy();states[...,:2]+=cond['initial_positions'][:,None,:]
    source=tmp_path/'trajectories'/'a';source.mkdir(parents=True)
    m=dict(raw={'quality_pass':True},decoded={'motion':{'comfort_pass_agents':2,'comfort_assessed_agents':2}},
           corrections={'velocity_correction_mps':{'max':.2}})
    (source/'metrics.json').write_text(json.dumps(m))
    np.savez(source/'stages.npz',decoded_absolute_states=states)
    failed=tmp_path/'trajectories'/'b';failed.mkdir()
    (failed/'metrics.json').write_text(json.dumps(dict(raw={},decoded={})))
    rows=[dict(window_id='a',generation_failed=False),dict(window_id='b',generation_failed=True)]
    aggregate=dict(pooled={},counts={})
    rows,out=extend_quality_rows(tmp_path,rows,aggregate,{'a':cond,'b':cond})
    assert out['strict_comfort']['pass_agents']==2 and out['strict_comfort']['assessed_agents']==2
    assert out['strict_comfort']['planned_agents']==4 and out['strict_comfort']['unassessed_agents']==2
    assert out['strict_comfort']['pass_rate']==.5 and rows[1]['strict_comfort_rate']==0
    assert rows[1]['decoded_boundary_jerk_p99'] is None
    assert rows[0]['velocity_correction_max']==.2 and rows[0]['decoded_boundary_jerk_p99']<1e-8

def test_single_vehicle_collision_is_na_and_not_unknown_or_zero_probability(tmp_path):
    c,t=fixture(agents=1,count=40)
    cond={k:v[0].numpy() for k,v in c.items()}
    source=tmp_path/'trajectories'/'single';source.mkdir(parents=True)
    (source/'metrics.json').write_text(json.dumps(dict(raw={},decoded={})))
    rows=[dict(window_id='single',generation_failed=True,collision=None)]
    _,out=extend_quality_rows(tmp_path,rows,dict(pooled={},counts={}),{'single':cond})
    assert out['counts']['collision_unknown']==0
    assert out['counts']['collision_not_applicable_scenes']==1
    assert out['collision_applicability']['applicable_scenes']==0
    assert out['collision_applicability']['confirmed_collision_rate'] is None
