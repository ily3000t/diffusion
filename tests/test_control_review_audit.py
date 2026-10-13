from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
import pytest
from sumodiff.decoding.controls import ControlConfig
from sumodiff.decoding.replay_audit import (
    numpy_inverse, numpy_rollout, numpy_gate, verify_source_report)
from sumodiff.decoding.control_replay import compatibility_gate


@pytest.mark.parametrize('point',['center','front'])
@pytest.mark.parametrize('basis',['midpoint','endpoint'])
@pytest.mark.parametrize('lateral',[False,True])
def test_numpy_label_inverse_recovers_known_closed_form_controls(point,basis,lateral):
    config=asdict(ControlConfig(reference_point=point,integration_heading=basis,lateral_velocity=lateral))
    angle=.4;turn=.025;dt=.1;offset=2. if point=='front' else 0.
    p0=np.array([[3.,-2.]])
    direction=lambda x:np.stack((np.cos(x),np.sin(x)),-1)
    initial_ref=p0+offset*direction(np.array([angle]))
    past_angle=angle-turn
    past_basis=angle-turn/2 if basis=='midpoint' else angle
    previous_ref=initial_ref-.8*direction(np.array([past_basis]))
    h=np.zeros((1,21,6))
    h[0,-2,:2]=(previous_ref-offset*direction(np.array([past_angle])))[0]
    h[0,-1,:2]=p0[0]
    h[0,-2,4:6]=[np.sin(past_angle),np.cos(past_angle)]
    h[0,-1,4:6]=[np.sin(angle),np.cos(angle)]
    k=np.arange(1,41)
    angles=angle+k*turn
    bases=angles-turn/2 if basis=='midpoint' else angles
    u=8.+k*.03
    r=.13 if lateral else 0.
    axis=direction(bases);side=np.stack((-axis[:,1],axis[:,0]),-1)
    reference=initial_ref+np.cumsum(dt*(u[:,None]*axis+r*side),axis=0)
    position=reference-offset*direction(angles)
    velocity=np.diff(np.concatenate((p0,position)),axis=0)/dt
    states=np.concatenate((position,velocity,np.stack((np.sin(angles),np.cos(angles)),-1)),axis=-1)[None]
    controls,residual,velocity_error=numpy_inverse(h,p0,np.array([4.]),np.array([True]),states,config)
    expected=np.broadcast_to(np.array([.3,.25,.13] if lateral else [.3,.25]),controls.shape)
    np.testing.assert_allclose(controls,expected,atol=1e-10,rtol=0)
    np.testing.assert_allclose(residual,r,atol=1e-11,rtol=0)
    np.testing.assert_allclose(velocity_error,0.,atol=1e-12,rtol=0)
    np.testing.assert_allclose(numpy_rollout(h,p0,np.array([4.]),np.array([True]),controls,config),states,atol=1e-10,rtol=0)


def test_inverse_audit_is_explicit_about_compute_precision():
    config=asdict(ControlConfig(compute_dtype='float32'))
    with pytest.raises(NotImplementedError,match='float64'):
        numpy_inverse(np.zeros((1,21,6)),np.zeros((1,2)),np.ones(1),np.array([True]),np.zeros((1,40,6)),config)


@pytest.mark.parametrize('change',['none','motion','road','collision','comfort','unknown'])
def test_independent_gate_agrees_on_quality_changes(change):
    original=dict(quality_pass=True,motion=dict(violation_counts={'jerk':0}),
                  roads={n:dict(violation_body_frames=0) for n in ('road','route','map_coverage')},
                  collisions=dict(pairs=[dict(slots=[0,1],collision=False)]))
    replay=deepcopy(original);old={'0':True,'1':False};new=dict(old)
    if change=='motion':replay['motion']['violation_counts']['jerk']=1
    if change=='road':replay['roads']['road']['violation_body_frames']=1
    if change=='collision':replay['collisions']['pairs'][0]['collision']=None
    if change=='comfort':new={'0':False,'1':True}
    if change=='unknown':replay['quality_pass']=None
    errors=dict(max_position_error_m=.01,max_velocity_error_mps=.01,max_heading_error_rad=0.)
    limits=dict(max_position_error_m=.05,max_velocity_error_mps=.05,max_heading_error_rad=1e-5)
    assert numpy_gate(original,replay,old,new,errors,limits)==compatibility_gate(original,replay,old,new,errors,limits)


@pytest.mark.parametrize('change',['window','denominator','method','metrics'])
def test_source_report_identity_and_denominator_tampering_are_detected(change):
    task=dict(window_id='x',split='train',group='ordinary')
    row={**task,'methods':{'label':{},'two':{}}}
    results=dict(planned=1,records=[row],saved_outputs=[],optimizer_updates=0,denoiser_calls=0,free_samples=0,test_arrays_loaded=0)
    plan=dict(tasks=[dict(task)]);config=dict(methods={'two':{}},method_configs={'two':{}})
    metrics={k:v for k,v in results.items()if k not in ('records','saved_outputs')}
    verify_source_report(results,plan,config,metrics)
    if change=='window':results['records'][0]['window_id']='other'
    if change=='denominator':results['planned']=2
    if change=='method':results['records'][0]['methods'].pop('two')
    if change=='metrics':metrics['planned']=9
    with pytest.raises(AssertionError):
        verify_source_report(results,plan,config,metrics)
