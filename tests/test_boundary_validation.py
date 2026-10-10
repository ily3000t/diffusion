import numpy as np
import pytest
import torch
from test_physical_losses import fixture
from sumodiff.decoding import decode_future,observed_heading
from sumodiff.diffusion.boundary_validation import (
    config,error_jet,prefix_position,select_training,quantities)

def test_first_centimeter_error_has_correct_jerk_units_and_stencil():
    e=np.zeros((1,40,2));e[0,0,0]=.01
    v,a,j=error_jet(e)
    np.testing.assert_allclose(v[0,:2,0],[.1,-.1],atol=1e-12)
    np.testing.assert_allclose(a[0,:3,0],[1,-2,1],atol=1e-12)
    np.testing.assert_allclose(j[0,:5,0],[10,-30,30,-10,0],atol=1e-12)

def test_oracle_prefix_can_move_a_position_discontinuity_to_a_later_join():
    label=np.zeros((1,40,6));raw=label.copy();raw[...,0]=.02;raw[...,2]=12.
    edited=prefix_position(raw,label,4)
    _,_,j=error_jet(edited[...,:2])
    assert np.linalg.norm(j[:,:4],axis=-1).max()==0
    assert np.linalg.norm(j[:,4:],axis=-1).max()>15
    assert np.array_equal(edited[...,2:],raw[...,2:])
    assert np.array_equal(raw[...,0],np.full((1,40),.02))

def test_decoded_boundary_identity_uses_observed_history_and_excludes_padding():
    c,t=fixture(count=40)
    c['agent_mask'][:,1]=False;c['history_mask'][:,1]=False
    raw=t['future'].clone();raw[0,0,0,0]+=.01;raw[0,1]=float('nan')
    out=decode_future(raw,c['initial_positions'],observed_heading(c['history'],c['history_mask'],c['agent_mask']),c['agent_mask'])
    cc={k:v[0].numpy() for k,v in c.items()}
    q=quantities(cc,raw[0].numpy(),out.states[0].numpy(),t['future'][0].numpy())
    assert q['raw_position_first'].shape==(1,)
    assert q['first4_position_linear_identity_error'][0]<1e-8
    np.testing.assert_allclose(q['decoded_jerk_error_boundary'][:,0],q['decoded_position_first']*1000,atol=1e-8)
    assert all(np.isfinite(v).all() for v in q.values())

def test_selection_keeps_all_dense_and_balances_metadata_groups_without_labels():
    d=[]
    for family in ('straight','merge','unsignalized_intersection'):
        for episode in ('e1','e2'):
            for i in range(5):
                d.append(dict(window_id=family+episode+str(i),family=family,
                    geometry_id=episode,episode_key=episode,agents=2))
    d.extend([dict(window_id='dense'+str(i),family='unsignalized_intersection',
        geometry_id='gd',episode_key='ed',agents=8+i) for i in range(3)])
    selected=select_training(d,4,10)
    assert set(range(30,33))<=set(selected) and len(selected)==15
    for family in ('straight','merge','unsignalized_intersection'):
        controls=[d[i] for i in selected if d[i]['family']==family and d[i]['agents']<8]
        assert len(controls)==4 and {x['episode_key'] for x in controls}=={'e1','e2'}
    assert select_training(d,4,10)==selected
    with pytest.raises(ValueError,match='Insufficient'):select_training(d,11,10)

@pytest.mark.parametrize('bad',[{'timesteps':[0,0]},{'timesteps':[301]},{'timesteps':list(range(6))},
    {'normal_train_per_family':25},{'batch_size':8},{'test_windows':10},{'seed':-1}])
def test_unbounded_or_test_data_options_rejected(bad):
    with pytest.raises(ValueError):config(bad)

@pytest.mark.parametrize('frames',[-1,41,True])
def test_invalid_prefix_rejected(frames):
    with pytest.raises(ValueError):prefix_position(np.zeros((1,40,6)),np.zeros((1,40,6)),frames)
