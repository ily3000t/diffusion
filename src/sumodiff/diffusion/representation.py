"""Explicit label transform and shared control rollout; inference has no labels."""
import torch
from sumodiff.decoding import DecodedTrajectory, DecoderConfig, decode_future, observed_heading
from sumodiff.decoding.controls import ControlConfig, initial_control_state, inverse_label_controls, rollout_controls
from .config import is_control
from .process import normalize

CONTROL = ControlConfig(reference_point='center', integration_heading='midpoint',
                        lateral_velocity=True, compute_dtype='float64')


def initial_state(c):
    return initial_control_state(c['history'],c['history_mask'],c['initial_positions'],
                                 c['attributes'][...,0],c['agent_mask'],CONTROL)


def training_target(targets,c,config):
    if not is_control(config):
        return normalize(targets['future'],config)
    a=c['agent_mask']
    truth=torch.where(a[...,None,None],targets['future'],0.).double()
    absolute=torch.cat((truth[...,:2]+c['initial_positions'][...,None,:].double(),truth[...,2:]),-1)
    with torch.no_grad():
        physical=inverse_label_controls(absolute,targets['future_mask'],initial_state(c)).controls
    return normalize(physical,config).to(targets['future'].dtype)


def decode_prediction(physical,c,config,decoder=None):
    if not is_control(config):
        return decode_future(physical,c['initial_positions'],
                             observed_heading(c['history'],c['history_mask'],c['agent_mask']),
                             c['agent_mask'],decoder)
    if decoder is not None and decoder != DecoderConfig():
        raise ValueError('Control rollout uses fixed float64 center/midpoint integration; LS decoder options do not apply')
    initial=initial_state(c)
    trajectory=rollout_controls(physical,initial)
    states=trajectory.states
    a=c['agent_mask']
    raw_future=torch.cat((states[...,:2]-initial.center_position[...,None,:],states[...,2:]),-1)
    raw_future=torch.where(a[...,None,None],raw_future,0.)
    zero2=torch.zeros_like(states[...,:2]);zero1=torch.zeros_like(states[...,0])
    valid=a[...,None].expand_as(zero1)
    heading=torch.stack((torch.sin(initial.yaw),torch.cos(initial.yaw)),-1)
    return DecodedTrajectory(raw_future,states,states,trajectory.positions_with_start,heading,
        zero2,zero2,zero2,zero1,valid,zero2,zero1,torch.zeros_like(valid),a)
