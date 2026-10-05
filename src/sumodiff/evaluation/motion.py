"""Independent motion derivatives and history/future boundary diagnostics."""
from dataclasses import dataclass
import math
import numpy as np
from sumodiff.geometry.boxes import wrap_angle


@dataclass(frozen=True)
class MotionConfig:
    max_speed_mps: float = 35.
    max_longitudinal_accel_mps2: float = 6.
    max_lateral_accel_mps2: float = 6.
    max_jerk_mps3: float = 15.
    max_yaw_rate_radps: float = 3.
    max_heading_velocity_angle_rad: float = math.pi/4
    moving_speed_min_mps: float = .5
    heading_norm_tolerance: float = .1
    max_velocity_residual_mps: float = .5
    comfort_longitudinal_accel_mps2: float = 3.
    comfort_lateral_accel_mps2: float = 3.
    comfort_jerk_mps3: float = 5.

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')


def summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return dict(count=0, mean=None, min=None, p50=None, p95=None, p99=None, max=None)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite value in an applicable metric')
    q = np.quantile(values,[.5,.95,.99])
    return dict(count=len(values), mean=float(values.mean()), min=float(values.min()),
                p50=float(q[0]), p95=float(q[1]), p99=float(q[2]), max=float(values.max()))


def _difference(values, valid, dt):
    result = np.zeros_like(values)
    mask = np.zeros_like(valid)
    result[:,1:] = np.diff(values,axis=1)/dt
    mask[:,1:] = valid[:,1:] & valid[:,:-1]
    return result, mask


def evaluate_motion(history, history_mask, states, future_mask, initial_positions, initial_heading, agent_mask, dt=.1, config=None):
    config = config or MotionConfig()
    h,f = np.asarray(history,float),np.asarray(states,float)
    hm,fm,agents = np.asarray(history_mask,bool),np.asarray(future_mask,bool),np.asarray(agent_mask,bool)
    initial_positions,initial_heading = np.asarray(initial_positions,float),np.asarray(initial_heading,float)
    if h.ndim != 3 or f.ndim != 3 or h.shape[0] != f.shape[0] or h.shape[-1] != 6 or f.shape[-1] != 6 or hm.shape != h.shape[:2] or fm.shape != f.shape[:2] or agents.shape != (len(h),):
        raise ValueError('Invalid motion input shapes')
    if not np.isfinite(dt) or dt <= 0 or h.shape[1] < 3 or not f.shape[1]:
        raise ValueError('Motion checks need positive dt, at least three history points and a future')
    if initial_positions.shape != (len(h),2) or initial_heading.shape != (len(h),2):
        raise ValueError('Invalid current pose shapes')
    hm, fm = hm & agents[:,None], fm & agents[:,None]
    if not np.isfinite(h[hm]).all() or not np.isfinite(f[fm]).all() or not np.isfinite(initial_positions[agents]).all() or not np.isfinite(initial_heading[agents]).all():
        raise ValueError('Non-finite active motion input')
    heading0_norm = np.linalg.norm(initial_heading,axis=-1)
    if (heading0_norm[agents] < 1e-6).any():
        raise ValueError('Current heading is unknown')
    h,f = np.where(hm[...,None],h,0.),np.where(fm[...,None],f,0.)
    context = np.concatenate((h,f),axis=1)
    valid = np.concatenate((hm,fm),axis=1)
    start = h.shape[1]-1
    context[:,start,:2] = np.where(agents[:,None],initial_positions,0.)
    context[:,start,4:6] = initial_heading / np.maximum(heading0_norm[:,None],1e-6)
    valid[:,start] = agents  # t0 geometric center/heading explicitly observed.
    velocity, vm = _difference(context[...,:2],valid,dt)
    acceleration, am = _difference(velocity,vm,dt)
    jerk, jm = _difference(acceleration,am,dt)
    heading_norm = np.linalg.norm(context[...,4:6],axis=-1)
    heading_valid = valid & (heading_norm >= 1e-6)
    yaw = np.arctan2(context[...,4],context[...,5])
    yaw_rate = np.zeros_like(yaw)
    yaw_rate[:,1:] = wrap_angle(np.diff(yaw,axis=1))/dt
    yaw_rate_valid = np.zeros_like(valid)
    yaw_rate_valid[:,1:] = heading_valid[:,1:] & heading_valid[:,:-1]
    axis = np.stack((np.cos(yaw),np.sin(yaw)),axis=-1)
    side = np.stack((-np.sin(yaw),np.cos(yaw)),axis=-1)
    longitudinal = (acceleration*axis).sum(axis=-1)
    lateral = (acceleration*side).sum(axis=-1)
    speed,jerk_norm = np.linalg.norm(velocity,axis=-1),np.linalg.norm(jerk,axis=-1)
    acceleration_norm = np.linalg.norm(acceleration,axis=-1)
    residual = np.linalg.norm(context[...,2:4]-velocity,axis=-1)
    angle = np.abs(wrap_angle(np.arctan2(velocity[...,1],velocity[...,0])-yaw))
    angle_mask = vm & heading_valid & (speed >= config.moving_speed_min_mps)
    slice_future = slice(start+1,None)
    sf,vf,af,jf = valid[:,slice_future],vm[:,slice_future],am[:,slice_future],jm[:,slice_future]
    hf = heading_valid[:,slice_future]
    long_mask = af & hf
    yf = yaw_rate_valid[:,slice_future]
    alignment_mask = angle_mask[:,slice_future]
    conditions = dict(speed=(speed[:,slice_future]>config.max_speed_mps)&vf,
        longitudinal_acceleration=(np.abs(longitudinal[:,slice_future])>config.max_longitudinal_accel_mps2)&long_mask,
        lateral_acceleration=(np.abs(lateral[:,slice_future])>config.max_lateral_accel_mps2)&long_mask,
        jerk=(jerk_norm[:,slice_future]>config.max_jerk_mps3)&jf,
        yaw_rate=(np.abs(yaw_rate[:,slice_future])>config.max_yaw_rate_radps)&yf,
        heading_velocity=(angle[:,slice_future]>config.max_heading_velocity_angle_rad)&alignment_mask,
        heading_norm=(np.abs(heading_norm[:,slice_future]-1)>config.heading_norm_tolerance)&sf,
        velocity_consistency=(residual[:,slice_future]>config.max_velocity_residual_mps)&vf)
    comfort = ((np.abs(longitudinal[:,slice_future])>config.comfort_longitudinal_accel_mps2)&long_mask) | \
              ((np.abs(lateral[:,slice_future])>config.comfort_lateral_accel_mps2)&long_mask) | \
              ((jerk_norm[:,slice_future]>config.comfort_jerk_mps3)&jf)
    applicable_comfort = long_mask & jf
    boundary = dict(displacement_m=summary(np.linalg.norm(context[:,start+1,:2]-context[:,start,:2],axis=-1)[valid[:,start+1]&agents]),
        velocity_change_mps=summary(np.linalg.norm(velocity[:,start+1]-velocity[:,start],axis=-1)[vm[:,start+1]&vm[:,start]]),
        acceleration_mps2=summary(acceleration_norm[:,start+1][am[:,start+1]]),
        acceleration_change_mps2=summary(np.linalg.norm(acceleration[:,start+1]-acceleration[:,start],axis=-1)[am[:,start+1]&am[:,start]]),
        jerk_mps3=summary(jerk_norm[:,start+1][jm[:,start+1]]),
        heading_change_rad=summary(np.abs(wrap_angle(yaw[:,start+1]-yaw[:,start]))[heading_valid[:,start+1]&heading_valid[:,start]]),
        assessed_agents=int(jm[:,start+1].sum()), missing_boundary_agents=int(agents.sum()-jm[:,start+1].sum()))
    full = bool(fm[agents].all()) if agents.any() else False
    hard_violation = any(value.any() for value in conditions.values())
    enough = bool(vf[agents].all() and af[agents].all() and jf[agents].all() and yf[agents].all()) if agents.any() else False
    hard_pass = False if hard_violation else (True if full and enough else None)
    comfort_agents = [bool(not comfort[a].any()) for a in np.flatnonzero(agents) if applicable_comfort[a].any()]
    return dict(schema_version='sumodiff.motion.v1', speed_mps=summary(speed[:,slice_future][vf]),
        predicted_speed_mps=summary(np.linalg.norm(f[...,2:4],axis=-1)[fm]),
        longitudinal_accel_abs_mps2=summary(np.abs(longitudinal[:,slice_future])[long_mask]),
        lateral_accel_abs_mps2=summary(np.abs(lateral[:,slice_future])[long_mask]),
        accel_norm_mps2=summary(acceleration_norm[:,slice_future][af]), jerk_norm_mps3=summary(jerk_norm[:,slice_future][jf]),
        yaw_rate_abs_radps=summary(np.abs(yaw_rate[:,slice_future])[yf]),
        heading_velocity_angle_rad=summary(angle[:,slice_future][alignment_mask]),
        heading_norm_error=summary(np.abs(heading_norm[:,slice_future]-1)[sf]), velocity_residual_mps=summary(residual[:,slice_future][vf]),
        invalid_heading_points=int((sf&~hf).sum()), violation_counts={k:int(v.sum()) for k,v in conditions.items()},
        comfort_violation_points=int(comfort.sum()), comfort_assessed_points=int(applicable_comfort.sum()),
        observed_comfort_pass_fraction_agents=sum(comfort_agents)/len(comfort_agents) if comfort_agents else None,
        hard_quality_pass=hard_pass, full_future_observed=full, boundary=boundary,
        thresholds_status='predeclared_engineering_candidates_not_behavioral_calibration',
        derivative_convention='backward interval means; acceleration/jerk from consecutive differences including history')
