"""Independent planar oriented rectangles and separating-axis test."""
import numpy as np


def wrap_angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


def box_corners(poses, sizes):
    poses, sizes = np.asarray(poses, dtype=np.float64), np.asarray(sizes, dtype=np.float64)
    if poses.shape[-1] != 3 or sizes.shape[-1] != 2 or not np.isfinite(poses).all() or not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError('Finite xy/yaw and positive length/width are required')
    signs = np.array([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=np.float64)
    offsets = sizes[..., None, :] * signs / 2
    cosine, sine = np.cos(poses[..., 2]), np.sin(poses[..., 2])
    x = offsets[..., 0] * cosine[..., None] - offsets[..., 1] * sine[..., None]
    y = offsets[..., 0] * sine[..., None] + offsets[..., 1] * cosine[..., None]
    return poses[..., None, :2] + np.stack((x, y), axis=-1)


def box_axes(pose):
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    return np.array([[cosine, sine], [-sine, cosine]])


def separation_margin(a, b, size_a, size_b):
    """Maximum signed projection gap; <=0 means intersection/contact.

    A positive gap is a separating-axis certificate, not collision probability
    and not necessarily Euclidean distance between the rectangles.
    """
    axes_a, axes_b = box_axes(a), box_axes(b)
    axes = np.vstack((axes_a, axes_b))
    radius_a = np.abs(axes @ axes_a.T) @ (np.asarray(size_a) / 2)
    radius_b = np.abs(axes @ axes_b.T) @ (np.asarray(size_b) / 2)
    return float(np.max(np.abs(axes @ (np.asarray(b[:2]) - a[:2])) - radius_a - radius_b))


def interpolate_pose(start, end, fraction):
    xy = start[:2] + fraction * (end[:2] - start[:2])
    yaw = start[2] + fraction * wrap_angle(end[2] - start[2])
    return np.r_[xy, yaw]
