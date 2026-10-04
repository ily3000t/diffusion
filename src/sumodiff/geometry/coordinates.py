"""SUMO coordinate conversion and one fixed frame per window."""
from dataclasses import dataclass
import numpy as np


def sumo_center(front, navigation_degrees, length):
    front = np.asarray(front, dtype=np.float64)
    yaw = np.deg2rad(90.0 - np.asarray(navigation_degrees, dtype=np.float64))
    heading = np.stack((np.cos(yaw), np.sin(yaw)), axis=-1)
    return front - np.asarray(length)[..., None] * 0.5 * heading, yaw


@dataclass(frozen=True)
class FixedFrame:
    origin: np.ndarray
    yaw: float

    def __post_init__(self):
        origin = np.array(self.origin, dtype=np.float64, copy=True)
        if origin.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(self.yaw):
            raise ValueError("A frame requires a finite 2D origin and yaw")
        origin.setflags(write=False)
        object.__setattr__(self, 'origin', origin)

    @property
    def rotation(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, s], [-s, c]])

    def positions(self, world):
        return (np.asarray(world) - self.origin) @ self.rotation.T

    def vectors(self, world):
        return np.asarray(world) @ self.rotation.T

    def inverse_positions(self, local):
        return np.asarray(local) @ self.rotation + self.origin

    def inverse_vectors(self, local):
        return np.asarray(local) @ self.rotation

    def heading(self, world_yaw):
        angle = np.asarray(world_yaw) - self.yaw
        return np.stack((np.sin(angle), np.cos(angle)), axis=-1)

    def to_dict(self):
        return {'origin_world_m': self.origin.tolist(), 'yaw_world_rad': float(self.yaw)}
