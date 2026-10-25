import numpy as np

# Constants (copied from plot_spider_pose.m)

NUM_LEGS = 8
LEG_ORDER = ("L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1")
SEGMENT_LENGTHS = np.array([1.2, 0.7, 1.0])
BODY_A, BODY_B = 1.5, 1.0
COXA_ELEVATION = np.deg2rad(30.0)

BASE_ANGLES = np.deg2rad([45, 75, 105, 135, -135, -105, -75, -45])
BASE_POSITIONS = np.stack(
    [BODY_A * np.cos(BASE_ANGLES),
     BODY_B * np.sin(BASE_ANGLES),
     np.zeros(NUM_LEGS)],
    axis=-1,
)

_UP = np.array([0.0, 0.0, 1.0])

def _as_legs(angles):
    a = np.asarray(angles, dtype=float)
    return a.reshape(*a.shape[:-1], NUM_LEGS, 3)

def leg_joints(angles):
    legs = _as_legs(angles)
    th1, th2, th3 = legs[..., 0], legs[..., 1], legs[..., 2]

    yaw = BASE_ANGLES + th1                                    # (..., 8)
    horiz = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=-1)

    coxa_elev = np.broadcast_to(COXA_ELEVATION, yaw.shape)
    femur_elev = COXA_ELEVATION + th2
    tibia_elev = femur_elev + th3
    elev = np.stack([coxa_elev, femur_elev, tibia_elev], axis=-1)  # (..., 8, 3)

    seg_dirs = (np.cos(elev)[..., None] * horiz[..., None, :]
                + np.sin(elev)[..., None] * _UP)               # (..., 8, 3, 3)

    steps = SEGMENT_LENGTHS[:, None] * seg_dirs
    base = np.broadcast_to(BASE_POSITIONS, horiz.shape)
    rest = base[..., None, :] + np.cumsum(steps, axis=-2)
    return np.concatenate([base[..., None, :], rest], axis=-2)

def foot_positions(angles):
    return leg_joints(angles)[..., 3, :]

