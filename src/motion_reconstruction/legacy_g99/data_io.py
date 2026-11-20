"""Legacy-model fitting and label alignment; raw reading lives in common."""
import numpy as np
from ..common.motion_io import MotionData, load_motion, trial_metadata


def kabsch(local, world):
    local, world = np.asarray(local), np.asarray(world)
    if (len(local) < 3 or np.linalg.matrix_rank(local - local.mean(0), tol=1e-8) < 2
            or np.linalg.matrix_rank(world - world.mean(0), tol=1e-8) < 2):
        raise ValueError('Need at least three non-collinear torso anchors')
    a, b = local.mean(0), world.mean(0)
    u, _, vt = np.linalg.svd((local-a).T @ (world-b))
    rot = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    T = np.eye(4)
    T[:3,:3], T[:3,3] = rot, b - rot @ a
    return T


def thorax_poses(data, model):
    """Fit calibrated Thorax-local markers; missing poses stay invalid."""
    names = [n for n,b in model.bindings.items() if b['segment'] == 'thorax' and n in data.names]
    idx = np.array([data.names.index(n) for n in names], int)
    local = np.array([model.bindings[n]['local_m'] for n in names]).reshape(-1,3)
    poses = np.full((len(data.t),4,4), np.nan)
    rms = np.full(len(data.t), np.nan)
    for t in range(len(data.t)):
        valid = data.valid[t,idx]
        try:
            T = kabsch(local[valid], data.p[t,idx[valid]])
        except ValueError:
            continue
        poses[t] = T
        predicted = local[valid] @ T[:3,:3].T + T[:3,3]
        rms[t] = np.sqrt(np.mean(np.sum((predicted-data.p[t,idx[valid]])**2,axis=1)))
    return poses, rms


def align_labels(t, q, flags, target_t, offset_s, clock_scale=1.0, max_gap_s=0.05):
    """Map source time to target clock: target = scale*source + offset.

    Interpolate only between usable adjacent samples; never bridge rejected frames
    or extrapolate. Clock parameters must come from external synchronization.
    """
    if clock_scale <= 0 or max_gap_s <= 0 or not np.isfinite([offset_s,clock_scale,max_gap_s]).all():
        raise ValueError('Invalid synchronization parameters')
    source = np.asarray(t)*clock_scale + offset_s
    target_t = np.asarray(target_t,float).reshape(-1)
    if not np.isfinite(target_t).all() or np.any(np.diff(target_t) <= 0):
        raise ValueError('Target timestamps must be finite and increasing')
    out = np.full((len(target_t),q.shape[1]),np.nan)
    quality = np.full(len(target_t),1,dtype=np.uint16)
    for j,time in enumerate(target_t):
        i = np.searchsorted(source,time)
        if i < len(source) and np.isclose(source[i],time,rtol=0,atol=1e-10):
            if flags[i] == 0:
                out[j], quality[j] = q[i], 0
        elif 0 < i < len(source) and source[i]-source[i-1] <= max_gap_s:
            if flags[i] == 0 and flags[i-1] == 0:
                a=(time-source[i-1])/(source[i]-source[i-1])
                out[j], quality[j] = (1-a)*q[i-1]+a*q[i], 0
    return out, quality
