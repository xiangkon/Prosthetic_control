"""Validated input and explicit clock alignment; no imputation of missing points."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
from ._matlab import read_tum


@dataclass
class MotionData:
    t: np.ndarray
    p: np.ndarray
    names: list
    valid: np.ndarray = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t = np.asarray(self.t, float).reshape(-1)
        self.p = np.asarray(self.p, float).copy()
        self.names = [str(n) for n in self.names]
        if not len(self.t) or self.p.shape != (len(self.t), len(self.names), 3):
            raise ValueError('Expected t[N], p[N,M,3], names[M]')
        if len(set(self.names)) != len(self.names):
            raise ValueError('Duplicate marker names')
        if not np.isfinite(self.t).all() or np.any(np.diff(self.t) <= 0):
            raise ValueError('Time must be finite and strictly increasing')
        measured = np.isfinite(self.p).all(axis=2) & ~np.all(self.p == 0, axis=2)
        if self.valid is not None:
            if np.shape(self.valid) != measured.shape:
                raise ValueError('Invalid validity mask shape')
            measured &= np.asarray(self.valid, bool)
        self.valid = measured
        self.p[~measured] = np.nan

    def select(self, stride=1, limit=None):
        if stride < 1 or (limit is not None and limit < 1):
            raise ValueError('stride and limit must be positive')
        idx = np.arange(len(self.t))[::stride]
        if limit is not None:
            idx = idx[:limit]
        return MotionData(self.t[idx], self.p[idx], self.names, self.valid[idx], self.metadata.copy())

    def save(self, path):
        np.savez_compressed(path, t=self.t, P=self.p, names=np.array(self.names),
                            valid=self.valid, units=np.array('m'),
                            metadata=np.array(json.dumps(self.metadata, ensure_ascii=False)))


def load_motion(path):
    path = Path(path)
    if path.suffix.lower() == '.mat':
        raw = read_tum(path)
        # Existing guarded MCOS reader normalizes the first timestamp to zero.
        return MotionData(raw.time, raw.points * 1e-3, raw.names, metadata={
            'source': str(path.resolve()), 'units': 'm', 'raw_units': 'mm',
            'unit_source': 'TUM plot_KIN.m Position [mm]',
            'clock': 'trial-relative seconds, first timestamp subtracted'})
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as z:
            if 'units' not in z or str(z['units'].item()) != 'm':
                raise ValueError('Reconstruction NPZ must explicitly declare units=m; use convert for raw MAT')
            meta = json.loads(str(z['metadata'].item())) if 'metadata' in z else {}
            return MotionData(z['t'], z['P'], z['names'].tolist(), z['valid'], meta)
    raise ValueError('Expected original TUM .mat or standardized .npz')


def trial_metadata(path):
    """Look up task IDs using the source table, never trial-number arithmetic."""
    path = Path(path)
    if path.parent.name != 'MoCap':
        return {}
    trial = int(path.stem.split('_')[-1])
    info = loadmat(path.parent.parent / 'trial_info.mat', simplify_cells=True)['trial_info']
    matches = []
    for task, row in enumerate(np.atleast_1d(info)):
        ids = [int(x) for x in np.atleast_1d(row['Kinematics']).ravel()]
        if trial in ids:
            matches.append(dict(task_id=task, description=str(row['Description']),
                                valid_list_position=ids.index(trial) + 1))
    if len(matches) > 1:
        raise ValueError('Trial mapped to multiple tasks')
    return dict(subject=path.parent.parent.name, trial_id=trial,
                referenced_by_trial_info=bool(matches), **(matches[0] if matches else {}))


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
