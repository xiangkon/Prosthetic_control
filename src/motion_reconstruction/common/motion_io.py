"""Validated raw motion input; no joint models or segment fitting."""
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
