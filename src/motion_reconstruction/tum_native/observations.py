"""Task-table indexing, gap-aware filtering and robust technical-frame tracking."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from ..common.motion_io import load_motion  # raw reader only; no legacy geometry
from .schema import SEGMENTS, GROUPS, runs


def tasks(subject_dir):
    info = np.atleast_1d(loadmat(Path(subject_dir) / 'trial_info.mat', simplify_cells=True)['trial_info'])
    result = []
    for task, row in enumerate(info):
        emg = {int(x) for x in np.atleast_1d(row['EMG']).ravel()}
        for repeat, value in enumerate(np.atleast_1d(row['Kinematics']).ravel(), 1):
            result.append(dict(task=task, trial=int(value), repeat=repeat, description=str(row['Description']), paired_emg=int(value) in emg))
    if len({r['trial'] for r in result}) != len(result):
        raise ValueError('A motion trial is referenced more than once')
    return result


def prepare(data, config):
    t, raw = data.t, data.p
    result = raw.copy()
    interpolated = np.zeros(raw.shape[:2], bool)
    edge = np.zeros(raw.shape[:2], bool)
    dt = float(np.median(np.diff(t)))
    fs = 1 / dt
    if not np.allclose(np.diff(t), dt, rtol=0, atol=1e-8):
        raise ValueError('Nonuniform clock must be explicitly resampled before filtering')
    cutoff = config['cutoff_hz']
    if cutoff is not None and not 0 < cutoff < fs / 2:
        raise ValueError('Invalid marker low-pass cutoff')
    sos = butter(config['filter_order'], cutoff, fs=fs, output='sos') if cutoff else None
    for marker in range(raw.shape[1]):
        valid = np.isfinite(result[:, marker]).all(1)
        for gap in runs(~valid):
            left, right = gap[0] - 1, gap[-1] + 1
            if left >= 0 and right < len(t) and len(gap) * dt <= config['max_interpolation_gap_s'] + 1e-10:
                for axis in range(3):
                    result[gap, marker, axis] = np.interp(t[gap], t[[left, right]], result[[left, right], marker, axis])
                interpolated[gap, marker] = True
        for part in runs(np.isfinite(result[:, marker]).all(1)):
            if sos is None:
                continue
            if len(part) <= 3 * (2 * len(sos) + 1):
                edge[part, marker] = True
                continue
            result[part, marker] = sosfiltfilt(sos, result[part, marker], axis=0)
            margin = max(1, int(np.ceil(config['filter_edge_s'] / dt)))
            edge[part[:margin], marker] = True
            edge[part[-margin:], marker] = True
    return dict(t=t, p=result, raw=raw, names=data.names, observed=data.valid,
        interpolated=interpolated, edge=edge)


def rigid_fit(local, measured, sigma=None):
    valid = np.isfinite(local).all(1) & np.isfinite(measured).all(1)
    if valid.sum() < 3:
        return None, None, np.nan, int(valid.sum()), 0.
    x, y = local[valid], measured[valid]
    singular = np.linalg.svd(x - x.mean(0), compute_uv=False)
    shape_ratio = singular[1] / max(singular[0], 1e-12)
    world_singular = np.linalg.svd(y - y.mean(0), compute_uv=False)
    if shape_ratio < .01 or world_singular[1] / max(world_singular[0], 1e-12) < .01:
        return None, None, np.nan, int(valid.sum()), float(shape_ratio)
    base = np.ones(len(x)) if sigma is None else 1 / np.asarray(sigma)[valid]**2
    weights = base.copy()
    for _ in range(3):
        weights /= weights.sum()
        a, b = weights @ x, weights @ y
        u, _, vt = np.linalg.svd(((x - a) * weights[:, None]).T @ (y - b))
        rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
        translation = b - rotation @ a
        residual = np.linalg.norm(x @ rotation.T + translation - y, axis=1)
        weights = base / np.sqrt(1 + (residual / .005)**2)
    return rotation, translation, float(np.sqrt(np.mean(residual**2))), int(valid.sum()), float(shape_ratio)


def track(obs, model):
    n = len(obs['t'])
    rotations = np.full((n, 4, 3, 3), np.nan)
    translations = np.full((n, 4, 3), np.nan)
    rms = np.full((n, 4), np.nan)
    counts = np.zeros((n, 4), int)
    degeneracy = np.zeros((n, 4), bool)
    edges = np.zeros((n, 4), bool)
    interpolated = np.zeros((n, 4), bool)
    for k, segment in enumerate(SEGMENTS):
        spec = model['segments'][segment]
        names = spec['fit_names']
        ids = [obs['names'].index(name) for name in names]
        local = np.array([spec['local_m'][name] for name in names])
        sigma = np.array([spec['sigma_m'][name] for name in names])
        for i in range(n):
            r, p, e, count, ratio = rigid_fit(local, obs['p'][i, ids], sigma)
            counts[i, k] = count
            degeneracy[i, k] = count >= 3 and r is None
            if r is not None:
                rotations[i, k], translations[i, k], rms[i, k] = r, p, e
            used = np.isfinite(obs['p'][i, ids]).all(1)
            edges[i, k] = bool(np.any(obs['edge'][i, ids][used]))
            interpolated[i, k] = bool(np.any(obs['interpolated'][i, ids][used]))
    return dict(R=rotations, p=translations, rms_m=rms, counts=counts,
        degenerate=degeneracy, edge=edges, interpolated=interpolated)
