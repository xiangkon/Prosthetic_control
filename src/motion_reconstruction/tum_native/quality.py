"""Per-coordinate measurement information, independent of priors and smoothing."""
import numpy as np
from scipy.spatial.transform import Rotation
from .model import forward
from .schema import *


def evaluate(obs, poses, model, solution, config):
    state = solution['state']
    n = len(state)
    flags = np.full((n, 7), REFERENCE_DEPENDENT | AXIS_PRIOR | CALIBRATION_WEAK, np.uint16)
    dependencies = [(0, 1)] * 3 + [(1, 2)] * 2 + [(2, 3)] * 2
    for k, segments in enumerate(dependencies):
        missing = ~np.isfinite(state[:, k])
        flags[missing, k] |= MISSING
        for segment in segments:
            flags[poses['degenerate'][:, segment], k] |= DEGENERATE
            flags[poses['edge'][:, segment], k] |= FILTER_EDGE
            flags[poses['interpolated'][:, segment], k] |= INTERPOLATED
    flags[solution['attempted'] & ~solution['converged']] |= NOT_CONVERGED
    chart = np.linalg.norm(state[:, :3], axis=1) > np.deg2rad(170)
    flags[chart, :3] |= ROTATION_CHART
    if 'm_acr' in obs['names']:
        shoulder_missing = ~np.isfinite(obs['p'][:, obs['names'].index('m_acr')]).all(1)
        flags[shoulder_missing, :3] |= LOW_INFORMATION
    full = np.isfinite(state).all(1)
    pred = np.full_like(obs['p'], np.nan)
    rotations = np.full((n, 4, 3, 3), np.nan)
    bones = np.full((n, 4, 3), np.nan)
    origins = np.full((n, 4, 3), np.nan)
    conditional_sd = np.full((n, 7), np.nan)
    local_error = solution['local_error'].copy()
    ids_full = np.flatnonzero(full)
    if len(ids_full):
        x = state[full]
        rt, pt = poses['R'][full, 0], poses['p'][full, 0]
        p, names, r, o, b = forward(model, x, rt, pt)
        ids = [obs['names'].index(name) for name in names]
        pred[np.ix_(ids_full, ids, np.arange(3))] = p
        rotations[full], origins[full], bones[full] = r, o, b
        for j, (a, bseg) in enumerate([(1, 2), (2, 3)]):
            target = np.swapaxes(poses['R'][full, a], 1, 2) @ poses['R'][full, bseg]
            fitted = np.swapaxes(r[:, a], 1, 2) @ r[:, bseg]
            local_error[full, j] = np.linalg.norm(Rotation.from_matrix(np.swapaxes(fitted, 1, 2) @ target).as_rotvec(), axis=1)
        fit_names = [name for spec in model['segments'].values() for name in spec['fit_names']]
        fit_ids = [names.index(name) for name in fit_names]
        raw_ids = [obs['names'].index(name) for name in fit_names]
        sigma = np.array([spec['sigma_m'][name] for spec in model['segments'].values() for name in spec['fit_names']])
        valid = np.isfinite(obs['p'][full][:, raw_ids]).all(2)
        # Only marker measurement rows: no temporal/position/initial priors.
        jac = []
        for k in range(10):
            shifted = x.copy()
            shifted[:, k] += 1e-5
            pp = forward(model, shifted, rt, pt)[0]
            column = (pp[:, fit_ids] - p[:, fit_ids]) / 1e-5 / sigma[None, :, None]
            jac.append(np.where(valid[:, :, None], column, 0).reshape(len(x), -1))
        jac = np.stack(jac, axis=2)
        for k in range(7):
            column = jac[:, :, k]
            other = np.delete(jac, k, axis=2)
            coefficients = np.linalg.pinv(other, rcond=1e-7) @ column[:, :, None]
            remaining = column - (other @ coefficients)[:, :, 0]
            information = np.sum(remaining**2, axis=1)
            conditional_sd[full, k] = 1 / np.sqrt(np.maximum(information, 1e-20))
    for j, columns in enumerate([(3, 4), (5, 6)]):
        bad = local_error[:, j] > np.deg2rad(config['orientation_mismatch_deg'])
        flags[np.ix_(np.flatnonzero(bad), columns)] |= MODEL_MISMATCH
    flags[(~np.isfinite(conditional_sd)) | (conditional_sd > np.deg2rad(config['conditional_angle_sd_deg']))] |= LOW_INFORMATION
    return dict(flags=flags, relative_valid=np.isfinite(state[:, :7]) & ((flags & RELATIVE_REJECT) == 0),
        absolute_valid=np.zeros((n, 7), bool), conditional_sd_rad=conditional_sd,
        predicted_markers_m=pred, segment_R=rotations, segment_origins_m=origins, bones_m=bones,
        local_orientation_error_rad=local_error)
