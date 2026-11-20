"""Additional frame-local checks for pose-anchored reconstruction."""
import numpy as np
from scipy.spatial.transform import Rotation
from .quality import evaluate
from .schema import MODEL_MISMATCH, QUALITY, RELATIVE_REJECT, SEGMENTS

MARKER_DETACHMENT = 4096
UNSUPPORTED_STEP = 8192
REJECT = RELATIVE_REJECT | MARKER_DETACHMENT | UNSUPPORTED_STEP
QUALITY_V2 = {**QUALITY, str(MARKER_DETACHMENT): 'segment marker fit exceeds limit',
              str(UNSUPPORTED_STEP): 'model rotation step unsupported by measured segment motion'}
DEPENDENCIES = [(0, 1)] * 3 + [(1, 2)] * 2 + [(2, 3)] * 2


def evaluate_anchored(obs, poses, model, solution, config):
    samples = len(obs['t'])
    qc = evaluate(obs, poses, model, {**solution, 'attempted': np.zeros(samples, bool),
                                     'converged': np.ones(samples, bool)}, config)
    # Use a separate wrist cutoff for output quality. The shared cutoff
    # remains in effect for elbow/forearm quality and axis calibration.
    wrist_limit = config.get('wrist_orientation_mismatch_deg', config['orientation_mismatch_deg'])
    wrist_flags = qc['flags'][:, 5:7]
    wrist_flags &= ~np.uint16(MODEL_MISMATCH)
    wrist_flags[qc['local_orientation_error_rad'][:, 1] > np.deg2rad(wrist_limit)] |= MODEL_MISMATCH
    error = np.linalg.norm(qc['predicted_markers_m'] - obs['p'], axis=2)
    segment_error = np.full((samples, 4), np.nan)
    spikes = np.zeros((samples, 4), bool)
    for index, segment in enumerate(SEGMENTS):
        columns = [obs['names'].index(n) for n in model['segments'][segment]['fit_names']]
        distances = error[:, columns]
        has_fit = np.isfinite(distances).any(1)
        segment_error[has_fit, index] = np.nanquantile(distances[has_fit], .95, axis=1)
        fitted, measured = qc['segment_R'][:, index], poses['R'][:, index]
        pair = np.isfinite(fitted).all((1, 2)) & np.isfinite(measured).all((1, 2))
        right = np.flatnonzero(pair[1:] & pair[:-1]) + 1
        if len(right):
            step = lambda r: Rotation.from_matrix(r[right-1].swapaxes(1, 2) @ r[right]).magnitude()
            tolerance = np.deg2rad(config['unsupported_rotation_excess_deg_s']) * np.diff(obs['t'])[right-1]
            bad = right[step(fitted) > step(measured) + tolerance]
            spikes[bad, index] = True
            spikes[bad-1, index] = True
    for column, segments in enumerate(DEPENDENCIES):
        detached = (segment_error[:, segments] > config['marker_fit_p95_limit_m']).any(1)
        qc['flags'][detached, column] |= MARKER_DETACHMENT
        qc['flags'][spikes[:, segments].any(1), column] |= UNSUPPORTED_STEP
    qc['relative_valid'] = np.isfinite(solution['state'][:, :7]) & ((qc['flags'] & REJECT) == 0)
    qc['segment_marker_error_p95_m'] = segment_error
    qc['unsupported_segment_step'] = spikes
    return qc
