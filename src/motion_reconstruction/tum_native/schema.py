"""Versioned functional coordinates and quality semantics (not legacy bits)."""
import json
from pathlib import Path
import hashlib
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT / 'datasets/TUM/TUM'
SCHEMA = 'tum-native-functional7-v2'
Q_NAMES = ['TH_RV_X', 'TH_RV_Y', 'TH_RV_Z', 'EL_FE_FUNC', 'FA_TWIST_PROXY', 'WR_FE_FUNC', 'WR_RD_FUNC']
SEGMENTS = ['thorax', 'humerus', 'forearm', 'hand']
GROUPS = {
    'thorax': ['m_xp', 'm_jn_h1', 'm_jn', 'm_c7', 'm_c7_h1', 'm_t8'] + [f'm_torso_h{i}' for i in range(1, 9)],
    'humerus': ['m_elb'] + [f'm_elb_h{i}' for i in range(1, 6)],
    'forearm': ['m_rp', 'm_up'] + [f'm_rp_h{i}' for i in range(1, 5)],
    'hand': ['m_mf'] + [f'm_mf_h{i}' for i in range(1, 6)],
}
MISSING, DEGENERATE, AXIS_PRIOR, MODEL_MISMATCH = 1, 2, 4, 8
REFERENCE_DEPENDENT, NOT_CONVERGED, FILTER_EDGE, INTERPOLATED = 16, 32, 64, 128
LOW_INFORMATION, ROTATION_CHART, CALIBRATION_WEAK, PREDICTED = 256, 512, 1024, 2048
QUALITY = {str(k): v for k, v in [(MISSING, 'missing observation'), (DEGENERATE, 'degenerate cluster'),
    (AXIS_PRIOR, 'functional geometric axes, not anatomical calibration'), (MODEL_MISMATCH, 'two-axis orientation mismatch'),
    (REFERENCE_DEPENDENT, 'subject reference, not anatomical zero'), (NOT_CONVERGED, 'numerical solver not converged'),
    (FILTER_EDGE, 'filter boundary sensitivity'), (INTERPOLATED, 'short-gap interpolated marker used'),
    (LOW_INFORMATION, 'weak measurement-only conditional information'), (ROTATION_CHART, 'rotation-vector chart near pi'),
    (CALIBRATION_WEAK, 'functional pivot calibration needs prior'), (PREDICTED, 'prediction without full observation')]}
RELATIVE_REJECT = MISSING | DEGENERATE | MODEL_MISMATCH | NOT_CONVERGED | FILTER_EDGE | LOW_INFORMATION | ROTATION_CHART | PREDICTED


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def finite_stat(values, fraction):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, fraction)) if len(a) else None


def runs(mask):
    indices = np.flatnonzero(mask)
    return [r for r in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1) if len(r)]
