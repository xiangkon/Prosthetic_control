"""Calibrate small functional axis corrections on reference-independent trials."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .model import two_axis


def calibrate_axes(model, tracks, config):
    """Accept only corrections supported by separate calibration records."""
    axis_limit = np.deg2rad(config['axis_max_adjustment_deg'])
    diagnostics = {}
    for label, pair, first, second in [
        ('HF', (1, 2), 'elbow_H', 'twist_F'),
        ('FM', (2, 3), 'wrist_FE_F', 'wrist_RD_M'),
    ]:
        relative = []
        for track in tracks:
            a, b = (track['R'][:, k] for k in pair)
            rotations = a.swapaxes(1, 2) @ b
            relative.append(rotations[np.isfinite(rotations).all((1, 2))])
        if len(relative) < 4:
            diagnostics[label] = dict(accepted=False, reason='insufficient calibration trials')
            continue
        train, validation = np.concatenate(relative[::2]), np.concatenate(relative[1::2])
        if min(len(train), len(validation)) < 30:
            diagnostics[label] = dict(accepted=False, reason='insufficient finite calibration frames')
            continue
        a, b = np.asarray(model['axes'][first]), np.asarray(model['axes'][second])
        reference = np.asarray(model['reference_rotations'][label])
        coordinates, baseline_train = two_axis(train, a, reference, b)
        span = np.quantile(coordinates, .9, axis=0) - np.quantile(coordinates, .1, axis=0)
        if np.min(span) < np.deg2rad(10):
            diagnostics[label] = dict(accepted=False, reason='insufficient excitation in both coordinates')
            continue

        def tangent(axis):
            u = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
            u /= np.linalg.norm(u)
            return np.stack([u, np.cross(axis, u)], axis=1)

        tangent_a, tangent_b = tangent(a), tangent(b)
        def axes(parameters):
            return (Rotation.from_rotvec(tangent_a @ parameters[:2]).apply(a),
                    Rotation.from_rotvec(tangent_b @ parameters[2:]).apply(b))

        # Alternating projection avoids differentiating an inner iterative solver.
        parameters = np.zeros(4)
        for _ in range(8):
            aa, bb = axes(parameters)
            coordinates, _ = two_axis(train, aa, reference, bb)
            def residual(candidate):
                aa, bb = axes(candidate)
                left = Rotation.from_rotvec(coordinates[:, :1] * aa).as_matrix()
                right = Rotation.from_rotvec(coordinates[:, 1:] * bb).as_matrix()
                error = Rotation.from_matrix((left @ reference @ right).swapaxes(1, 2) @ train).as_rotvec()
                return np.r_[error.ravel(), candidate * np.sqrt(len(train)) * .15]
            fit = least_squares(residual, parameters, bounds=(-axis_limit/np.sqrt(2), axis_limit/np.sqrt(2)),
                                loss='soft_l1', f_scale=.12, max_nfev=16)
            parameters = fit.x
        aa, bb = axes(parameters)
        _, old_validation = two_axis(validation, a, reference, b)
        _, new_validation = two_axis(validation, aa, reference, bb)
        _, new_train = two_axis(train, aa, reference, bb)
        p95 = lambda x: float(np.rad2deg(np.quantile(x, .95)))
        cutoff = np.deg2rad(config['orientation_mismatch_deg'])
        old_reject, new_reject = (float(np.mean(error > cutoff)) for error in (old_validation, new_validation))
        accepted = (p95(new_validation) < .90 * p95(old_validation) and
                    np.median(new_validation) < np.median(old_validation) and
                    new_reject <= old_reject and
                    p95(new_train) < p95(baseline_train) and abs(aa @ reference @ bb) < .94)
        if accepted:
            model['axes'][first], model['axes'][second] = aa.tolist(), bb.tolist()
        diagnostics[label] = dict(accepted=bool(accepted), training_frames=len(train),
            validation_frames=len(validation), validation_p95_before_deg=p95(old_validation),
            validation_p95_after_deg=p95(new_validation), validation_median_before_deg=float(np.rad2deg(np.median(old_validation))),
            validation_median_after_deg=float(np.rad2deg(np.median(new_validation))),
            training_p95_before_deg=p95(baseline_train), training_p95_after_deg=p95(new_train),
            validation_rejection_before=old_reject, validation_rejection_after=new_reject,
            attempted_adjustment_deg=np.rad2deg(np.linalg.norm(parameters.reshape(2, 2), axis=1)).tolist())
    return diagnostics
