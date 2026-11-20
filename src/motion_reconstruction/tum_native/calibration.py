"""Task-0 reference and training-only functional template/pivot calibration."""
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from .schema import SEGMENTS, GROUPS, SCHEMA, sha256, canonical_hash
from .observations import load_motion, prepare, rigid_fit, track, tasks
from .model import unit, frame, decompose


def choose_reference(subject_dir, jobs, config):
    required = ['m_jn', 'm_c7', 'm_xp', 'm_t8', 'm_acr', 'm_elb', 'm_rp', 'm_up', 'm_mf']
    candidates = []
    for job in [r for r in jobs if r['task'] == 0]:
        path = subject_dir / 'MoCap' / f'trial_{job["trial"]:03d}.mat'
        data = load_motion(path)
        ids = [data.names.index(name) for name in required]
        width = max(5, int(round(config['reference_window_s'] / np.median(np.diff(data.t)))))
        step = max(1, width // 3)
        for begin in range(0, len(data.t) - width + 1, step):
            end = begin + width
            points = data.p[begin:end, ids]
            valid = np.isfinite(points).all(axis=(1, 2))
            if valid.mean() < .9:
                continue
            center = np.nanmedian(points, axis=0)
            deviation = np.linalg.norm(points - center, axis=2)
            score = float(np.nanmedian(deviation))
            anchor = begin + int(np.nanargmin(np.nanmean(deviation, axis=1)))
            if not np.isfinite(data.p[anchor, ids]).all():
                continue
            if any(np.isfinite(data.p[anchor, [data.names.index(x) for x in group]]).all(1).sum() < 4 for group in GROUPS.values()):
                continue
            candidates.append((score, job['trial'], begin, end, anchor, data, path))
    if not candidates:
        raise ValueError('No common stable Task-0 reference; no evaluation-trial fallback is allowed')
    score, trial, begin, end, anchor, data, path = min(candidates, key=lambda item: item[0])
    return data, dict(trial=trial, task=0, start_index=begin, end_index_exclusive=end, anchor_index=anchor,
        start_s=float(data.t[begin]), end_s=float(data.t[end-1]), score_m=score,
        source=str(path.resolve()), source_sha256=sha256(path), anatomical_zero=False)


def initial_model(data, reference, config):
    points = data.p[reference['anchor_index']]
    get = lambda name: points[data.names.index(name)]
    jn, c7, xp, t8 = [get(name) for name in ['m_jn', 'm_c7', 'm_xp', 'm_t8']]
    z = unit((jn + c7 - xp - t8) / 2)
    y = unit(jn - c7 - z * np.dot(jn - c7, z))
    rt = np.column_stack([unit(np.cross(y, z)), y, z])
    shoulder, elbow, wrist, tip = get('m_acr'), get('m_elb'), (get('m_rp') + get('m_up')) / 2, get('m_mf')
    humerus, forearm = unit(elbow - shoulder), unit(wrist - elbow)
    hinge = unit(np.cross(humerus, forearm))
    rh, rf = frame(humerus, hinge), frame(forearm, hinge)
    transverse = unit(get('m_rp') - get('m_up') - forearm * np.dot(get('m_rp') - get('m_up'), forearm))
    rm = frame(tip - wrist, transverse)
    origins, rotations = [jn, shoulder, elbow, wrist], [rt, rh, rf, rm]
    segments = {}
    for k, segment in enumerate(SEGMENTS):
        group = GROUPS[segment]
        local = {name: (rotations[k].T @ (get(name) - origins[k])).tolist() for name in group if np.isfinite(get(name)).all()}
        # Align each static frame into the same anchored technical frame.
        values = {name: [] for name in local}
        ids = [data.names.index(name) for name in local]
        template = np.array(list(local.values()))
        for i in range(reference['start_index'], reference['end_index_exclusive'], 5):
            r, p, _, _, _ = rigid_fit(template, data.p[i, ids])
            if r is None:
                continue
            aligned = (data.p[i, ids] - p) @ r
            for name, value in zip(local, aligned):
                if np.isfinite(value).all():
                    values[name].append(value)
        local = {name: np.median(values[name], axis=0).tolist() if values[name] else local[name] for name in local}
        segments[segment] = dict(local_m=local, fit_names=list(local), sigma_m={name: config['marker_sigma_floor_m'] for name in local})
    return dict(schema=SCHEMA, units='m,rad,s', reference=reference, segments=segments,
        axes=dict(elbow_H=(rh.T @ hinge).tolist(), twist_F=(rf.T @ forearm).tolist(),
            wrist_FE_F=(rf.T @ transverse).tolist(), wrist_RD_M=(rm.T @ unit(np.cross(forearm, transverse))).tolist()),
        reference_rotations=dict(TH=(rt.T @ rh).tolist(), HF=(rh.T @ rf).tolist(), FM=(rf.T @ rm).tolist()),
        reference_frames=dict(R=np.array(rotations).tolist(), p=np.array(origins).tolist()),
        shoulder_reference_T_m=(rt.T @ (shoulder - jn)).tolist(), elbow_in_H_m=(rh.T @ (elbow-shoulder)).tolist(),
        wrist_in_F_m=(rf.T @ (wrist-elbow)).tolist(), hand_tip_M_m=(rm.T @ (tip-wrist)).tolist(),
        axis_calibration=dict(status='geometric_functional_reference', anatomical_validation=False,
            reason='No separately verified radius/ulna frames or known neutral angles; no forced personalized-axis fit'))


def fit_pivot(a, b, prior_a, prior_b, config):
    valid = np.isfinite(a['R']).all(axis=(1, 2)) & np.isfinite(b['R']).all(axis=(1, 2)) & np.isfinite(a['p']).all(1) & np.isfinite(b['p']).all(1)
    ra, rb = a['R'][valid], b['R'][valid]
    prior = np.r_[prior_a, prior_b]
    if len(ra) < 10:
        return prior, dict(status='prior_only', frames=len(ra), measurement_rank=0, prior_dependent=True)
    matrix = np.concatenate([ra, -rb], axis=2).reshape(-1, 6)
    rhs = (b['p'][valid] - a['p'][valid]).ravel()
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-3))
    fit = least_squares(lambda x: np.r_[(matrix @ x - rhs) / .006, (x - prior) / config['center_prior_sigma_m']],
        prior, bounds=(prior-config['center_max_shift_m'], prior+config['center_max_shift_m']), loss='soft_l1', max_nfev=70)
    residual = (matrix @ fit.x - rhs).reshape(-1, 3)
    return fit.x, dict(status='regularized_functional_pivot', frames=len(ra), measurement_rank=rank,
        singular_values=singular.tolist(), rms_m=float(np.sqrt(np.mean(residual**2))),
        prior_dependent=True, converged=bool(fit.success), anatomical_validation=False)


def calibrate(subject_dir, config):
    subject_dir = Path(subject_dir)
    jobs = tasks(subject_dir)
    data, reference = choose_reference(subject_dir, jobs, config)
    model = initial_model(data, reference, config)
    chosen = []
    for task in config['calibration_tasks']:
        options = [r for r in jobs if r['task'] == task]
        if options:
            chosen.append(min(options, key=lambda r: r['repeat']))
    calibration_observations = []
    provenance = []
    for job in chosen:
        path = subject_dir / 'MoCap' / f'trial_{job["trial"]:03d}.mat'
        original = load_motion(path)
        sample = original.select(max(1, len(original.t) // config['calibration_frames_per_trial']))
        obs = dict(t=sample.t, p=sample.p, raw=sample.p, names=sample.names, observed=sample.valid,
            interpolated=np.zeros(sample.valid.shape, bool), edge=np.zeros(sample.valid.shape, bool))
        calibration_observations.append(obs)
        provenance.append(dict(job, source=str(path.resolve()), source_sha256=sha256(path)))
    # Leave-one-marker-out errors on calibration data, never on evaluation trials.
    for segment in SEGMENTS:
        spec = model['segments'][segment]
        names = list(spec['fit_names'])
        errors = {name: [] for name in names}
        for obs in calibration_observations:
            ids = [obs['names'].index(name) for name in names]
            local = np.array([spec['local_m'][name] for name in names])
            for measured in obs['p'][:, ids]:
                for k, name in enumerate(names):
                    if not np.isfinite(measured[k]).all():
                        continue
                    keep = np.arange(len(names)) != k
                    r, p, _, _, _ = rigid_fit(local[keep], measured[keep])
                    if r is not None:
                        errors[name].append(float(np.linalg.norm(r @ local[k] + p - measured[k])))
        scores = {name: float(np.quantile(values, .95)) if values else config['marker_sigma_ceiling_m'] for name, values in errors.items()}
        droppable = sorted([name for name in names if '_h' in name and scores[name] > config['drop_marker_p95_m']], key=lambda name: scores[name], reverse=True)
        for name in droppable:
            if len(spec['fit_names']) > config['minimum_cluster_markers']:
                spec['fit_names'].remove(name)
        spec['sigma_m'] = {name: float(np.clip(scores[name] / 2, config['marker_sigma_floor_m'], config['marker_sigma_ceiling_m'])) for name in names}
        spec['calibration_leave_one_out_p95_m'] = scores
    tracks = [track(obs, model) for obs in calibration_observations]
    if tracks:
        combined = dict(R=np.concatenate([tr['R'] for tr in tracks]), p=np.concatenate([tr['p'] for tr in tracks]))
        segment = lambda k: dict(R=combined['R'][:, k], p=combined['p'][:, k])
        elbow, elbow_qc = fit_pivot(segment(1), segment(2), model['elbow_in_H_m'], np.zeros(3), config)
        wrist, wrist_qc = fit_pivot(segment(2), segment(3), model['wrist_in_F_m'], np.zeros(3), config)
    else:
        elbow = np.r_[model['elbow_in_H_m'], np.zeros(3)]
        wrist = np.r_[model['wrist_in_F_m'], np.zeros(3)]
        elbow_qc = wrist_qc = dict(status='prior_only', prior_dependent=True, measurement_rank=0)
    r0, p0 = np.array(model['reference_frames']['R']), np.array(model['reference_frames']['p'])
    effective_elbow = (r0[1] @ elbow[:3] + p0[1] + r0[2] @ elbow[3:] + p0[2]) / 2
    effective_wrist = (r0[2] @ wrist[:3] + p0[2] + r0[3] @ wrist[3:] + p0[3]) / 2
    for k, origin in [(2, effective_elbow), (3, effective_wrist)]:
        spec = model['segments'][SEGMENTS[k]]
        shift = r0[k].T @ (origin - p0[k])
        spec['local_m'] = {name: (np.array(value) - shift).tolist() for name, value in spec['local_m'].items()}
    model['elbow_in_H_m'] = (r0[1].T @ (effective_elbow - p0[1])).tolist()
    model['wrist_in_F_m'] = (r0[2].T @ (effective_wrist - effective_elbow)).tolist()
    tip = data.p[reference['anchor_index'], data.names.index('m_mf')]
    model['hand_tip_M_m'] = (r0[3].T @ (tip - effective_wrist)).tolist()
    model['reference_frames']['p'][2:4] = [effective_elbow.tolist(), effective_wrist.tolist()]
    model.update(subject=subject_dir.name, calibration_sources=provenance,
        calibration_trial_ids=[r['trial'] for r in provenance], pivot_quality=dict(elbow=elbow_qc, wrist=wrist_qc),
        reference_policy='One common Task-0 reference; no per-evaluation-trial zeroing', anatomy_reviewed=False)
    reference_obs = prepare(data, config)
    reference_track = track(reference_obs, model)
    reference_q, _, _ = decompose(reference_track, model)
    window = reference_q[reference['start_index']:reference['end_index_exclusive']]
    medians = []
    for k in range(7):
        finite = window[:, k][np.isfinite(window[:, k])]
        medians.append(float(np.median(finite)) if len(finite) else None)
    model['reference_q_median_rad'] = medians
    model['reference_id'] = canonical_hash(reference)
    return model
