"""Targeted repair with explicit candidate validation and numbered backups."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from datetime import datetime

import numpy as np

from ..paths import PROJECT, OUTPUT_ROOT, DATA_ROOT, resolve_historical_path
from .model import Model, SEGMENTS
from .data_io import load_motion, thorax_poses, MotionData
from .filtering import prepare_observations
from .kinematics import solve_trajectory
from .quality import diagnostics
from .pipeline import save_method, summary_metrics
from .state_filter import trajectory_state_ekf, FILTER_METHOD, FILTER_PARAMETERS, project_state
from .reference_geometry import compile_reference
from ..common.emg_sync import resample_estimates
from .pronation_audit import baseline, marker_twist, fingerprint, write_json

SUBJECT = 'subj_005'
TRIAL = 'trial_011'
TARGET = OUTPUT_ROOT / SUBJECT / 'trials' / TRIAL
WORK = OUTPUT_ROOT / '_checks/trial_repairs' / SUBJECT / TRIAL


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def self_check():
    cfg = compile_reference()
    cfg.update(status='synthetic_demo', lower_rad=[-3.14] * 12, upper_rad=[3.14] * 12,
        initial_q_rad=[0.] * 12, scapulothoracic=None,
        calibration={'prior_dominated_angles': [True] * 8 + [False] * 4})
    offsets = [[.04, 0, 0], [0, .05, 0], [0, 0, -.07], [-.03, .02, -.04]]
    cfg['markers'] = {f'{segment}_{i}': dict(segment=segment, local_m=point, sigma_m=.004)
        for segment in SEGMENTS for i, point in enumerate(offsets)}
    model = Model(cfg, allow_demo=True)
    t = np.arange(8) * .02
    q = .1 + .03 * np.sin(t[:, None] * 4 + np.arange(12)[None])
    poses = np.repeat(np.eye(4)[None], len(t), axis=0)
    from .kinematics import forward
    measured = forward(model, q, poses)[0]
    motion = MotionData(t, measured, list(model.bindings))
    recovered, _, _ = trajectory_state_ekf(motion, model, poses, q)
    corrupted = q.copy()
    corrupted[3:6, 9:] = [-2.5, 1.4, -1.2]
    corrupted[4] = np.nan
    robust, _, _ = trajectory_state_ekf(motion, model, poses, corrupted)
    np.testing.assert_allclose(recovered, robust, atol=1e-12)
    assert np.sqrt(np.mean((recovered - q) ** 2)) < .02
    state = np.r_[model.q0, np.zeros(12)]
    state[0], state[12] = model.upper[0] + .1, 5
    covariance = np.eye(24)
    assert project_state(state, covariance, model)
    assert state[0] == model.upper[0] and state[12] == 0
    assert np.linalg.eigvalsh(covariance).min() > 0
    return dict(synthetic_recovery='passed', corrupted_later_ik_invariance='passed', boundary_velocity_covariance='passed')


def metrics(q, diag, t, proxy):
    deg = np.rad2deg(q)
    rel = deg[::10, 9] - baseline(deg[:, 9], t)
    valid = np.isfinite(rel) & np.isfinite(proxy)
    return dict(q10_range_deg=[float(np.nanmin(deg[:, 9])), float(np.nanmax(deg[:, 9]))],
        q10_max_step_deg=float(np.nanmax(abs(np.diff(deg[:, 9])))),
        q10_lower_bound_samples=int(np.sum(deg[:, 9] <= -29.9)),
        q11_range_deg=[float(np.nanmin(deg[:, 10])), float(np.nanmax(deg[:, 10]))],
        q12_range_deg=[float(np.nanmin(deg[:, 11])), float(np.nanmax(deg[:, 11]))],
        marker_rms_p95_mm=float(np.nanquantile(diag['marker_rms_m'], .95) * 1000),
        proxy_relative_rmse_deg=float(np.sqrt(np.mean((rel[valid] - proxy[valid]) ** 2))))


def prepare():
    if WORK.exists():
        raise RuntimeError('Repair workspace already exists; inspect it before retrying')
    WORK.mkdir(parents=True)
    candidate = WORK / 'candidate'
    candidate.mkdir()
    model_path = OUTPUT_ROOT / SUBJECT / 'estimated_model.json'
    model = Model.load(model_path, allow_estimated=True)
    report = read(TARGET / 'report.json')
    source = DATA_ROOT / SUBJECT / 'MoCap' / f'{TRIAL}.mat'
    inventory = {str(path.relative_to(PROJECT)): fingerprint(path) for path in OUTPUT_ROOT.glob('subj_*/trials/trial_*/*.npz')}
    write_json(WORK / 'before_arrays.json', inventory)
    before_files = {str(path.relative_to(PROJECT)): fingerprint(path) for path in [*TARGET.iterdir(), model_path,
        OUTPUT_ROOT / SUBJECT / 'trial_manifest.json', source] if path.is_file()}
    write_json(WORK / 'before_files.json', before_files)
    checks = self_check()
    data = load_motion(source)
    with np.load(TARGET / 'ik_angles.npz', allow_pickle=False) as z:
        old_ik = z['q'].astype(float)
        assert np.array_equal(z['t'], data.t)
    with np.load(TARGET / 'ekf_angles.npz', allow_pickle=False) as z:
        old_ekf = z['q'].astype(float)
    poses, torso_rms = thorax_poses(data, model)
    print('Computing state-linearized candidate from raw markers', flush=True)
    seed, _, _ = trajectory_state_ekf(data, model, poses, old_ik)
    measured, valid = prepare_observations(data, model)
    print('Refining IK on the continuous seed branch', flush=True)
    ik, details = solve_trajectory(model, measured, valid, poses, seed, iterations=60, prior_sigma=2.)
    di = diagnostics(data, model, poses, ik, details['converged'])
    print('Filtering repaired IK with current-state marker updates', flush=True)
    ekf, sd, clipped = trajectory_state_ekf(data, model, poses, ik)
    de = diagnostics(data, model, poses, ekf, clipped=clipped)
    proxy = marker_twist(data.select(10), model.config['calibration']['selected_groups']['humerus']['markers'], True)[0]
    before_diag = diagnostics(data, model, poses, old_ekf)
    comparison = dict(before_ekf=metrics(old_ekf, before_diag, data.t, proxy),
        after_ik=metrics(ik, di, data.t, proxy), after_ekf=metrics(ekf, de, data.t, proxy),
        clipped_frames=int(clipped.sum()), ik_converged_fraction=float(details['converged'].mean()), self_checks=checks)
    write_json(WORK / 'comparison.json', comparison)
    assert np.isfinite(ik).all() and np.isfinite(ekf).all()
    assert comparison['after_ik']['q10_max_step_deg'] < 5
    assert comparison['after_ekf']['q10_max_step_deg'] < 5
    assert comparison['after_ekf']['q10_lower_bound_samples'] == 0
    assert comparison['after_ekf']['marker_rms_p95_mm'] < comparison['before_ekf']['marker_rms_p95_mm']
    assert comparison['after_ekf']['proxy_relative_rmse_deg'] < min(30., comparison['before_ekf']['proxy_relative_rmse_deg'] * .5)
    assert not clipped.any()
    for angle in [ik, ekf]:
        assert np.all(angle >= model.lower - 1e-8) and np.all(angle <= model.upper + 1e-8)
    assert np.all(di['angle_quality_flag'][:, 9] & 64) and np.all(de['angle_quality_flag'][:, 9] & 64)
    for method, q, diag, uncertainty in [('ik', ik, di, None), ('ekf', ekf, de, sd)]:
        save_method(candidate, method, data, poses, torso_rms, q, diag, uncertainty)
    source_hashes = {name: fingerprint(PROJECT / name) for name in ['src/motion_reconstruction/legacy_g99/state_filter.py',
        'src/motion_reconstruction/legacy_g99/trial_repair.py', 'src/motion_reconstruction/legacy_g99/kinematics.py',
        'src/motion_reconstruction/legacy_g99/quality.py', 'src/motion_reconstruction/common/emg_sync.py',
        'src/motion_reconstruction/legacy_g99/data_io.py', 'src/motion_reconstruction/legacy_g99/pipeline.py']}
    calculation_hash = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    for method in ['ik', 'ekf']:
        path = candidate / f'{method}_angles.npz'
        with np.load(path, allow_pickle=False) as z:
            payload = {key: z[key] for key in z.files}
        payload.update(repair_code_sha256=np.array(calculation_hash), reconstruction_method=np.array(
            'state_seeded_independent_ik' if method == 'ik' else FILTER_METHOD))
        np.savez_compressed(path, **payload)
    # Reuse the existing measured synchronization evidence and EMG clock exactly.
    with np.load(TARGET / 'emg_aligned_distal.npz', allow_pickle=False) as z:
        aligned = {key: z[key] for key in z.files}
    mapped = aligned['emg_time'] - report['sync']['offset_s']
    labels, flags = resample_estimates(data.t, ekf[:, 8:], de['angle_quality_flag'][:, 8:], mapped, max_gap_s=.01)
    aligned.update(q_distal_estimated=labels, angle_quality_flag=flags, strict_valid=flags == 0,
        filter_method=np.array(FILTER_METHOD), repair_code_sha256=np.array(calculation_hash))
    assert not aligned['strict_valid'][:, 1].any()
    np.savez_compressed(candidate / 'emg_aligned_distal.npz', **aligned)
    new_report = dict(report)
    new_report.pop('cache_code_sha256', None)
    new_report.update(ik=summary_metrics(ik, di), ekf=summary_metrics(ekf, de), solver_iterations=60,
        code_sha256=calculation_hash, filter_method=FILTER_METHOD, filter_parameters=FILTER_PARAMETERS,
        ik_method='state_seeded_independent_ik',
        repair=dict(created_at=datetime.now().astimezone().isoformat(),
            scope='subj_005/trial_011 only; model, raw data and synchronization evidence unchanged',
            previous_code_sha256=report['code_sha256'], previous_cache_code_sha256=report.get('cache_code_sha256'),
            source_hashes=source_hashes, ik_initialization='raw-marker state EKF seeded by first valid original IK; per-frame refinement with 60 iterations and 2 rad start prior',
            default_batch_algorithm_changed=False, validation=str((WORK / 'comparison.json').relative_to(PROJECT))))
    write_json(candidate / 'report.json', new_report)
    np.savetxt(WORK / 'comparison_curves.csv', np.column_stack([data.t[::10], np.rad2deg(old_ik[::10, 9]),
        np.rad2deg(old_ekf[::10, 9]), np.rad2deg(ik[::10, 9]), np.rad2deg(ekf[::10, 9]), proxy]),
        delimiter=',', header='time_s,old_ik_deg,old_ekf_deg,new_ik_deg,new_ekf_deg,marker_relative_deg', comments='')
    write_json(WORK / 'candidate_ready.json', dict(files={path.name: fingerprint(path) for path in candidate.iterdir()}, model_sha256=fingerprint(model_path)))
    print(json.dumps(comparison, indent=2), flush=True)


def publish():
    ready = read(WORK / 'candidate_ready.json')
    assert fingerprint(OUTPUT_ROOT / SUBJECT / 'estimated_model.json') == ready['model_sha256']
    for relative, expected in read(WORK / 'before_files.json').items():
        assert fingerprint(resolve_historical_path(relative)) == expected, relative
    for name, expected in ready['files'].items():
        assert fingerprint(WORK / 'candidate' / name) == expected, name
    base = WORK / '_backups'
    base.mkdir(exist_ok=True)
    number = 1
    while (base / f'backup_{number:03d}').exists():
        number += 1
    backup = base / f'backup_{number:03d}'
    backup.mkdir()
    # Preserve exact pre-repair bytes for every file this publish step changes.
    paths = [*TARGET.iterdir(), OUTPUT_ROOT / SUBJECT / 'trial_manifest.json']
    for path in paths:
        if path.is_file():
            destination = backup / path.relative_to(OUTPUT_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    write_json(backup / 'manifest.json', {str(path.relative_to(OUTPUT_ROOT)): fingerprint(path) for path in paths if path.is_file()})
    # Atomic per-file replacement, with recoverable originals and a completion marker.
    for name in ready['files']:
        temporary = TARGET / (name + '.repair_tmp')
        shutil.copy2(WORK / 'candidate' / name, temporary)
        temporary.replace(TARGET / name)
    manifest_path = OUTPUT_ROOT / SUBJECT / 'trial_manifest.json'
    rows = read(manifest_path)
    matches = [i for i, row in enumerate(rows) if row['trial'] == 11]
    assert len(matches) == 1
    rows[matches[0]] = read(TARGET / 'report.json')
    temporary = manifest_path.with_suffix('.repair_tmp')
    write_json(temporary, rows)
    temporary.replace(manifest_path)
    changed = {str((TARGET / name).relative_to(PROJECT)) for name in ready['files'] if name.endswith('.npz')}
    verified = 0
    for relative, expected in read(WORK / 'before_arrays.json').items():
        if relative not in changed:
            assert fingerprint(resolve_historical_path(relative)) == expected, relative
            verified += 1
    write_json(WORK / 'published.json', dict(published_at=datetime.now().astimezone().isoformat(),
        backup=str(backup.relative_to(PROJECT)), changed_arrays=sorted(changed),
        unchanged_array_files_verified=verified, default_batch_algorithm_changed=False))
    print(f'Published trial_011; {verified} other array files unchanged; backup: {backup}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'publish'], default='prepare')
    args = parser.parse_args(argv)
    {'prepare': prepare, 'publish': publish}[args.stage]()
