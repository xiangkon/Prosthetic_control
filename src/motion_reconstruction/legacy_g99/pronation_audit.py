"""Task-informed plausibility audit, without treating marker agreement as truth."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
from datetime import datetime

import numpy as np

from .data_io import load_motion
from .marker_analysis import fit_template, fit_frame
from ..paths import PROJECT, OUTPUT_ROOT, DATA_ROOT
from src.task_metadata import tum_task_description_zh


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quantile(value, fraction):
    finite = np.asarray(value)[np.isfinite(value)]
    return float(np.quantile(finite, fraction)) if len(finite) else None


def baseline(value, t):
    valid = np.isfinite(value) & (t <= t[0] + .5)
    return float(np.median(value[valid])) if valid.any() else np.nan


def unwrap_segments(values):
    result = np.full_like(values, np.nan)
    indices = np.flatnonzero(np.isfinite(values))
    for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
        if len(run):
            result[run] = np.unwrap(values[run])
    return result


def marker_twist(data, cluster_names=None, transport_axis=False):
    """Track wrist transverse direction in a raw skin-cluster frame.

    The upper-arm variant removes minimal forearm-axis swing; the ulna variant
    tests the disputed rigid forearm binding. No fitted subject geometry, IK
    angles, calibrated offsets or temporal filter enters either calculation.
    Sign follows the approximate distal-to-proximal forearm axis.
    """
    index = lambda name: data.names.index(name)
    cluster_names = cluster_names or ['m_rp_h2', 'm_rp_h3', 'm_rp_h4', 'm_up']
    cluster = data.p[:, [index(name) for name in cluster_names]]
    template = fit_template(cluster)
    if template is None:
        raise ValueError('No usable forearm cluster template')
    rotation = np.full((len(data.t), 3, 3), np.nan)
    rms = np.full(len(data.t), np.nan)
    for i, frame in enumerate(cluster):
        pose, errors = fit_frame(template, frame)
        if pose is not None:
            rotation[i] = pose[:3, :3]
            rms[i] = np.sqrt(np.nanmean(errors ** 2))
    longitudinal = data.p[:, index('m_elb')] - data.p[:, index('m_up')]
    local_longitudinal = np.einsum('nji,nj->ni', rotation, longitudinal)
    if transport_axis:
        axis = local_longitudinal / np.linalg.norm(local_longitudinal, axis=1)[:, None]
    else:
        constant_axis = np.nanmedian(local_longitudinal, axis=0)
        constant_axis /= np.linalg.norm(constant_axis)
        axis = np.repeat(constant_axis[None], len(data.t), axis=0)
    transverse = data.p[:, index('m_rp')] - data.p[:, index('m_up')]
    local = np.einsum('nji,nj->ni', rotation, transverse)
    local -= np.sum(local * axis, axis=1)[:, None] * axis
    lever = np.linalg.norm(local, axis=1)
    good = np.isfinite(local).all(1) & (lever > .005)
    first = np.flatnonzero(good)
    if not len(first):
        raise ValueError('No usable off-axis wrist direction')
    anchor = first[0]
    x0 = local[anchor] / lever[anchor]
    if transport_axis:
        # Shortest rotation from initial forearm axis to each current axis,
        # in the tracked upper-arm frame. Remove bending without imposing twist.
        a0 = axis[anchor]
        cross = np.cross(a0, axis)
        cosine = np.clip(axis @ a0, -1, 1)
        x = x0 + np.cross(cross, x0) + np.cross(cross, np.cross(cross, x0)) / np.maximum(1 + cosine[:, None], 1e-8)
        x[cosine < -.99] = np.nan
    else:
        x = np.repeat(x0[None], len(data.t), axis=0)
    y = np.cross(axis, x)
    angle = np.arctan2(np.sum(local * y, axis=1), np.sum(local * x, axis=1))
    angle[~good] = np.nan
    angle = np.rad2deg(unwrap_segments(angle))
    angle -= baseline(angle, data.t)
    return angle, rms, lever * 1000


def select_trials(results, seed):
    rng = random.Random(seed)
    candidates = []
    for subject in sorted(results.glob('subj_*')):
        for row in json.loads((subject / 'trial_manifest.json').read_text(encoding='utf-8')):
            if row['task'] in range(1, 31) and row['status'] != 'failed':
                candidates.append(dict(subject=subject.name, trial=row['trial'], task=row['task'], description=row['description']))
    selected = []
    for subject in sorted({row['subject'] for row in candidates}):
        selected.append(dict(rng.choice([row for row in candidates if row['subject'] == subject]), selection='random_subject_balanced'))
    key = lambda row: (row['subject'], row['trial'])
    extra = [row for row in candidates if key(row) not in {key(r) for r in selected}]
    selected.extend(dict(row, selection='random_extra') for row in rng.sample(extra, 2))
    for task in [2, 21, 30]:
        if not any(row['task'] == task for row in selected):
            selected.append(dict(rng.choice([row for row in candidates if row['task'] == task]), selection='targeted_task_control'))
    return selected


def analyze(row, results, data_root, output):
    trial_name = f'trial_{row["trial"]:03d}'
    folder = results / row['subject'] / 'trials' / trial_name
    with np.load(folder / 'ik_angles.npz', allow_pickle=False) as z:
        t = z['t'].copy()
        ik = np.rad2deg(z['q'][:, 9].astype(float))
    with np.load(folder / 'ekf_angles.npz', allow_pickle=False) as z:
        assert np.array_equal(t, z['t'])
        ekf = np.rad2deg(z['q'][:, 9].astype(float))
        flags = z['angle_quality_flag'][:, 9].copy()
        marker_error = z['marker_rms_m'].copy() * 1000
    raw_path = data_root / row['subject'] / 'MoCap' / f'{trial_name}.mat'
    motion = load_motion(raw_path)
    assert len(t) == len(motion.t) and np.allclose(t, motion.t, atol=1e-8)
    # Evaluate the direct marker proxy at 50 Hz; jumps/speeds use full 500 Hz arrays.
    stride = 10
    proxy_data = motion.select(stride)
    model = json.loads((results / row['subject'] / 'estimated_model.json').read_text(encoding='utf-8'))
    humerus_names = model['calibration']['selected_groups']['humerus']['markers']
    ulna_proxy, ulna_rms, _ = marker_twist(proxy_data)
    proxy, cluster_rms, lever = marker_twist(proxy_data, humerus_names, transport_axis=True)
    tt, qi, qe = t[::stride], ik[::stride], ekf[::stride]
    qi_rel, qe_rel = qi - baseline(ik, t), qe - baseline(ekf, t)
    pair = np.isfinite(proxy) & np.isfinite(qe_rel)
    correlation = None
    slope = None
    if pair.sum() >= 20 and np.std(proxy[pair]) > 2 and np.std(qe_rel[pair]) > 2:
        correlation = float(np.corrcoef(proxy[pair], qe_rel[pair])[0, 1])
        slope = float(np.polyfit(proxy[pair], qe_rel[pair], 1)[0])
    proxy_error = qe_rel[pair] - proxy[pair]
    ik_pair = np.isfinite(proxy) & np.isfinite(qi_rel)
    ik_error = qi_rel[ik_pair] - proxy[ik_pair]
    end = np.isfinite(ekf) & (t >= t[-1] - .5)
    result = dict(row, task_zh=tum_task_description_zh(row['task']), duration_s=float(t[-1] - t[0]),
        ik_min_deg=quantile(ik, 0), ik_max_deg=quantile(ik, 1),
        ekf_min_deg=quantile(ekf, 0), ekf_max_deg=quantile(ekf, 1),
        ekf_span90_deg=quantile(ekf, .95) - quantile(ekf, .05),
        start_deg=baseline(ekf, t), end_deg=float(np.median(ekf[end])) if end.any() else None,
        start_end_difference_deg=float(np.median(ekf[end]) - baseline(ekf, t)) if end.any() else None,
        ik_max_step_deg=quantile(abs(np.diff(ik)), 1), ekf_max_step_deg=quantile(abs(np.diff(ekf)), 1),
        ekf_speed_p99_deg_s=quantile(abs(np.diff(ekf) / np.diff(t)), .99),
        ik_ekf_rmse_deg=float(np.sqrt(np.nanmean((ik - ekf) ** 2))),
        proxy_span90_deg=quantile(proxy, .95) - quantile(proxy, .05),
        proxy_coverage=float(pair.mean()), proxy_ekf_corr=correlation, proxy_ekf_slope=slope,
        proxy_ekf_relative_rmse_deg=float(np.sqrt(np.mean(proxy_error ** 2))),
        proxy_ik_relative_rmse_deg=float(np.sqrt(np.mean(ik_error ** 2))),
        upper_arm_cluster_rms_p95_mm=quantile(cluster_rms, .95), off_axis_lever_p05_mm=quantile(lever, .05),
        ulna_proxy_span90_deg=quantile(ulna_proxy, .95) - quantile(ulna_proxy, .05),
        ulna_cluster_rms_p95_mm=quantile(ulna_rms, .95),
        marker_rms_p95_mm=quantile(marker_error, .95),
        nonprior_quality_clean_fraction=float(np.mean((flags & np.uint16(0xFFFF ^ 64)) == 0)),
        prior_fraction=float(np.mean((flags & 64) != 0)),
        sources={str(path.relative_to(PROJECT)): fingerprint(path) for path in [raw_path, folder / 'ik_angles.npz', folder / 'ekf_angles.npz']})
    identifier = f'{row["subject"]}_{trial_name}'
    np.savetxt(output / f'{identifier}.csv', np.column_stack([tt, qi, qe, qi_rel, qe_rel, proxy, cluster_rms, lever, ulna_proxy]),
        delimiter=',', header='time_s,ik_deg,ekf_deg,ik_relative_deg,ekf_relative_deg,upperarm_referenced_twist_deg,upperarm_cluster_rms_mm,wrist_lever_mm,ulna_referenced_twist_deg', comments='')
    return result, (tt, qi, qe, qi_rel, qe_rel, proxy)


def draw(records, curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    for group_name, indices in [('random', [i for i, r in enumerate(records) if r['selection'].startswith('random')]),
                                 ('task_controls', [i for i, r in enumerate(records) if r['selection'] == 'targeted_task_control'])]:
        if not indices:
            continue
        fig, axes = plt.subplots(len(indices), 2, figsize=(11, 2.05 * len(indices)), squeeze=False, constrained_layout=True)
        for j, i in enumerate(indices):
            row = records[i]
            t, ik, ekf, ir, er, proxy = curves[i]
            a, b = axes[j]
            a.plot(t, ik, color='#9B9B9B', lw=.8, label='IK')
            a.plot(t, ekf, color='#386A9A', lw=1, label='EKF')
            a.set(title=f'{row["subject"]} / trial_{row["trial"]:03d} / Task {row["task"]}: {row["description"]}', ylabel='q10 (deg)', xlabel='Time (s)')
            b.plot(t, ir, color='#9B9B9B', lw=.7, label='IK - initial')
            b.plot(t, er, color='#386A9A', lw=1, label='EKF - initial')
            b.plot(t, proxy, color='#B77942', lw=1, label='Raw marker twist - initial')
            b.set(xlabel='Time (s)', ylabel='Relative rotation (deg)', title=f'Marker trend check; r={row["proxy_ekf_corr"]:.2f}' if row['proxy_ekf_corr'] is not None else 'Marker trend check; r unavailable')
            for ax in (a, b):
                ax.grid(alpha=.15)
                ax.margins(x=.01)
            if j == 0:
                a.legend(loc='upper center', bbox_to_anchor=(.5, 1.35), ncol=2, frameon=False)
                b.legend(loc='upper center', bbox_to_anchor=(.5, 1.35), ncol=3, frameon=False, fontsize=7)
        fig.savefig(output / f'{group_name}.png', dpi=150)
        fig.savefig(output / f'{group_name}.svg')
        plt.close(fig)


def draw_failure_detail(records, results, output):
    """Show the worst sampled full-rate branch jump and its neighboring joints."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    row = max(records, key=lambda r: r['ik_max_step_deg'])
    trial = f'trial_{row["trial"]:03d}'
    folder = results / row['subject'] / 'trials' / trial
    with np.load(folder / 'ik_angles.npz', allow_pickle=False) as z:
        t, ik = z['t'], np.rad2deg(z['q'].astype(float))
    with np.load(folder / 'ekf_angles.npz', allow_pickle=False) as z:
        ekf = np.rad2deg(z['q'].astype(float))
    jump = int(np.argmax(abs(np.diff(ik[:, 9]))))
    lower = ekf[:, 9] < -29.99
    detail = dict(subject=row['subject'], trial=row['trial'], task=row['task'],
        max_ik_jump_time_before_s=float(t[jump]), max_ik_jump_time_after_s=float(t[jump + 1]),
        max_ik_jump_before_deg=float(ik[jump, 9]), max_ik_jump_after_deg=float(ik[jump + 1, 9]),
        max_ik_jump_speed_deg_s=float(abs(ik[jump + 1, 9] - ik[jump, 9]) / (t[jump + 1] - t[jump])),
        ekf_at_lower_bound_first_s=float(t[lower][0]) if lower.any() else None,
        ekf_at_lower_bound_last_s=float(t[lower][-1]) if lower.any() else None,
        ekf_at_lower_bound_duration_s=float(np.sum(np.diff(t)[lower[:-1]])))
    write_json(output / 'failure_detail.json', detail)
    proxy = np.genfromtxt(output / f'{row["subject"]}_{trial}.csv', delimiter=',', names=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for ax, k, label in zip(axes.ravel(), range(8, 12), ['q9: elbow flexion', 'q10: forearm rotation', 'q11: wrist flexion', 'q12: wrist deviation']):
        ax.plot(t, ik[:, k], color='#9B9B9B', lw=.8, label='IK')
        ax.plot(t, ekf[:, k], color='#386A9A', lw=1, label='EKF')
        ax.set(title=label, xlabel='Time (s)', ylabel='Angle (deg)')
        ax.grid(alpha=.15)
        if k == 9:
            ax.plot(proxy['time_s'], proxy['upperarm_referenced_twist_deg'] + row['start_deg'], color='#B77942', lw=1,
                label='Marker twist + initial q10 (offset only)')
            ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.38), frameon=False, fontsize=7)
        elif k == 8:
            ax.legend(frameon=False)
    fig.suptitle(f'{row["subject"]} / {trial}: {row["description"]}', fontsize=12)
    fig.savefig(output / 'failure_detail.png', dpi=160)
    fig.savefig(output / 'failure_detail.svg')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--data', type=Path, default=DATA_ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT / '_checks/pronation_audit')
    parser.add_argument('--seed', type=int, default=20261118)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT / 'outputs').resolve()):
        parser.error('Output must stay within project outputs/')
    if output.exists() and any(output.iterdir()):
        parser.error('Output exists; choose a new output path to preserve previous audit')
    output.mkdir(parents=True, exist_ok=True)
    selection = select_trials(args.results, args.seed)
    write_json(output / 'selection.json', dict(seed=args.seed, method='One uniform draw within each subject, plus two draws without replacement from remaining trials; missing task controls 2/21/30 added separately', trials=selection))
    records, curves = [], []
    for row in selection:
        result, curve = analyze(row, args.results, args.data, output)
        records.append(result)
        curves.append(curve)
        print(f'{row["subject"]} trial_{row["trial"]:03d} task={row["task"]}: r={result["proxy_ekf_corr"]}, proxy RMSE={result["proxy_ekf_relative_rmse_deg"]:.1f} deg', flush=True)
    fields = [key for key in records[0] if key != 'sources']
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    write_json(output / 'summary.json', dict(created_at=datetime.now().astimezone().isoformat(), seed=args.seed, records=records,
        limitations=['Task labels do not establish frame-level phase or angle truth.', 'Main marker proxy tracks wrist transverse direction relative to upper-arm skin cluster, removing the shortest forearm-axis swing; it uses no calibrated offsets or IK angles.', 'The alternative ulna-referenced proxy shares the disputed ulna binding hypothesis and may absorb true forearm twist.', 'The proxy axis is an approximate elbow-to-wrist direction, not the fitted G99 RU axis; upper-arm skin motion and bend/twist decomposition remain error sources.', 'Proxy gaps are unwrapped separately; no inference should bridge gaps.', 'Relative agreement cannot validate absolute zero or clinical pronation/supination sign.', 'No reconstruction or EMG output was recomputed or replaced.']))
    draw(records, curves, output)
    draw_failure_detail(records, args.results, output)
    print(f'Saved {len(records)} trials to {output}', flush=True)
