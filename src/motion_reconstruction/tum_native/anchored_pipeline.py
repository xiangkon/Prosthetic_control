"""Run the maintained v002 pose-anchored reconstruction."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import numpy as np
from .anchored import solve_anchored
from .anchored_quality import evaluate_anchored, QUALITY_V2, REJECT
from .calibration import calibrate
from .functional_axes import calibrate_axes
from .model import decompose
from .observations import load_motion, prepare, tasks, track
from .provenance import runtime_fingerprint
from .schema import PROJECT, DATA_ROOT, Q_NAMES, SEGMENTS, canonical_hash, finite_stat, sha256, write_json
from ..common.emg_sync import read_emg_clock, infer_pulse_alignment, resample_estimates
from ..paths import NATIVE_ROOT

SCHEMA = 'tum-native-functional7-v2'
CONFIG = PROJECT / 'configs/motion_reconstruction/tum_native/v002.json'


def calibrate_v2(folder, jobs, config):
    model = calibrate(folder, config)
    tracks, sources = [], []
    for job in jobs:
        if job['task'] not in config['axis_calibration_tasks'] or job['repeat'] != 1:
            continue
        source = folder/'MoCap'/f"trial_{job['trial']:03d}.mat"
        original = load_motion(source)
        sample = original.select(max(1, len(original.t) // config['calibration_frames_per_trial']))
        mask = np.zeros_like(sample.valid)
        tracks.append(track(dict(t=sample.t, p=sample.p, names=sample.names, edge=mask, interpolated=mask), model))
        sources.append(dict(job, source=str(source.resolve()), source_sha256=sha256(source),
                            axis_role='fit' if len(tracks) % 2 else 'validation'))
    model['axis_calibration']['functional_refinement'] = calibrate_axes(model, tracks, config)
    model['axis_calibration_sources'] = sources
    model['calibration_trial_ids'] = sorted(set(model['calibration_trial_ids']) | {r['trial'] for r in sources})
    model['schema'] = SCHEMA
    reference = model['reference']
    obs = prepare(load_motion(reference['source']), config)
    q, _, _ = decompose(track(obs, model), model)
    window = q[reference['start_index']:reference['end_index_exclusive']]
    model['reference_q_median_rad'] = [finite_stat(v, .5) for v in window.T]
    return model


def reconstruct_v2(source, job, model, config, output, emg, fingerprint, source_hash, emg_hash):
    start = perf_counter()
    data = load_motion(source)
    obs = prepare(data, config)
    poses = track(obs, model)
    tracked = perf_counter()
    solution = solve_anchored(obs, poses, model, config)
    solved = perf_counter()
    quality = evaluate_anchored(obs, poses, model, solution, config)
    checked = perf_counter()
    q = solution['state'][:, :7]
    # Use measured marker movement in the thorax frame to define the diagnostic motion subset.
    ids = [obs['names'].index(name) for segment in SEGMENTS[1:] for name in model['segments'][segment]['fit_names']]
    local = np.einsum('nji,nkj->nki', poses['R'][:, 0], obs['p'][:, ids]-poses['p'][:, 0, None])
    speeds = np.linalg.norm(np.diff(local, axis=0), axis=2) / np.diff(data.t)[:, None]
    moving = np.r_[False, np.ma.filled(np.ma.median(np.ma.masked_invalid(speeds), axis=1), -np.inf) > .05]
    motion_valid = quality['relative_valid'][moving]
    mismatch = quality['local_orientation_error_rad']
    raw_error = np.linalg.norm(quality['predicted_markers_m']-obs['raw'], axis=-1)
    report = dict(schema=SCHEMA, job=job, source=str(source.resolve()), source_sha256=source_hash,
        source_frames=len(data.t), output_frames=len(data.t), output_rate_hz=float(1/np.median(np.diff(data.t))),
        algorithm='tum_native', model_hash=canonical_hash(model), fingerprint=fingerprint, config=config,
        quality_bits=QUALITY_V2, quality_reject_mask=REJECT, q_names=Q_NAMES,
        role='reference' if job['task'] == 0 else 'calibration' if job['trial'] in model['calibration_trial_ids'] else 'evaluation',
        relative_valid_fraction=quality['relative_valid'].mean(0).tolist(),
        motion_marker_speed_threshold_m_s=.05, motion_frames=int(moving.sum()),
        motion_valid_fraction=motion_valid.mean(0).tolist() if moving.any() else None,
        motion_quality_flag_counts={bit: ((quality['flags'][moving] & int(bit)) != 0).sum(0).tolist()
                                    for bit in QUALITY_V2 if int(bit) & REJECT},
        raw_marker_error_median_m=finite_stat(raw_error, .5), raw_marker_error_p95_m=finite_stat(raw_error, .95),
        local_orientation_error_p95_deg=[finite_stat(np.rad2deg(v), .95) for v in mismatch.T],
        maximum_adjacent_angle_step_deg=[finite_stat(abs(np.rad2deg(np.diff(v))), 1.) for v in q.T],
        maximum_adjacent_bone_step_m=[finite_stat(v, 1.) for v in np.linalg.norm(np.diff(quality['bones_m'], axis=0), axis=-1).T],
        solver_refinement=solution['refinement'], sync=dict(status='not_requested'))
    with TemporaryDirectory(prefix='.v002-', dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        np.savez_compressed(staging/'reconstruction.npz', schema=np.array(SCHEMA), t=data.t,
            source_indices=np.arange(len(data.t)), q=q, q_initial=solution['initial'][:, :7],
            q_names=np.array(Q_NAMES), angle_unit=np.array('rad'), length_unit=np.array('m'), time_unit=np.array('s'),
            shoulder_displacement_T_m=solution['state'][:, 7:],
            observed_segment_R=poses['R'], observed_segment_p_m=poses['p'],
            cluster_rms_m=poses['rms_m'], cluster_marker_count=poses['counts'],
            raw_markers_m=obs['raw'], filtered_markers_m=obs['p'], marker_names=np.array(obs['names']),
            marker_observed=obs['observed'], marker_interpolated=obs['interpolated'], marker_filter_edge=obs['edge'],
            solver_refined=solution['refined'], solver_correction_rejected=solution['correction_rejected'],
            **quality)
        if emg and job['paired_emg']:
            try:
                emg_source = source.parent.parent/'EMG'/source.name
                target, pulse, metadata = read_emg_clock(emg_source)
                sync = infer_pulse_alignment(target, pulse, data.t)
                clean = np.where(quality['relative_valid'], q, np.nan)
                aligned, flags = resample_estimates(data.t, clean, quality['flags'],
                    (target-sync['offset_s']) / sync['clock_scale'])
                valid = np.isfinite(aligned) & ((flags & REJECT) == 0)
                np.savez_compressed(staging/'emg_aligned_native.npz', t=target, q=aligned, flags=flags,
                    relative_valid=valid, q_names=np.array(Q_NAMES), schema=np.array(SCHEMA), angle_unit=np.array('rad'))
                report['sync'] = dict(sync, emg_source_sha256=emg_hash, metadata=metadata)
            except (ValueError, OSError) as exc:
                report['sync'] = dict(status='unavailable', reason=str(exc), emg_source_sha256=emg_hash)
        report['timing_s'] = dict(track=tracked-start, solve=solved-tracked, quality=checked-solved,
                                  total=perf_counter()-start)
        write_json(staging/'report.json', report)
        staging.rename(output)
    return report


def _run_trial(arguments):
    source, job, model, config, output, emg, fingerprint = arguments
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    emg_file = source.parent.parent/'EMG'/source.name
    emg_hash = sha256(emg_file) if emg and job['paired_emg'] and emg_file.is_file() else None
    if output.exists():
        report = json.loads((output/'report.json').read_text(encoding='utf-8'))
        sync = report['sync']
        sync_matches = not emg or not job['paired_emg'] or (
            sync['status'] != 'not_requested' and sync.get('emg_source_sha256') == emg_hash and
            (sync['status'] == 'unavailable' or (output/'emg_aligned_native.npz').is_file()))
        if (report['fingerprint'] == fingerprint and report['source_sha256'] == source_hash and
                report['model_hash'] == canonical_hash(model) and sync_matches and (output/'reconstruction.npz').is_file()):
            return model['subject'], output.name, 'cached'
        raise FileExistsError(f'Existing v002 result differs from requested inputs: {output}; use a separate --output-root')
    report = reconstruct_v2(source, job, model, config, output, emg, fingerprint, source_hash, emg_hash)
    return model['subject'], output.name, f"{report['timing_s']['total']:.2f}s, valid={np.round(report['relative_valid_fraction'], 3).tolist()}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-root', type=Path, default=NATIVE_ROOT)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 7)))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--trials', nargs='+', type=int)
    group.add_argument('--all', action='store_true')
    group.add_argument('--calibrate-only', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--emg', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or any(not 1 <= s <= 6 for s in args.subjects):
        parser.error('Expected 1–8 workers and subject IDs in 1–6')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['version'] != 'v002_anchored7' or config['solver'] != 'anchored':
        parser.error('v002 requires the anchored solver and v002_anchored7 output version')
    root = args.output_root.resolve() / config['version']
    fingerprint = runtime_fingerprint(config)
    queue = []
    for number in dict.fromkeys(args.subjects):
        subject = f'subj_{number:03d}'
        folder, output = args.data_root/subject, root/subject
        jobs = tasks(folder)
        selected = jobs if args.all else [job for job in jobs if job['trial'] in (args.trials or [])]
        if args.trials and set(args.trials) - {job['trial'] for job in selected}:
            parser.error(f'{subject}: requested trial not in task table')
        path = output/'model.json'
        if path.exists():
            model = json.loads(path.read_text(encoding='utf-8'))
            sources = [model['reference'], *model['calibration_sources'], *model['axis_calibration_sources']]
            if model['runtime_fingerprint'] != fingerprint or any(sha256(s['source']) != s['source_sha256'] for s in sources):
                raise ValueError(f'{path}: calibration input changed; select a separate --output-root')
        else:
            print(f'{subject}: calibrating {config["version"]}', flush=True)
            model = calibrate_v2(folder, jobs, config)
            model['runtime_fingerprint'] = fingerprint
            write_json(path, model)
        for job in selected:
            name = f"trial_{job['trial']:03d}"
            queue.append((folder/'MoCap'/f'{name}.mat', job, model, config, output/'trials'/name, args.emg, fingerprint))

    def display(rows):
        for subject, trial, status in rows:
            print(f'{subject}/{trial}: {status}', flush=True)

    if args.workers == 1 or len(queue) <= 1:
        display(map(_run_trial, queue))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            display(pool.map(_run_trial, queue))
