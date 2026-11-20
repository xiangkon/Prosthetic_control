"""Versioned, bounded-memory offline preprocessing with explicit provenance."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import scipy

from src.motion_reconstruction.paths import PROJECT, DATA_ROOT
from src.task_metadata import tum_task_description_zh
from .methods.hu2018 import preprocess
from .reader import discover_trials, read_selected


DEFAULT_CONFIG = PROJECT / 'configs/emg_preprocessing/v002_hu2018_with_rest.json'
DEFAULT_OUTPUT = PROJECT / 'outputs/TUM/emg_preprocessed'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def write_npz(path, arrays):
    path = Path(path)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def publish_result(staging, destination):
    """Publish only completed results; preserve the previous method result."""
    backup = None
    if destination.exists():
        backup_root = destination.parent / '_backups' / destination.name
        backup_root.mkdir(parents=True, exist_ok=True)
        index = 1
        while (backup_root / f'backup_{index:03d}').exists():
            index += 1
        backup = backup_root / f'backup_{index:03d}'
        destination.rename(backup)
    try:
        staging.rename(destination)
    except OSError:
        if backup is not None:
            backup.rename(destination)
        raise
    return backup


def validate_config(config):
    if not re.fullmatch(r'v[0-9]{3}_[a-z0-9_]+', config['version']):
        raise ValueError('Version must be vNNN_method_name')
    if config['method'] != 'hu2018' or config['sample_rate_hz'] != 2000:
        raise ValueError('Only hu2018 on the guarded 2000 Hz TUM reader is implemented')
    channels = config['channels']
    indices = [c['bipolar_index_1based'] for c in channels]
    if (not channels or any(type(i) is not int or i not in range(1, 33) or i in (19, 25, 26)
                            for i in indices) or len(set(indices)) != len(indices)
            or len({c['name'] for c in channels}) != len(channels)):
        raise ValueError('Invalid, duplicate, or officially unused bipolar channels')
    tasks = config['task_ids']
    if not tasks or len(set(tasks)) != len(tasks) or any(type(t) is not int or t not in range(31) for t in tasks):
        raise ValueError('Select unique task IDs from 0 through 30')
    reference_tasks = config.get('normalization_task_ids', tasks)
    if (not reference_tasks or len(set(reference_tasks)) != len(reference_tasks)
            or any(type(t) is not int or t not in range(1, 31) for t in reference_tasks)
            or not set(reference_tasks).issubset(tasks)):
        raise ValueError('Normalization tasks must be unique motion IDs 1-30 included in task_ids')
    reference_tasks = config.get('normalization_task_ids', tasks)
    if (not reference_tasks or len(set(reference_tasks)) != len(reference_tasks)
            or any(type(t) is not int or t not in tasks for t in reference_tasks)):
        raise ValueError('Normalization task IDs must be a nonempty unique subset of exported tasks')
    bp, lp, zp = config['bandpass'], config['lowpass'], config['zero_phase']
    if (type(bp['prototype_order']) is not int or bp['prototype_order'] < 1
            or not 0 < bp['low_hz'] < bp['high_hz'] < 1000):
        raise ValueError('Invalid Butterworth bandpass')
    if (lp['order'] != 2 or lp['damping_ratio'] != 1 or not 0 < lp['cutoff_hz'] < 1000
            or lp['cutoff_definition'] != 'single_pass_minus3db'):
        raise ValueError('Expected second-order critically damped lowpass with single-pass -3 dB cutoff')
    if (zp['implementation'] != 'sosfiltfilt' or zp['padtype'] != 'odd'
            or not 0 < zp['pad_seconds'] < 60 or not 0 <= config['edge_margin_s'] < 60):
        raise ValueError('Unsupported zero-phase or edge parameters')
    if config['normalization'] != 'subject_channel_max_over_configured_tasks' or config['amplitude_unit'] != 'uV':
        raise ValueError('Unsupported normalization or amplitude unit')


def process_subject(root, destination, subject, config, limit_trials=None):
    trials = discover_trials(root, subject, config['task_ids'])
    total = len(trials)
    if limit_trials is not None:
        trials = trials[:limit_trials]
    reference_tasks = config.get('normalization_task_ids', config['task_ids'])
    if not any(info['task_id'] in reference_tasks for info in trials):
        raise ValueError('No normalization reference trials in the selection')
    folder = destination / f'subj_{subject:03d}'
    folder.mkdir()
    peaks = np.zeros(len(config['channels']), dtype=float)
    reports = []
    for info in trials:
        source = Path(info['source'])
        digest = sha256(source)
        time, bipolar, clock = read_selected(source, config['channels'])
        if not np.isclose(1 / clock['dt_s'], config['sample_rate_hz'], rtol=0, atol=1e-6):
            raise ValueError(f'Sampling rate mismatch: {source}')
        filtered, envelope, flags, edge_valid, negatives = preprocess(bipolar, config)
        filtered, envelope = filtered.astype(np.float32), envelope.astype(np.float32)
        if not np.isfinite(filtered[:, flags == 0]).all() or not np.isfinite(envelope[:, flags == 0]).all():
            raise ValueError(f'Float32 overflow: {source}')
        local_max = np.zeros(len(peaks))
        good = flags == 0
        if good.any():
            local_max[good] = envelope[:, good].max(axis=0)
        if info['task_id'] in reference_tasks:
            peaks = np.maximum(peaks, local_max)
        report = dict(**info, source_sha256=digest, clock=clock, samples=len(time),
                      description_zh=tum_task_description_zh(info['task_id']),
                      used_for_normalization=info['task_id'] in reference_tasks,
                      duration_s=len(time) * clock['dt_s'], channel_quality_flag=flags.tolist(),
                      negative_envelope_samples_clipped=negatives.tolist(),
                      peak_envelope_uV=local_max.tolist(), edge_valid_samples=int(edge_valid.sum()))
        trial_folder = folder / f'trial_{info["trial_id"]:03d}'
        trial_folder.mkdir()
        path = trial_folder / 'emg_preprocessed.npz'
        write_npz(path, dict(t=time, emg_bandpass=filtered, emg_envelope=envelope,
                            channel_names=np.array([c['name'] for c in config['channels']]),
                            bipolar_indices_1based=np.array([c['bipolar_index_1based'] for c in config['channels']]),
                            channel_quality_flag=flags, edge_valid=edge_valid,
                            amplitude_unit=np.array(config['amplitude_unit']), time_unit=np.array('s'),
                            sample_rate_hz=np.array(config['sample_rate_hz']),
                            task_id=np.array(info['task_id']), trial_id=np.array(info['trial_id']),
                            task_description_zh=np.array(tum_task_description_zh(info['task_id'])),
                            subject_id=np.array(subject), source_sha256=np.array(digest),
                            source=np.array(str(source)), version=np.array(config['version'])))
        reports.append((path, report))
        print(f'subj_{subject:03d} trial_{info["trial_id"]:03d}: filtered {len(time)} samples', flush=True)
    valid_scale = np.isfinite(peaks) & (peaks > 0)
    scales = np.where(valid_scale, peaks, np.nan)
    for path, report in reports:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        flags = arrays['channel_quality_flag'].copy()
        flags[~valid_scale] |= 4
        arrays['channel_quality_flag'] = flags
        normalized = np.full_like(arrays['emg_envelope'], np.nan)
        good = flags == 0
        normalized[:, good] = arrays['emg_envelope'][:, good] / scales[good]
        arrays['emg_normalized'] = normalized
        arrays['normalization_max_uV'] = scales
        arrays['normalization_unit'] = np.array('dimensionless')
        arrays['normalization_task_ids'] = np.array(reference_tasks, dtype=int)
        arrays['normalization_scope'] = np.array('selected_trials_smoke_only' if len(trials) < total
                                                else ('subject_channel_max_over_motion_tasks_1_30'
                                                      if set(reference_tasks) == set(range(1, 31))
                                                      and 0 in config['task_ids'] else config['normalization']))
        write_npz(path, arrays)
        report['channel_quality_flag'] = flags.tolist()
        report['output_sha256'] = sha256(path)
        write_json(path.parent / 'report.json', report)
    summary = dict(subject=subject, trials=len(trials), available_trials=total,
                   samples=sum(r['samples'] for _, r in reports),
                   normalization_max_uV=[float(v) if np.isfinite(v) else None for v in scales],
                   normalization_task_ids=reference_tasks,
                   normalization_trial_ids=[r['trial_id'] for _, r in reports if r['used_for_normalization']],
                   resting_trial_ids=[r['trial_id'] for _, r in reports if r['task_id'] == 0],
                   trials_with_quality_flags=sum(any(f != 0 for f in r['channel_quality_flag']) for _, r in reports),
                   trial_info_sha256=sha256(Path(root) / f'subj_{subject:03d}' / 'trial_info.mat'))
    write_json(folder / 'normalization.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Versioned offline TUM EMG preprocessing (not a training split)')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Base directory under project outputs/')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 7)))
    parser.add_argument('--limit-trials', type=int, help='Diagnostic only: first N selected trials per subject')
    parser.add_argument('--self-check', action='store_true', help='Synthetic filter checks; no dataset processing')
    args = parser.parse_args(argv)
    if args.self_check:
        from .self_check import run_checks
        run_checks()
        return
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    if (not args.subjects or len(set(args.subjects)) != len(args.subjects)
            or any(s not in range(1, 7) for s in args.subjects)):
        parser.error('Select unique subject IDs from 1 through 6')
    if args.limit_trials is not None and args.limit_trials < 1:
        parser.error('--limit-trials must be positive')
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT / 'outputs').resolve()):
        parser.error('--output must remain inside the project outputs/ directory')
    # Discover all requested inputs before creating a working directory.
    for subject in args.subjects:
        discover_trials(args.root, subject, config['task_ids'])
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest()
    version_folder = output / config['version']
    previous_manifests = list(version_folder.glob('run_*/manifest.json'))
    if (version_folder / 'manifest.json').exists():
        previous_manifests.append(version_folder / 'manifest.json')
    for previous in previous_manifests:
        previous_manifest = json.loads(previous.read_text(encoding='utf-8'))
        if previous_manifest['config_sha256'] != config_hash:
            raise ValueError('This version already has different parameters; create a new vNNN configuration')
        if args.limit_trials is not None and not previous_manifest.get('diagnostic', False):
            raise ValueError('Diagnostic processing cannot replace full results; use --output outputs/TUM/_checks/emg_preprocessed')
    local = datetime.now(timezone(timedelta(hours=8)))
    output.mkdir(parents=True, exist_ok=True)
    # A transient workspace avoids exposing incomplete arrays at the stable path.
    destination = Path(tempfile.mkdtemp(prefix=f'.{config["version"]}_working_', dir=output))
    code_files = sorted((PROJECT / 'src/emg_preprocessing').rglob('*.py')) + [
        PROJECT / 'src/motion_reconstruction/emg_sync.py',
        PROJECT / 'src/task_metadata.py',
        PROJECT / 'src/motion_reconstruction/_matlab.py']
    manifest = dict(status='running', started_at=local.isoformat(), config=config,
                    config_sha256=config_hash, input_root=str(args.root.resolve()),
                    config_file_sha256=sha256(args.config), subjects=args.subjects,
                    limit_trials=args.limit_trials, diagnostic=args.limit_trials is not None,
                    software=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
                    code_sha256={str(p.relative_to(PROJECT)): sha256(p) for p in code_files},
                    normalization_warning='Per-subject maxima use all selected motion trials, including potential future evaluation trials. Refit on training/calibration data before decoding evaluation.',
                    subject_reports=[])
    manifest_path = destination / 'manifest.json'
    write_json(manifest_path, manifest)
    write_json(destination / 'config.json', config)
    print(f'Output: {version_folder}', flush=True)
    try:
        for subject in args.subjects:
            manifest['subject_reports'].append(process_subject(args.root, destination, subject, config, args.limit_trials))
            write_json(manifest_path, manifest)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        manifest['finished_at'] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        write_json(manifest_path, manifest)
    backup = publish_result(destination, version_folder)
    if backup is not None:
        print(f'Previous results backed up: {backup}', flush=True)
    print(f'Complete: {sum(s["trials"] for s in manifest["subject_reports"])} trials; {version_folder}', flush=True)
