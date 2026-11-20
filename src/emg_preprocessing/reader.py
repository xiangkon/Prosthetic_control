"""TUM input adapters. Source files and original sample clocks are untouched."""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.motion_reconstruction.common.emg_sync import read_emg_record


def discover_trials(root, subject, task_ids):
    """Use the official valid EMG lists, rather than infer tasks from filenames."""
    folder = Path(root) / f'subj_{subject:03d}'
    rows = np.atleast_1d(loadmat(folder / 'trial_info.mat', simplify_cells=True)['trial_info'])
    trials, seen = [], set()
    for task in task_ids:
        if not 0 <= task < len(rows):
            raise ValueError(f'Unknown task {task}')
        row = rows[task]
        for position, value in enumerate(np.atleast_1d(row['EMG']).ravel(), 1):
            number = float(value)
            if not np.isfinite(number) or number < 1 or not number.is_integer():
                raise ValueError(f'Invalid EMG trial index: {value}')
            trial = int(number)
            if trial in seen:
                raise ValueError(f'Duplicate EMG trial mapping: {subject}/{trial}')
            seen.add(trial)
            path = folder / 'EMG' / f'trial_{trial:03d}.mat'
            if not path.is_file():
                raise FileNotFoundError(path)
            trials.append(dict(subject=subject, task_id=task, trial_id=trial,
                               valid_list_position=position, description=str(row['Description']),
                               source=str(path.resolve())))
    if not trials:
        raise ValueError(f'No selected EMG trials for subject {subject}')
    return trials


def read_selected(path, channels):
    time, raw, clock = read_emg_record(path)
    indices = np.array([c['bipolar_index_1based'] for c in channels], dtype=int) - 1
    bipolar = raw[:, 2 * indices] - raw[:, 2 * indices + 1]
    return time, bipolar, clock
