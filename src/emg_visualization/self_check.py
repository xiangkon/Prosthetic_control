"""Exercise real data, Tk controls, peak-preserving rendering and rejection."""
import hashlib
from pathlib import Path
import tempfile
from copy import copy

import numpy as np

from .data import EmgData, SIGNALS, extrema_trace
from .viewer import EmgViewer
from .plotting import wave_plot
from src.task_metadata import tum_task_description_zh


def run_checks(data):
    import tkinter as tk
    original_hash = hashlib.sha256(data.path.read_bytes()).hexdigest()
    assert tum_task_description_zh(0) == '初始放松姿态'
    assert tum_task_description_zh(1) == 'OK 手势'
    assert tum_task_description_zh(999) == '未知任务（Task 999）'
    assert tum_task_description_zh(data.task) in data.summary()
    time = np.arange(10000, dtype=float)
    values = np.zeros(len(time))
    values[125] = 8
    values[126] = -9
    x, y = extrema_trace(time, values, 0, time[-1], 200)
    assert 8 in y and -9 in y and len(x) < 1000
    x, y = extrema_trace(time, values, 120, 130, 200)
    assert 8 in y and -9 in y
    root = tk.Tk()
    root.withdraw()
    viewer = EmgViewer(root, data)
    try:
        root.update()
        for label in SIGNALS:
            viewer.signal.set(label)
            if label == '归一化包络':
                viewer.uniform.set(False)
                viewer.refresh()
                root.update()
                for index, ax in enumerate(viewer.wave_state['axes']):
                    finite = data.arrays['emg_normalized'][:, index]
                    finite = finite[np.isfinite(finite)]
                    maximum = float(np.max(np.abs(finite))) if len(finite) else 1.
                    np.testing.assert_allclose(ax.get_ylim(), [0, max(maximum, 1e-9) * 1.05])
            viewer.uniform.set(True)
            viewer.refresh()
            root.update()
            state = viewer.wave_state
            if label == '归一化包络':
                assert all(np.allclose(ax.get_ylim(), state['axes'][0].get_ylim()) for ax in state['axes'])
                assert state['axes'][0].get_ylim()[1] >= 1.05
            state['axes'][0].set_xlim(state['time'][0] + .5, state['time'][0] + 1.)
            viewer.redraw_wave()
            assert all(np.allclose(ax.get_xlim(), state['axes'][0].get_xlim()) for ax in state['axes'])
        for index in range(3):
            viewer.notebook.select(index)
            viewer.render()
            root.update()
            viewer.canvases[viewer.notebook.tab(index, 'text')].draw()
        viewer.relative.set(False)
        viewer.notebook.select(0)
        viewer.refresh()
        root.update()
        if len(viewer.paths) > 1:
            viewer.step(1)
            root.update()
            assert viewer.data.path != data.path
            assert viewer.signal.get() == '归一化包络' and viewer.uniform.get()
            assert not viewer.relative.get()
            assert viewer.load(data.path)
        with tempfile.TemporaryDirectory() as directory:
            malformed = Path(directory) / 'invalid.npz'
            np.savez(malformed, t=[0, 1])
            current = viewer.data
            assert not viewer.load(malformed, show_error=False)
            assert viewer.data is current and viewer.last_error
            arrays = {k: v.copy() for k, v in data.arrays.items()}
            arrays['t'][1] = arrays['t'][0]
            np.savez(malformed, **arrays)
            try:
                EmgData(malformed)
            except ValueError:
                pass
            else:
                raise AssertionError('Non-monotonic time must be rejected')
        assert hashlib.sha256(data.path.read_bytes()).hexdigest() == original_hash
        # Resting EMG can exceed the motion-derived reference; do not crop it.
        from matplotlib.figure import Figure
        larger = copy(data)
        larger.arrays = dict(data.arrays)
        larger.arrays['emg_normalized'] = data.arrays['emg_normalized'].copy()
        larger.arrays['emg_normalized'][0, 0] = 1.8
        figure = Figure()
        state = wave_plot(figure, larger, signal='emg_normalized', uniform=True)
        assert all(ax.get_ylim()[1] > 1.8 for ax in state['axes'])
        figure.clear()
        print('PASS: NPZ validation, extrema preservation, three signals/tabs, shared zoom, data switching, failed-load retention, source unchanged')
    finally:
        viewer.close()
