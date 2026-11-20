"""Verify synchronization, FK, linked time axes, controls and missing-angle UI."""
import hashlib
import time
from types import SimpleNamespace
from copy import copy

import numpy as np

from .data import interpolate_poses
from .viewer import CombinedViewer


def run_checks(data, reconstruction):
    import tkinter as tk
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (data.emg.path, data.label_path)]
    assert data.q.shape == (len(data.emg.time), 4)
    expected_columns = 7 if data.algorithm == 'tum_native' else 12
    assert data.full_q.shape == (len(data.emg.time), expected_columns)
    if data.algorithm == 'legacy_g99':
        np.testing.assert_allclose(data.full_q[:, 8:], data.q, atol=1e-6, rtol=0, equal_nan=True)
    finite = np.flatnonzero(data.overlap)
    mid = int(np.flatnonzero(data.pose_valid)[len(np.flatnonzero(data.pose_valid)) // 2])
    base = data.pose(mid)
    changed = copy(data)
    if data.algorithm == 'legacy_g99':
        changed.full_q = data.full_q.copy()
        changed.full_q[mid, 0] += .2
        assert np.linalg.norm(changed.pose(mid)['upper'] - base['upper']) > .01
        changed = copy(data)
        changed.poses = data.poses.copy()
        changed.poses[mid, 0, 3] += .01
    else:
        changed.segment_poses = data.segment_poses.copy()
        changed.bones = data.bones.copy()
        changed.segment_poses[mid, :, 0, 3] += .01
        changed.bones[mid, :, 0] += .01
    np.testing.assert_allclose(changed.pose(mid)['upper'] - base['upper'], [[10, 0, 0], [10, 0, 0]], atol=1e-5)
    from scipy.spatial.transform import Rotation
    source_poses = np.tile(np.eye(4), (2, 1, 1))
    source_poses[1, :3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    source_poses[1, 0, 3] = 2
    resampled = interpolate_poses(np.array([0., 1.]), source_poses, np.array([-.1, .5, 1.1]), 2)
    np.testing.assert_allclose(resampled[1, :3, :3], Rotation.from_euler('z', 45, degrees=True).as_matrix(), atol=1e-10)
    assert resampled[1, 0, 3] == 1 and np.isnan(resampled[[0, 2]]).all()
    assert np.isnan(interpolate_poses(np.array([0., 1.]), source_poses, np.array([.5]), .1)).all()
    root = tk.Tk()
    root.withdraw()
    viewer = CombinedViewer(root, data, reconstruction)
    try:
        root.update()
        # Real Tk geometry, kept transparent so no test window is displayed.
        root.attributes('-alpha', 0)
        root.deiconify()
        for width, height in ((1440, 1000), (1280, 900)):
            root.geometry(f'{width}x{height}')
            root.update()
            viewer.emg_canvas.draw()
            viewer.angle_canvas.draw()
            assert viewer.emg_canvas.get_tk_widget().winfo_height() > 1.8 * viewer.angle_canvas.get_tk_widget().winfo_height()
            assert len(viewer.emg_axes) == 10 and len(viewer.angle_axes) == 4
            for ax in viewer.emg_axes:
                assert ax.bbox.y0 >= 0 and ax.bbox.y1 <= viewer.emg_figure.bbox.height
                assert ax.bbox.height >= 20
        middle = int(finite[len(finite) // 2])
        viewer.set_index(middle)
        root.update()
        for line in viewer.angle_cursors + viewer.emg_cursors:
            np.testing.assert_allclose(line.get_xdata(), data.time[middle])
        original_range = viewer.angle_axes[0].get_xlim()
        zoom = (data.time[middle] - .15, data.time[middle] + .15)
        viewer.angle_axes[2].set_xlim(*zoom)
        assert all(np.allclose(ax.get_xlim(), zoom) for ax in viewer.linked_axes)
        viewer.redraw_trace()
        viewer.click_time(SimpleNamespace(canvas=viewer.emg_canvas, button=1, inaxes=viewer.emg_axes[0], xdata=data.time[middle] + .03))
        assert abs(data.time[viewer.index] - data.time[middle] - .03) < 1 / data.emg.fs
        viewer.step_sample(-1)
        for name in ('带通 sEMG', '肌电包络', '归一化包络'):
            viewer.signal.set(name)
            viewer.refresh_emg()
            root.update()
        viewer.heat.set(True)
        viewer.refresh_emg()
        root.update()
        assert len(viewer.emg_axes) == 1
        viewer.emg_axes[0].set_xlim(*zoom)
        assert all(np.allclose(ax.get_xlim(), zoom) for ax in viewer.linked_axes)
        viewer.heat.set(False)
        viewer.refresh_emg()
        viewer.full.set(True)
        viewer.reset_range()
        missing = np.flatnonzero(~data.pose_valid)
        if len(missing):
            viewer.set_index(missing[0])
            assert data.pose(missing[0]) is None
            assert len(viewer.scene['forearm'].get_data_3d()[0]) == 0
        viewer.full.set(False)
        viewer.reset_range()
        viewer.set_index(middle)
        viewer.toggle_play()
        time.sleep(.06)
        root.update()
        viewer.pause()
        assert viewer.index > middle and viewer.play_job is None
        for name in data.angle_names:
            assert name in viewer.values.get()
        if len(viewer.paths) > 1:
            original = viewer.data.path
            viewer.step(1)
            root.update()
            assert viewer.data.path != original
            assert viewer.load(original)
        previous = viewer.aligned
        assert not viewer.load(data.emg.path.with_name('absent.npz'), show_error=False)
        assert viewer.aligned is previous and viewer.last_error
        other = 'legacy_g99' if data.algorithm == 'tum_native' else 'tum_native'
        if viewer.paired([viewer.data.path], other):
            assert viewer.load(viewer.data.path, algorithm=other)
            assert viewer.aligned.algorithm == other
            assert viewer.load(data.emg.path, algorithm=data.algorithm)
        after = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (data.emg.path, data.label_path)]
        assert before == after
        print('PASS: clocks/source, FK, 14 linked axes, seeking, playback, 3 EMG modes, heatmap, missing-angle hiding, navigation, failed-load retention, source unchanged')
    finally:
        viewer.close()
