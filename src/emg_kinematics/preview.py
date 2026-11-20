"""Static synchronized overview using the same data and plot builders as the UI."""
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.emg_visualization.plotting import heatmap_plot, wave_plot
from .plotting import angle_tracks, build_scene, update_scene


def save_preview(data, path, seconds=None, signal='emg_normalized', uniform=False, heat=False, limits=None):
    if seconds is None:
        seconds = np.mean(data.common_range)
    index = data.nearest(seconds)
    figure = Figure(figsize=(18, 13), dpi=120)
    FigureCanvasAgg(figure)
    heading, body = figure.subfigures(2, 1, height_ratios=[.065, .935])
    left, right = body.subfigures(1, 2, width_ratios=[.31, .69])
    motion, text = left.subfigures(2, 1, height_ratios=[1.15, 1])
    angles, emg = right.subfigures(2, 1, height_ratios=[3, 7])
    scene = build_scene(motion, data)
    update_scene(scene, data, index)
    angle_axes = angle_tracks(angles, data)
    if heat:
        emg_axes = [heatmap_plot(emg, data.emg)]
    else:
        state = wave_plot(emg, data.emg, signal=signal, uniform=uniform)
        emg_axes = state['axes']
    if heat:
        emg_axes[0].set_position([.23, .1, .69, .82])
        emg.axes[-1].set_position([.94, .1, .02, .82])
    else:
        emg.subplots_adjust(left=.23, right=.92)
    for ax in [*angle_axes, *emg_axes]:
        ax.set_xlim(*(limits or data.common_range))
        ax.axvline(data.time[index], color='#bb463e', lw=.9)
    # Refresh min/max reduction to the exported time range.
    if not heat:
        from src.emg_visualization.plotting import update_wave
        update_wave(state, 1400)
    lines = [f'当前时间 {data.time[index]:.4f} s', '']
    for j, name in enumerate(data.angle_names):
        value = data.degrees[index, j]
        lines.extend([f'{name}：{value:.2f}°' if np.isfinite(value) else f'{name}：缺失',
                      '  ' + data.quality_text(data.flags[index, j]), ''])
    lines.extend([f'动画使用{data.motion_description}。', '手指为示意形状；缺失完整运动学时隐藏姿态。', '',
                  f'同步偏移：{data.sync["offset_s"]:.4f} s', '脉冲时长匹配候选；硬件延迟未独立验证。'])
    text.text(.08, .92, '\n'.join(lines), va='top', fontsize=11, linespacing=1.5)
    heading.text(.5, .5, f'subj_{data.emg.subject:03d} / trial_{data.emg.trial:03d} · '
                    f'Task {data.emg.task} {data.emg.task_description}\n'
                    f'{data.algorithm_label} · {data.emg.version} · 同步角度估计与肌电', fontsize=13, ha='center', va='center')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    print(path.resolve())
