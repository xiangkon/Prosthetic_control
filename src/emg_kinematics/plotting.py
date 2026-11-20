"""Angle tracks and calibrated four-coordinate motion illustration."""
import numpy as np
from matplotlib.ticker import MaxNLocator

from src.emg_visualization.plotting import spans
from .data import ANGLE_COLORS


def angle_tracks(figure, data):
    figure.clear()
    axes = figure.subplots(4, 1, sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(data.time, data.degrees[:, j], color=ANGLE_COLORS[j], lw=1.)
        # Prior dependence is displayed separately from other quality problems.
        other = (data.flags[:, j].astype(np.int64) & data.reject_mask) != 0
        for begin, end in spans(other):
            ax.axvspan(data.time[begin], data.time[end - 1] + 1 / data.emg.fs,
                       color='#d48869', alpha=.2, lw=0)
        prior = np.any((data.flags[:, j] & data.prior_mask) != 0)
        ax.set_ylabel(data.angle_names[j] + (' *' if prior else '') + '\n°',
                      rotation=0, ha='right', va='center', color=ANGLE_COLORS[j], fontsize=8)
        ax.tick_params(labelsize=7, pad=1)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.grid(alpha=.16)
    axes[-1].tick_params(labelbottom=False)
    figure.suptitle(f'{data.algorithm_label}肘腕角度估计 · * 模型／参考依赖 · 橙色区：质量未通过', fontsize=9, y=.99)
    figure.subplots_adjust(left=.23, right=.92, top=.88, bottom=.035, hspace=.22)
    return list(axes)


def build_scene(figure, data):
    figure.clear()
    ax = figure.add_subplot(111, projection='3d')
    ax.set_title(f'{data.algorithm_label}上肢运动 · 含胸廓位姿\n基于估计角度，非独立测量', fontsize=10, pad=12)
    ax.set_xlabel('X (mm)', fontsize=8)
    ax.set_ylabel('Y (mm)', fontsize=8)
    ax.set_zlabel('Z (mm)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=22, azim=-65)
    ax.set_box_aspect((1, 1, 1))
    scene = dict(ax=ax)
    for key, color, width in [('torso', '#b4bdc5', 2), ('proximal', '#8096aa', 5),
                              ('upper', '#b4bdc5', 8), ('forearm', '#49798f', 7),
                              ('palm', '#6e8f86', 2), ('band', ANGLE_COLORS[1], 4)]:
        scene[key], = ax.plot([], [], [], color=color, lw=width, solid_capstyle='round')
    scene['fingers'] = [ax.plot([], [], [], color='#6e8f86', lw=2)[0] for _ in range(4)]
    scene['fingers'].append(ax.plot([], [], [], color='#ba8346', lw=2.8, label='拇指示意')[0])
    scene['joint_axes'] = [ax.plot([], [], [], color=color, lw=2, label=name)[0]
                           for color, name in zip(ANGLE_COLORS, data.angle_names)]
    scene['hand_axes'] = [ax.plot([], [], [], color=color, lw=1)[0] for color in ('#c74a45', '#4a9957', '#4176b6')]
    scene['note'] = ax.text2D(.02, .02, '', transform=ax.transAxes, fontsize=9, color='#a55b36')
    ax.legend(loc='upper left', fontsize=7, framealpha=.85)
    points = []
    valid = np.flatnonzero(data.pose_valid)
    for index in valid[np.linspace(0, len(valid) - 1, min(50, len(valid)), dtype=int)]:
        pose = data.pose(index)
        points.extend([pose['torso'], pose['proximal'], pose['upper'], pose['forearm'], pose['palm'], *pose['fingers']])
    points = np.concatenate(points)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(100., float(np.ptp(points, axis=0).max()) * .65)
    scene['limits'] = [(c - radius, c + radius) for c in center]
    reset_camera(scene)
    figure.subplots_adjust(left=0, right=1, bottom=.03, top=.90)
    return scene


def reset_camera(scene):
    ax = scene['ax']
    ax.view_init(elev=22, azim=-65)
    for setter, limits in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), scene['limits']):
        setter(*limits)


def set_line(line, points):
    points = np.asarray(points).reshape(-1, 3)
    line.set_data_3d(points[:, 0], points[:, 1], points[:, 2])


def update_scene(scene, data, index):
    pose = data.pose(index)
    keys = ('torso', 'proximal', 'upper', 'forearm', 'palm', 'band')
    all_lines = [scene[k] for k in keys] + scene['fingers'] + scene['joint_axes'] + scene['hand_axes']
    if pose is None:
        for line in all_lines:
            set_line(line, np.empty((0, 3)))
        scene['note'].set_text('当前时刻缺少完整运动学／骨段位姿；不延续上一姿态')
        return
    for key in keys:
        set_line(scene[key], pose[key])
    for line, points in zip(scene['fingers'], pose['fingers']):
        set_line(line, points)
    for line, center, direction in zip(scene['joint_axes'], pose['centers'], pose['axes']):
        set_line(line, np.stack([center - 25 * direction, center + 25 * direction]))
    wrist = pose['centers'][-1]
    for line, direction in zip(scene['hand_axes'], pose['hand_axes']):
        set_line(line, np.stack([wrist, wrist + 40 * direction]))
    flag = int(np.bitwise_or.reduce(data.full_flags[index]))
    scene['note'].set_text('含质量未通过项；请查看逐角说明' if flag & data.reject_mask else
                          ('含模型／参考依赖 *' if flag & data.prior_mask else '通过当前数值质量检查'))
