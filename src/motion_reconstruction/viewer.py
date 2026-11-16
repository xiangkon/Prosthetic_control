"""Synchronized TUM marker, model-skeleton and 12-angle desktop viewer.

Run from any directory. Defaults to subject 001 / trial 011. Uses the existing
raw-data reader, saved angle results and the exact subject-model snapshot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from .kinematics import forward
from .data_io import load_motion
from .reference_geometry import read_tables
from .model import Model, Q_NAMES
from .paths import PROJECT, OUTPUT_ROOT


DEFAULT_RESULT = OUTPUT_ROOT / 'subj_001/trials/trial_011'
ANGLE_BONES = [0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6]
BONE_COLORS = ['#537bb0', '#8b78ac', '#315e8b', '#3a8977', '#56a5a1', '#bd873b', '#b26152']
TASK_LABELS = [
    '初始放松姿势', 'OK手势', '拇指向下', '欢呼', '搭便车手势', '抬手遮阳', '打招呼',
    '军礼', '停止手势', '指向', '安静手势', '伸手抓小手提箱', '伸手抓玻璃杯',
    '伸手抓电话听筒', '伸手抓书', '伸手抓小杯子', '伸手抓苹果', '伸手抓帽子',
    '伸手抓杯子', '接过托盘', '伸手抓锁中的钥匙', '伸手抓瓶子', '伸手抓智能手机',
    '伸手抓牙刷', '伸手抓笔记本电脑', '伸手抓钢笔', '伸手抓铅笔', '伸手抓茶包',
    '伸手抓门把手', '伸手抓网球', '伸手抓瓶盖',
]
FRAME_NAMES = ['Ulna', 'Radius', 'Carpal', 'Hand']
ANGLE_CHOICES = [f'q{k + 1} {name}' for k, name in enumerate(Q_NAMES)]
ANGLE_CHOICES[8:] = ['q9 肘屈伸', 'q10 前臂旋前 / 旋后', 'q11 腕屈伸', 'q12 腕桡偏 / 尺偏']
# label, model frame, first downstream coordinate, reference bone line
REFERENCE_FRAMES = {
    '肱骨 Humerus': ('humerus', 8, 2),
    '尺骨 Ulna': ('ulna', 9, 3),
    '桡骨 Radius': ('radius', 10, 4),
    '腕骨 Carpal': ('carpal', 11, 5),
    '胸廓 Thorax': ('thorax', 0, 0),
    '锁骨 Clavicle': ('clavicle', 2, 0),
    '肩胛骨 Scapula': ('scapula', 5, 1),
    '手 Hand（反向观察）': ('hand', 12, 6),
}
DISTAL_COLORS = ['#287b65', '#168eac', '#b47a19', '#ad426f']
QUALITY_NAMES = {
    1: '观测或躯干位姿缺失',
    2: 'IK未收敛',
    4: '位置残差较大',
    8: '角度弱可观测',
    16: '肩胛接触误差',
    32: '角度边界或裁剪',
    64: '绝对坐标依赖先验',
    128: '标定验证未通过',
}


def enable_dpi_awareness():
    if sys.platform == 'win32':
        import ctypes
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except AttributeError:
            ctypes.windll.user32.SetProcessDPIAware()


def quality_text(flag):
    flag = int(flag)
    if flag == 0:
        return '通过当前数值质量检查'
    return '、'.join(text for bit, text in QUALITY_NAMES.items() if flag & bit)


def nearest_frames(source_time, target_time, tolerance=1e-8):
    right = np.clip(np.searchsorted(source_time, target_time), 0, len(source_time) - 1)
    left = np.maximum(right - 1, 0)
    use_left = abs(source_time[left] - target_time) < abs(source_time[right] - target_time)
    indices = np.where(use_left, left, right)
    if np.any(abs(source_time[indices] - target_time) > tolerance):
        raise ValueError('原始轨迹与角度结果时间轴不匹配。请选择对应的结果目录。')
    return indices


class ReconstructionData:
    def __init__(self, result_dir, motion_path=None, model_path=None):
        self.directory = Path(result_dir).resolve()
        self.report = json.loads((self.directory / 'report.json').read_text(encoding='utf-8'))
        if motion_path is None:
            motion_path = Path(self.report['source'])
            if not motion_path.is_absolute():
                motion_path = PROJECT / motion_path
        if model_path is None:
            model_path = self.directory / self.report.get('model_file', '../../estimated_model.json')
        self.model_path = Path(model_path).resolve()
        self.model = Model.load(self.model_path, allow_estimated=True)
        expected_hash = self.report.get('model_sha256')
        actual_hash = hashlib.sha256(json.dumps(self.model.config, sort_keys=True).encode()).hexdigest()
        if expected_hash and expected_hash != actual_hash:
            raise ValueError('模型文件与该次角度结果的模型哈希不一致。')
        self.motion_path = Path(motion_path).resolve()
        raw = load_motion(self.motion_path)
        self.methods = {}
        self.names = raw.names
        self.time = None
        self.marker_indices = None
        for method in ('ekf', 'ik'):
            path = self.directory / f'{method}_angles.npz'
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as archive:
                t = archive['t'].copy()
                q = archive['q'].astype(float)
                poses = archive['T_world_thorax'].astype(float)
                flags = archive['angle_quality_flag'].copy()
                if str(archive['angle_unit'].item()) != 'rad':
                    raise ValueError('角度文件应声明单位 rad。')
                if q.shape != (len(t), 12) or poses.shape != (len(t), 4, 4):
                    raise ValueError('角度或躯干位姿数组形状错误。')
                if flags.shape != q.shape:
                    raise ValueError('逐角度质量标记形状错误。')
                if self.time is None:
                    self.time = t
                    self.marker_indices = nearest_frames(raw.t, t)
                elif t.shape != self.time.shape or not np.allclose(t, self.time, rtol=0, atol=1e-9):
                    raise ValueError('IK与滤波结果的时间轴不一致。')
                rms = archive['marker_rms_m'].copy() * 1000
            self.methods[method] = self.build_method(q, poses, flags, rms)
        if not self.methods:
            raise ValueError('结果目录中没有 ik_angles.npz 或 ekf_angles.npz。')
        self.measured_mm = raw.p[self.marker_indices] * 1000
        self.bound_names = list(self.model.bindings)
        self.bound_to_raw = np.array([self.names.index(name) for name in self.bound_names], dtype=int)
        self.raw_to_bound = {self.names.index(name): i for i, name in enumerate(self.bound_names)}
        self.main_indices = [i for i, name in enumerate(self.names) if '_h' not in name]
        self.active_limits = self.find_limits()

    def segment_scale(self, name):
        calibration = self.model.config.get('calibration', {})
        initial = calibration.get('initial_segment_scales', {}).get(name, 1.0)
        refinement = calibration.get('scale_refinement', {}).get(name, 1.0)
        return initial * refinement

    def action_text(self):
        task = int(self.report['task'])
        chinese = TASK_LABELS[task] if 0 <= task < len(TASK_LABELS) else self.report['description']
        return f'动作 {task} · {chinese} / {self.report["description"]}'

    def build_method(self, q, poses, flags, rms):
        valid = np.isfinite(q).all(axis=1) & np.isfinite(poses).all(axis=(1, 2))
        safe_q = q.copy()
        safe_q[~valid] = self.model.q0
        safe_poses = poses.copy()
        safe_poses[~valid] = np.eye(4)
        predicted, jacobian, frames, axes, centers = forward(self.model, safe_q, safe_poses)
        del jacobian
        joints = []
        for joint in self.model.joints:
            parent = frames[joint['parent']]
            point = np.array(joint['center_parent_m'])
            joints.append(np.einsum('nij,j->ni', parent[:, :3, :3], point) + parent[:, :3, 3])
        joints = np.stack(joints, axis=1)

        ulna_tip_local = np.array([0, 0, -.25711]) * self.segment_scale('ulna')
        ulna_frame = frames['ulna']
        ulna_tip = np.einsum('nij,j->ni', ulna_frame[:, :3, :3], ulna_tip_local) + ulna_frame[:, :3, 3]
        features = read_tables()['table_II']
        scapula_points = {}
        for name in ('superior_medial_border', 'inferior_medial_border'):
            feature = next(f for f in features if f['segment'] == 'scapula' and f['name'] == name)
            local = np.array(feature['in_segment']) * .001 * self.segment_scale('scapula')
            T = frames['scapula']
            scapula_points[name] = np.einsum('nij,j->ni', T[:, :3, :3], local) + T[:, :3, 3]
        bones = [
            np.stack([joints[:, 0], joints[:, 1]], axis=1),
            np.stack([joints[:, 1], scapula_points['superior_medial_border'],
                      scapula_points['inferior_medial_border'], joints[:, 2], joints[:, 1]], axis=1),
            np.stack([joints[:, 2], joints[:, 3]], axis=1),
            np.stack([joints[:, 3], ulna_tip], axis=1),
            np.stack([joints[:, 4], joints[:, 5]], axis=1),
            np.stack([joints[:, 5], joints[:, 6]], axis=1),
        ]
        # Fixed open-hand drawing in the G99 hand frame. +Y is palm normal,
        # +Z points proximally, so fingers extend along -Z.
        scale = self.segment_scale('hand')
        palm_local = np.array([[-.027, 0, -.012], [.027, 0, -.012],
                               [.038, 0, -.085], [-.034, 0, -.085]]) * scale
        finger_local = []
        for x, length in [(-.028, .058), (-.009, .080), (.012, .087), (.032, .075)]:
            finger_local.append(np.array([[x, 0, -.080], [x * 1.12, 0, -.080 - length * .48],
                                          [x * 1.2, 0, -.080 - length * .78],
                                          [x * 1.25, 0, -.080 - length]]) * scale)
        finger_local.append(np.array([[.025, 0, -.028], [.052, -.005, -.047],
                                      [.069, -.008, -.067], [.079, -.010, -.082]]) * scale)
        hand_frame = frames['hand']
        def hand_world(local):
            return np.einsum('nij,kj->nki', hand_frame[:, :3, :3], local) + hand_frame[:, None, :3, 3]
        palm = hand_world(palm_local)
        bones.append(np.concatenate([palm, palm[:, :1]], axis=1))
        fingers = [hand_world(local) for local in finger_local]
        hand_description = '手掌和五指为固定张手示意，随手骨段运动；手指姿态未重建。'
        frame_origins = np.stack([joints[:, 3], (joints[:, 4] + joints[:, 5]) / 2,
                                  joints[:, 5], joints[:, 6]], axis=1)
        frame_rotations = np.stack([frames[name][:, :3, :3]
                                    for name in ('ulna', 'radius', 'carpal', 'hand')], axis=1)
        arrays = [predicted, joints, axes, centers, palm, frame_origins, frame_rotations, *frames.values(), *fingers, *bones]
        for array in arrays:
            array[~valid] = np.nan
        return dict(q=q, degrees=np.rad2deg(q), poses=poses, flags=flags, rms_mm=rms,
                    predicted_mm=predicted * 1000, joints_mm=joints * 1000,
                    axes=axes, centers_mm=centers * 1000,
                    bones_mm=[bone * 1000 for bone in bones], hand_description=hand_description,
                    palm_mm=palm * 1000, fingers_mm=[f * 1000 for f in fingers],
                    frame_origins_mm=frame_origins * 1000, frame_rotations=frame_rotations,
                    segment_frames=frames)

    def find_limits(self):
        points = [self.measured_mm.reshape(-1, 3)]
        for method in self.methods.values():
            points.extend(bone.reshape(-1, 3) for bone in method['bones_mm'])
            points.extend(finger.reshape(-1, 3) for finger in method['fingers_mm'])
        finite = np.concatenate(points)
        finite = finite[np.isfinite(finite).all(axis=1)]
        low, high = finite.min(axis=0), finite.max(axis=0)
        center = (low + high) / 2
        radius = max(float(np.max(high - low)) * .57, 100)
        return center, radius


def segment_local_view(result, segment):
    """Express all drawing geometry in the selected bone's own frame."""
    reference = result['segment_frames'][segment]
    rotation = reference[:, :3, :3]
    inverse = np.linalg.inv(rotation)
    origin = reference[:, :3, 3] * 1000
    def transform_points(points):
        offset = origin.reshape((len(origin),) + (1,) * (points.ndim - 2) + (3,))
        return np.einsum('nij,n...j->n...i', inverse, points - offset)
    local = dict(result)
    for name in ('predicted_mm', 'joints_mm', 'centers_mm', 'palm_mm', 'frame_origins_mm'):
        local[name] = transform_points(result[name])
    for name in ('bones_mm', 'fingers_mm'):
        local[name] = [transform_points(points) for points in result[name]]
    local['axes'] = np.einsum('nij,nkj->nki', inverse, result['axes'])
    local['frame_rotations'] = np.einsum('nij,najk->naik', inverse, result['frame_rotations'])
    world_to_reference = np.linalg.inv(reference)
    local['segment_frames'] = {name: world_to_reference @ frame for name, frame in result['segment_frames'].items()}
    return local


def configure_3d(ax, data):
    center, radius = data.active_limits
    for dimension, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(center[dimension] - radius, center[dimension] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel='World X (mm)', ylabel='World Y (mm)', zlabel='World Z (mm)')
    ax.view_init(elev=22, azim=-65)
    ax.tick_params(labelsize=8)


def create_3d_scene(ax, data):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    configure_3d(ax, data)
    measured = ax.plot([], [], [], 'o', color='#477b9d', ms=4, alpha=.8, label='Vicon markers')[0]
    predicted = ax.plot([], [], [], 'o', color='#d58429', ms=5, mfc='none', mew=1,
                        label='FK marker predictions')[0]
    joints = ax.plot([], [], [], 'o', color='#233647', ms=4, label='Model joint centers')[0]
    bones = [ax.plot([], [], [], lw=2.6, color=color)[0] for color in BONE_COLORS]
    palm = Poly3DCollection([], facecolors='#e1ab81', edgecolors='none', alpha=.45)
    ax.add_collection3d(palm, autolim=False)
    fingers = [ax.plot([], [], [], color='#985a39', lw=2, marker='o', ms=2.5)[0] for _ in range(5)]
    metacarpals = Line3DCollection([], colors='#985a39', linewidths=1, alpha=.8)
    ax.add_collection3d(metacarpals, autolim=False)
    triads = []
    for index in range(4):
        lines = [ax.plot([], [], [], color=color, lw=1.5, ls='--' if index == 2 else '-')[0]
                 for color in ('#d44840', '#389447', '#386ed0')]
        texts = [ax.text(0, 0, 0, '', fontsize=6.5) for _ in range(3)]
        triads.append((lines, texts))
    joint_axes = [ax.plot([], [], [], color=DISTAL_COLORS[k - 8] if k >= 8 else '#537bb0', lw=1.8, ls='--')[0]
                  for k in range(12)]
    joint_axis_texts = [ax.text(0, 0, 0, '', fontsize=7, color=line.get_color()) for line in joint_axes]
    frame_key = ax.text2D(.02, .02, 'Local frames  X red · Y green · Z blue\nOpen hand schematic · fixed fingers',
                          transform=ax.transAxes, fontsize=7, color='#4c5560')
    residuals = Line3DCollection([], colors='#bc5b39', linewidths=.7, alpha=.6)
    ax.add_collection3d(residuals, autolim=False)
    axis = ax.plot([], [], [], color='#b0478b', lw=2, ls='--')[0]
    axis_label = ax.text(0, 0, 0, '', fontsize=8, color='#963571')
    selected = ax.plot([], [], [], 'o', color='#d03f3b', ms=8, mfc='none', mew=1.8)[0]
    trail = ax.plot([], [], [], color='#bd4d49', alpha=.65, lw=1)[0]
    labels = [ax.text(0, 0, 0, '', fontsize=7, color='#29465b') for _ in data.names]
    ax.legend(loc='upper left', fontsize=7, framealpha=.8)
    return dict(measured=measured, predicted=predicted, joints=joints, bones=bones,
                residuals=residuals, axis=axis, axis_label=axis_label, selected=selected,
                trail=trail, labels=labels, palm=palm, fingers=fingers, metacarpals=metacarpals, triads=triads,
                frame_key=frame_key, joint_axes=joint_axes, joint_axis_texts=joint_axis_texts)


def set_line(line, points):
    points = np.asarray(points).reshape(-1, 3)
    line.set_data_3d(points[:, 0], points[:, 1], points[:, 2])


def update_3d(scene, data, method, frame, chosen, selected_marker, focus_angle=8,
              show_model=True, show_predicted=True, show_residuals=False,
              label_mode='main', show_axis=True, show_trail=True, show_frames=True, show_distal=True,
              reference_segment=None, first_bone=0, angle_indices=None, result=None):
    if result is None:
        result = data.methods[method]
    local = reference_segment is not None
    measured = data.measured_mm[frame]
    chosen = np.asarray(chosen, dtype=int)
    set_line(scene['measured'], measured[chosen])
    predicted = result['predicted_mm'][frame]
    bound_visible = np.isin(data.bound_to_raw, chosen)
    set_line(scene['predicted'], predicted[bound_visible])
    scene['predicted'].set_visible(show_predicted)
    set_line(scene['joints'], result['joints_mm'][frame, first_bone:])
    scene['joints'].set_visible(show_model)
    focused_bone = ANGLE_BONES[focus_angle]
    for k, (line, points) in enumerate(zip(scene['bones'], result['bones_mm'])):
        set_line(line, points[frame])
        line.set_visible(show_model and k >= first_bone)
        line.set_alpha(.5 if local and k == first_bone else 1)
        line.set_linewidth(4 if k == focused_bone else 2.6)
    palm = result['palm_mm'][frame]
    scene['palm'].set_verts([palm] if np.isfinite(palm).all() else [])
    scene['palm'].set_visible(show_model)
    wrist = result['joints_mm'][frame, 6]
    scene['metacarpals'].set_segments([[wrist, finger[frame, 0]] for finger in result['fingers_mm']]
                                       if np.isfinite(wrist).all() else [])
    scene['metacarpals'].set_visible(show_model)
    for line, points in zip(scene['fingers'], result['fingers_mm']):
        set_line(line, points[frame])
        line.set_visible(show_model)
        line.set_linewidth(3 if focus_angle >= 10 else 2)
    for index, (lines, texts) in enumerate(scene['triads']):
        origin = result['frame_origins_mm'][frame, index]
        rotation = result['frame_rotations'][frame, index]
        length = [38, 38, 48, 30][index]
        visible = show_frames and np.isfinite(origin).all() and np.isfinite(rotation).all()
        if local:
            origin, rotation, length = np.zeros(3), np.eye(3), 60
            visible = index == 0
        for dimension, (line, text) in enumerate(zip(lines, texts)):
            tip = origin + rotation[:, dimension] * length
            set_line(line, [origin, tip])
            line.set_visible(visible)
            text.set_visible(visible)
            if visible:
                text.set_position((tip[0], tip[1]))
                text.set_3d_properties(tip[2], zdir=None)
                label = f'{FRAME_NAMES[index]} Z' if dimension == 2 else 'XY'[dimension]
                text.set_text('Fixed ' + 'XYZ'[dimension] if local else label)
    visible_angles = []
    if local and show_axis:
        visible_angles = angle_indices if angle_indices is not None else []
    elif not local and show_distal:
        visible_angles = range(8, 12)
    for angle, (line, text) in enumerate(zip(scene['joint_axes'], scene['joint_axis_texts'])):
        center = result['centers_mm'][frame, angle]
        direction = result['axes'][frame, angle]
        visible = angle in visible_angles and np.isfinite(center).all() and np.isfinite(direction).all()
        reach = 60 + angle % 3 * 12 if local else 55 + (angle - 8) * 9
        tip = center + direction * reach
        set_line(line, [center - direction * 20, tip])
        line.set_visible(visible)
        line.set_linewidth(3 if local and angle == focus_angle else 1.8)
        text.set_visible(visible)
        if visible:
            text.set_position((tip[0], tip[1]))
            text.set_3d_properties(tip[2], zdir=None)
            text.set_text(f'q{angle + 1}' if local else f'q{angle + 1} {result["degrees"][frame, angle]:.1f}°')
    scene['frame_key'].set_visible(not local and (show_frames or show_model))
    ax = scene['measured'].axes
    ax.get_legend().set_visible(not local)
    prefix = reference_segment.capitalize() if local else 'World'
    ax.set(xlabel=prefix + ' X (mm)', ylabel=prefix + ' Y (mm)', zlabel=prefix + ' Z (mm)')
    segments = []
    if show_residuals:
        for k, raw_index in enumerate(data.bound_to_raw):
            if bound_visible[k] and np.isfinite(measured[raw_index]).all() and np.isfinite(predicted[k]).all():
                segments.append([measured[raw_index], predicted[k]])
    scene['residuals'].set_segments(segments)
    axis = result['axes'][frame, focus_angle]
    center = result['centers_mm'][frame, focus_angle]
    axis_finite = np.isfinite(center).all() and np.isfinite(axis).all()
    scene['axis'].set_visible(show_axis and not local and axis_finite)
    scene['axis_label'].set_visible(show_axis and not local and axis_finite)
    if axis_finite:
        set_line(scene['axis'], [center - axis * 40, center + axis * 40])
        tip = center + axis * 48
        scene['axis_label'].set_position((tip[0], tip[1]))
        scene['axis_label'].set_3d_properties(tip[2], zdir=None)
        scene['axis_label'].set_text(f'+q{focus_angle + 1}')
    selected_point = measured[selected_marker]
    set_line(scene['selected'], [selected_point])
    scene['selected'].set_visible(selected_marker in chosen)
    begin = np.searchsorted(data.time, data.time[frame] - .5)
    set_line(scene['trail'], data.measured_mm[begin:frame + 1, selected_marker])
    scene['trail'].set_visible(show_trail and selected_marker in chosen)
    chosen_set = set(chosen.tolist())
    for k, text in enumerate(scene['labels']):
        visible = k in chosen_set and np.isfinite(measured[k]).all()
        visible &= label_mode == 'all' or (label_mode == 'main' and k in data.main_indices)
        text.set_visible(visible)
        if visible:
            point = measured[k] + [4, 4, 4]
            text.set_position((point[0], point[1]))
            text.set_3d_properties(point[2], zdir=None)
            text.set_text(data.names[k])


def build_angle_axes(figure, grid, data, active_method, animated, indices=None, active_only=False):
    indices = list(range(12)) if indices is None else indices
    single = len(indices) == 1
    axes, artists = [], []
    for position, k in enumerate(indices):
        ax = figure.add_subplot(grid[position])
        for method, result in data.methods.items():
            if (single or active_only) and method != active_method:
                continue
            is_active = method == active_method
            ax.plot(data.time, result['degrees'][:, k], color='#2d668a' if is_active else '#a5acb4',
                    lw=1.1 if is_active else .75, alpha=1 if is_active else .65,
                    label='Filter' if method == 'ekf' else 'IK')
        prior = np.any((data.methods[active_method]['flags'][:, k] & 64) != 0)
        title = f'q{k + 1} {Q_NAMES[k]}' + (' *' if prior else '')
        ax.set_title(title, color='#aa672a' if prior else '#253746', fontsize=14 if single else 9, loc='left', pad=3)
        ax.tick_params(labelsize=10 if single else 7, pad=2)
        ax.grid(alpha=.16)
        ax.set_xlim(data.time[0], data.time[-1])
        if position % grid.ncols == 0:
            ax.set_ylabel('deg', fontsize=8)
        if position >= len(indices) - grid.ncols:
            ax.set_xlabel('Time (s)', fontsize=8)
        else:
            ax.tick_params(labelbottom=False)
        cursor = ax.axvline(data.time[0], color='#b44842', lw=.8, animated=animated)
        dot, = ax.plot([], [], 'o', color='#b44842', ms=3, animated=animated)
        value = ax.text(.98, .95, '', transform=ax.transAxes, ha='right', va='top', fontsize=20 if single else 8,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.75, pad=1), animated=animated)
        axes.append(ax)
        artists.append((cursor, dot, value))
    axes[0].legend(fontsize=6, loc='lower left')
    return axes, artists


def update_angles(artists, data, method, frame, indices=None):
    indices = range(12) if indices is None else indices
    result = data.methods[method]
    t = data.time[frame]
    for k, (cursor, dot, value) in zip(indices, artists):
        angle = result['degrees'][frame, k]
        flag = int(result['flags'][frame, k])
        cursor.set_xdata([t, t])
        dot.set_data([t], [angle])
        value.set_text(f'{angle:+.1f}°' if np.isfinite(angle) else 'missing')
        color = '#aa672a' if flag & 64 else '#254965'
        if flag & ~64:
            color = '#b33c35'
        value.set_color(color)


class ReconstructionViewer:
    def __init__(self, root, data, method='ekf'):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        self.tk, self.root = tk, root
        self.data = data
        self.method = method if method in data.methods else next(iter(data.methods))
        self.frame = 0
        self.playhead = float(data.time[0])
        self.playing = False
        self.last_tick = time.perf_counter()
        self.updating_slider = False
        self.focus_angle = 10
        self.angle_background = None
        self.wrist_radius = 180
        self.display_result = data.methods[self.method]
        self.camera_mode = 'local'
        self.local_view_key = None
        root.title('TUM 标记点、骨架与关节角核验')
        width = min(1600, root.winfo_screenwidth() - 80)
        height = min(1000, root.winfo_screenheight() - 100)
        root.geometry(f'{width}x{height}+40+40')
        root.minsize(min(1150, width), min(760, height))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        style = ttk.Style(root)
        style.configure('Header.TLabel', font=('Microsoft YaHei UI', 12, 'bold'))
        header = ttk.Frame(root, padding=(12, 8))
        header.grid(row=0, column=0, sticky='ew')
        identity = ttk.Frame(header)
        identity.pack(fill='x')
        ttk.Label(identity, text='Vicon 重建核验', style='Header.TLabel').pack(side='left')
        self.file_label = tk.StringVar(value=self.description())
        ttk.Label(identity, textvariable=self.file_label).pack(side='left', padx=16)
        ttk.Button(identity, text='打开结果目录', command=self.open_result).pack(side='right')
        self.action_label = tk.StringVar(value=data.action_text())
        ttk.Label(header, textvariable=self.action_label, style='Header.TLabel').pack(anchor='w', pady=(4, 0))
        observation = ttk.Frame(header)
        observation.pack(fill='x', pady=(5, 0))
        self.observation = tk.StringVar(value='局部坐标系观察')
        self.reference_choice = tk.StringVar(value='桡骨 Radius')
        for label, variable, values, width in [('观察方式', self.observation, ['全部自由度', '局部坐标系观察'], 18),
                                               ('固定参考坐标系', self.reference_choice, list(REFERENCE_FRAMES), 27)]:
            ttk.Label(observation, text=label).pack(side='left', padx=(0, 4))
            control = ttk.Combobox(observation, textvariable=variable, values=values, state='readonly', width=width)
            control.pack(side='left', padx=(0, 6))
            control.bind('<<ComboboxSelected>>', self.change_observation)
        ttk.Label(observation, text='选择固定参考骨段，自动显示下游关节角').pack(side='left')
        self.mode_note = tk.StringVar()
        ttk.Label(header, textvariable=self.mode_note).pack(anchor='w', pady=(3, 0))
        content = ttk.Frame(root)
        content.grid(row=1, column=0, sticky='nsew')
        sidebar_shell = ttk.Frame(content, width=240)
        sidebar_shell.pack(side='left', fill='y')
        sidebar_shell.pack_propagate(False)
        self.sidebar_canvas = tk.Canvas(sidebar_shell, width=220, highlightthickness=0)
        sidebar_scroll = ttk.Scrollbar(sidebar_shell, orient='vertical', command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar_scroll.pack(side='right', fill='y')
        self.sidebar_canvas.pack(side='left', fill='both', expand=True)
        sidebar = ttk.Frame(self.sidebar_canvas, padding=10)
        sidebar_window = self.sidebar_canvas.create_window((0, 0), window=sidebar, anchor='nw')
        sidebar.bind('<Configure>', lambda event: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox('all')))
        self.sidebar_canvas.bind('<Configure>', lambda event: self.sidebar_canvas.itemconfigure(sidebar_window, width=event.width))
        ttk.Label(sidebar, text='显示原始标记点（多选）').pack(anchor='w')
        box = ttk.Frame(sidebar)
        box.pack(fill='x', pady=4)
        self.marker_list = tk.Listbox(box, selectmode=tk.MULTIPLE, exportselection=False, height=12, width=24)
        scrollbar = ttk.Scrollbar(box, orient='vertical', command=self.marker_list.yview)
        self.marker_list.configure(yscrollcommand=scrollbar.set)
        self.marker_list.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        for name in data.names:
            self.marker_list.insert(tk.END, name)
        self.marker_list.select_set(0, tk.END)
        self.marker_list.bind('<<ListboxSelect>>', lambda event: self.draw())
        buttons = ttk.Frame(sidebar)
        buttons.pack(fill='x', pady=3)
        for label, choice in [('全部', 'all'), ('主点', 'main'), ('清空', 'none')]:
            ttk.Button(buttons, text=label, width=5, command=lambda mode=choice: self.select_markers(mode)).pack(side='left', padx=1)
        self.show_model = tk.BooleanVar(value=True)
        self.show_predicted = tk.BooleanVar(value=True)
        self.show_residuals = tk.BooleanVar(value=False)
        self.show_axis = tk.BooleanVar(value=True)
        self.show_trail = tk.BooleanVar(value=True)
        self.show_frames = tk.BooleanVar(value=True)
        self.show_distal = tk.BooleanVar(value=True)
        for text, variable in [('模型骨架与关节中心', self.show_model), ('FK预测标记点', self.show_predicted),
                               ('显示位置误差连线', self.show_residuals), ('关节旋转轴', self.show_axis),
                               ('肘 / 前臂 / 腕局部坐标系', self.show_frames),
                               ('q9–q12旋转轴与角度', self.show_distal), ('检查点0.5秒轨迹', self.show_trail)]:
            ttk.Checkbutton(sidebar, text=text, variable=variable, command=self.draw).pack(anchor='w', pady=1)
        ttk.Label(sidebar, text='局部坐标轴 X红 / Y绿 / Z蓝\nUlna尺骨 / Radius桡骨\nCarpal腕骨 / Hand手\n腕部虚线为腕骨系，实线为手系\nq9肘屈伸 / q10旋前旋后\nq11腕屈伸 / q12腕桡尺偏\n五指保持固定张手示意',
                  wraplength=195, justify='left').pack(anchor='w', pady=5)
        ttk.Label(sidebar, text='点名显示').pack(anchor='w', pady=(6, 1))
        self.labels = tk.StringVar(value='主点')
        labels = ttk.Combobox(sidebar, textvariable=self.labels, values=['隐藏', '主点', '全部'], state='readonly', width=12)
        labels.pack(anchor='w')
        labels.bind('<<ComboboxSelected>>', lambda event: self.draw())
        ttk.Label(sidebar, text='检查单个标记点').pack(anchor='w', pady=(7, 1))
        self.inspected_marker = tk.StringVar(value='m_elb' if 'm_elb' in data.names else data.names[0])
        self.inspected_box = ttk.Combobox(sidebar, textvariable=self.inspected_marker, values=data.names, state='readonly', width=18)
        self.inspected_box.pack(anchor='w')
        self.inspected_box.bind('<<ComboboxSelected>>', lambda event: self.draw())
        self.inspector = tk.StringVar()
        ttk.Label(sidebar, textvariable=self.inspector, wraplength=180, justify='left').pack(anchor='w', pady=6)
        ttk.Label(sidebar, text='三维视角').pack(anchor='w', pady=(4, 1))
        views = ttk.Frame(sidebar)
        views.pack(fill='x')
        for text, elevation, azimuth in [('斜视', 22, -65), ('沿X', 0, 0), ('沿Y', 0, 90), ('沿Z', 90, -90)]:
            ttk.Button(views, text=text, width=3,
                       command=lambda e=elevation, a=azimuth: self.change_view(e, a)).pack(side='left')
        framing = ttk.Frame(sidebar)
        framing.pack(fill='x', pady=4)
        ttk.Button(framing, text='当前姿态', width=8, command=self.fit_current).pack(side='left')
        ttk.Button(framing, text='全程范围', width=8, command=lambda: self.change_view(22, -65)).pack(side='left')
        ttk.Button(sidebar, text='腕部近景（跟随运动）', command=self.focus_wrist).pack(anchor='w', pady=3)
        self.quality_label = tk.StringVar()
        ttk.Label(sidebar, textvariable=self.quality_label, wraplength=180, justify='left',
                  foreground='#9a5e27').pack(anchor='w', pady=(10, 0))

        split = ttk.Panedwindow(content, orient='horizontal')
        split.pack(side='left', fill='both', expand=True)
        left = ttk.Frame(split)
        right = ttk.Frame(split)
        split.add(left, weight=4)
        split.add(right, weight=6)
        self.figure3d = Figure(figsize=(6, 7), dpi=100)
        self.figure3d.subplots_adjust(left=.01, right=.92, bottom=.07, top=.94)
        self.ax3d = self.figure3d.add_subplot(111, projection='3d')
        self.scene = create_3d_scene(self.ax3d, data)
        self.refresh_display_result()
        self.set_local_camera()
        self.canvas3d = FigureCanvasTkAgg(self.figure3d, master=left)
        self.canvas3d.mpl_connect('scroll_event', self.zoom3d)
        self.toolbar3d = NavigationToolbar2Tk(self.canvas3d, left, pack_toolbar=False)
        self.toolbar3d.pack(side='bottom', fill='x')
        self.canvas3d.get_tk_widget().pack(side='top', fill='both', expand=True)
        self.toolbar3d.update()
        self.figure_angles = Figure(figsize=(8, 7), dpi=100)
        self.canvas_angles = FigureCanvasTkAgg(self.figure_angles, master=right)
        self.canvas_angles.get_tk_widget().pack(fill='both', expand=True)
        self.canvas_angles.mpl_connect('draw_event', self.cache_angle_background)
        self.canvas_angles.mpl_connect('button_press_event', self.click_angle)
        self.build_angles()

        transport = ttk.Frame(root, padding=(10, 4))
        transport.grid(row=2, column=0, sticky='ew')
        self.play_button = ttk.Button(transport, text='播放', command=self.toggle)
        self.play_button.pack(side='left')
        ttk.Button(transport, text='上一帧', command=lambda: self.step(-1)).pack(side='left', padx=2)
        ttk.Button(transport, text='下一帧', command=lambda: self.step(1)).pack(side='left', padx=2)
        ttk.Button(transport, text='起点', command=self.rewind).pack(side='left', padx=2)
        ttk.Label(transport, text='速度').pack(side='left', padx=(10, 3))
        self.speed = tk.StringVar(value='1')
        ttk.Combobox(transport, textvariable=self.speed, values=['0.1', '0.25', '0.5', '1', '2', '4'],
                     state='readonly', width=5).pack(side='left')
        self.loop = tk.BooleanVar(value=True)
        ttk.Checkbutton(transport, text='循环', variable=self.loop).pack(side='left', padx=8)
        ttk.Label(transport, text='骨架采用').pack(side='left', padx=(8, 3))
        self.method_choice = tk.StringVar(value='滤波 EKF' if self.method == 'ekf' else '原始 IK')
        choices = ['滤波 EKF' if name == 'ekf' else '原始 IK' for name in data.methods]
        self.method_box = ttk.Combobox(transport, textvariable=self.method_choice, values=choices, state='readonly', width=10)
        self.method_box.pack(side='left')
        self.method_box.bind('<<ComboboxSelected>>', self.change_method)
        ttk.Label(transport, text='定位帧').pack(side='left', padx=(12, 3))
        self.frame_entry = tk.StringVar(value='1')
        self.frame_box = ttk.Spinbox(transport, from_=1, to=len(data.time), textvariable=self.frame_entry,
                                     width=7, command=self.go_to_entry)
        self.frame_box.pack(side='left')
        self.frame_box.bind('<Return>', lambda event: self.go_to_entry())
        self.position = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(root, variable=self.position, from_=0, to=len(data.time) - 1, command=self.seek)
        self.slider.grid(row=3, column=0, sticky='ew', padx=12)
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, padding=(12, 4)).grid(row=4, column=0, sticky='ew')
        ttk.Label(root, text='拖动三维图旋转，滚轮/工具栏缩放。点击右侧曲线选择关节，点击曲线时间位置可同步定位。* 表示先验依赖。',
                  padding=(12, 0, 12, 7)).grid(row=5, column=0, sticky='ew')
        root.bind('<space>', self.space)
        root.bind('<Left>', lambda event: self.step(-1))
        root.bind('<Right>', lambda event: self.step(1))
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.closed = False
        self.draw()
        self.timer = root.after(33, self.tick)
        root.after(100, self.synchronize_canvas_sizes)

    def synchronize_canvas_sizes(self):
        # Tk can keep a canvas at the same allocated size after a high-DPI Map
        # event. Force a resize using its actual pixel size in that case.
        if self.closed:
            return
        from types import SimpleNamespace
        for canvas in (self.canvas3d, self.canvas_angles):
            widget = canvas.get_tk_widget()
            width, height = widget.winfo_width(), widget.winfo_height()
            if width > 10 and height > 10:
                canvas.resize(SimpleNamespace(width=width, height=height))

    def description(self):
        return f'{self.data.model.config.get("subject", "")} / trial {self.data.report["trial"]:03d}  ·  {self.data.report["description"]}'

    def build_angles(self):
        self.angle_background = None
        self.figure_angles.clear()
        local = self.observation.get() == '局部坐标系观察'
        _, first_angle, _ = REFERENCE_FRAMES[self.reference_choice.get()]
        self.angle_indices = list(range(8, 12)) if local and first_angle == 12 else list(range(first_angle if local else 0, 12))
        columns = 1 if len(self.angle_indices) <= 3 else 2 if len(self.angle_indices) <= 4 else 3
        rows = (len(self.angle_indices) + columns - 1) // columns
        grid = self.figure_angles.add_gridspec(rows, columns,
                                               left=.10, right=.98, bottom=.09, top=.90,
                                               hspace=.40, wspace=.33)
        self.angle_axes, self.angle_artists = build_angle_axes(self.figure_angles, grid, self.data, self.method, True,
                                                             self.angle_indices, active_only=local)
        self.figure_angles.suptitle('Joint angles (deg)   |   * prior dependent', fontsize=11, y=.992)
        update_angles(self.angle_artists, self.data, self.method, self.frame, self.angle_indices)
        self.canvas_angles.draw()

    def refresh_display_result(self):
        self.display_result = self.data.methods[self.method]
        if self.observation.get() == '局部坐标系观察':
            segment, _, _ = REFERENCE_FRAMES[self.reference_choice.get()]
            key = (id(self.data), self.method, segment)
            if key != self.local_view_key:
                self.local_result = segment_local_view(self.display_result, segment)
                self.local_view_key = key
            self.display_result = self.local_result

    def change_observation(self, event=None):
        if event is not None and str(event.widget.cget('textvariable')) != str(self.observation):
            self.observation.set('局部坐标系观察')
        _, first_angle, _ = REFERENCE_FRAMES[self.reference_choice.get()]
        self.focus_angle = first_angle if first_angle < 12 else 8
        self.refresh_display_result()
        self.camera_mode = 'local' if self.observation.get() == '局部坐标系观察' else 'manual'
        if self.camera_mode == 'local':
            self.set_local_camera()
        else:
            configure_3d(self.ax3d, self.data)
        self.build_angles()
        self.draw()

    def set_local_camera(self):
        segment, _, first_bone = REFERENCE_FRAMES[self.reference_choice.get()]
        first_bone = 2 if segment == 'hand' else first_bone
        result = self.display_result
        points = [bone.reshape(-1, 3) for bone in result['bones_mm'][first_bone:]]
        points.extend(finger.reshape(-1, 3) for finger in result['fingers_mm'])
        points.append(np.zeros((1, 3)))
        finite = np.concatenate(points)
        finite = finite[np.isfinite(finite).all(axis=1)]
        low, high = finite.min(axis=0), finite.max(axis=0)
        self.local_center = (low + high) / 2
        self.local_radius = max(float(np.max(high - low)) * .62, 100)

    def cache_angle_background(self, event):
        self.angle_background = self.canvas_angles.copy_from_bbox(self.figure_angles.bbox)
        self.blit_angles()

    def blit_angles(self):
        if self.angle_background is None:
            return
        self.canvas_angles.restore_region(self.angle_background)
        for ax, artists in zip(self.angle_axes, self.angle_artists):
            for artist in artists:
                ax.draw_artist(artist)
        self.canvas_angles.blit(self.figure_angles.bbox)

    def draw(self):
        if not hasattr(self, 'status'):
            return
        inspected = self.data.names.index(self.inspected_marker.get())
        local = self.observation.get() == '局部坐标系观察'
        segment, _, first_bone = REFERENCE_FRAMES[self.reference_choice.get()]
        first_bone = 2 if segment == 'hand' else first_bone
        label_mode = {'隐藏': 'none', '主点': 'main', '全部': 'all'}[self.labels.get()]
        chosen = [] if local else self.marker_list.curselection()
        update_3d(self.scene, self.data, self.method, self.frame, chosen, inspected,
                  focus_angle=self.focus_angle, show_model=self.show_model.get(),
                  show_predicted=self.show_predicted.get() and not local,
                  show_residuals=self.show_residuals.get() and not local,
                  label_mode='none' if local else label_mode, show_axis=self.show_axis.get(),
                  show_trail=self.show_trail.get() and not local, show_frames=self.show_frames.get(),
                  show_distal=self.show_distal.get(), reference_segment=segment if local else None,
                  first_bone=first_bone if local else 0, angle_indices=self.angle_indices,
                  result=self.display_result)
        update_angles(self.angle_artists, self.data, self.method, self.frame, self.angle_indices)
        result = self.display_result
        if self.camera_mode == 'local':
            center = self.local_center
            radius = self.local_radius
            if np.isfinite(center).all():
                for dimension, setter in enumerate((self.ax3d.set_xlim, self.ax3d.set_ylim, self.ax3d.set_zlim)):
                    setter(center[dimension] - radius, center[dimension] + radius)
        elif self.camera_mode == 'wrist':
            center = result['joints_mm'][self.frame, 6]
            if np.isfinite(center).all():
                for dimension, setter in enumerate((self.ax3d.set_xlim, self.ax3d.set_ylim, self.ax3d.set_zlim)):
                    setter(center[dimension] - self.wrist_radius, center[dimension] + self.wrist_radius)
        self.ax3d.set_title(f'{self.method.upper()}   t = {self.data.time[self.frame]:.3f} s'
                           + (f'   Fixed {segment}' if local else ''), fontsize=10)
        self.mode_note.set((f'固定 {self.reference_choice.get()} · 原点和方向固定 · '
                            + ('显示上游相对运动及 q9–q12 原始角度' if segment == 'hand' else
                               '下游角度 ' + '、'.join(f'q{k + 1}' for k in self.angle_indices)))
                           if local else '世界坐标总览 · 全部重建关节运动')
        point = self.data.measured_mm[self.frame, inspected]
        text = f'{self.data.names[inspected]}\n测量世界 XYZ (mm)\n' + ', '.join(f'{v:.1f}' for v in point)
        if inspected in self.data.raw_to_bound:
            predicted = self.data.methods[self.method]['predicted_mm'][self.frame, self.data.raw_to_bound[inspected]]
            error = np.linalg.norm(point - predicted)
            text += '\n预测世界 XYZ (mm)\n' + ', '.join(f'{v:.1f}' for v in predicted)
            text += f'\n位置差 {error:.2f} mm'
        else:
            text += '\n该点未参与当前模型拟合'
        self.inspector.set(text)
        flag = result['flags'][self.frame, self.focus_angle]
        self.quality_label.set(f'所选 {ANGLE_CHOICES[self.focus_angle]}\n'
                               f'{quality_text(flag)}\n\n骨段连线采用模型关节中心。\n{result["hand_description"]}')
        self.status.set(f'{self.data.time[self.frame]:.3f} / {self.data.time[-1]:.3f} s   '
                        f'帧 {self.frame + 1}/{len(self.data.time)}   '
                        f'显示 {len(chosen)}/{len(self.data.names)} 个原始点   '
                        f'绑定点 RMS {result["rms_mm"][self.frame]:.2f} mm')
        self.canvas3d.draw_idle()
        self.blit_angles()

    def set_frame(self, frame):
        self.frame = int(np.clip(frame, 0, len(self.data.time) - 1))
        self.updating_slider = True
        self.position.set(self.frame)
        self.frame_entry.set(str(self.frame + 1))
        self.updating_slider = False
        self.draw()

    def pause(self):
        self.playing = False
        self.play_button.configure(text='播放')

    def toggle(self):
        self.playing = not self.playing
        self.play_button.configure(text='暂停' if self.playing else '播放')
        self.playhead = float(self.data.time[self.frame])
        self.last_tick = time.perf_counter()

    def space(self, event):
        if event.widget.winfo_class() not in ('TCombobox', 'TSpinbox', 'Entry', 'TEntry'):
            self.toggle()
            return 'break'

    def seek(self, value):
        if self.updating_slider:
            return
        self.pause()
        self.set_frame(round(float(value)))
        self.playhead = float(self.data.time[self.frame])

    def step(self, amount):
        self.pause()
        self.set_frame(self.frame + amount)
        self.playhead = float(self.data.time[self.frame])

    def rewind(self):
        self.pause()
        self.set_frame(0)
        self.playhead = float(self.data.time[0])

    def go_to_entry(self):
        try:
            frame = int(self.frame_entry.get()) - 1
        except ValueError:
            return
        self.pause()
        self.set_frame(frame)
        self.playhead = float(self.data.time[self.frame])

    def click_angle(self, event):
        if event.inaxes not in self.angle_axes:
            return
        self.focus_angle = self.angle_indices[self.angle_axes.index(event.inaxes)]
        if event.xdata is not None:
            index = int(np.argmin(abs(self.data.time - event.xdata)))
            self.pause()
            self.set_frame(index)
            self.playhead = float(self.data.time[index])
        else:
            self.draw()

    def change_method(self, event=None):
        self.method = 'ekf' if self.method_choice.get() == '滤波 EKF' else 'ik'
        self.refresh_display_result()
        if self.camera_mode == 'local':
            self.set_local_camera()
        self.build_angles()
        self.draw()

    def select_markers(self, mode):
        self.marker_list.select_clear(0, self.tk.END)
        if mode == 'all':
            self.marker_list.select_set(0, self.tk.END)
        elif mode == 'main':
            for index in self.data.main_indices:
                self.marker_list.select_set(index)
        self.draw()

    def change_view(self, elevation, azimuth):
        self.camera_mode = 'manual'
        if self.observation.get() == '局部坐标系观察':
            for dimension, setter in enumerate((self.ax3d.set_xlim, self.ax3d.set_ylim, self.ax3d.set_zlim)):
                setter(self.local_center[dimension] - self.local_radius, self.local_center[dimension] + self.local_radius)
        else:
            configure_3d(self.ax3d, self.data)
        self.ax3d.view_init(elev=elevation, azim=azimuth)
        self.canvas3d.draw_idle()

    def fit_current(self):
        self.camera_mode = 'manual'
        self.pause()
        points = [bone[self.frame] for bone, line in zip(self.display_result['bones_mm'], self.scene['bones'])
                  if line.get_visible()]
        points.extend(finger[self.frame] for finger in self.display_result['fingers_mm'])
        points = np.concatenate(points)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = max(float(np.max(np.ptp(points, axis=0))) * .6, 50)
        for dimension, setter in enumerate((self.ax3d.set_xlim, self.ax3d.set_ylim, self.ax3d.set_zlim)):
            setter(center[dimension] - radius, center[dimension] + radius)
        self.canvas3d.draw_idle()

    def focus_wrist(self):
        self.camera_mode = 'wrist'
        self.wrist_radius = 180
        self.draw()

    def zoom3d(self, event):
        if event.inaxes != self.ax3d:
            return
        factor = .85 ** float(event.step)
        if self.camera_mode == 'local':
            self.local_radius *= factor
        elif self.camera_mode == 'wrist':
            self.wrist_radius *= factor
        for getter, setter in ((self.ax3d.get_xlim, self.ax3d.set_xlim),
                               (self.ax3d.get_ylim, self.ax3d.set_ylim),
                               (self.ax3d.get_zlim, self.ax3d.set_zlim)):
            low, high = getter()
            center = (low + high) / 2
            half = (high - low) * factor / 2
            setter(center - half, center + half)
        self.canvas3d.draw_idle()

    def tick(self):
        if self.closed:
            return
        now = time.perf_counter()
        if self.playing:
            self.playhead += (now - self.last_tick) * float(self.speed.get())
            if self.playhead > self.data.time[-1]:
                if self.loop.get():
                    self.playhead = self.data.time[0] + (self.playhead - self.data.time[0]) % (self.data.time[-1] - self.data.time[0])
                else:
                    self.playhead = float(self.data.time[-1])
                    self.pause()
            frame = np.searchsorted(self.data.time, self.playhead, side='right') - 1
            if frame != self.frame:
                self.set_frame(frame)
        self.last_tick = now
        self.timer = self.root.after(33, self.tick)

    def open_result(self):
        from tkinter import filedialog, messagebox
        self.pause()
        folder = filedialog.askdirectory(title='选择含 ik_angles.npz / ekf_angles.npz 的结果目录',
                                         initialdir=self.data.directory.parent, parent=self.root)
        if not folder:
            return
        try:
            data = ReconstructionData(folder)
        except Exception as exc:
            messagebox.showerror('读取失败', str(exc), parent=self.root)
            return
        self.data = data
        self.local_view_key = None
        self.camera_mode = 'local' if self.observation.get() == '局部坐标系观察' else 'manual'
        self.method = 'ekf' if 'ekf' in data.methods else 'ik'
        self.frame = 0
        self.playhead = float(data.time[0])
        self.file_label.set(self.description())
        self.action_label.set(data.action_text())
        self.marker_list.delete(0, self.tk.END)
        for name in data.names:
            self.marker_list.insert(self.tk.END, name)
        self.marker_list.select_set(0, self.tk.END)
        self.inspected_box.configure(values=data.names)
        self.inspected_marker.set('m_elb' if 'm_elb' in data.names else data.names[0])
        choices = ['滤波 EKF' if name == 'ekf' else '原始 IK' for name in data.methods]
        self.method_box.configure(values=choices)
        self.method_choice.set('滤波 EKF' if self.method == 'ekf' else '原始 IK')
        self.frame_box.configure(to=len(data.time))
        self.slider.configure(to=len(data.time) - 1)
        self.figure3d.clear()
        self.ax3d = self.figure3d.add_subplot(111, projection='3d')
        self.scene = create_3d_scene(self.ax3d, data)
        self.refresh_display_result()
        if self.camera_mode == 'local':
            self.set_local_camera()
        self.toolbar3d.update()
        self.build_angles()
        self.set_frame(0)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.root.after_cancel(self.timer)
        self.root.destroy()


def preview(data, output, frame=0, method='ekf'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    method = method if method in data.methods else next(iter(data.methods))
    frame = int(np.clip(frame, 0, len(data.time) - 1))
    figure = plt.figure(figsize=(17, 10), layout='constrained')
    grid = figure.add_gridspec(1, 2, width_ratios=[1, 1.4])
    ax = figure.add_subplot(grid[0], projection='3d')
    scene = create_3d_scene(ax, data)
    inspected = data.names.index('m_elb') if 'm_elb' in data.names else 0
    update_3d(scene, data, method, frame, list(range(len(data.names))), inspected, label_mode='main')
    ax.set_title(f'Vicon / model skeleton   t = {data.time[frame]:.3f} s')
    _, artists = build_angle_axes(figure, grid[1].subgridspec(4, 3), data, method, False)
    update_angles(artists, data, method, frame)
    figure.suptitle(f'{data.model.config.get("subject", "")} trial {data.report["trial"]:03d}  |  '
                   f'{data.report["description"]}\n'
                   'synchronized markers, skeleton and joint angles · * prior-dependent coordinate', fontsize=13)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def self_check(data):
    import tkinter as tk
    from scipy.spatial.transform import Rotation
    # Moving the reference bone and world pose must preserve distal geometry
    # expressed in that bone's frame.
    source = data.methods[next(iter(data.methods))]
    valid = np.isfinite(source['q']).all(axis=1) & np.isfinite(source['poses']).all(axis=(1, 2))
    anchor = np.flatnonzero(valid)[0]
    for segment, cutoff, _ in REFERENCE_FRAMES.values():
        q = np.repeat(source['q'][anchor:anchor + 1], 3, axis=0)
        q[1, :cutoff] += .07
        q[2, :cutoff] -= .04
        poses = np.repeat(source['poses'][anchor:anchor + 1], 3, axis=0)
        poses[1, :3, :3] = Rotation.from_euler('xyz', [.2, -.3, .4]).as_matrix() @ poses[1, :3, :3]
        poses[2, :3, 3] += [.2, -.1, .3]
        local = segment_local_view(data.build_method(q, poses, np.zeros((3, 12), dtype=np.uint8), np.zeros(3)), segment)
        np.testing.assert_allclose(local['segment_frames'][segment], np.repeat(np.eye(4)[None], 3, axis=0), atol=1e-9)
        np.testing.assert_allclose(local['palm_mm'], np.repeat(local['palm_mm'][:1], 3, axis=0), atol=1e-8)
    enable_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    viewer = ReconstructionViewer(root, data)
    root.update()
    viewer.set_frame(len(data.time) // 2)
    assert viewer.frame == len(data.time) // 2
    assert viewer.reference_choice.get() == '桡骨 Radius'
    assert viewer.angle_indices == [10, 11]
    for k, (cursor, dot, value) in zip(viewer.angle_indices, viewer.angle_artists):
        np.testing.assert_allclose(cursor.get_xdata(), data.time[viewer.frame])
        np.testing.assert_allclose(dot.get_ydata(), data.methods[viewer.method]['degrees'][viewer.frame, k])
    original_q = data.methods[viewer.method]['q'].copy()
    for label, (segment, cutoff, _) in REFERENCE_FRAMES.items():
        viewer.reference_choice.set(label)
        viewer.change_observation()
        result = viewer.display_result
        expected = list(range(8, 12)) if cutoff == 12 else list(range(cutoff, 12))
        assert viewer.angle_indices == expected
        assert len(viewer.angle_axes) == len(expected)
        assert viewer.ax3d.get_xlabel() == segment.capitalize() + ' X (mm)'
        np.testing.assert_allclose(result['q'], original_q)
        finite = np.isfinite(result['segment_frames'][segment]).all(axis=(1, 2))
        np.testing.assert_allclose(result['segment_frames'][segment][finite],
                                   np.repeat(np.eye(4)[None], finite.sum(), axis=0), atol=1e-8)
        viewer.set_frame(20)
        visible_axes = [k for k, line in enumerate(viewer.scene['joint_axes']) if line.get_visible()]
        assert visible_axes == expected
        viewer.show_model.set(False)
        viewer.draw()
        assert not any(line.get_visible() for line in viewer.scene['bones'])
        assert not viewer.scene['palm'].get_visible()
        viewer.show_model.set(True)
        viewer.show_axis.set(False)
        viewer.draw()
        assert not any(line.get_visible() for line in viewer.scene['joint_axes'])
        viewer.show_axis.set(True)
        viewer.draw()
        for angle, artists in zip(viewer.angle_indices, viewer.angle_artists):
            np.testing.assert_allclose(artists[1].get_ydata(), original_q[20, angle] * 180 / np.pi)
        old_limits = viewer.ax3d.get_xlim()
        viewer.set_frame(30)
        np.testing.assert_allclose(viewer.ax3d.get_xlim(), old_limits)
        viewer.change_view(0, 90)
        np.testing.assert_allclose(np.mean(viewer.ax3d.get_xlim()), viewer.local_center[0])
    viewer.reference_choice.set('桡骨 Radius')
    viewer.change_observation()
    if 'ik' in data.methods:
        viewer.method_choice.set('原始 IK')
        viewer.change_method()
        np.testing.assert_allclose(viewer.display_result['q'], data.methods['ik']['q'])
        viewer.method_choice.set('滤波 EKF')
        viewer.change_method()
    np.testing.assert_allclose(data.methods[viewer.method]['q'], original_q)
    viewer.observation.set('全部自由度')
    viewer.change_observation()
    assert len(viewer.angle_axes) == 12 and viewer.display_result is viewer.data.methods[viewer.method]
    np.testing.assert_allclose(np.mean(viewer.ax3d.get_xlim()), data.active_limits[0][0])
    viewer.set_frame(len(data.time) // 2)
    viewer.step(1)
    viewer.select_markers('main')
    assert len(viewer.marker_list.curselection()) == len(data.main_indices)
    viewer.show_residuals.set(True)
    viewer.focus_angle = 11
    viewer.draw()
    assert viewer.action_label.get() == data.action_text()
    for method in data.methods.values():
        # The fixed hand must stay rigid through the complete motion.
        palm_width = np.linalg.norm(method['palm_mm'][:, 1] - method['palm_mm'][:, 0], axis=1)
        np.testing.assert_allclose(palm_width[np.isfinite(palm_width)], .054 * data.segment_scale('hand') * 1000)
    viewer.show_frames.set(False)
    viewer.show_distal.set(False)
    viewer.draw()
    assert not any(line.get_visible() for lines, _ in viewer.scene['triads'] for line in lines)
    assert not any(line.get_visible() for line in viewer.scene['joint_axes'])
    viewer.show_frames.set(True)
    viewer.show_distal.set(True)
    viewer.focus_wrist()
    center = data.methods[viewer.method]['joints_mm'][viewer.frame, 6]
    np.testing.assert_allclose(np.mean(viewer.ax3d.get_xlim()), center[0])
    from types import SimpleNamespace
    old_span = np.ptp(viewer.ax3d.get_xlim())
    viewer.zoom3d(SimpleNamespace(inaxes=viewer.ax3d, step=1))
    assert np.ptp(viewer.ax3d.get_xlim()) < old_span
    if 'ik' in data.methods:
        viewer.method_choice.set('原始 IK')
        viewer.change_method()
        assert viewer.method == 'ik'
    viewer.toggle()
    viewer.last_tick -= .1
    previous = viewer.frame
    root.after_cancel(viewer.timer)
    viewer.tick()
    assert viewer.frame > previous
    viewer.pause()
    viewer.rewind()
    assert viewer.frame == 0
    other = data.directory.parent / 'trial_001'
    opened_other = False
    if other.exists() and other != data.directory:
        from unittest.mock import patch
        viewer.observation.set('局部坐标系观察')
        viewer.reference_choice.set('桡骨 Radius')
        viewer.change_observation()
        with patch('tkinter.filedialog.askdirectory', return_value=str(other)):
            viewer.open_result()
        assert viewer.data.directory == other
        assert viewer.frame == 0
        assert viewer.action_label.get() == viewer.data.action_text()
        assert viewer.angle_indices == [10, 11]
        assert viewer.ax3d.get_xlabel() == 'Radius X (mm)'
        assert int(float(viewer.slider.cget('to'))) == len(viewer.data.time) - 1
        opened_other = True
    root.update()
    viewer.close()
    print(json.dumps(dict(gui_check='passed',reference_frames_checked=8,frames=len(data.time),markers=len(data.names),
                          methods=list(data.methods),open_other_trial_checked=opened_other), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default=str(DEFAULT_RESULT))
    parser.add_argument('--motion')
    parser.add_argument('--model')
    parser.add_argument('--method', choices=['ik', 'ekf'], default='ekf')
    parser.add_argument('--frame', type=int, default=0, help='0-based initial frame')
    parser.add_argument('--preview', help='Export a PNG without opening the desktop UI')
    parser.add_argument('--self-check', action='store_true', help='Exercise Tk controls then close')
    args = parser.parse_args()
    data = ReconstructionData(args.results, args.motion, args.model)
    if args.preview:
        preview(data, args.preview, args.frame, args.method)
    elif args.self_check:
        self_check(data)
    else:
        import tkinter as tk
        enable_dpi_awareness()
        root = tk.Tk()
        viewer = ReconstructionViewer(root, data, args.method)
        viewer.set_frame(args.frame)
        root.mainloop()


if __name__ == '__main__':
    main()
