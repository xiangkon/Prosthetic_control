"""Join existing labels and EMG only after validating source and sample clocks."""
import hashlib
import json
from pathlib import Path

import numpy as np

from src.emg_visualization.data import DEFAULT_DATA, EmgData
from src.motion_reconstruction.paths import OUTPUT_ROOT, NATIVE_ROOT, resolve_historical_path
from src.motion_reconstruction.legacy_g99.model import Model
from src.motion_reconstruction.legacy_g99.kinematics import forward
from src.motion_reconstruction.common.emg_sync import resample_estimates
from scipy.spatial.transform import Rotation, Slerp


ANGLE_NAMES = ['q9 肘屈伸', 'q10 前臂旋前／旋后', 'q11 腕屈伸', 'q12 腕桡偏／尺偏']
ANGLE_COLORS = ['#277e70', '#7262ab', '#287fa8', '#b07929']
QUALITY_BITS = {1: '缺失', 2: 'IK未收敛', 4: '残差偏大', 8: '弱观测', 16: '肩胛约束异常',
                32: '角度边界', 64: '先验依赖', 128: '标定验证异常'}


def quality_text(flag):
    flag = int(flag)
    known = '、'.join(text for bit, text in QUALITY_BITS.items() if flag & bit)
    return (known or ('未知标记' if flag else '通过当前数值检查')) + (f' [{flag}]' if flag else '')


class SynchronizedData:
    algorithm = 'legacy_g99'
    algorithm_label = '老方法 G99'
    label_filename = 'emg_aligned_distal.npz'
    angle_names = ANGLE_NAMES
    prior_mask = 64
    reject_mask = 0xFFFF ^ 64
    motion_description = '完整12角和胸廓位姿'
    status_note = 'q10绝对相位依赖先验'
    quality_text = staticmethod(quality_text)

    def __init__(self, path=DEFAULT_DATA, reconstruction=OUTPUT_ROOT):
        self.emg = EmgData(path)
        self.result_root = Path(reconstruction).resolve()
        self.directory = resolve_historical_path(Path(reconstruction).resolve() / f'subj_{self.emg.subject:03d}' / 'trials' / f'trial_{self.emg.trial:03d}')
        self.label_path = self.directory / 'emg_aligned_distal.npz'
        if not self.label_path.is_file():
            raise ValueError(f'此肌电试验没有现成的同步角度标签：{self.label_path}')
        self.report = json.loads((self.directory / 'report.json').read_text(encoding='utf-8'))
        if self.report['task'] != self.emg.task or self.report['trial'] != self.emg.trial:
            raise ValueError('角度报告与肌电任务／试验编号不一致')
        with np.load(self.label_path, allow_pickle=False) as z:
            if str(z['angle_unit'].item()) != 'rad':
                raise ValueError('角度文件须声明单位rad')
            if not np.array_equal(z['emg_time'], self.emg.time):
                raise ValueError('预处理肌电与同步角度时间轴不完全一致，不能按行关联')
            if Path(str(z['source_emg'].item())).resolve() != Path(str(self.emg.arrays['source'].item())).resolve():
                raise ValueError('同步标签与预处理信号引用了不同的原始肌电文件')
            self.q = np.asarray(z['q_distal_estimated'], dtype=float)
            self.flags = z['angle_quality_flag'].copy()
            self.strict_valid = z['strict_valid'].copy()
            self.sync = json.loads(str(z['sync_metadata'].item()))
        if self.q.shape != (len(self.emg.time), 4) or self.flags.shape != self.q.shape or self.strict_valid.shape != self.q.shape:
            raise ValueError('预期四个远端角度和逐角质量标记')
        if (not np.issubdtype(self.flags.dtype, np.integer) or self.strict_valid.dtype != np.bool_
                or not np.array_equal(self.strict_valid, self.flags == 0) or np.isinf(self.q).any()):
            raise ValueError('角度或质量标记不符合导出约定')
        if self.sync.get('status') != 'duration_validated_pulse_candidate':
            raise ValueError('此结果缺少已验证的脉冲持续时间匹配')
        self.time = self.emg.time - self.emg.time[0]
        self.degrees = np.rad2deg(self.q)
        self.overlap = np.isfinite(self.q).all(axis=1)
        if not self.overlap.any():
            raise ValueError('无四角均有限的共同时间范围')
        self.common_range = (self.time[self.overlap][0], self.time[self.overlap][-1])
        model_path = (self.directory / self.report['model_file']).resolve()
        self.model = Model.load(model_path, allow_estimated=True)
        digest = hashlib.sha256(json.dumps(self.model.config, sort_keys=True).encode()).hexdigest()
        if digest != self.report['model_sha256']:
            raise ValueError('个体模型哈希与角度报告不一致')
        self.full_label_path = self.directory / 'ekf_angles.npz'
        with np.load(self.full_label_path, allow_pickle=False) as z:
            t = np.asarray(z['t'], float)
            q = np.asarray(z['q'], float)
            flags = z['angle_quality_flag']
            poses = np.asarray(z['T_world_thorax'], float)
            if str(z['angle_unit'].item()) != 'rad' or q.shape != (len(t), 12) or poses.shape != (len(t), 4, 4):
                raise ValueError('完整运动学须包含12角、胸廓位姿及rad单位')
        if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('完整运动学时间轴无效')
        target = self.emg.time - self.sync['offset_s']
        max_gap = max(.01, 2.1 * float(np.median(np.diff(t))))
        self.full_q, self.full_flags = resample_estimates(t, q, flags, target, max_gap)
        if not np.allclose(self.full_q[:, 8:], self.q, atol=1e-6, rtol=0, equal_nan=True):
            raise ValueError('完整运动学与既有四角同步标签不一致')
        self.poses = interpolate_poses(t, poses, target, max_gap)
        self.pose_valid = np.isfinite(self.full_q).all(axis=1) & np.isfinite(self.poses).all(axis=(1, 2))
        if not self.pose_valid.any():
            raise ValueError('无完整12角与胸廓位姿可用于动画')
        self.reference_origin = self.poses[np.flatnonzero(self.pose_valid)[0], :3, 3].copy()

    def nearest(self, seconds):
        right = int(np.clip(np.searchsorted(self.time, seconds), 0, len(self.time) - 1))
        left = max(0, right - 1)
        return left if abs(self.time[left] - seconds) < abs(self.time[right] - seconds) else right

    def pose(self, index):
        """Use all 12 reconstructed angles and measured thorax pose in world axes."""
        if not self.pose_valid[index]:
            return None
        _, _, frames, axes, centers = forward(self.model, self.full_q[index], self.poses[index:index + 1])
        humerus = frames['humerus'][0]
        elbow = centers[0, 8]

        def local(points):
            return (np.asarray(points) - self.reference_origin) * 1000

        shoulder = humerus[:3, :3] @ np.asarray(self.model.joints[2]['center_child_m']) + humerus[:3, 3]
        wrist = centers[0, 10]
        hand = frames['hand'][0]
        palm_local = np.array([[-.027, 0, -.012], [.027, 0, -.012], [.038, 0, -.085],
                               [-.034, 0, -.085], [-.027, 0, -.012]])
        palm = palm_local @ hand[:3, :3].T + hand[:3, 3]
        fingers = []
        for x, length in [(-.028, .058), (-.009, .080), (.012, .087), (.032, .075)]:
            finger = np.array([[x, 0, -.080], [x * 1.2, 0, -.080 - length]])
            fingers.append(local(finger @ hand[:3, :3].T + hand[:3, 3]))
        # Fixed abducted thumb on the same +X side as the reconstruction viewer.
        thumb = np.array([[.027, 0, -.025], [.050, -.006, -.044],
                          [.070, -.010, -.064], [.082, -.012, -.080]])
        fingers.append(local(thumb @ hand[:3, :3].T + hand[:3, 3]))
        # A transverse radius bar makes axial forearm rotation observable.
        radial = frames['radius'][0][:3, 0]
        band = np.array([wrist - radial * .025, wrist + radial * .025])
        thorax = frames['thorax'][0]
        torso = np.array([[-.12, 0, -.22], [.12, 0, -.22], [.12, 0, .05], [-.12, 0, .05], [-.12, 0, -.22]])
        return dict(torso=local(torso @ thorax[:3, :3].T + thorax[:3, 3]),
                    proximal=local(centers[0, [0, 2, 5]]),
                    upper=local([shoulder, elbow]), forearm=local([elbow, wrist]),
                    palm=local(palm), fingers=fingers, band=local(band),
                    centers=local(centers[0, 8:]), axes=axes[0, 8:],
                    hand_axes=hand[:3, :3].T)


def interpolate_poses(t, poses, target, max_gap):
    """Linear translation plus SLERP rotation, with no extrapolation or gap filling."""
    out = np.full((len(target), 4, 4), np.nan)
    valid = np.isfinite(poses).all(axis=(1, 2))
    right = np.searchsorted(t, target)
    clipped = np.clip(right, 0, len(t) - 1)
    exact = (right < len(t)) & np.isclose(t[clipped], target, atol=1e-9, rtol=0)
    exact &= valid[clipped]
    out[exact] = poses[clipped[exact]]
    inside = (right > 0) & (right < len(t)) & ~exact
    positions = np.flatnonzero(inside)
    r = right[positions]
    l = r - 1
    good = valid[l] & valid[r] & (t[r] - t[l] <= max_gap)
    positions, l, r = positions[good], l[good], r[good]
    if len(positions):
        ratio = (target[positions] - t[l]) / (t[r] - t[l])
        out[positions] = np.eye(4)
        out[positions, :3, 3] = (1 - ratio[:, None]) * poses[l, :3, 3] + ratio[:, None] * poses[r, :3, 3]
        rotation = Slerp(t[valid], Rotation.from_matrix(poses[valid, :3, :3]))
        out[positions, :3, :3] = rotation(target[positions]).as_matrix()
    return out


def reconstruction_root(algorithm):
    return NATIVE_ROOT / 'v002_anchored7' if algorithm == 'tum_native' else OUTPUT_ROOT


def load_synchronized(path=DEFAULT_DATA, reconstruction=None, algorithm='tum_native'):
    root = reconstruction_root(algorithm) if reconstruction is None else Path(reconstruction)
    if algorithm == 'tum_native':
        from .native_data import NativeSynchronizedData
        return NativeSynchronizedData(path, root)
    if algorithm == 'legacy_g99':
        return SynchronizedData(path, root)
    raise ValueError('不支持的运动重建方法')
