"""Read-only v002 motion and EMG adapter, using saved pulse alignment and poses."""
import json
from pathlib import Path

import numpy as np

from src.emg_visualization.data import DEFAULT_DATA, EmgData
from src.motion_reconstruction.common.emg_sync import resample_estimates
from src.motion_reconstruction.tum_native.anchored_quality import REJECT
from src.motion_reconstruction.tum_native.schema import Q_NAMES, canonical_hash, sha256
from src.motion_reconstruction.tum_native.viewer import hand_geometry
from .data import SynchronizedData, interpolate_poses, reconstruction_root

SCHEMA = 'tum-native-functional7-v2'
NATIVE_QUALITY = {1: '缺失观测', 2: '点簇退化', 4: '功能轴依赖几何先验', 8: '两轴姿态失配',
    16: '受试者参考零位', 32: '数值求解未收敛', 64: '滤波边界', 128: '短缺口插值',
    256: '测量信息不足', 512: '旋转向量分支风险', 1024: '功能枢轴依赖先验',
    2048: '缺少完整观测的预测', 4096: '点簇拟合异常', 8192: '观测不支持的旋转步长'}


def native_quality_text(flag):
    flag = int(flag)
    text = [name for bit, name in NATIVE_QUALITY.items() if flag & bit]
    unknown = flag & ~sum(NATIVE_QUALITY)
    if unknown:
        text.append(f'未知质量位{unknown}')
    return '、'.join(text) + f' [{flag}]' if flag else '通过当前数值检查'


def validate_angles(q, flags, valid, samples):
    if (q.shape != (samples, 7) or flags.shape != q.shape or valid.shape != q.shape
            or not np.issubdtype(flags.dtype, np.integer) or valid.dtype != np.bool_ or np.isinf(q).any()):
        raise ValueError('v002须包含七列角度及逐角质量、有效掩码')
    if not np.array_equal(valid, np.isfinite(q) & ((flags & REJECT) == 0)):
        raise ValueError('v002相对质量掩码与角度／质量位不一致')


class NativeSynchronizedData(SynchronizedData):
    algorithm = 'tum_native'
    algorithm_label = '新方法 v002'
    label_filename = 'emg_aligned_native.npz'
    angle_names = ['肘屈伸', '前臂旋转／扭转代理', '腕屈伸', '腕桡偏／尺偏']
    prior_mask = 4 | 16 | 1024
    reject_mask = REJECT
    motion_description = 'v002七坐标、肩带位移和保存的骨段位姿'
    status_note = '前臂为技术段扭转代理；参考零位与功能轴依赖模型'
    quality_text = staticmethod(native_quality_text)

    def __init__(self, path=DEFAULT_DATA, reconstruction=None):
        self.emg = EmgData(path)
        self.result_root = Path(reconstruction or reconstruction_root(self.algorithm)).resolve()
        subject, trial = f'subj_{self.emg.subject:03d}', f'trial_{self.emg.trial:03d}'
        self.directory = self.result_root / subject / 'trials' / trial
        self.label_path = self.directory / self.label_filename
        self.full_label_path = self.directory / 'reconstruction.npz'
        self.report = json.loads((self.directory/'report.json').read_text(encoding='utf-8'))
        self.model = json.loads((self.directory.parents[1]/'model.json').read_text(encoding='utf-8'))
        job = self.report['job']
        if (self.report.get('schema') != SCHEMA or self.model.get('schema') != SCHEMA
                or self.model.get('subject') != subject or job['trial'] != self.emg.trial or job['task'] != self.emg.task):
            raise ValueError('需要受试者、试验和任务匹配的v002运动重建')
        if self.report.get('model_hash') != canonical_hash(self.model):
            raise ValueError('v002模型哈希与重建报告不一致')
        self.sync = self.report['sync']
        if self.sync.get('status') != 'duration_validated_pulse_candidate':
            raise ValueError('v002缺少已核实的脉冲时长匹配证据')
        scale, offset = float(self.sync.get('clock_scale', 1)), float(self.sync['offset_s'])
        if not np.isfinite([scale, offset]).all() or scale <= 0:
            raise ValueError('同步偏移或时钟尺度无效')
        source = Path(str(self.emg.arrays['source'].item())).resolve()
        motion_source = Path(self.report['source']).resolve()
        expected_source = motion_source.parent.parent/'EMG'/motion_source.name
        if source != expected_source.resolve() or sha256(source) != self.sync.get('emg_source_sha256'):
            raise ValueError('v002与预处理肌电的原始文件路径或哈希不匹配')
        if sha256(motion_source) != self.report.get('source_sha256'):
            raise ValueError('v002原始运动文件哈希与报告不匹配')
        if ('source_sha256' in self.emg.arrays
                and str(self.emg.arrays['source_sha256'].item()) != self.sync['emg_source_sha256']):
            raise ValueError('预处理肌电源哈希与v002同步来源不一致')
        with np.load(self.label_path, allow_pickle=False) as z:
            if z['schema'].item() != SCHEMA or z['angle_unit'].item() != 'rad' or z['q_names'].tolist() != Q_NAMES:
                raise ValueError('同步标签版本、角度列或单位不匹配')
            if not np.array_equal(z['t'], self.emg.time):
                raise ValueError('预处理肌电与v002同步角度时间轴不完全一致')
            aligned, aligned_flags, valid = z['q'].astype(float), z['flags'].copy(), z['relative_valid'].copy()
        validate_angles(aligned, aligned_flags, valid, len(self.emg.time))
        with np.load(self.full_label_path, allow_pickle=False) as z:
            if (z['schema'].item() != SCHEMA or z['q_names'].tolist() != Q_NAMES
                    or z['angle_unit'].item() != 'rad' or z['length_unit'].item() != 'm' or z['time_unit'].item() != 's'):
                raise ValueError('v002完整运动学的版本、角度列或单位不正确')
            t, q, flags, source_valid = z['t'], z['q'], z['flags'], z['relative_valid']
            rotations, origins, bones = z['segment_R'], z['segment_origins_m'], z['bones_m']
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('v002运动时间轴无效')
        validate_angles(q, flags, source_valid, len(t))
        if (rotations.shape != (len(t), 4, 3, 3) or origins.shape != (len(t), 4, 3)
                or bones.shape != (len(t), 4, 3)):
            raise ValueError('v002骨段位姿或骨架维度不正确')
        target = (self.emg.time-offset)/scale
        expected, expected_flags = resample_estimates(t, np.where(source_valid, q, np.nan), flags, target)
        if (not np.allclose(expected, aligned, atol=1e-6, rtol=0, equal_nan=True)
                or not np.array_equal(expected_flags, aligned_flags)):
            raise ValueError('已有v002同步标签与源运动学及脉冲映射不一致')
        self.q, self.flags, self.strict_valid = aligned[:, 3:], aligned_flags[:, 3:], valid[:, 3:]
        self.full_q, self.full_flags = resample_estimates(t, q, flags, target)
        self.segment_poses = np.full((len(target), 4, 4, 4), np.nan)
        for segment in range(4):
            poses = np.tile(np.eye(4), (len(t), 1, 1))
            poses[:, :3, :3], poses[:, :3, 3] = rotations[:, segment], origins[:, segment]
            self.segment_poses[:, segment] = interpolate_poses(t, poses, target, .01)
        values, _ = resample_estimates(t, bones.reshape(len(t), 12), np.zeros((len(t), 12), np.uint16), target)
        self.bones = values.reshape(len(target), 4, 3).astype(float)
        self.poses = self.segment_poses[:, 0]
        self.pose_valid = (np.isfinite(self.full_q).all(1) & np.isfinite(self.segment_poses).all((1, 2, 3))
                           & np.isfinite(self.bones).all((1, 2)))
        self.time = self.emg.time-self.emg.time[0]
        self.degrees = np.rad2deg(self.q)
        self.overlap = np.isfinite(self.q).all(1)
        if not self.overlap.any():
            raise ValueError('v002无肘腕四角均有效的共同时间范围')
        if not self.pose_valid.any():
            raise ValueError('v002无完整骨段位姿可用于动画')
        self.common_range = (self.time[self.overlap][0], self.time[self.overlap][-1])
        self.reference_origin = self.poses[np.flatnonzero(self.pose_valid)[0], :3, 3].copy()

    def pose(self, index):
        if not self.pose_valid[index]:
            return None
        poses = self.segment_poses[index]
        rt, rh, rf, rm = poses[:, :3, :3]
        shoulder, elbow, wrist, tip = self.bones[index]
        hand = hand_geometry(rm[None], poses[3, :3, 3][None], self.model['hand_tip_M_m'])
        def local(points):
            return (np.asarray(points)-self.reference_origin)*1000
        torso = np.array([[-.12, 0, -.22], [.12, 0, -.22], [.12, 0, .05], [-.12, 0, .05], [-.12, 0, -.22]])
        radial = rf[:, 0]
        axes = self.model['axes']
        return dict(torso=local(torso@rt.T+poses[0, :3, 3]),
            proximal=local([poses[0, :3, 3], shoulder]), upper=local([shoulder, elbow]),
            forearm=local([elbow, wrist]), palm=local(np.vstack([hand['palm'][0], hand['palm'][0, :1]])),
            fingers=[local(f) for f in hand['fingers'][0]], band=local([wrist-radial*.025, wrist+radial*.025]),
            centers=local([elbow, elbow, wrist, wrist]),
            axes=np.stack([rh@axes['elbow_H'], rf@axes['twist_F'], rf@axes['wrist_FE_F'], rm@axes['wrist_RD_M']]),
            hand_axes=rm.T)
