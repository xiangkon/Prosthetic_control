"""Read-only paired TUM results and coordinates for their visual comparison."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .angle_conversion import AngleConversion
from ..legacy_g99.kinematics import forward as legacy_forward
from ..legacy_g99.viewer import ReconstructionData, TASK_LABELS
from ..paths import LEGACY_ROOT, NATIVE_ROOT, PROJECT
from ..tum_native.observations import rigid_fit
from ..tum_native.schema import Q_NAMES, canonical_hash, sha256
from ..tum_native.viewer import hand_geometry

V002_ROOT = NATIVE_ROOT / 'v002_anchored7'
ANGLE_LABELS = ('Thorax-humerus X', 'Thorax-humerus Y', 'Thorax-humerus Z',
                'Elbow flexion', 'Forearm rotation / twist proxy', 'Wrist flexion', 'Wrist deviation')
ANGLE_LABELS_ZH = ('胸廓—上臂 X', '胸廓—上臂 Y', '胸廓—上臂 Z',
                   '肘屈伸', '前臂旋转／扭转代理', '腕屈伸', '桡尺偏')
LEGACY_REJECT = np.uint16(0xFFFF ^ 64)


@dataclass(frozen=True, order=True)
class TrialKey:
    subject: int
    trial: int

    @property
    def relative(self):
        return Path(f'subj_{self.subject:03d}') / 'trials' / f'trial_{self.trial:03d}'

    def __str__(self):
        return f'subj_{self.subject:03d} / trial_{self.trial:03d}'


def paired_trials(legacy_root=LEGACY_ROOT, native_root=V002_ROOT):
    pairs = []
    for path in native_root.glob('subj_[0-9][0-9][0-9]/trials/trial_[0-9][0-9][0-9]/reconstruction.npz'):
        trial = path.parent
        key = TrialKey(int(trial.parents[1].name[5:]), int(trial.name[6:]))
        old = legacy_root / key.relative
        if ((trial / 'report.json').is_file() and (old / 'report.json').is_file()
                and any((old / f'{method}_angles.npz').is_file() for method in ('ik', 'ekf'))):
            pairs.append(key)
    return sorted(pairs)


def common_frames(old_time, new_time, tolerance=1e-8):
    """Return only shared MoCap timestamps, with no interpolation."""
    for clock in (old_time, new_time):
        if clock.ndim != 1 or not len(clock) or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
            raise ValueError('结果时间轴必须为有限、严格递增的时间序列')
    right = np.clip(np.searchsorted(old_time, new_time), 0, len(old_time) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(old_time[left] - new_time) < abs(old_time[right] - new_time), left, right)
    shared = abs(old_time[indices] - new_time) <= tolerance
    if not shared.any() or np.any(np.diff(indices[shared]) <= 0):
        raise ValueError('两套结果没有可一一匹配的 MoCap 时刻')
    return indices[shared], np.flatnonzero(shared)


def _read_report(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _check_sources(old_report, new_report):
    old_source = (PROJECT / old_report['source']).resolve()
    new_source = (PROJECT / new_report['source']).resolve()
    if old_source != new_source or old_report['task'] != new_report['job']['task']:
        raise ValueError('两套结果引用的 MoCap 文件或任务编号不同')
    new_hash = new_report.get('source_sha256')
    if new_hash and sha256(new_source) != new_hash:
        raise ValueError('当前 MoCap 文件哈希与 v002 结果记录不同')
    if old_report.get('source_sha256') and old_report['source_sha256'] != new_hash:
        raise ValueError('老方法与 v002 的 MoCap 来源哈希不同')


def _bridge_rotation(old, native, segment):
    spec = native['segments'][segment]
    names = [name for name in spec['fit_names'] if name in old.bindings
             and old.bindings[name]['segment'] == segment]
    local_new = np.array([spec['local_m'][name] for name in names]).reshape(-1, 3)
    local_old = np.array([old.bindings[name]['local_m'] for name in names]).reshape(-1, 3)
    rotation, _, _, _, _ = rigid_fit(local_new, local_old)
    if rotation is None:
        raise ValueError(f'{segment} 的共同标记不足或退化，无法建立参考轴')
    return rotation


def _old_thoracohumeral(frames, bridge):
    thorax, humerus = (frames[name][:, :3, :3] for name in ('thorax', 'humerus'))
    return bridge[0].T @ thorax.swapaxes(1, 2) @ humerus @ bridge[1]


def _native_thoracohumeral(q, model):
    result = np.full((len(q), 3, 3), np.nan)
    finite = np.isfinite(q[:, :3]).all(1)
    if finite.any():
        result[finite] = np.array(model['reference_rotations']['TH']) @ (
            Rotation.from_rotvec(q[finite, :3]).as_matrix())
    return result


def _relative_rotvec(rotation, reference):
    result = np.full((len(rotation), 3), np.nan)
    finite = np.isfinite(rotation).all((1, 2))
    if finite.any():
        result[finite] = Rotation.from_matrix(reference.T @ rotation[finite]).as_rotvec()
    return result


@dataclass
class ReferenceMapping:
    bridge: tuple | None
    old_shoulder_zero: np.ndarray | None
    native_shoulder_zero: np.ndarray | None
    signs: np.ndarray
    offset: np.ndarray
    reference: dict
    notes: list


def _reference_mapping(old, native, legacy_root, native_root, subject, method):
    reference = native['reference']
    notes = []
    signs, offset = np.full(4, np.nan), np.full(4, np.nan)
    bridge = old_zero = native_zero = None
    key = TrialKey(subject, reference['trial'])
    legacy_dir, native_dir = legacy_root / key.relative, native_root / key.relative
    try:
        previous, current = (_read_report(path / 'report.json') for path in (legacy_dir, native_dir))
        _check_sources(previous, current)
        if (previous['task'] != 0 or current['model_hash'] != canonical_hash(native)
                or previous['model_sha256'] != canonical_hash(old.config)
                or current.get('source_sha256') != reference.get('source_sha256')):
            raise ValueError('Task 0 参考试验与当前模型或原始输入不匹配')
        with np.load(legacy_dir / f'{method}_angles.npz', allow_pickle=False) as z, \
                np.load(native_dir / 'reconstruction.npz', allow_pickle=False) as n:
            a, b = common_frames(z['t'], n['t'])
            time = n['t'][b]
            window = (time >= reference['start_s'] - 1e-8) & (time <= reference['end_s'] + 1e-8)
            old_q, native_q = z['q'][a[window]].astype(float), n['q'][b[window]].astype(float)
        if not len(old_q):
            raise ValueError('Task 0 稳定窗口没有共有采样时刻')
        old_ok = np.isfinite(old_q).all(axis=1)
        if not old_ok.any():
            raise ValueError('Task 0 稳定窗口缺少完整老方法角度')
    except (OSError, ValueError, KeyError) as exc:
        return ReferenceMapping(bridge, old_zero, native_zero, signs, offset, reference, [f'参考对齐不可用：{exc}'])

    try:
        bridge = tuple(_bridge_rotation(old, native, segment) for segment in ('thorax', 'humerus'))
        old_rotation = _old_thoracohumeral(legacy_forward(old, old_q[old_ok])[2], bridge)
        native_rotation = _native_thoracohumeral(native_q, native)
        native_finite = np.isfinite(native_rotation).all((1, 2))
        if not native_finite.any():
            raise ValueError('Task 0 稳定窗口无可用 v002 肩部姿态')
        old_zero = Rotation.from_matrix(old_rotation).mean().as_matrix()
        native_zero = Rotation.from_matrix(native_rotation[native_finite]).mean().as_matrix()
    except ValueError as exc:
        notes.append(f'肩部对齐不可用：{exc}')

    try:
        anchor = np.median(old_q[old_ok], axis=0)
        conversion = AngleConversion(old, native)
        _, jacobian = conversion.linearize(anchor)
        _, check = conversion.linearize(anchor, step=5e-5)
        derivatives = jacobian[3:7, 8:12].diagonal()
        stable = (abs(derivatives) >= .05) & (abs(derivatives - check[3:7, 8:12].diagonal()) < .01)
        for k in range(4):
            use = np.isfinite(old_q[:, k+8]) & np.isfinite(native_q[:, k+3])
            if stable[k] and use.any():
                signs[k] = np.sign(derivatives[k])
                offset[k] = np.median(old_q[use, k+8]) - signs[k] * np.median(native_q[use, k+3])
            else:
                notes.append(f'{ANGLE_LABELS_ZH[k+3]}方向或参考角不可用')
    except (ValueError, np.linalg.LinAlgError) as exc:
        notes.append(f'远端四角对齐不可用：{exc}')
    return ReferenceMapping(bridge, old_zero, native_zero, signs, offset, reference, notes)


@dataclass
class Skeleton:
    bones: list
    palm: np.ndarray
    fingers: list
    predicted: np.ndarray
    wrist: np.ndarray
    color: str


@dataclass
class PairedMotion:
    key: TrialKey
    method: str
    methods: tuple
    task: str
    time: np.ndarray
    source_frames: np.ndarray
    markers: np.ndarray
    legacy: Skeleton
    native: Skeleton
    angles: tuple
    valid: tuple
    flags: tuple
    native_bits: dict
    mapping: ReferenceMapping
    limits: tuple
    shared_marker_count: int


class MotionLibrary:
    def __init__(self, legacy_root=LEGACY_ROOT, native_root=V002_ROOT):
        self.legacy_root, self.native_root = Path(legacy_root), Path(native_root)
        self.keys = paired_trials(self.legacy_root, self.native_root)
        self.references = {}

    def load(self, key, preferred_method='ekf'):
        old_dir, new_dir = self.legacy_root / key.relative, self.native_root / key.relative
        new_report = _read_report(new_dir / 'report.json')
        new_model = _read_report(new_dir.parents[1] / 'model.json')
        _check_sources(_read_report(old_dir / 'report.json'), new_report)
        if new_report['schema'] != 'tum-native-functional7-v2' or new_report['model_hash'] != canonical_hash(new_model):
            raise ValueError('需要与 v002 模型匹配的七坐标重建结果')
        old = ReconstructionData(old_dir)
        method = preferred_method if preferred_method in old.methods else next(iter(old.methods))
        legacy = old.methods[method]
        with np.load(new_dir / 'reconstruction.npz', allow_pickle=False) as z:
            a, b = common_frames(old.time, z['t'])
            if z['q_names'].tolist() != Q_NAMES or z['angle_unit'].item() != 'rad' or z['length_unit'].item() != 'm':
                raise ValueError('v002 角度列或坐标单位不符合七坐标约定')
            time, source_frames = z['t'][b], z['source_indices'][b]
            if not np.array_equal(source_frames, old.marker_indices[a]):
                raise ValueError('两套结果的原始采样帧索引不一致')
            new_names = z['marker_names'].tolist()
            order = [new_names.index(name) for name in old.names]
            raw = z['raw_markers_m'][b][:, order] * 1000
            if not np.allclose(raw, old.measured_mm[a], rtol=0, atol=1e-5, equal_nan=True):
                raise ValueError('共同时刻的原始标记坐标不一致')
            q = z['q'][b].astype(float)
            native_valid, native_flags = z['relative_valid'][b], z['flags'][b]
            geometry = hand_geometry(z['segment_R'][b, 3], z['segment_origins_m'][b, 3], new_model['hand_tip_M_m'])
            new_bones = z['bones_m'][b] * 1000
            native_skeleton = Skeleton([new_bones], geometry['palm'] * 1000,
                [f * 1000 for f in geometry['fingers'].transpose(1, 0, 2, 3)],
                z['predicted_markers_m'][b][:, order] * 1000, new_bones[:, 2], '#d3742f')

        old_predicted = np.full_like(raw, np.nan)
        old_predicted[:, old.bound_to_raw] = legacy['predicted_mm'][a]
        supported = np.isfinite(old_predicted).any(axis=(0, 2)) & np.isfinite(native_skeleton.predicted).any(axis=(0, 2))
        # Overlay and summarize the same markers for both methods.
        for prediction in (old_predicted, native_skeleton.predicted):
            prediction[:, ~supported] = np.nan
        old_skeleton = Skeleton([bone[a] for bone in legacy['bones_mm']], legacy['palm_mm'][a],
            [finger[a] for finger in legacy['fingers_mm']], old_predicted, legacy['joints_mm'][a, 6], '#27688e')

        map_key = (key.subject, method, new_report['model_hash'], canonical_hash(old.model.config))
        if map_key not in self.references:
            self.references[map_key] = _reference_mapping(old.model, new_model, self.legacy_root, self.native_root, key.subject, method)
        mapping = self.references[map_key]
        old_angles = np.full_like(q, np.nan)
        old_angles[:, 3:] = legacy['q'][a, 8:]
        new_angles = np.full_like(q, np.nan)
        new_angles[:, 3:] = q[:, 3:] * mapping.signs + mapping.offset
        if mapping.old_shoulder_zero is not None and mapping.native_shoulder_zero is not None:
            frames = {name: legacy['segment_frames'][name][a] for name in ('thorax', 'humerus')}
            old_angles[:, :3] = _relative_rotvec(_old_thoracohumeral(frames, mapping.bridge), mapping.old_shoulder_zero)
            new_angles[:, :3] = _relative_rotvec(_native_thoracohumeral(q, new_model), mapping.native_shoulder_zero)
        angles = tuple(np.rad2deg(item) for item in (old_angles, new_angles))
        legacy_flags = legacy['flags'][a]
        shoulder_flags = np.bitwise_or.reduce(legacy_flags[:, :8], axis=1)
        old_flags = np.column_stack([np.repeat(shoulder_flags[:, None], 3, axis=1), legacy_flags[:, 8:]])
        old_valid = (old_flags & LEGACY_REJECT) == 0
        for values, valid in zip(angles, (old_valid, native_valid)):
            valid &= np.isfinite(values)
            valid[:, :3] &= np.linalg.norm(values[:, :3], axis=1)[:, None] <= 170

        points = np.concatenate([item.reshape(-1, 3) for item in
            (raw, *old_skeleton.bones, *native_skeleton.bones, *old_skeleton.fingers, *native_skeleton.fingers)])
        finite = points[np.isfinite(points).all(axis=1)]
        low, high = finite.min(axis=0), finite.max(axis=0)
        limits = ((low + high) / 2, max(100., float((high - low).max()) * .57))
        task_id = new_report['job']['task']
        action = TASK_LABELS[task_id] if 0 <= task_id < len(TASK_LABELS) else new_report['job']['description']
        return PairedMotion(key, method, tuple(old.methods), f'Task {task_id:02d} · {action}', time, source_frames,
            raw, old_skeleton, native_skeleton, angles, (old_valid, native_valid), (old_flags, native_flags),
            new_report['quality_bits'], mapping, limits, int(supported.sum()))
