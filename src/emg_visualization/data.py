"""Validated NPZ loading without pickle or modifying preprocessing outputs."""
import json
from pathlib import Path

import numpy as np

from src.motion_reconstruction.paths import PROJECT
from src.task_metadata import tum_task_description_zh


DEFAULT_DATA = PROJECT / 'outputs/TUM/emg_preprocessed/v001_hu2018/subj_001/trial_005/emg_preprocessed.npz'
SIGNALS = {'带通 sEMG': 'emg_bandpass', '肌电包络': 'emg_envelope', '归一化包络': 'emg_normalized'}
MUSCLES = {'BICl': '肱二头肌长头', 'BICs': '肱二头肌短头', 'TRClg': '肱三头肌长头',
           'TRClt': '肱三头肌外侧头', 'PRNT': '旋前圆肌', 'BRD': '肱桡肌',
           'FCR': '桡侧腕屈肌', 'FCU': '尺侧腕屈肌', 'ECU': '尺侧腕伸肌', 'ECR': '桡侧腕伸肌'}


class EmgData:
    def __init__(self, path):
        self.path = Path(path).resolve()
        with np.load(self.path, allow_pickle=False) as archive:
            required = {'t', *SIGNALS.values(), 'channel_names', 'channel_quality_flag',
                        'edge_valid', 'sample_rate_hz', 'amplitude_unit', 'time_unit',
                        'normalization_scope', 'subject_id', 'trial_id', 'task_id', 'version'}
            missing = required - set(archive.files)
            if missing:
                raise ValueError('缺少字段：' + ', '.join(sorted(missing)))
            self.arrays = {key: archive[key].copy() for key in archive.files}
        z = self.arrays
        self.time = np.asarray(z['t'], dtype=float)
        if self.time.ndim != 1 or len(self.time) < 2 or not np.isfinite(self.time).all() or np.any(np.diff(self.time) <= 0):
            raise ValueError('时间轴须为至少两个有限、严格递增的采样点')
        self.names = np.asarray(z['channel_names']).tolist()
        if z['channel_names'].ndim != 1 or not self.names or any(not isinstance(n, str) for n in self.names) or len(set(self.names)) != len(self.names):
            raise ValueError('通道名称须为非空且无重复的一维字符串数组')
        self.fs = float(z['sample_rate_hz'].item())
        if not np.isfinite(self.fs) or self.fs <= 0 or not np.allclose(np.diff(self.time), 1 / self.fs, rtol=1e-5, atol=1e-8):
            raise ValueError('采样率与时间步长不一致')
        if str(z['time_unit'].item()) != 's' or str(z['amplitude_unit'].item()) != 'uV':
            raise ValueError('当前支持时间单位 s、幅值单位 uV')
        for name in SIGNALS.values():
            if z[name].shape != (len(self.time), len(self.names)) or not np.issubdtype(z[name].dtype, np.number) or np.iscomplexobj(z[name]):
                raise ValueError(f'{name} 维度或数值类型不正确')
            if np.isinf(z[name]).any():
                raise ValueError(f'{name} 包含 Inf')
        if z['channel_quality_flag'].shape != (len(self.names),) or not np.issubdtype(z['channel_quality_flag'].dtype, np.integer):
            raise ValueError('通道质量标记维度或类型不正确')
        if z['edge_valid'].shape != self.time.shape or z['edge_valid'].dtype != np.bool_:
            raise ValueError('边界掩码维度或类型不正确')
        self.flags = z['channel_quality_flag']
        self.edge_valid = z['edge_valid']
        self.subject, self.trial, self.task = (int(z[k].item()) for k in ('subject_id', 'trial_id', 'task_id'))
        self.version = str(z['version'].item())
        self.scope = str(z['normalization_scope'].item())
        self.report = {}
        report_path = self.path.parent / 'report.json'
        if report_path.is_file():
            self.report = json.loads(report_path.read_text(encoding='utf-8'))
        self.manifest = {}
        for parent in list(self.path.parents)[:4]:
            manifest_path = parent / 'manifest.json'
            if manifest_path.is_file():
                self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                if self.manifest.get('status') != 'complete':
                    raise ValueError('此结果的 manifest 尚未标记为 complete')
                break
        # Invalid samples remain visible as gaps; never fill missing EMG values.
        for array in z.values():
            array.setflags(write=False)

    @property
    def diagnostic(self):
        return self.scope == 'selected_trials_smoke_only' or self.manifest.get('diagnostic', False)

    @property
    def task_description(self):
        chinese = tum_task_description_zh(self.task)
        english = self.report.get('description', '')
        return f'{chinese}（{english}）' if english else chinese

    def summary(self):
        scope = ('诊断结果：仅所选试验范围内归一化' if self.diagnostic else
                 ('归一化范围：该受试者 Task 1–30 运动峰值' if self.scope == 'subject_channel_max_over_motion_tasks_1_30'
                  else '归一化范围：' + self.scope))
        invalid = sum(np.any(~np.isfinite(self.arrays[k])) for k in SIGNALS.values())
        return (f'{self.version}  |  受试者 {self.subject:03d} · 试验 {self.trial:03d} · '
                f'Task {self.task}: {self.task_description}\n'
                f'{len(self.names)} 通道 · {self.fs:g} Hz · {len(self.time) / self.fs:.3f} s · '
                f'{len(self.time):,} 点  |  {scope}  |  '
                f'{np.count_nonzero(self.flags)} 个通道有质量标记 · {invalid} 类信号含缺失值')


def extrema_trace(time, values, left, right, pixels=1000):
    """Keep minima AND maxima in each display bin; use full data on zoom-in."""
    begin = max(0, np.searchsorted(time, left, side='left') - 1)
    end = min(len(time), np.searchsorted(time, right, side='right') + 1)
    x, y = time[begin:end], values[begin:end]
    pixels = max(1, int(pixels))
    if len(x) <= 2 * pixels or not np.isfinite(y).all():
        return x, y
    width = int(np.ceil(len(x) / pixels))
    groups = len(y) // width
    matrix = y[:groups * width].reshape(groups, width)
    offsets = np.arange(groups) * width
    indices = np.r_[0, len(y) - 1, offsets + matrix.argmin(axis=1),
                    offsets + matrix.argmax(axis=1), np.arange(groups * width, len(y))]
    indices = np.unique(indices)
    return x[indices], y[indices]
