"""Shared plot builders for the desktop viewer and reproducible PNG previews."""
import numpy as np
from scipy.signal import welch
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator

from .data import MUSCLES, extrema_trace


COLORS = ['#286c96', '#397eaa', '#95624a', '#ab785c', '#6e66a3',
          '#397f73', '#4b9185', '#b17d38', '#557b9b', '#8a694b']


def configure_fonts():
    import matplotlib as mpl
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in ('Microsoft YaHei', 'Microsoft YaHei UI', 'SimHei', 'Noto Sans CJK SC') if name in available), 'DejaVu Sans')
    mpl.rcParams.update({'font.sans-serif': [chosen, 'DejaVu Sans'], 'axes.unicode_minus': False,
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff'})


def spans(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def shade_edges(ax, time, valid, fs):
    for begin, end in spans(~valid):
        ax.axvspan(time[begin], time[end - 1] + 1 / fs, color='#e4b974', alpha=.23, lw=0)


def display_time(data, relative):
    return data.time - data.time[0] if relative else data.time


def wave_plot(figure, data, signal='emg_bandpass', overlay=True, uniform=False, relative=True):
    figure.clear()
    time = display_time(data, relative)
    count = len(data.names)
    axes = figure.subplots(count, 1, sharex=True, squeeze=False)[:, 0]
    lines, envelope_lines = [], []
    values = data.arrays[signal]
    global_values = np.abs(values[np.isfinite(values)])
    global_max = float(global_values.max()) if len(global_values) else 1.0
    for index, (name, ax) in enumerate(zip(data.names, axes)):
        color = COLORS[index % len(COLORS)]
        line, = ax.plot([], [], color=color, lw=.65, label='带通 sEMG' if signal == 'emg_bandpass' else '包络')
        lines.append(line)
        envelope_line = None
        if overlay and signal == 'emg_bandpass':
            envelope_line, = ax.plot([], [], color='#d05e38', lw=1.2, label='2 Hz 包络')
        envelope_lines.append(envelope_line)
        shade_edges(ax, time, data.edge_valid, data.fs)
        finite = values[:, index][np.isfinite(values[:, index])]
        maximum = global_max if uniform else (float(np.max(np.abs(finite))) if len(finite) else 1.)
        if envelope_line is not None:
            env = data.arrays['emg_envelope'][:, index]
            if np.isfinite(env).any():
                maximum = max(maximum, float(np.nanmax(env)))
        maximum = max(maximum, 1e-9)
        if signal == 'emg_normalized':
            ax.set_ylim(0, max(1.05, maximum * 1.05) if uniform else maximum * 1.05)
        elif signal == 'emg_envelope':
            ax.set_ylim(0, maximum * 1.1)
        else:
            ax.set_ylim(-maximum * 1.08, maximum * 1.08)
        ax.set_ylabel(f'{name}\n{MUSCLES.get(name, name)}', rotation=0, ha='right', va='center',
                      color='#a64b31' if data.flags[index] else '#253746', fontsize=8)
        ax.tick_params(axis='both', labelsize=7, pad=2)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.grid(axis='x', alpha=.15)
        if data.flags[index] or not len(finite):
            ax.text(.99, .7, f'质量标记 {data.flags[index]}' if data.flags[index] else '无有效样本',
                    transform=ax.transAxes, ha='right', color='#a64b31', fontsize=8)
    axes[-1].set_xlabel('从记录起点计时 (s)' if relative else '原始 EMG 时间 (s)')
    axes[0].set_xlim(time[0], time[-1])
    axes[0].legend(loc='upper right', fontsize=7, ncol=2, frameon=False)
    unit = '无量纲' if signal == 'emg_normalized' else 'μV'
    figure.suptitle(f'逐通道信号 · {unit}    |    浅橙底纹：滤波边界敏感区', fontsize=10, y=.993)
    figure.subplots_adjust(left=.155, right=.985, bottom=.06, top=.955, hspace=.22)
    state = dict(axes=axes, lines=lines, envelopes=envelope_lines, time=time,
                 data=data, signal=signal)
    update_wave(state, 1000)
    return state


def update_wave(state, pixels):
    left, right = state['axes'][0].get_xlim()
    for index, line in enumerate(state['lines']):
        line.set_data(*extrema_trace(state['time'], state['data'].arrays[state['signal']][:, index],
                                    left, right, pixels))
        envelope_line = state['envelopes'][index]
        if envelope_line is not None:
            envelope_line.set_data(*extrema_trace(state['time'], state['data'].arrays['emg_envelope'][:, index],
                                                 left, right, pixels))


def heatmap_plot(figure, data, relative=True):
    figure.clear()
    ax = figure.subplots()
    time = display_time(data, relative)
    values = data.arrays['emg_normalized']
    boundaries = np.linspace(0, len(time), min(len(time), 1800) + 1, dtype=int)
    means = np.full((len(boundaries) - 1, len(data.names)), np.nan)
    for i, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        block = values[begin:end]
        counts = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), counts, out=means[i], where=counts > 0)
    edges = np.r_[time[boundaries[:-1]], time[-1] + 1 / data.fs]
    palette = colormaps['Blues'].copy()
    palette.set_bad('#d8d8d8')
    mesh = ax.pcolormesh(edges, np.arange(len(data.names) + 1), np.ma.masked_invalid(means.T),
                         cmap=palette, vmin=0, vmax=1, shading='flat', rasterized=True)
    ax.set_yticks(np.arange(len(data.names)) + .5, [f'{n} · {MUSCLES.get(n, n)}' for n in data.names])
    ax.invert_yaxis()
    shade_edges(ax, time, data.edge_valid, data.fs)
    ax.set_xlabel('从记录起点计时 (s)' if relative else '原始 EMG 时间 (s)')
    ax.set_title('归一化包络热图 · 时间分箱均值；灰色为缺失值；浅橙为边界区', fontsize=10, pad=15)
    figure.colorbar(mesh, ax=ax, label='归一化激活（0–1）', pad=.025)
    figure.subplots_adjust(left=.18, right=.96, bottom=.1, top=.92)
    return ax


def spectrum_plot(figure, data):
    figure.clear()
    count = len(data.names)
    rows = int(np.ceil(count / 2))
    axes = figure.subplots(rows, 2, squeeze=False, sharex=True).ravel()
    for index, name in enumerate(data.names):
        ax = axes[index]
        values = data.arrays['emg_bandpass'][:, index]
        segments = spans(data.edge_valid & np.isfinite(values))
        begin, end = max(segments, key=lambda pair: pair[1] - pair[0], default=(0, 0))
        if end - begin >= 32:
            frequency, power = welch(values[begin:end], fs=data.fs, nperseg=min(2048, end - begin))
            ax.plot(frequency, 10 * np.log10(np.maximum(power, np.finfo(float).tiny)),
                    color=COLORS[index % len(COLORS)], lw=.9)
        else:
            ax.text(.5, .5, '有效连续样本不足', transform=ax.transAxes, ha='center')
        for frequency in (20, 600):
            ax.axvline(frequency, color='#56847c', ls='--', lw=.7, alpha=.7)
        for frequency in (50, 100):
            ax.axvline(frequency, color='#ba8f50', ls=':', lw=.7, alpha=.7)
        ax.set_title(f'{name} · {MUSCLES.get(name, name)}', fontsize=9, loc='left')
        ax.set_xlim(0, data.fs / 2)
        ax.grid(alpha=.15)
        ax.tick_params(labelsize=7)
        if index % 2 == 0:
            ax.set_ylabel('PSD [dB re 1 μV²/Hz]', fontsize=7)
        if index >= 2 * (rows - 1):
            ax.set_xlabel('频率 (Hz)')
    for ax in axes[count:]:
        ax.set_visible(False)
    figure.suptitle('带通信号 Welch 功率谱 · 最长连续有效非边界片段\n'
                    '虚线：20/600 Hz；点线：50/100 Hz 参考，不代表已检测到干扰', fontsize=10, y=.99)
    figure.subplots_adjust(left=.085, right=.98, bottom=.07, top=.9, hspace=.62, wspace=.25)
