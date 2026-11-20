"""Synchronized four-coordinate motion, angles and ten-channel EMG viewer."""
import argparse
from pathlib import Path
import time

import numpy as np

from src.emg_visualization.data import DEFAULT_DATA, SIGNALS
from src.emg_visualization.plotting import configure_fonts, heatmap_plot, update_wave, wave_plot
from src.emg_visualization.viewer import EmgViewer
from src.motion_reconstruction.paths import OUTPUT_ROOT, PROJECT
from .data import ANGLE_COLORS, load_synchronized, reconstruction_root
from .plotting import angle_tracks, build_scene, reset_camera, update_scene


class CombinedViewer(EmgViewer):
    """Reuse the EMG viewer's file chooser and directory-navigation contract."""
    def __init__(self, root, aligned, reconstruction=None):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.root, self.aligned, self.data = root, aligned, aligned.emg
        self.reconstruction = aligned.result_root
        self.reconstruction_roots = {name: reconstruction_root(name) for name in ('legacy_g99', 'tum_native')}
        self.reconstruction_roots[aligned.algorithm] = self.reconstruction
        self.pending = self.play_job = None
        self.playing = False
        self.slider_busy = False
        self.last_error = ''
        self.index = int(np.flatnonzero(aligned.overlap)[0])
        self.backgrounds, self.dynamic = {}, {}
        self.linked_axes = []
        self.library = self.infer_library(self.data.path)
        root.title('肘腕运动与 sEMG 同步查看器')
        root.geometry(f'{min(1600, root.winfo_screenwidth() - 60)}x{min(1050, root.winfo_screenheight() - 90)}')
        root.minsize(1120, 780)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)
        header = ttk.Frame(root, padding=(8, 6))
        header.grid(row=0, column=0, sticky='ew')
        ttk.Button(header, text='打开肌电数据…', command=self.open_file).pack(side='left')
        ttk.Button(header, text='选择数据根目录…', command=self.open_library).pack(side='left', padx=5)
        self.method, self.subject, self.trial = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.choices = {}
        for label, variable, width, callback in (
                ('方法版本', self.method, 25, self.choose_method), ('受试者', self.subject, 11, self.choose_subject),
                ('试验', self.trial, 11, self.choose_trial)):
            ttk.Label(header, text=label).pack(side='left', padx=(8, 3))
            box = ttk.Combobox(header, textvariable=variable, width=width, state='readonly')
            box.pack(side='left')
            box.bind('<<ComboboxSelected>>', callback)
            self.choices[label] = box
        self.previous = ttk.Button(header, text='上一条', command=lambda: self.step(-1))
        self.previous.pack(side='left', padx=5)
        self.next = ttk.Button(header, text='下一条', command=lambda: self.step(1))
        self.next.pack(side='left')
        self.info = tk.StringVar()
        ttk.Label(root, textvariable=self.info, padding=(10, 3), wraplength=1500).grid(row=1, column=0, sticky='ew')
        playback = ttk.Frame(root, padding=(8, 3))
        playback.grid(row=2, column=0, sticky='ew')
        playback.columnconfigure(4, weight=1)
        self.play_button = ttk.Button(playback, text='播放', command=self.toggle_play)
        self.play_button.grid(row=0, column=0, padx=3)
        ttk.Button(playback, text='−1采样', command=lambda: self.step_sample(-1)).grid(row=0, column=1, padx=3)
        ttk.Button(playback, text='+1采样', command=lambda: self.step_sample(1)).grid(row=0, column=2, padx=3)
        self.clock = tk.StringVar()
        ttk.Label(playback, textvariable=self.clock, width=23).grid(row=0, column=3, padx=5)
        self.slider = ttk.Scale(playback, command=self.seek_slider)
        self.slider.grid(row=0, column=4, sticky='ew', padx=5)
        self.speed = tk.StringVar(value='1')
        ttk.Label(playback, text='倍速').grid(row=0, column=5)
        speed_box = ttk.Combobox(playback, textvariable=self.speed, values=['0.25', '0.5', '1', '2', '4'], width=5, state='readonly')
        speed_box.grid(row=0, column=6, padx=5)
        speed_box.bind('<<ComboboxSelected>>', self.restart_clock)
        controls = ttk.Frame(root, padding=(8, 3))
        controls.grid(row=3, column=0, sticky='ew')
        self.algorithm = tk.StringVar(value=aligned.algorithm)
        ttk.Label(controls, text='运动重建').pack(side='left', padx=(0, 4))
        algorithm_box = ttk.Combobox(controls, textvariable=self.algorithm,
                                    values=['tum_native', 'legacy_g99'], width=12, state='readonly')
        algorithm_box.pack(side='left', padx=(0, 8))
        algorithm_box.bind('<<ComboboxSelected>>', self.choose_algorithm)
        self.signal = tk.StringVar(value='归一化包络')
        box = ttk.Combobox(controls, textvariable=self.signal, values=list(SIGNALS), state='readonly', width=15)
        box.pack(side='left')
        box.bind('<<ComboboxSelected>>', self.refresh_emg)
        self.uniform, self.overlay, self.full, self.heat = (tk.BooleanVar(value=False) for _ in range(4))
        for text, variable, callback in [('统一纵轴', self.uniform, self.refresh_emg), ('叠加包络', self.overlay, self.refresh_emg),
                                          ('肌电热图', self.heat, self.refresh_emg), ('完整记录', self.full, self.reset_range)]:
            ttk.Checkbutton(controls, text=text, variable=variable, command=callback).pack(side='left', padx=7)
        ttk.Button(controls, text='重置时间范围', command=self.reset_range).pack(side='left', padx=5)
        ttk.Button(controls, text='重置三维视角', command=self.reset_scene).pack(side='left', padx=5)
        ttk.Button(controls, text='导出同步总览…', command=self.export).pack(side='right')
        body = ttk.Panedwindow(root, orient='horizontal')
        body.grid(row=4, column=0, sticky='nsew', padx=6, pady=3)
        left, right = ttk.Frame(body), ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=3)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.scene_figure = Figure(figsize=(5, 5), dpi=100)
        self.scene_canvas = FigureCanvasTkAgg(self.scene_figure, master=left)
        self.scene_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self.values = tk.StringVar()
        ttk.Label(left, textvariable=self.values, padding=(10, 6), wraplength=390,
                  font=('Microsoft YaHei UI', 10), justify='left').grid(row=1, column=0, sticky='ew')
        self.motion_info = tk.StringVar()
        ttk.Label(left, textvariable=self.motion_info,
                  padding=10, wraplength=390, foreground='#61717b').grid(row=2, column=0, sticky='ew')
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=3, uniform='plot_area')
        right.rowconfigure(1, weight=7, uniform='plot_area')
        angle_panel = ttk.Frame(right)
        angle_panel.grid(row=0, column=0, sticky='nsew')
        angle_panel.pack_propagate(False)
        self.angle_figure = Figure(figsize=(10, 3.5), dpi=100)
        self.angle_canvas = FigureCanvasTkAgg(self.angle_figure, master=angle_panel)
        self.angle_toolbar = NavigationToolbar2Tk(self.angle_canvas, angle_panel, pack_toolbar=False)
        self.angle_toolbar.pack(side='bottom', fill='x')
        self.angle_canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        emg_panel = ttk.Frame(right)
        emg_panel.grid(row=1, column=0, sticky='nsew')
        emg_panel.pack_propagate(False)
        self.emg_figure = Figure(figsize=(10, 6), dpi=100)
        self.emg_canvas = FigureCanvasTkAgg(self.emg_figure, master=emg_panel)
        self.emg_toolbar = NavigationToolbar2Tk(self.emg_canvas, emg_panel, pack_toolbar=False)
        self.emg_toolbar.pack(side='bottom', fill='x')
        self.emg_canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, padding=(8, 4), wraplength=1500).grid(row=5, column=0, sticky='ew')
        for canvas in (self.angle_canvas, self.emg_canvas):
            canvas.mpl_connect('button_press_event', self.click_time)
            canvas.mpl_connect('draw_event', self.cache_background)
        self.emg_canvas.mpl_connect('resize_event', self.schedule_trace)
        self.sync_navigation()
        self.build_plots()
        root.protocol('WM_DELETE_WINDOW', self.close)

    def paired(self, paths, algorithm=None):
        algorithm = algorithm or self.aligned.algorithm
        filename = 'emg_aligned_native.npz' if algorithm == 'tum_native' else 'emg_aligned_distal.npz'
        root = self.reconstruction_roots[algorithm]
        return sorted(p.resolve() for p in paths if (root / p.parent.parent.name / 'trials' /
                                                     p.parent.name / filename).is_file())

    def choose_algorithm(self, event=None):
        requested = self.algorithm.get()
        previous = self.aligned.algorithm
        paths = [self.data.path, *self.paired(self.data.path.parents[2].glob('subj_*/trial_*/emg_preprocessed.npz'), requested)]
        for path in dict.fromkeys(paths):
            if self.load(path, show_error=False, algorithm=requested):
                return
        self.algorithm.set(previous)
        from tkinter import messagebox
        messagebox.showerror('无法切换重建方法', self.last_error or '没有匹配的同步结果', parent=self.root)

    def sync_navigation(self):
        super().sync_navigation()
        self.paths = self.paired(self.paths)
        if self.library is not None:
            method_dir = self.data.path.parents[2]
            self.choices['受试者'].configure(values=sorted({p.parent.parent.name for p in self.paths}))
            self.choices['试验'].configure(values=[p.parent.name for p in self.paths if p.parent.parent.name == self.subject.get()])
        index = self.paths.index(self.data.path) if self.data.path in self.paths else -1
        self.previous.configure(state='normal' if index > 0 else 'disabled')
        self.next.configure(state='normal' if 0 <= index < len(self.paths) - 1 else 'disabled')
        self.algorithm.set(self.aligned.algorithm)
        self.root.title(f'肘腕运动与 sEMG 同步查看器 · {self.aligned.algorithm_label}')
        self.motion_info.set(f'动画使用{self.aligned.motion_description}；右侧显示肘腕四角。\n'
                             '紫色横杆提示前臂旋转；手指为固定示意。\n正负方向沿用各自模型定义。')
        self.info.set(self.aligned.algorithm_label + ' | ' + self.data.summary())
        self.status.set(f'脉冲时长匹配候选 · 偏移 {self.aligned.sync["offset_s"]:.4f} s · '
                        f'时钟尺度 {self.aligned.sync.get("clock_scale", 1):g} · '
                        f'硬件延迟／漂移未独立验证；{self.aligned.status_note}。默认范围仅要求四角有限。')

    def choose_method(self, event=None):
        if self.library is not None:
            paths = self.paired((self.library / self.method.get()).glob('subj_*/trial_*/emg_preprocessed.npz'))
            if paths:
                self.load(paths[0])
            else:
                self.sync_navigation()

    def choose_subject(self, event=None):
        if self.library is not None:
            paths = self.paired((self.library / self.method.get() / self.subject.get()).glob('trial_*/emg_preprocessed.npz'))
            if paths:
                self.load(paths[0])

    def open_library(self):
        from tkinter import filedialog, messagebox
        directory = filedialog.askdirectory(parent=self.root, initialdir=self.library or self.data.path.parent)
        if directory:
            paths = self.paired(p for p in Path(directory).glob('*/subj_*/trial_*/emg_preprocessed.npz')
                                if not p.relative_to(directory).parts[0].startswith(('_', '.')))
            if paths:
                self.load(paths[0])
            else:
                messagebox.showinfo('无匹配数据', '没有同时包含预处理肌电和同步角度标签的试验。', parent=self.root)

    def load(self, path, show_error=True, algorithm=None):
        from tkinter import messagebox
        try:
            algorithm = algorithm or self.aligned.algorithm
            candidate = load_synchronized(path, self.reconstruction_roots[algorithm], algorithm)
        except Exception as error:
            self.last_error = str(error)
            if show_error:
                messagebox.showerror('无法同步加载', str(error), parent=self.root)
            self.sync_navigation()
            return False
        self.pause()
        self.last_error = ''
        self.aligned, self.data = candidate, candidate.emg
        self.reconstruction = candidate.result_root
        self.library = self.infer_library(self.data.path)
        self.index = int(np.flatnonzero(candidate.overlap)[0])
        self.sync_navigation()
        self.build_plots()
        return True

    def build_plots(self):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        self.backgrounds.clear()
        self.angle_axes = angle_tracks(self.angle_figure, self.aligned)
        self.scene = build_scene(self.scene_figure, self.aligned)
        self.angle_cursors = [ax.axvline(self.aligned.time[self.index], color='#bb463e', lw=.8, animated=True) for ax in self.angle_axes]
        self.angle_dots = [ax.plot([], [], 'o', ms=4, color=color, animated=True)[0]
                           for ax, color in zip(self.angle_axes, ANGLE_COLORS)]
        self.dynamic[self.angle_canvas] = self.angle_cursors + self.angle_dots
        self.refresh_emg()
        self.reset_range()

    def refresh_emg(self, event=None):
        if not hasattr(self, 'angle_axes'):
            return
        old_range = self.angle_axes[0].get_xlim()
        self.backgrounds.pop(self.emg_canvas, None)
        self.wave_state = None
        if self.heat.get():
            axis = heatmap_plot(self.emg_figure, self.data)
            axes = [axis]
            axis.set_position([.23, .1, .69, .82])
            self.emg_figure.axes[-1].set_position([.94, .1, .02, .82])
        else:
            self.wave_state = wave_plot(self.emg_figure, self.data, SIGNALS[self.signal.get()],
                                        self.overlay.get(), self.uniform.get())
            axes = list(self.wave_state['axes'])
            self.emg_figure.subplots_adjust(left=.23, right=.92, top=.95, bottom=.065, hspace=.12)
            from matplotlib.ticker import MaxNLocator
            for ax in axes:
                ax.yaxis.set_major_locator(MaxNLocator(2))
                ax.yaxis.label.set_fontsize(6.5)
                ax.tick_params(labelsize=6, pad=1)
            self.emg_figure._suptitle.set_fontsize(9)
        self.emg_axes = axes
        # wave_plot already links its ten axes; link that group's root once.
        axes[0].sharex(self.angle_axes[0])
        self.linked_axes = self.angle_axes + axes
        self.emg_cursors = [ax.axvline(self.aligned.time[self.index], color='#bb463e', lw=.8, animated=True) for ax in axes]
        self.dynamic[self.emg_canvas] = self.emg_cursors
        for ax in self.linked_axes:
            ax.callbacks.connect('xlim_changed', self.schedule_trace)
        self.angle_axes[0].set_xlim(old_range)
        self.angle_toolbar.update()
        self.emg_toolbar.update()
        self.angle_canvas.draw_idle()
        self.emg_canvas.draw_idle()
        self.set_index(self.index)

    def reset_range(self):
        self.pause()
        self.bounds = (self.aligned.time[0], self.aligned.time[-1]) if self.full.get() else self.aligned.common_range
        self.angle_axes[0].set_xlim(*self.bounds)
        self.slider.configure(from_=self.bounds[0], to=self.bounds[1])
        self.set_index(self.aligned.nearest(np.clip(self.aligned.time[self.index], *self.bounds)))
        self.angle_canvas.draw_idle()
        self.emg_canvas.draw_idle()

    def schedule_trace(self, event=None):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
        self.pending = self.root.after(80, self.redraw_trace)

    def redraw_trace(self):
        self.pending = None
        if self.wave_state is not None:
            update_wave(self.wave_state, max(400, self.emg_canvas.get_tk_widget().winfo_width()))
        self.angle_canvas.draw_idle()
        self.emg_canvas.draw_idle()

    def cache_background(self, event):
        canvas = event.canvas
        self.backgrounds[canvas] = canvas.copy_from_bbox(canvas.figure.bbox)
        self.blit(canvas)

    def blit(self, canvas):
        if canvas not in self.backgrounds:
            canvas.draw_idle()
            return
        canvas.restore_region(self.backgrounds[canvas])
        for artist in self.dynamic.get(canvas, []):
            artist.axes.draw_artist(artist)
        canvas.blit(canvas.figure.bbox)

    def set_index(self, index):
        self.index = int(np.clip(index, 0, len(self.aligned.time) - 1))
        seconds = self.aligned.time[self.index]
        self.slider_busy = True
        self.slider.set(seconds)
        self.slider_busy = False
        self.clock.set(f't = {seconds:.4f} s')
        for line in self.angle_cursors + self.emg_cursors:
            line.set_xdata([seconds, seconds])
        lines = []
        for j, dot in enumerate(self.angle_dots):
            value = self.aligned.degrees[self.index, j]
            dot.set_data([seconds] if np.isfinite(value) else [], [value] if np.isfinite(value) else [])
            text = f'{value:.2f}°' if np.isfinite(value) else '缺失'
            lines.append(f'{self.aligned.angle_names[j]}：{text}\n  {self.aligned.quality_text(self.aligned.flags[self.index, j])}')
        self.values.set('\n\n'.join(lines))
        update_scene(self.scene, self.aligned, self.index)
        self.scene_canvas.draw_idle()
        self.blit(self.angle_canvas)
        self.blit(self.emg_canvas)

    def seek_slider(self, value):
        if not self.slider_busy and hasattr(self, 'angle_cursors'):
            self.pause()
            self.set_index(self.aligned.nearest(float(value)))

    def click_time(self, event):
        toolbar = self.angle_toolbar if event.canvas is self.angle_canvas else self.emg_toolbar
        if event.button == 1 and event.xdata is not None and event.inaxes in self.linked_axes and not toolbar.mode:
            self.pause()
            self.set_index(self.aligned.nearest(np.clip(event.xdata, *self.bounds)))

    def step_sample(self, delta):
        self.pause()
        target = np.clip(self.index + delta, self.aligned.nearest(self.bounds[0]), self.aligned.nearest(self.bounds[1]))
        self.set_index(int(target))

    def restart_clock(self, event=None):
        self.play_started = time.monotonic()
        self.play_base = self.aligned.time[self.index]

    def toggle_play(self):
        if self.playing:
            self.pause()
        else:
            if self.aligned.time[self.index] >= self.bounds[1]:
                self.set_index(self.aligned.nearest(self.bounds[0]))
            self.playing = True
            self.play_button.configure(text='暂停')
            self.restart_clock()
            self.tick()

    def tick(self):
        self.play_job = None
        if not self.playing:
            return
        target = self.play_base + (time.monotonic() - self.play_started) * float(self.speed.get())
        self.set_index(self.aligned.nearest(min(target, self.bounds[1])))
        if target >= self.bounds[1]:
            self.pause()
        else:
            self.play_job = self.root.after(40, self.tick)

    def pause(self):
        self.playing = False
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None
        if hasattr(self, 'play_button'):
            self.play_button.configure(text='播放')

    def reset_scene(self):
        reset_camera(self.scene)
        self.scene_canvas.draw_idle()

    def export(self):
        from tkinter import filedialog
        directory = self.reconstruction / 'analysis/emg_kinematics'
        directory.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(parent=self.root, initialdir=directory,
                                           initialfile=f'subj_{self.data.subject:03d}_trial_{self.data.trial:03d}_overview.png',
                                           defaultextension='.png', filetypes=[('PNG', '*.png')])
        if path:
            from .preview import save_preview
            save_preview(self.aligned, path, self.aligned.time[self.index], SIGNALS[self.signal.get()],
                         self.uniform.get(), self.heat.get(), self.angle_axes[0].get_xlim())

    def close(self):
        self.pause()
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--algorithm', choices=['tum_native', 'legacy_g99'], default='tum_native',
                        help='Motion reconstruction method; tum_native uses v002 only')
    parser.add_argument('--reconstruction', type=Path, help='Selected method result root, containing subj_*/trials/')
    parser.add_argument('--preview', type=Path, help='Export synchronized overview without a window')
    parser.add_argument('--time', type=float, help='Initial time in EMG recording-relative seconds')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args(argv)
    configure_fonts()
    try:
        data = load_synchronized(args.data, args.reconstruction, args.algorithm)
    except Exception as error:
        parser.exit(1, f'无法同步加载：{error}\n')
    if args.preview:
        from .preview import save_preview
        save_preview(data, args.preview, args.time)
    elif args.self_check:
        from .self_check import run_checks
        run_checks(data, args.reconstruction)
    else:
        import tkinter as tk
        root = tk.Tk()
        viewer = CombinedViewer(root, data, args.reconstruction)
        if args.time is not None:
            if args.time < viewer.bounds[0] or args.time > viewer.bounds[1]:
                viewer.full.set(True)
                viewer.reset_range()
            viewer.set_index(data.nearest(args.time))
        root.mainloop()
