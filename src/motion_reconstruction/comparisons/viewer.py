"""Interactive comparison of the legacy and v002 reconstructions."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .viewer_data import ANGLE_LABELS, ANGLE_LABELS_ZH, MotionLibrary, TrialKey, V002_ROOT
from ..legacy_g99.viewer import QUALITY_NAMES, enable_dpi_awareness
from ..paths import LEGACY_ROOT
from ..tum_native.anchored_quality import REJECT

LEGACY_COLOR = '#27688e'
NATIVE_COLOR = '#d3742f'


def _set_line(artist, points):
    artist.set_data_3d(*points.T)


class Scene:
    def __init__(self, ax, skeleton, title):
        self.ax = ax
        self.ax.set_title(title, loc='left', fontsize=10)
        self.ax.set(xlabel='X mm', ylabel='Y mm', zlabel='Z mm')
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.tick_params(labelsize=7, pad=0)
        self.raw, = ax.plot([], [], [], '.', color='#617585', ms=4, alpha=.65, label='Raw')
        self.pred, = ax.plot([], [], [], 'x', color=skeleton.color, ms=4, label='Predicted')
        self.residuals = Line3DCollection([], colors='#b04e43', alpha=.62, linewidths=.8)
        self.palm = Poly3DCollection([], facecolors=skeleton.color, alpha=.25)
        ax.add_collection3d(self.residuals, autolim=False)
        ax.add_collection3d(self.palm, autolim=False)
        self.bones = [ax.plot([], [], [], 'o-', lw=2, ms=3, color=skeleton.color)[0] for _ in skeleton.bones]
        self.fingers = [ax.plot([], [], [], '-', lw=1.5, color=skeleton.color)[0] for _ in skeleton.fingers]
        ax.legend(fontsize=7, loc='upper right')
        self.error = ax.text2D(.02, .94, '', transform=ax.transAxes, fontsize=8)

    def draw(self, skeleton, raw, frame, comparable, show_pred, show_residuals):
        for lines, points in [(self.bones, skeleton.bones), (self.fingers, skeleton.fingers)]:
            for line, values in zip(lines, points):
                _set_line(line, values[frame])
        palm = skeleton.palm[frame]
        self.palm.set_verts([palm] if np.isfinite(palm).all() else [])
        _set_line(self.raw, raw)
        prediction = skeleton.predicted[frame]
        _set_line(self.pred, prediction)
        self.pred.set_visible(show_pred)
        valid = np.isfinite(prediction).all(1) & np.isfinite(raw).all(1)
        self.residuals.set_segments(np.stack([raw[valid], prediction[valid]], axis=1) if show_residuals else [])
        error = np.linalg.norm(prediction[comparable] - raw[comparable], axis=1)
        self.error.set_text(f'Median {np.median(error):.1f} mm · n={len(error)}'
                            if len(error) else 'No common valid markers')


class ComparisonFigure:
    def __init__(self, figure):
        self.fig = figure
        self.selected = 0
        self.show_pred = False
        self.show_residuals = False
        self.follow_wrist = False
        self.frame = 0

    def load(self, motion):
        self.motion = motion
        self.fig.clear()
        layout = self.fig.add_gridspec(1, 2, width_ratios=(1, 1.4),
                                       left=.025, right=.985, bottom=.074, top=.947, wspace=.18)
        left = layout[0, 0].subgridspec(2, 1, hspace=.12)
        right = layout[0, 1].subgridspec(7, 1, hspace=.41)
        self.scenes = [Scene(self.fig.add_subplot(left[k], projection='3d'), skeleton, title)
                       for k, (skeleton, title) in enumerate(zip(
                           (motion.legacy, motion.native), (f'Legacy G99 · {motion.method.upper()}', 'TUM native · v002')))]
        self.angle_axes, self.cursors, self.dots = [], [], []
        for k, name in enumerate(ANGLE_LABELS):
            ax = self.fig.add_subplot(right[k], sharex=self.angle_axes[0] if self.angle_axes else None)
            self.angle_axes.append(ax)
            ax.set_title(name, loc='left', fontsize=8, pad=1)
            for series, valid, color, label, style in zip(motion.angles, motion.valid,
                    (LEGACY_COLOR, NATIVE_COLOR), ('Legacy', 'v002 aligned'), ('-', '--')):
                values = series[:, k]
                ax.plot(motion.time, values, color='#a9afb3', lw=.8, ls=style)
                ax.plot(motion.time, np.where(valid[:, k], values, np.nan),
                        color=color, lw=1.15, ls=style, label=label)
            if all(not np.isfinite(series[:, k]).any() for series in motion.angles):
                ax.text(.5, .5, 'Reference alignment unavailable', ha='center', va='center',
                        color='#666666', transform=ax.transAxes, fontsize=8)
            self.cursors.append(ax.axvline(motion.time[0], color='#505050', lw=.8))
            self.dots.append([ax.plot([], [], 'o', color=color, ms=3)[0] for color in (LEGACY_COLOR, NATIVE_COLOR)])
            ax.set_ylabel('°', rotation=0, labelpad=6, fontsize=8)
            ax.grid(alpha=.22)
            ax.tick_params(labelsize=7, labelbottom=k == 6)
        self.angle_axes[0].legend(loc='upper right', fontsize=7)
        self.angle_axes[-1].set_xlabel('MoCap time (s) · gray: quality flagged', fontsize=9)
        self.angle_axes[-1].set_xlim(motion.time[0], max(motion.time[-1], motion.time[0] + .002))
        self.select(self.selected)
        self.reset_camera()
        self.draw(0)

    def select(self, index):
        self.selected = index
        for k, axis in enumerate(self.angle_axes):
            axis.set_facecolor('#eef5f8' if k == index else 'white')

    def set_limits(self, center, radius):
        limits = [(value - radius, value + radius) for value in center]
        for scene in self.scenes:
            scene.ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])

    def reset_camera(self):
        self.follow_wrist = False
        self.wrist_radius = 160.
        self.set_limits(*self.motion.limits)
        for scene in self.scenes:
            scene.ax.view_init(elev=22, azim=-65)

    def sync_camera(self, origin):
        if not any(scene.ax is origin for scene in self.scenes):
            return
        for scene in self.scenes:
            if scene.ax is not origin:
                scene.ax.view_init(elev=origin.elev, azim=origin.azim, roll=origin.roll)
                scene.ax.set(xlim=origin.get_xlim(), ylim=origin.get_ylim(), zlim=origin.get_zlim())
        self.fig.canvas.draw_idle()

    def zoom(self, axis, factor):
        if self.follow_wrist:
            self.wrist_radius = float(np.clip(self.wrist_radius * factor, 40, 3000))
            self.draw(self.frame)
        else:
            limits = np.asarray([axis.get_xlim(), axis.get_ylim(), axis.get_zlim()])
            radius = float(np.clip((limits[0, 1] - limits[0, 0]) * factor / 2, 40, 3000))
            self.set_limits(limits.mean(axis=1), radius)
        self.fig.canvas.draw_idle()

    def draw(self, frame):
        self.frame = frame
        motion = self.motion
        raw = motion.markers[frame]
        comparable = (np.isfinite(raw).all(1) & np.isfinite(motion.legacy.predicted[frame]).all(1)
                      & np.isfinite(motion.native.predicted[frame]).all(1))
        for scene, model in zip(self.scenes, (motion.legacy, motion.native)):
            scene.draw(model, raw, frame, comparable, self.show_pred, self.show_residuals)
        if self.follow_wrist:
            wrists = np.stack([motion.legacy.wrist[frame], motion.native.wrist[frame]])
            wrists = wrists[np.isfinite(wrists).all(1)]
            if len(wrists):
                self.set_limits(wrists.mean(0), self.wrist_radius)
        timestamp = motion.time[frame]
        for k, (line, dots) in enumerate(zip(self.cursors, self.dots)):
            line.set_xdata([timestamp, timestamp])
            for dot, values, passing, color in zip(dots, motion.angles, motion.valid, (LEGACY_COLOR, NATIVE_COLOR)):
                dot.set_data([timestamp], [values[frame, k]])
                dot.set_color(color if passing[frame, k] else '#888888')
        self.fig.suptitle(f'{motion.key}  ·  t = {timestamp:.3f} s  ·  MoCap frame {motion.source_frames[frame] + 1}',
                           fontsize=11)


class ComparisonWindow:
    def __init__(self, root, library, key, method):
        import tkinter as tk
        from tkinter import ttk, messagebox
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.root, self.library, self.messagebox = root, library, messagebox
        self.index = library.keys.index(key)
        self.playing = False
        self.updating_slider = False
        self.last_tick = perf_counter()
        self.playhead = 0.
        self.method = tk.StringVar(value=method.upper())
        self.label = tk.StringVar()
        self.detail = tk.StringVar()
        self.speed = tk.StringVar(value='1')
        self.loop = tk.BooleanVar(value=True)
        self.pred = tk.BooleanVar()
        self.residuals = tk.BooleanVar()
        self.location = tk.StringVar()

        root.title('TUM 重建对照 · Legacy G99 / v002')
        root.geometry(f'{min(1450, root.winfo_screenwidth()-60)}x{min(950, root.winfo_screenheight()-130)}')
        root.minsize(980, 680)
        header = ttk.Frame(root, padding=(8, 5))
        header.pack(fill='x')
        self.previous = ttk.Button(header, text='上一条', command=lambda: self.change_trial(-1))
        self.following = ttk.Button(header, text='下一条', command=lambda: self.change_trial(1))
        self.previous.pack(side='left')
        self.following.pack(side='left')
        self.trial_choice = ttk.Combobox(header, state='readonly', width=24, values=[str(k) for k in library.keys])
        self.trial_choice.pack(side='left', padx=6)
        self.trial_choice.bind('<<ComboboxSelected>>', lambda _: self.open_trial(self.trial_choice.current()))
        ttk.Label(header, textvariable=self.label).pack(side='left', padx=5)
        self.method_choice = ttk.Combobox(header, state='readonly', width=6, textvariable=self.method)
        self.method_choice.pack(side='right')
        ttk.Label(header, text='老方法').pack(side='right', padx=4)
        self.method_choice.bind('<<ComboboxSelected>>', lambda _: self.open_trial(self.index, keep_frame=True))

        footer = ttk.Frame(root, padding=(8, 5))
        footer.pack(side='bottom', fill='x')
        transport = ttk.Frame(footer)
        transport.pack(fill='x')
        self.play_button = ttk.Button(transport, text='播放', command=self.toggle)
        self.play_button.pack(side='left')
        for title, callback in [('上一帧', lambda: self.step(-1)), ('下一帧', lambda: self.step(1)),
                                ('起点', lambda: self.jump(0))]:
            ttk.Button(transport, text=title, command=callback).pack(side='left', padx=2)
        self.slider = ttk.Scale(transport, orient='horizontal', command=self.seek)
        self.slider.pack(side='left', fill='x', expand=True, padx=6)
        ttk.Label(transport, text='MoCap 帧').pack(side='left')
        entry = ttk.Entry(transport, width=7, textvariable=self.location)
        entry.pack(side='left', padx=3)
        entry.bind('<Return>', self.go_to_frame)
        ttk.Combobox(transport, width=5, state='readonly', textvariable=self.speed,
                     values=('0.1', '0.25', '0.5', '1', '2', '4')).pack(side='left', padx=3)
        ttk.Checkbutton(transport, text='循环', variable=self.loop).pack(side='left')

        display = ttk.Frame(footer)
        display.pack(fill='x', pady=(5, 0))
        ttk.Button(display, text='联动全景', command=self.reset_camera).pack(side='left')
        ttk.Button(display, text='腕部跟随', command=self.focus_wrist).pack(side='left', padx=4)
        ttk.Checkbutton(display, text='预测标记', variable=self.pred, command=self.draw).pack(side='left')
        ttk.Checkbutton(display, text='误差连线', variable=self.residuals, command=self.draw).pack(side='left', padx=5)
        ttk.Label(display, text='空格播放，方向键逐帧，点击曲线定位，滚轮联动缩放').pack(side='left', padx=5)
        ttk.Label(footer, textvariable=self.detail, wraplength=1220, justify='left').pack(anchor='w', pady=(4, 0))

        self.view = ComparisonFigure(Figure(figsize=(13, 8), dpi=100))
        self.canvas = FigureCanvasTkAgg(self.view.fig, master=root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        for event, callback in [('button_press_event', self.click), ('button_release_event', self.sync_camera),
                                ('motion_notify_event', self.drag), ('scroll_event', self.scroll)]:
            self.canvas.mpl_connect(event, callback)
        for keypress, action in [('<space>', self.toggle), ('<Left>', lambda: self.step(-1)),
                                ('<Right>', lambda: self.step(1))]:
            root.bind(keypress, lambda event, action=action: self.key_command(event, action))
        self.open_trial(self.index, startup=True)
        self.timer = root.after(33, self.tick)
        root.protocol('WM_DELETE_WINDOW', self.close)

    def open_trial(self, index, keep_frame=False, startup=False):
        if not 0 <= index < len(self.library.keys):
            return
        self.pause()
        previous = hasattr(self.view, 'motion')
        previous_frame = self.view.frame if previous and keep_frame else 0
        if previous:
            ax = self.view.scenes[0].ax
            camera = dict(elev=ax.elev, azim=ax.azim, roll=ax.roll)
            look_at_wrist = self.view.follow_wrist
            wrist_radius = self.view.wrist_radius
            limits = np.asarray([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
            default_center, default_radius = self.view.motion.limits
            pan = (limits.mean(axis=1) - default_center) / default_radius
            zoom = np.diff(limits[0])[0] / (2 * default_radius)
        try:
            motion = self.library.load(self.library.keys[index], self.method.get().lower())
        except (OSError, ValueError, KeyError, IndexError) as exc:
            if startup:
                raise
            self.messagebox.showerror('读取试验失败', str(exc), parent=self.root)
            self.trial_choice.current(self.index)
            self.method.set(self.view.motion.method.upper())
            return
        self.index = index
        self.view.load(motion)
        if previous:
            for scene in self.view.scenes:
                scene.ax.view_init(**camera)
            self.view.follow_wrist, self.view.wrist_radius = look_at_wrist, wrist_radius
            if not look_at_wrist:
                center, radius = motion.limits
                self.view.set_limits(center + pan * radius, radius * zoom)
        self.method.set(motion.method.upper())
        self.method_choice['values'] = [value.upper() for value in motion.methods]
        self.trial_choice.current(index)
        self.previous.state(['!disabled'] if index else ['disabled'])
        self.following.state(['!disabled'] if index + 1 < len(self.library.keys) else ['disabled'])
        self.slider.configure(to=max(0, len(motion.time) - 1))
        self.label.set(f'{index+1} / {len(self.library.keys)} · {motion.task}')
        self.jump(previous_frame)

    def change_trial(self, delta):
        self.open_trial(self.index + delta)

    def pause(self):
        self.playing = False
        self.play_button.configure(text='播放')

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            if self.view.frame == len(self.view.motion.time) - 1:
                self.jump(0)
            self.playing = True
            self.last_tick = perf_counter()
            self.play_button.configure(text='暂停')

    def key_command(self, event, action):
        if event.widget.winfo_class() not in ('TEntry', 'Entry', 'TCombobox', 'TButton', 'TCheckbutton'):
            action()
            return 'break'

    def jump(self, index):
        self.pause()
        self.view.frame = int(np.clip(index, 0, len(self.view.motion.time) - 1))
        self.playhead = float(self.view.motion.time[self.view.frame])
        self.draw()

    def step(self, amount):
        self.jump(self.view.frame + amount)

    def seek(self, value):
        if not self.updating_slider:
            self.jump(round(float(value)))

    def go_to_frame(self, _=None):
        try:
            requested = int(self.location.get()) - 1
        except ValueError:
            requested = -1
        matches = np.flatnonzero(self.view.motion.source_frames == requested)
        if len(matches):
            self.jump(matches[0])
        else:
            self.root.bell()
            self.location.set(str(self.view.motion.source_frames[self.view.frame] + 1))
        return 'break'

    def click(self, event):
        if event.button == 1 and event.inaxes in self.view.angle_axes and event.xdata is not None:
            self.view.select(self.view.angle_axes.index(event.inaxes))
            self.jump(np.argmin(abs(self.view.motion.time - event.xdata)))

    def sync_camera(self, event):
        self.view.sync_camera(event.inaxes)

    def drag(self, event):
        if event.button:
            self.sync_camera(event)

    def scroll(self, event):
        if any(scene.ax is event.inaxes for scene in self.view.scenes) and event.step:
            self.view.zoom(event.inaxes, .85 if event.step > 0 else 1 / .85)

    def reset_camera(self):
        self.view.reset_camera()
        self.draw()

    def focus_wrist(self):
        self.view.follow_wrist = not self.view.follow_wrist
        if not self.view.follow_wrist:
            self.view.set_limits(*self.view.motion.limits)
        self.draw()

    def tick(self):
        now = perf_counter()
        if self.playing:
            time = self.view.motion.time
            self.playhead += (now - self.last_tick) * float(self.speed.get())
            if self.playhead >= time[-1]:
                if self.loop.get() and time[-1] > time[0]:
                    self.playhead = time[0] + (self.playhead - time[0]) % (time[-1] - time[0])
                else:
                    self.playhead = time[-1]
                    self.pause()
            frame = np.searchsorted(time, self.playhead, side='right') - 1
            if frame != self.view.frame:
                self.view.frame = frame
                self.draw()
        self.last_tick = now
        self.timer = self.root.after(33, self.tick)

    def draw(self):
        frame, col, motion = self.view.frame, self.view.selected, self.view.motion
        self.view.show_pred, self.view.show_residuals = self.pred.get(), self.residuals.get()
        self.view.draw(frame)
        self.updating_slider = True
        self.slider.set(frame)
        self.updating_slider = False
        self.location.set(str(motion.source_frames[frame] + 1))
        old_flags, new_flags = (int(flags[frame, col]) for flags in motion.flags)
        old_text = [text for bit, text in QUALITY_NAMES.items() if old_flags & bit]
        new_text = [text for bit, text in motion.native_bits.items() if new_flags & int(bit) & REJECT]
        values = [q[frame, col] for q in motion.angles]
        statuses = []
        for label, value, quality, passing in zip(('旧', 'v002'), values, (old_text, new_text), motion.valid):
            if np.isfinite(value):
                if col < 3 and np.linalg.norm(motion.angles[0 if label == '旧' else 1][frame, :3]) > 170:
                    quality.append('旋转向量分支风险')
                text = '、'.join(quality) or ('通过' if passing[frame, col] else '旋转向量分支风险')
                statuses.append(f'{label} {value:.2f}°  {text}')
            else:
                statuses.append(f'{label} 当前帧无可对照角度')
        reference = motion.mapping.reference
        basis = f"参考 trial_{reference['trial']:03d}  {reference['start_s']:.3f}–{reference['end_s']:.3f} s"
        if col >= 3 and np.isfinite(motion.mapping.signs[col-3]):
            sign, offset = motion.mapping.signs[col-3], motion.mapping.offset[col-3]
            basis += f' · 方向 {sign:+.0f} · 偏移 {np.rad2deg(offset):+.2f}°'
        if motion.mapping.notes:
            basis += ' · ' + '；'.join(motion.mapping.notes)
        self.detail.set(f'{ANGLE_LABELS_ZH[col]}   ' + '   |   '.join(statuses) + '\n' + basis)
        self.canvas.draw_idle()

    def close(self):
        self.root.after_cancel(self.timer)
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare saved legacy G99 and TUM native v002 reconstructions.')
    parser.add_argument('--subject', type=int, default=1)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--method', choices=('ekf', 'ik'), default='ekf')
    parser.add_argument('--legacy-root', type=Path, default=LEGACY_ROOT)
    parser.add_argument('--native-root', type=Path, default=V002_ROOT)
    parser.add_argument('--preview', type=Path, help='Export a still frame instead of opening a window')
    parser.add_argument('--frame', type=int, default=1, help='One-based MoCap source frame for the preview')
    args = parser.parse_args(argv)
    library = MotionLibrary(args.legacy_root, args.native_root)
    key = TrialKey(args.subject, args.trial)
    if key not in library.keys:
        parser.error(f'{key} has no paired legacy and v002 results')
    if args.preview:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        view = ComparisonFigure(Figure(figsize=(13, 8), dpi=130))
        FigureCanvasAgg(view.fig)
        motion = library.load(key, args.method)
        matching = np.flatnonzero(motion.source_frames + 1 == args.frame)
        if not len(matching):
            parser.error(f'Frame {args.frame} is not shared by the two files')
        view.load(motion)
        view.show_pred = view.show_residuals = True
        view.draw(matching[0])
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        view.fig.savefig(args.preview)
    else:
        import tkinter as tk

        enable_dpi_awareness()
        root = tk.Tk()
        try:
            viewer = ComparisonWindow(root, library, key, args.method)
        except (OSError, ValueError, KeyError) as exc:
            root.destroy()
            parser.exit(1, f'无法打开对照查看器：{exc}\n')
        root.mainloop()
