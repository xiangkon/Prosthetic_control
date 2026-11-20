"""Interactive ten-channel EMG viewer with waveform, heatmap and PSD tabs."""
import argparse
from pathlib import Path

from .data import DEFAULT_DATA, SIGNALS, EmgData
from .plotting import configure_fonts, heatmap_plot, spectrum_plot, update_wave, wave_plot


class EmgViewer:
    def __init__(self, root, data):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.root, self.data = root, data
        self.pending = None
        self.wave_state = None
        self.last_error = ''
        root.title('sEMG 信号查看器')
        root.geometry('1450x1000')
        root.minsize(1050, 760)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)
        style = ttk.Style(root)
        style.configure('Info.TLabel', font=('Microsoft YaHei UI', 10))
        header = ttk.Frame(root, padding=(10, 8))
        header.grid(row=0, column=0, sticky='ew')
        ttk.Button(header, text='打开数据…', command=self.open_file).pack(side='left', padx=(0, 5))
        ttk.Button(header, text='选择数据根目录…', command=self.open_library).pack(side='left', padx=5)
        self.method = tk.StringVar()
        self.subject = tk.StringVar()
        self.trial = tk.StringVar()
        self.choices = {}
        for label, variable, width, callback in (
                ('方法版本', self.method, 22, self.choose_method),
                ('受试者', self.subject, 12, self.choose_subject),
                ('试验', self.trial, 12, self.choose_trial)):
            ttk.Label(header, text=label).pack(side='left', padx=(9, 4))
            box = ttk.Combobox(header, textvariable=variable, width=width, state='readonly')
            box.pack(side='left')
            box.bind('<<ComboboxSelected>>', callback)
            self.choices[label] = box
        self.previous = ttk.Button(header, text='上一条', command=lambda: self.step(-1))
        self.previous.pack(side='left', padx=(10, 4))
        self.next = ttk.Button(header, text='下一条', command=lambda: self.step(1))
        self.next.pack(side='left')
        self.info = tk.StringVar()
        ttk.Label(root, textvariable=self.info, style='Info.TLabel', padding=(12, 5),
                  wraplength=1400).grid(row=1, column=0, sticky='ew')
        controls = ttk.Frame(root, padding=(10, 5))
        controls.grid(row=2, column=0, sticky='ew')
        ttk.Label(controls, text='波形类型').pack(side='left', padx=(0, 5))
        self.signal = tk.StringVar(value='带通 sEMG')
        self.signal_box = ttk.Combobox(controls, textvariable=self.signal, values=list(SIGNALS), width=16, state='readonly')
        self.signal_box.pack(side='left')
        self.signal_box.bind('<<ComboboxSelected>>', self.refresh)
        self.overlay = tk.BooleanVar(value=True)
        self.uniform = tk.BooleanVar(value=False)
        self.relative = tk.BooleanVar(value=True)
        for label, variable in (('叠加包络（仅带通信号）', self.overlay), ('统一纵轴', self.uniform),
                                ('从记录起点计时', self.relative)):
            ttk.Checkbutton(controls, text=label, variable=variable, command=self.refresh).pack(side='left', padx=10)
        ttk.Button(controls, text='重置当前视图', command=self.refresh).pack(side='left', padx=5)
        ttk.Button(controls, text='导出当前图…', command=self.export).pack(side='right')
        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=3, column=0, sticky='nsew', padx=8, pady=(2, 4))
        self.figures, self.canvases, self.toolbars = {}, {}, {}
        for name in ('通道波形', '激活热图', '功率谱'):
            panel = ttk.Frame(self.notebook)
            self.notebook.add(panel, text=name)
            figure = Figure(figsize=(13, 9), dpi=100)
            canvas = FigureCanvasTkAgg(figure, master=panel)
            toolbar = NavigationToolbar2Tk(canvas, panel, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side='bottom', fill='x')
            canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
            self.figures[name], self.canvases[name], self.toolbars[name] = figure, canvas, toolbar
            if name == '通道波形':
                canvas.mpl_connect('resize_event', self.schedule_wave)
        self.notebook.bind('<<NotebookTabChanged>>', self.render)
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, padding=(10, 4), wraplength=1400).grid(row=4, column=0, sticky='ew')
        self.library = self.infer_library(data.path)
        self.sync_navigation()
        self.refresh()
        root.protocol('WM_DELETE_WINDOW', self.close)

    @staticmethod
    def infer_library(path):
        if path.parent.name.startswith('trial_') and path.parent.parent.name.startswith('subj_'):
            return path.parents[3]
        return None

    def sync_navigation(self):
        self.info.set(self.data.summary())
        self.status.set(str(self.data.path))
        self.root.title(f'sEMG · {self.data.version} · subj_{self.data.subject:03d} / trial_{self.data.trial:03d} · '
                        f'Task {self.data.task} {self.data.task_description}')
        self.paths = []
        if self.library is None:
            for box in self.choices.values():
                box.configure(values=[])
            self.method.set('自选文件')
            self.subject.set('')
            self.trial.set('')
        else:
            method_dir = self.data.path.parents[2]
            methods = sorted(p.name for p in self.library.iterdir()
                             if p.is_dir() and not p.name.startswith(('_', '.'))
                             and any(p.glob('subj_*/trial_*/emg_preprocessed.npz')))
            subjects = sorted(p.name for p in method_dir.glob('subj_*') if any(p.glob('trial_*/emg_preprocessed.npz')))
            trials = sorted(p.parent.name for p in self.data.path.parent.parent.glob('trial_*/emg_preprocessed.npz'))
            for label, values in (('方法版本', methods), ('受试者', subjects), ('试验', trials)):
                self.choices[label].configure(values=values)
            self.method.set(method_dir.name)
            self.subject.set(self.data.path.parent.parent.name)
            self.trial.set(self.data.path.parent.name)
            self.paths = sorted(p.resolve() for p in method_dir.glob('subj_*/trial_*/emg_preprocessed.npz'))
        index = self.paths.index(self.data.path) if self.data.path in self.paths else -1
        self.previous.configure(state='normal' if index > 0 else 'disabled')
        self.next.configure(state='normal' if 0 <= index < len(self.paths) - 1 else 'disabled')

    def load(self, path, show_error=True):
        try:
            candidate = EmgData(path)
        except Exception as error:
            from tkinter import messagebox
            self.last_error = str(error)
            if show_error:
                messagebox.showerror('无法加载数据', str(error), parent=self.root)
            self.sync_navigation()
            return False
        self.last_error = ''
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        self.data = candidate
        self.library = self.infer_library(candidate.path)
        self.sync_navigation()
        self.refresh()
        return True

    def open_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(parent=self.root, title='选择预处理肌电数据',
                                          initialdir=self.data.path.parent,
                                          filetypes=[('预处理 NPZ', '*.npz')])
        if path:
            self.load(path)

    def open_library(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askdirectory(parent=self.root, title='选择包含方法版本目录的数据根目录',
                                        initialdir=self.library or self.data.path.parent)
        if path:
            candidates = sorted(p for p in Path(path).glob('*/subj_*/trial_*/emg_preprocessed.npz')
                                if not p.relative_to(path).parts[0].startswith(('_', '.')))
            if candidates:
                self.load(candidates[0])
            else:
                messagebox.showinfo('没有可选数据', '未找到 方法版本/subj_*/trial_*/emg_preprocessed.npz', parent=self.root)

    def choose_method(self, event=None):
        if self.library is not None:
            candidates = sorted((self.library / self.method.get()).glob('subj_*/trial_*/emg_preprocessed.npz'))
            if candidates:
                self.load(candidates[0])

    def choose_subject(self, event=None):
        if self.library is not None:
            candidates = sorted((self.library / self.method.get() / self.subject.get()).glob('trial_*/emg_preprocessed.npz'))
            if candidates:
                self.load(candidates[0])

    def choose_trial(self, event=None):
        if self.library is not None:
            self.load(self.library / self.method.get() / self.subject.get() / self.trial.get() / 'emg_preprocessed.npz')

    def step(self, delta):
        if self.data.path in self.paths:
            index = self.paths.index(self.data.path) + delta
            if 0 <= index < len(self.paths):
                self.load(self.paths[index])

    def refresh(self, event=None):
        self.dirty = set(self.figures)
        self.render()

    def render(self, event=None):
        if not hasattr(self, 'dirty'):
            return
        name = self.notebook.tab(self.notebook.select(), 'text')
        if name not in self.dirty:
            return
        figure = self.figures[name]
        if name == '通道波形':
            self.wave_state = wave_plot(figure, self.data, SIGNALS[self.signal.get()],
                                        self.overlay.get(), self.uniform.get(), self.relative.get())
            for ax in self.wave_state['axes']:
                ax.callbacks.connect('xlim_changed', self.schedule_wave)
        elif name == '激活热图':
            heatmap_plot(figure, self.data, self.relative.get())
        else:
            spectrum_plot(figure, self.data)
        self.toolbars[name].update()
        self.canvases[name].draw_idle()
        self.dirty.discard(name)

    def schedule_wave(self, event=None):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
        self.pending = self.root.after(70, self.redraw_wave)

    def redraw_wave(self):
        self.pending = None
        if self.wave_state is not None:
            pixels = self.canvases['通道波形'].get_tk_widget().winfo_width()
            update_wave(self.wave_state, max(400, int(pixels * .82)))
            self.canvases['通道波形'].draw_idle()

    def export(self):
        from tkinter import filedialog, messagebox
        from src.motion_reconstruction.paths import PROJECT
        directory = PROJECT / 'outputs/TUM/analysis/figures/emg'
        directory.mkdir(parents=True, exist_ok=True)
        name = self.notebook.tab(self.notebook.select(), 'text')
        path = filedialog.asksaveasfilename(parent=self.root, title='导出当前图', initialdir=directory,
                                           initialfile=f'subj_{self.data.subject:03d}_trial_{self.data.trial:03d}_{name}.png',
                                           defaultextension='.png', filetypes=[('PNG', '*.png')])
        if path:
            try:
                self.figures[name].savefig(path, dpi=180)
                self.status.set('已导出：' + path)
            except Exception as error:
                messagebox.showerror('导出失败', str(error), parent=self.root)

    def close(self):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--preview', type=Path, help='Export PNG without opening a window')
    parser.add_argument('--view', choices=['waveform', 'heatmap', 'spectrum'], default='waveform')
    parser.add_argument('--signal', choices=['bandpass', 'envelope', 'normalized'], default='bandpass')
    parser.add_argument('--self-check', action='store_true', help='Exercise controls, switching and validation, then close')
    args = parser.parse_args(argv)
    configure_fonts()
    try:
        data = EmgData(args.data)
    except Exception as error:
        parser.exit(1, f'无法加载 {args.data}: {error}\n')
    if args.preview:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        figure = Figure(figsize=(14, 11), dpi=130)
        FigureCanvasAgg(figure)
        if args.view == 'waveform':
            wave_plot(figure, data, 'emg_' + args.signal)
        elif args.view == 'heatmap':
            heatmap_plot(figure, data)
        else:
            spectrum_plot(figure, data)
        figure.text(.155, .015, f'subj_{data.subject:03d} / trial_{data.trial:03d} · {data.version} · '
                    f'Task {data.task} {data.task_description}'
                    + (' · 诊断归一化' if data.diagnostic else ''), fontsize=8)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.preview, dpi=130)
        print(args.preview.resolve())
    elif args.self_check:
        from .self_check import run_checks
        run_checks(data)
    else:
        import tkinter as tk
        root = tk.Tk()
        viewer = EmgViewer(root, data)
        viewer.signal.set({'bandpass': '带通 sEMG', 'envelope': '肌电包络', 'normalized': '归一化包络'}[args.signal])
        viewer.notebook.select(['waveform', 'heatmap', 'spectrum'].index(args.view))
        viewer.refresh()
        root.mainloop()
