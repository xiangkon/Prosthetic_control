"""Display-only seven-coordinate animation; no changes to reconstruction data."""
import argparse
import json
from pathlib import Path
import numpy as np
from .schema import SCHEMA, Q_NAMES, canonical_hash


def hand_geometry(rotations, origins, tip_local):
    """Rigid open-hand drawing in native hand coordinates (distal direction +Z)."""
    rotations, origins = np.asarray(rotations), np.asarray(origins)
    tip = np.asarray(tip_local, dtype=float)
    length = np.linalg.norm(tip)
    if tip.shape != (3,) or not np.isfinite(length) or length < 1e-6:
        raise ValueError('Invalid reference wrist-to-hand vector')
    if rotations.shape != (len(origins), 3, 3) or origins.shape != (len(origins), 3):
        raise ValueError('Expected hand rotations[N,3,3] and wrist origins[N,3]')
    z = tip / length
    transverse = np.eye(3)[np.argmin(abs(z))] if abs(z[0]) > .95 else np.array([1., 0, 0])
    x = transverse - z * np.dot(z, transverse)
    x /= np.linalg.norm(x)
    basis = np.column_stack([x, np.cross(z, x), z])
    scale = float(np.clip(length / .085, .5, 1.8))
    palm_local = np.array([[-.027, 0, .012], [.027, 0, .012], [.038, 0, .085], [-.034, 0, .085]]) * scale
    fingers_local = []
    for lateral, finger_length in [(-.028, .058), (-.009, .080), (.012, .087), (.032, .075)]:
        fingers_local.append(np.array([[lateral, 0, .080], [lateral * 1.12, 0, .080 + finger_length * .48],
            [lateral * 1.2, 0, .080 + finger_length * .78], [lateral * 1.25, 0, .080 + finger_length]]) * scale)
    fingers_local.append(np.array([[.025, 0, .028], [.052, -.005, .047], [.069, -.008, .067], [.079, -.010, .082]]) * scale)
    def world(points):
        return np.einsum('nij,kj->nki', rotations, points @ basis.T) + origins[:, None]
    palm = world(palm_local)
    fingers = np.stack([world(points) for points in fingers_local], axis=1)
    roots = fingers[:, :, 0]
    metacarpals = np.stack([np.broadcast_to(origins[:, None], roots.shape), roots], axis=2)
    return dict(palm=palm, fingers=fingers, metacarpals=metacarpals, scale=scale,
                drawing_basis_M=basis, description='Fixed open-hand schematic; finger motion is not reconstructed')


def create_view(trial_dir, fps=30., frame=0, interactive=True):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    trial_dir = Path(trial_dir)
    report = json.loads((trial_dir / 'report.json').read_text(encoding='utf-8'))
    model = json.loads((trial_dir.parents[1] / 'model.json').read_text(encoding='utf-8'))
    if report.get('schema') != SCHEMA or model.get('schema') != SCHEMA or report.get('model_hash') != canonical_hash(model):
        raise ValueError('需要模型与报告匹配的v002重建结果')
    with np.load(trial_dir / 'reconstruction.npz', allow_pickle=False) as z:
        if z['schema'].item() != SCHEMA or z['q_names'].tolist() != Q_NAMES or z['angle_unit'].item() != 'rad' or z['length_unit'].item() != 'm':
            raise ValueError('v002结果的版本、角度列或单位不正确')
        t, q, names, bones, raw, valid, segment_r, segment_p = [z[k] for k in
            ['t', 'q', 'q_names', 'bones_m', 'raw_markers_m', 'relative_valid', 'segment_R', 'segment_origins_m']]
    if not 0 <= frame < len(t):
        raise ValueError(f'--frame must be between 0 and {len(t)-1}')
    hand = hand_geometry(segment_r[:, 3], segment_p[:, 3], model['hand_tip_M_m'])
    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(7, 2, width_ratios=[1.05, 1.3], left=.04, right=.98,
                           top=.95, bottom=.16 if interactive else .07, hspace=.7, wspace=.18)
    ax = fig.add_subplot(grid[:, 0], projection='3d')
    line, = ax.plot([], [], [], 'o-', color='#287b65', lw=2.6, ms=4, label='Functional chain')
    scatter, = ax.plot([], [], [], '.', color='#477b9d', alpha=.5, label='Raw markers')
    palm = Poly3DCollection([], facecolors='#e1ab81', edgecolors='#985a39', alpha=.6, linewidths=1)
    ax.add_collection3d(palm, autolim=False)
    fingers = [ax.plot([], [], [], color='#985a39', lw=2, marker='o', ms=2)[0] for _ in range(5)]
    metacarpals = Line3DCollection([], colors='#985a39', linewidths=1, alpha=.8)
    ax.add_collection3d(metacarpals, autolim=False)
    axes = [ax.plot([], [], [], color=color, lw=1.6)[0] for color in ['#c74a43', '#3b9656', '#437bc7']]
    ax.text2D(.02, .96, 'Open-hand schematic; fixed fingers\nHand frame: X red / Y green / Z blue',
              transform=ax.transAxes, fontsize=8, color='#606060')
    points = np.concatenate([bones.reshape(-1, 3), hand['palm'].reshape(-1, 3), hand['fingers'].reshape(-1, 3)])
    points = points[np.isfinite(points).all(1)]
    measured = raw.reshape(-1, 3)
    measured = measured[np.isfinite(measured).all(1)]
    if len(measured):
        points = np.concatenate([points, np.quantile(measured, [.005, .995], axis=0)])
    if not len(points):
        raise ValueError('No finite geometry or markers to display')
    lower, upper = points.min(0), points.max(0)
    center = (lower + upper) / 2
    radius = max(.15, float(np.max(upper-lower) / 2 * 1.1))
    limits = [(v-radius, v+radius) for v in center]
    def reset_view(_=None):
        ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=22, azim=-65)
        fig.canvas.draw_idle()
    reset_view()
    ax.legend(loc='upper right', fontsize=8)
    cursors = []
    colors = ['#536e9a'] * 3 + ['#287b65', '#168eac', '#b47a19', '#ad426f']
    for k in range(7):
        axis = fig.add_subplot(grid[k, 1])
        axis.plot(t, np.rad2deg(q[:, k]), color='.7', lw=.8)
        axis.plot(t, np.where(valid[:, k], np.rad2deg(q[:, k]), np.nan), color=colors[k], lw=1)
        axis.set(ylabel='deg', xlim=(t[0], t[-1]))
        axis.set_title(str(names[k]), loc='left', fontsize=9, pad=2)
        cursors.append(axis.axvline(t[frame], color='#b33c35', lw=.7))
    axis.set_xlabel('Time (s); gray = quality rejected')
    status = fig.suptitle('', fontsize=11)
    def set_line(artist, values):
        artist.set_data_3d(values[:, 0], values[:, 1], values[:, 2])
    def update(i):
        set_line(line, bones[i]); set_line(scatter, raw[i])
        pp = hand['palm'][i]
        good = np.isfinite(pp).all()
        palm.set_verts([pp] if good else [])
        metacarpals.set_segments(hand['metacarpals'][i] if good else [])
        for artist, values in zip(fingers, hand['fingers'][i]):
            set_line(artist, values)
        origin, rotation = segment_p[i, 3], segment_r[i, 3]
        for k, artist in enumerate(axes):
            set_line(artist, np.stack([origin, origin + rotation[:, k] * .05]))
        for cursor in cursors:
            cursor.set_xdata([t[i], t[i]])
        status.set_text(f'{trial_dir.parents[1].name} / {trial_dir.name}   t={t[i]:.3f}s   Frame {i+1}/{len(t)}')
        fig.canvas.draw_idle()
    update(frame)
    controls = []
    timer = None
    if interactive:
        state = dict(playing=True, time=float(t[frame]))
        play = Button(fig.add_axes([.04, .075, .09, .04]), 'Pause')
        slider = Slider(fig.add_axes([.22, .08, .59, .025]), 'Time (s)', t[0], t[-1], valinit=t[frame], valfmt='%.3f')
        reset = Button(fig.add_axes([.86, .075, .1, .04]), 'Reset view')
        markers = Button(fig.add_axes([.25, .015, .14, .035]), 'Markers on/off')
        frames = Button(fig.add_axes([.41, .015, .14, .035]), 'Hand axes on/off')
        speed = Slider(fig.add_axes([.72, .025, .23, .02]), 'Speed', .1, 2., valinit=1., valfmt='%.1fx')
        def seek(value):
            state['time'] = float(value)
            update(min(int(np.searchsorted(t, value)), len(t)-1))
        def toggle(_):
            state['playing'] = not state['playing']
            play.label.set_text('Pause' if state['playing'] else 'Play')
            fig.canvas.draw_idle()
        def show_markers(_):
            scatter.set_visible(not scatter.get_visible()); fig.canvas.draw_idle()
        def show_axes(_):
            visible = not axes[0].get_visible()
            for artist in axes:
                artist.set_visible(visible)
            fig.canvas.draw_idle()
        def tick():
            if state['playing']:
                value = state['time'] + float(speed.val) / fps
                slider.set_val(float(t[0]) if value > t[-1] else value)
        play.on_clicked(toggle); slider.on_changed(seek); reset.on_clicked(reset_view)
        markers.on_clicked(show_markers); frames.on_clicked(show_axes)
        timer = fig.canvas.new_timer(interval=max(1, int(round(1000/fps))))
        timer.add_callback(tick)
        fig.canvas.mpl_connect('close_event', lambda _: timer.stop())
        controls = [play, slider, reset, markers, frames, speed, state, seek, tick]
    return fig, dict(update=update, hand=hand, controls=controls, timer=timer)


def main():
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description='Native seven-coordinate animation with palm and fixed fingers')
    p.add_argument('trial_dir', type=Path)
    p.add_argument('--save', type=Path, help='Save a static diagnostic image')
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--frame', type=int, default=0, help='Initial/source-array frame index, starting at zero')
    a = p.parse_args()
    if not np.isfinite(a.fps) or a.fps <= 0:
        p.error('--fps must be positive and finite')
    fig, scene = create_view(a.trial_dir, a.fps, a.frame, interactive=a.save is None)
    if a.save:
        a.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.save, dpi=140)
    else:
        scene['timer'].start()
        plt.show()
    plt.close(fig)
