"""Batched marker refinement with a bounded correction to measured poses."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from .model import decompose, forward
from .schema import runs


def solve_anchored(obs, poses, model, config):
    q, displacement, local_error = decompose(poses, model)
    state = np.column_stack([q, displacement])
    result = dict(state=state, initial=state.copy(), local_error=local_error,
                  refined=np.zeros(len(state), bool), correction_rejected=np.zeros(len(state), bool),
                  refinement=dict(iterations=0, refined_frames=0, reverted_frames=0))
    complete = np.isfinite(state).all(1)
    if not complete.any():
        return result
    initial = state[complete].copy()
    rt, pt = poses['R'][complete, 0], poses['p'][complete, 0]
    fit_names = [name for spec in model['segments'].values() for name in spec['fit_names']]
    target = obs['p'][complete][:, [obs['names'].index(n) for n in fit_names]]
    sigma_lookup = {n: v for spec in model['segments'].values() for n, v in spec['sigma_m'].items()}
    sigma = np.array([sigma_lookup[n] for n in fit_names])[None, :, None]
    observed = np.isfinite(target).all(2)
    target = np.nan_to_num(target)
    reference_rotation = Rotation.from_rotvec(initial[:, :3]).as_matrix()
    angle_limit = np.deg2rad(config['correction_limit_deg'])
    position_limit = config['correction_limit_m']
    limits = np.r_[np.repeat(angle_limit, 7), np.repeat(position_limit, 3)]
    prior = limits / 2
    names = forward(model, initial[:1], rt[:1], pt[:1])[1]
    output_indices = [names.index(n) for n in fit_names]
    measured = poses['R'][complete]
    pairs = ((1, 2), (2, 3))
    relative = [measured[:, a].swapaxes(1, 2) @ measured[:, b] for a, b in pairs]
    orientation_sigma = np.deg2rad(config['orientation_fit_sigma_deg'])

    def apply(delta):
        values = initial + delta
        values[:, :3] = Rotation.from_matrix(reference_rotation @
            Rotation.from_rotvec(delta[:, :3]).as_matrix()).as_rotvec()
        return values

    def residual(delta):
        predicted, _, fitted, _, _ = forward(model, apply(delta), rt, pt)
        predicted = predicted[:, output_indices]
        marker = np.where(observed[:, :, None], (predicted - target) / sigma, 0)
        orientation = [Rotation.from_matrix(
            (fitted[:, a].swapaxes(1, 2) @ fitted[:, b]).swapaxes(1, 2) @ target).as_rotvec()
            for (a, b), target in zip(pairs, relative)]
        return np.column_stack([marker.reshape(len(delta), -1), delta / prior,
                                np.column_stack(orientation) / orientation_sigma])

    def cost(r):
        return np.sum(np.sqrt(1 + r*r) - 1, axis=1)

    def bounded(delta):
        for start, stop, limit in [(0, 3, angle_limit), (3, 5, angle_limit),
                                   (5, 7, angle_limit), (7, 10, position_limit)]:
            block = delta[:, start:stop]
            block *= np.minimum(1, limit / np.maximum(np.linalg.norm(block, axis=1), 1e-12))[:, None]
        return delta

    delta = np.zeros_like(initial)
    r = residual(delta)
    base_cost = cost(r)
    for _ in range(config['refinement_iterations']):
        step_size = 1e-5
        jac = np.stack([(residual(delta + shift * step_size) - r) / step_size for shift in np.eye(10)], axis=2)
        weight = 1 / np.sqrt(1 + r*r)
        jt = jac.swapaxes(1, 2)
        hessian = jt @ (jac * weight[:, :, None])
        gradient = (jt @ (r * weight)[:, :, None])[:, :, 0]
        damping = .05 * np.maximum(np.diagonal(hessian, axis1=1, axis2=2), 1.)
        step = np.linalg.solve(hessian + np.eye(10) * damping[:, None, :], -gradient[:, :, None])[:, :, 0]
        old_cost = cost(r)
        improved = np.zeros(len(delta), bool)
        for scale in (1., .5, .25):
            candidate = bounded(delta + scale * step)
            rr = residual(candidate)
            better = ~improved & (cost(rr) < old_cost)
            delta[better], r[better] = candidate[better], rr[better]
            improved |= better
            if improved.all():
                break
    # Smooth only the small correction. Keep the measured trajectory at full rate.
    correction = np.zeros_like(state)
    correction[complete] = delta
    dt = np.median(np.diff(obs['t']))
    for part in runs(complete):
        correction[part] = gaussian_filter1d(correction[part],
            config['correction_smoothing_s'] / dt, axis=0, mode='nearest')
    smooth = correction[complete]
    for _ in range(4):
        worse = cost(residual(smooth)) > base_cost + 1e-10
        if not worse.any():
            break
        smooth[worse] *= .5
    worse = cost(residual(smooth)) > base_cost + 1e-10
    smooth[worse] = 0
    state[complete] = apply(smooth)
    result['refined'][complete] = np.linalg.norm(smooth, axis=1) > 1e-9
    result['correction_rejected'][complete] = worse
    result['refinement'] = dict(iterations=config['refinement_iterations'],
        refined_frames=int(result['refined'].sum()), reverted_frames=int(worse.sum()),
        initial_cost=float(base_cost.sum()), final_cost=float(cost(residual(smooth)).sum()))
    return result
