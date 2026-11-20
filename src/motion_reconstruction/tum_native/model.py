"""Independent functional seven-coordinate FK and rotation decomposition."""
import numpy as np
from scipy.spatial.transform import Rotation
from .schema import SEGMENTS, runs


def unit(v):
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-8:
        raise ValueError('Reference axis is degenerate')
    return np.asarray(v) / n


def frame(z, x):
    z = unit(z)
    x = unit(x - z * np.dot(x, z))
    y = unit(np.cross(z, x))
    return np.column_stack([x, y, z])


def two_axis(target, first, reference, second, iterations=25):
    """Batched geodesic projection; input axes belong to different local frames."""
    count = len(target)
    valid = np.isfinite(target).all(axis=(1, 2))
    safe = target.copy()
    safe[~valid] = np.eye(3)
    q = np.zeros((count, 2))
    damping = np.full(count, .01)

    def evaluate(value):
        predicted = Rotation.from_rotvec(value[:, :1] * first).as_matrix() @ reference @ Rotation.from_rotvec(value[:, 1:] * second).as_matrix()
        return Rotation.from_matrix(np.swapaxes(predicted, 1, 2) @ safe).as_rotvec()

    residual = evaluate(q)
    for _ in range(iterations):
        jac = np.stack([(evaluate(q + np.eye(2)[k] * 1e-5) - residual) / 1e-5 for k in range(2)], axis=2)
        normal = np.swapaxes(jac, 1, 2) @ jac
        gradient = np.einsum('ndi,nd->ni', jac, residual)
        step = np.linalg.solve(normal + damping[:, None, None] * np.eye(2), -gradient[:, :, None])[:, :, 0]
        step *= np.minimum(1., .35 / np.maximum(np.linalg.norm(step, axis=1), 1e-12))[:, None]
        candidate = q + step
        rr = evaluate(candidate)
        accepted = np.sum(rr**2, axis=1) <= np.sum(residual**2, axis=1)
        q[accepted], residual[accepted] = candidate[accepted], rr[accepted]
        damping = np.where(accepted, np.maximum(damping * .4, 1e-8), np.minimum(damping * 5, 1e6))
        if np.max(abs(step[valid]), initial=0) < 1e-6:
            break
    for k in range(2):
        for part in runs(valid):
            q[part, k] = np.unwrap(q[part, k])
    q[~valid] = np.nan
    error = np.linalg.norm(residual, axis=1)
    error[~valid] = np.nan
    return q, error


def decompose(observed, model):
    rotation = observed['R']
    r0 = model['reference_rotations']
    rt, rh, rf, rm = [rotation[:, k] for k in range(4)]
    n = len(rt)
    q = np.full((n, 7), np.nan)
    th = np.swapaxes(rt, 1, 2) @ rh
    good = np.isfinite(th).all(axis=(1, 2))
    q[good, :3] = Rotation.from_matrix(np.array(r0['TH']).T @ th[good]).as_rotvec()
    hf, fm = np.swapaxes(rh, 1, 2) @ rf, np.swapaxes(rf, 1, 2) @ rm
    q[:, 3:5], elbow_error = two_axis(hf, np.array(model['axes']['elbow_H']), np.array(r0['HF']), np.array(model['axes']['twist_F']))
    q[:, 5:7], wrist_error = two_axis(fm, np.array(model['axes']['wrist_FE_F']), np.array(r0['FM']), np.array(model['axes']['wrist_RD_M']))
    displacement = np.einsum('nji,nj->ni', rt, observed['p'][:, 1] - observed['p'][:, 0]) - np.array(model['shoulder_reference_T_m'])
    return q, displacement, np.column_stack([elbow_error, wrist_error])


def forward(model, state, thorax_R, thorax_p):
    """state[N,10] = seven rotation coordinates + shoulder displacement in m."""
    state = np.atleast_2d(state)
    q, d = state[:, :7], state[:, 7:]
    axes, ref = model['axes'], model['reference_rotations']
    th = np.array(ref['TH']) @ Rotation.from_rotvec(q[:, :3]).as_matrix()
    hf = Rotation.from_rotvec(q[:, 3:4] * axes['elbow_H']).as_matrix() @ np.array(ref['HF']) @ Rotation.from_rotvec(q[:, 4:5] * axes['twist_F']).as_matrix()
    fm = Rotation.from_rotvec(q[:, 5:6] * axes['wrist_FE_F']).as_matrix() @ np.array(ref['FM']) @ Rotation.from_rotvec(q[:, 6:7] * axes['wrist_RD_M']).as_matrix()
    rh = thorax_R @ th
    rf = rh @ hf
    rm = rf @ fm
    shoulder = thorax_p + np.einsum('nij,nj->ni', thorax_R, np.array(model['shoulder_reference_T_m']) + d)
    elbow = shoulder + np.einsum('nij,j->ni', rh, model['elbow_in_H_m'])
    wrist = elbow + np.einsum('nij,j->ni', rf, model['wrist_in_F_m'])
    hand_tip = wrist + np.einsum('nij,j->ni', rm, model['hand_tip_M_m'])
    rotations = np.stack([thorax_R, rh, rf, rm], axis=1)
    origins = np.stack([thorax_p, shoulder, elbow, wrist], axis=1)
    predictions = []
    names = []
    for k, segment in enumerate(SEGMENTS):
        spec = model['segments'][segment]
        for name, local in spec['local_m'].items():
            names.append(name)
            predictions.append(np.einsum('nij,j->ni', rotations[:, k], local) + origins[:, k])
    return np.stack(predictions, axis=1), names, rotations, origins, np.stack([shoulder, elbow, wrist, hand_tip], axis=1)
