"""Opt-in state-linearized robust iterated EKF for targeted reconstruction repair.

The default batch filter is unchanged. The update and boundary handling follow
the candidate validated for subj_005/trial_009 and trial_010.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .filtering import prepare_observations
from .kinematics import residual_jacobian

FILTER_METHOD = 'robust_iterated_state_ekf'
FILTER_PARAMETERS = dict(state='q,qdot', process_intensity=2., max_iterations=3,
    max_iteration_angle_step_rad=.25, contact_sigma=.05,
    initial_angle_sd_rad=.1, initial_velocity_sd_rad_s=1.,
    bound_angle_variance_rad2=1e-8, reset_outward_velocity=True)


def project_state(x, covariance, model):
    clipped = (x[:12] <= model.lower + 1e-10) | (x[:12] >= model.upper - 1e-10)
    if not clipped.any():
        return False
    outward = ((x[:12] <= model.lower + 1e-10) & (x[12:] < 0)) | ((x[:12] >= model.upper - 1e-10) & (x[12:] > 0))
    x[:12] = np.clip(x[:12], model.lower, model.upper)
    x[12:][outward] = 0
    for index in np.flatnonzero(clipped):
        covariance[index, :] = 0
        covariance[:, index] = 0
        covariance[index, index] = FILTER_PARAMETERS['bound_angle_variance_rad2']
    for index in np.flatnonzero(outward) + 12:
        covariance[index, :] = 0
        covariance[:, index] = 0
        covariance[index, index] = FILTER_PARAMETERS['initial_velocity_sd_rad_s'] ** 2
    return True


def trajectory_state_ekf(data, model, poses, ik):
    """Use IK only to initialize; re-evaluate raw-marker observations at state.

    The existing vectorized residual/Jacobian is used with one frame per update,
    avoiding a second kinematic implementation in the production source tree.
    """
    measured, valid = prepare_observations(data, model)
    count = len(data.t)
    pose_ok = np.isfinite(poses).all(axis=(1, 2))
    output = np.full((count, 12), np.nan)
    sd = np.full_like(output, np.nan)
    clipped = np.zeros(count, bool)
    state = covariance = None
    identity = np.eye(24)
    for i in range(count):
        observed = pose_ok[i] and valid[i].sum() >= 4
        if state is None:
            if not observed or not np.isfinite(ik[i]).all():
                continue
            state = np.r_[ik[i], np.zeros(12)]
            covariance = np.diag(np.r_[np.full(12, FILTER_PARAMETERS['initial_angle_sd_rad'] ** 2),
                np.full(12, FILTER_PARAMETERS['initial_velocity_sd_rad_s'] ** 2)])
        else:
            dt = data.t[i] - data.t[i - 1]
            transition = identity.copy()
            transition[:12, 12:] = np.eye(12) * dt
            process = FILTER_PARAMETERS['process_intensity'] * np.kron([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]], np.eye(12))
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
            clipped[i] |= project_state(state, covariance, model)
            if observed:
                prediction = state.copy()
                information = cho_solve(cho_factor(covariance, lower=True), identity)

                def evaluate(candidate):
                    r, jac, _ = residual_jacobian(model, candidate[None, :12], poses[i:i + 1], measured[i:i + 1], valid[i:i + 1], FILTER_PARAMETERS['contact_sigma'])
                    residual, jacobian = r[0], jac[0]
                    delta = candidate - prediction
                    cost = np.sum(2 * (np.sqrt(1 + residual ** 2) - 1)) + delta @ information @ delta
                    return residual, jacobian, cost

                residual, jacobian, cost = evaluate(state)
                accepted = False
                posterior = covariance
                for _ in range(FILTER_PARAMETERS['max_iterations']):
                    weights = 1 / np.sqrt(1 + residual ** 2)
                    updated_information = information.copy()
                    updated_information[:12, :12] += jacobian.T @ (jacobian * weights[:, None])
                    innovation = -residual + jacobian @ (state[:12] - prediction[:12])
                    factor = cho_factor(updated_information, lower=True)
                    estimate = prediction + cho_solve(factor, np.r_[jacobian.T @ (weights * innovation), np.zeros(12)])
                    step = estimate - state
                    factor_step = min(1., FILTER_PARAMETERS['max_iteration_angle_step_rad'] / max(np.linalg.norm(step[:12]), 1e-12))
                    trial = state + factor_step * step
                    trial[:12] = np.clip(trial[:12], model.lower, model.upper)
                    candidate_residual, candidate_jacobian, candidate_cost = evaluate(trial)
                    if candidate_cost > cost + 1e-8:
                        break
                    state = trial
                    residual, jacobian, cost = candidate_residual, candidate_jacobian, candidate_cost
                    posterior = cho_solve(factor, identity)
                    accepted = True
                    if np.linalg.norm(factor_step * step[:12]) < 1e-5:
                        break
                if accepted:
                    covariance = (posterior + posterior.T) / 2
                clipped[i] |= project_state(state, covariance, model)
        output[i] = state[:12]
        sd[i] = np.sqrt(np.maximum(np.diag(covariance)[:12], 0))
    return output, sd, clipped
