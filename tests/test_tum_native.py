"""Independent synthetic geometry; no legacy models or datasets required."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from src.motion_reconstruction.common.motion_io import MotionData
from src.motion_reconstruction.tum_native.model import forward, decompose, two_axis
from src.motion_reconstruction.tum_native.observations import rigid_fit, prepare, track
from src.motion_reconstruction.tum_native.anchored import solve_anchored
from src.motion_reconstruction.tum_native.anchored_quality import evaluate_anchored, MARKER_DETACHMENT, UNSUPPORTED_STEP
from src.motion_reconstruction.tum_native.schema import SEGMENTS, MISSING, ROTATION_CHART
import json
from pathlib import Path


def synthetic():
    rng = np.random.default_rng(42)
    model = dict(reference_rotations={s: np.eye(3).tolist() for s in ['TH', 'HF', 'FM']},
        axes=dict(elbow_H=[1, 0, 0], twist_F=[0, 0, 1], wrist_FE_F=[1, 0, 0], wrist_RD_M=[0, 1, 0]),
        shoulder_reference_T_m=[.2, 0, 0], elbow_in_H_m=[0, 0, -.3], wrist_in_F_m=[0, 0, -.25], hand_tip_M_m=[0, 0, -.1], segments={})
    for s in SEGMENTS:
        names = [f'{s}_{i}' for i in range(5)]
        model['segments'][s] = dict(local_m={name: point.tolist() for name, point in zip(names, rng.normal(size=(5, 3)) * .05)},
            fit_names=names, sigma_m={name: .003 for name in names})
    return model


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.model = synthetic()
        self.config = json.loads((Path(__file__).parents[1] / 'configs/motion_reconstruction/tum_native/v002.json').read_text())

    def observation(self, state):
        rt = np.tile(np.eye(3), (len(state), 1, 1))
        p, names, r, origins, _ = forward(self.model, state, rt, np.zeros((len(state), 3)))
        obs = dict(t=np.arange(len(state)) * .02, p=p, raw=p, names=names,
            observed=np.ones(p.shape[:2], bool), interpolated=np.zeros(p.shape[:2], bool), edge=np.zeros(p.shape[:2], bool))
        # Independent shoulder observation, not a fitting point.
        obs['names'] = names + ['m_acr']
        obs['p'] = np.concatenate([p, origins[:, 1:2]], axis=1)
        obs['raw'] = obs['p'].copy()
        for key in ['observed', 'interpolated', 'edge']:
            obs[key] = np.pad(obs[key], ((0, 0), (0, 1)), constant_values=key == 'observed')
        return obs

    def test_roundtrip_mixed_axes_and_translation(self):
        x = np.tile([.2, -.3, .1, .7, -.6, .2, -.4, .01, -.02, .03], (8, 1))
        obs = self.observation(x)
        q, d, err = decompose(track(obs, self.model), self.model)
        np.testing.assert_allclose(q, x[:, :7], atol=1e-5)
        np.testing.assert_allclose(d, x[:, 7:], atol=1e-7)
        self.assertLess(err.max(), 1e-6)

    def test_partial_hand_missing_keeps_proximal(self):
        obs = self.observation(np.zeros((10, 10)))
        obs['p'][:, 15:20] = np.nan
        poses = track(obs, self.model)
        result = solve_anchored(obs, poses, self.model, self.config)
        self.assertTrue(np.isfinite(result['state'][:, :5]).all())
        self.assertTrue(np.isnan(result['state'][:, 5:7]).all())
        qc = evaluate_anchored(obs, poses, self.model, result, self.config)
        self.assertTrue(((qc['flags'][:, 5:7] & MISSING) != 0).all())

    def test_collinear_rejected(self):
        local = np.column_stack([np.arange(4), np.zeros((4, 2))])
        self.assertIsNone(rigid_fit(local, local)[0])

    def test_long_gap_not_filtered_across(self):
        t = np.arange(200) * .002
        p = np.ones((200, 1, 3))
        p[80:120] = np.nan
        p[120:] = 10
        obs = prepare(MotionData(t, p, ['a']), self.config)
        self.assertTrue(np.isnan(obs['p'][80:120]).all())
        np.testing.assert_allclose(obs['p'][:80], 1, atol=1e-8)
        np.testing.assert_allclose(obs['p'][120:], 10, atol=1e-8)

    def test_chart_and_no_absolute_claim(self):
        x = np.zeros((5, 10)); x[:, 0] = np.deg2rad(175)
        obs = self.observation(x); poses = track(obs, self.model)
        qc = evaluate_anchored(obs, poses, self.model, solve_anchored(obs, poses, self.model, self.config), self.config)
        self.assertTrue(((qc['flags'][:, :3] & ROTATION_CHART) != 0).all())
        self.assertFalse(qc['absolute_valid'].any())

    def test_anchored_refinement_respects_pose_and_frame_cost(self):
        cfg = json.loads((Path(__file__).parents[1] / 'configs/motion_reconstruction/tum_native/v002.json').read_text())
        x = np.zeros((50, 10))
        x[:, 3] = np.linspace(0, .3, len(x))
        obs = self.observation(x)
        obs['t'] = np.arange(len(x)) * .002
        poses = track(obs, self.model)
        # Simulate a marker error after the pose anchor has been determined.
        obs['p'][25, 5:10, 0] += .25
        solution = solve_anchored(obs, poses, self.model, cfg)
        rotation_error = Rotation.from_rotvec(solution['state'][:, :3]).magnitude()
        self.assertTrue(np.all(rotation_error <= np.deg2rad(cfg['correction_limit_deg']) + 1e-9))
        self.assertLessEqual(solution['refinement']['final_cost'], solution['refinement']['initial_cost'])

    def test_v002_flags_an_unsupported_spike_and_detachment(self):
        cfg = json.loads((Path(__file__).parents[1] / 'configs/motion_reconstruction/tum_native/v002.json').read_text())
        obs = self.observation(np.zeros((20, 10)))
        obs['t'] = np.arange(20) * .002
        poses = track(obs, self.model)
        solution = solve_anchored(obs, poses, self.model, cfg)
        solution['state'][10, 0] = .4
        solution['state'][10, 7] = .10
        qc = evaluate_anchored(obs, poses, self.model, solution, cfg)
        self.assertTrue(qc['flags'][10, 0] & MARKER_DETACHMENT)
        self.assertTrue(np.all(qc['flags'][9:12, 0] & UNSUPPORTED_STEP))
        self.assertFalse(qc['relative_valid'][10, 0])
        self.assertTrue(qc['relative_valid'][5, 0])


if __name__ == '__main__':
    unittest.main()
