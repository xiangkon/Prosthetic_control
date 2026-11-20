"""Comparison-only coordinate conversion tests on frozen subject geometry."""
import json
import unittest
import numpy as np
from src.motion_reconstruction.paths import LEGACY_ROOT, NATIVE_ROOT
from src.motion_reconstruction.comparisons.angle_conversion import AngleConversion
from src.motion_reconstruction.legacy_g99.kinematics import forward


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = AngleConversion.load(LEGACY_ROOT/'subj_005/estimated_model.json',
            NATIVE_ROOT/'v002_anchored7/subj_005/model.json')
        ref = cls.bridge.native['reference']
        with np.load(LEGACY_ROOT/f'subj_005/trials/trial_{ref["trial"]:03d}/ik_angles.npz', allow_pickle=False) as z:
            keep = (z['t'] >= ref['start_s']-1e-9) & (z['t'] <= ref['end_s']+1e-9)
            cls.anchor = np.median(z['q'][keep], axis=0)

    def test_local_jacobian_and_directions(self):
        value, jac = self.bridge.linearize(self.anchor)
        _, other = self.bridge.linearize(self.anchor, step=5e-5)
        np.testing.assert_allclose(jac, other, atol=1e-5, rtol=0)
        self.assertEqual(np.linalg.matrix_rank(jac[:, 5:], tol=1e-5), 7)
        self.assertGreater(jac[3, 8], 0)
        self.assertLess(jac[4, 9], 0)
        self.assertGreater(jac[5, 10], 0)
        self.assertLess(jac[6, 11], 0)
        self.assertGreater(abs(jac[3, 9]), .05)  # RU changes native elbow projection too

    def test_last_seven_alone_are_insufficient(self):
        changed = self.anchor.copy(); changed[0] += .02
        a, b = self.bridge.forward(np.stack([self.anchor, changed]))
        self.assertGreater(np.linalg.norm(a[:3]-b[:3]), .005)
        np.testing.assert_allclose(a[3:], b[3:], atol=1e-8)

    def test_conditional_local_inverse_roundtrip(self):
        q = self.anchor.copy()
        q[5:] += np.deg2rad([1., -1., .5, .8, -.7, .4, -.3])
        target = self.bridge.forward(q)[0]
        recovered, quality = self.bridge.inverse_conditional(target, q[:5], self.anchor[5:])
        self.assertTrue(quality['coordinate_match'])
        self.assertEqual(quality['local_rank'], 7)
        np.testing.assert_allclose(recovered, q[5:], atol=2e-5, rtol=0)
        self.assertFalse(quality['global_uniqueness_established'])

    def test_technical_frame_bridge_and_forearm_mixture(self):
        q = np.tile(self.anchor, (3, 1))
        q[1, :9] += .01
        q[2, 9] += .1
        native = self.bridge.frames(q)['R']
        old = forward(self.bridge.legacy, q)[2]
        for index, segment in [(0, 'thorax'), (1, 'humerus'), (3, 'hand')]:
            change = np.swapaxes(old[segment][:, :3, :3], 1, 2) @ native[:, index]
            np.testing.assert_allclose(change, np.repeat(change[:1], 3, axis=0), atol=1e-10)
        cf = np.swapaxes(old['ulna'][:, :3, :3], 1, 2) @ native[:, 2]
        np.testing.assert_allclose(cf[0], cf[1], atol=1e-10)
        self.assertGreater(np.linalg.norm(cf[0]-cf[2]), .001)
        self.assertEqual(set(self.bridge.support['forearm']['legacy_segments'].values()), {'ulna', 'radius'})


if __name__ == '__main__':
    unittest.main()
