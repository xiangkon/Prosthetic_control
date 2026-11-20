"""Migration safety checks that never invoke a real reconstruction job."""
import json
import unittest
from src.motion_reconstruction.paths import (PROJECT, LEGACY_ROOT, NATIVE_ROOT,
    require_writable_legacy_output, resolve_historical_path)
from src.motion_reconstruction.tum_native.provenance import runtime_fingerprint
from unittest.mock import patch


class LayoutTests(unittest.TestCase):
    def test_frozen_output_guard_without_invoking_runner(self):
        for path in [LEGACY_ROOT, LEGACY_ROOT/'subj_001', LEGACY_ROOT.parent, PROJECT]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                require_writable_legacy_output(path)
        require_writable_legacy_output(PROJECT/'outputs/TUM/_checks/legacy_g99_reruns/test')

    def test_historical_result_path(self):
        old = PROJECT/'outputs/TUM/reconstruction/subj_005/trials/trial_011/ik_angles.npz'
        self.assertEqual(resolve_historical_path(old), LEGACY_ROOT/'subj_005/trials/trial_011/ik_angles.npz')

    def test_native_cache_and_config_invalidation(self):
        model = json.loads((NATIVE_ROOT/'v002_anchored7/subj_001/model.json').read_text())
        config = json.loads((PROJECT/'configs/motion_reconstruction/tum_native/v002.json').read_text())
        current = runtime_fingerprint(config)
        self.assertEqual(model['runtime_fingerprint'], current)
        self.assertNotEqual(runtime_fingerprint(dict(config, cutoff_hz=7.)), current)
        with patch('src.motion_reconstruction.tum_native.provenance.sha256', return_value='changed-source'):
            self.assertNotEqual(runtime_fingerprint(config), current)

    def test_existing_native_trial_reuses_without_calibration_or_solving(self):
        from src.motion_reconstruction.tum_native import anchored_pipeline
        with patch.object(anchored_pipeline, 'calibrate_v2', side_effect=AssertionError('must reuse model')), \
                patch.object(anchored_pipeline, 'reconstruct_v2', side_effect=AssertionError('must reuse trial')):
            anchored_pipeline.main(['--subjects', '1', '--trials', '5', '--emg'])


if __name__ == '__main__':
    unittest.main()
