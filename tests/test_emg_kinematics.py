"""Source identity, time mapping and quality checks for synchronized v002 display."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.emg_kinematics.data import load_synchronized, interpolate_poses, quality_text
from src.motion_reconstruction.common.emg_sync import resample_estimates
from src.motion_reconstruction.tum_native.anchored_quality import REJECT
from src.motion_reconstruction.tum_native.schema import Q_NAMES, canonical_hash, sha256


class NativeSynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root/'results'
        self.trial = self.results/'subj_001/trials/trial_005'
        self.trial.mkdir(parents=True)
        self.source = self.root/'raw/subj_001/EMG/trial_005.mat'
        motion = self.root/'raw/subj_001/MoCap/trial_005.mat'
        for p in (self.source, motion):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'immutable source '+p.parent.name.encode())
        self.data = self.root/'emg/subj_001/trial_005/emg_preprocessed.npz'
        self.data.parent.mkdir(parents=True)
        time = np.arange(30)*.0005
        wave = np.ones((len(time), 10), np.float32)
        np.savez(self.data, t=time, emg_bandpass=wave, emg_envelope=wave, emg_normalized=wave,
            channel_names=np.array([f'ch{i}' for i in range(10)]), channel_quality_flag=np.zeros(10, np.uint16),
            edge_valid=np.ones(len(time), bool), sample_rate_hz=2000., amplitude_unit='uV', time_unit='s',
            normalization_scope='fixture', subject_id=1, trial_id=5, task_id=1, version='fixture',
            source=str(self.source), source_sha256=sha256(self.source))
        schema = 'tum-native-functional7-v2'
        model = dict(schema=schema, subject='subj_001', hand_tip_M_m=[0, 0, .085],
            axes=dict(elbow_H=[1,0,0], twist_F=[0,0,1], wrist_FE_F=[1,0,0], wrist_RD_M=[0,1,0]))
        (self.trial.parents[1]/'model.json').write_text(json.dumps(model), encoding='utf-8')
        self.report_path = self.trial/'report.json'
        report = dict(schema=schema, job=dict(task=1, trial=5), model_hash=canonical_hash(model),
            source=str(motion), source_sha256=sha256(motion),
            sync=dict(status='duration_validated_pulse_candidate', offset_s=.002, clock_scale=2.,
                      emg_source_sha256=sha256(self.source)))
        self.report_path.write_text(json.dumps(report), encoding='utf-8')
        t = np.arange(6)*.002
        q = np.zeros((len(t), 7)); q[:, 3] = t
        flags = np.full(q.shape, 4|16|1024, np.uint16)
        flags[0] |= 64  # Native filter edge, not the old model's prior bit.
        valid = (flags & REJECT) == 0
        rotations = np.tile(np.eye(3), (len(t), 4, 1, 1))
        origins = np.tile([[0,0,0],[0,0,.1],[0,0,.4],[0,0,.65]], (len(t), 1, 1))
        bones = np.tile([[0,0,.1],[0,0,.4],[0,0,.65],[0,0,.735]], (len(t), 1, 1))
        np.savez(self.trial/'reconstruction.npz', schema=schema, t=t, q=q, flags=flags,
            relative_valid=valid, q_names=Q_NAMES, angle_unit='rad', length_unit='m', time_unit='s',
            segment_R=rotations, segment_origins_m=origins, bones_m=bones)
        aligned, af = resample_estimates(t, np.where(valid, q, np.nan), flags, (time-.002)/2)
        self.label = self.trial/'emg_aligned_native.npz'
        np.savez(self.label, schema=schema, t=time, q=aligned, flags=af,
            relative_valid=np.isfinite(aligned)&((af&REJECT)==0), q_names=Q_NAMES, angle_unit='rad')

    def change_label(self, edit):
        with np.load(self.label, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        edit(arrays)
        np.savez(self.label, **arrays)

    def test_clock_scale_geometry_and_read_only_sources(self):
        paths = [self.data, self.label, self.trial/'reconstruction.npz', self.report_path, self.source]
        before = [sha256(p) for p in paths]
        data = load_synchronized(self.data, self.results)
        self.assertEqual(data.algorithm, 'tum_native')
        self.assertAlmostEqual(data.q[12, 0], (.006-.002)/2, places=8)
        self.assertFalse(data.overlap[0])
        self.assertIsNone(data.pose(0))
        pose = data.pose(12)
        np.testing.assert_allclose(pose['upper'], [[0,0,100],[0,0,400]], atol=1e-4)
        self.assertEqual(len(pose['fingers']), 5)
        self.assertIn('滤波边界', data.quality_text(64))
        self.assertNotIn('先验', data.quality_text(64))
        self.assertIn('先验', quality_text(64))
        self.assertEqual([sha256(p) for p in paths], before)

    def test_rejects_shifted_emg_clock(self):
        self.change_label(lambda a: a.__setitem__('t', a['t']+.0001))
        with self.assertRaisesRegex(ValueError, '时间轴'):
            load_synchronized(self.data, self.results)

    def test_rejects_wrong_label_values_even_with_correct_shape(self):
        def edit(a): a['q'][12,3] += .1
        self.change_label(edit)
        with self.assertRaisesRegex(ValueError, '同步标签与源运动学'):
            load_synchronized(self.data, self.results)

    def test_rejects_inconsistent_quality_mask(self):
        def edit(a): a['relative_valid'][12,3] = False
        self.change_label(edit)
        with self.assertRaisesRegex(ValueError, '质量掩码'):
            load_synchronized(self.data, self.results)

    def test_rejects_changed_raw_source(self):
        self.source.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, '路径或哈希'):
            load_synchronized(self.data, self.results)

    def test_rejects_wrong_model_hash(self):
        report = json.loads(self.report_path.read_text())
        report['model_hash'] = 'incorrect'
        self.report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, '模型哈希'):
            load_synchronized(self.data, self.results)

    def test_poses_do_not_cross_gaps_or_extrapolate(self):
        t = np.array([0., .1])
        poses = np.tile(np.eye(4), (2,1,1))
        result = interpolate_poses(t, poses, np.array([-.001,0,.05,.1,.101]), .01)
        self.assertTrue(np.isnan(result[[0,2,4]]).all())
        self.assertTrue(np.isfinite(result[[1,3]]).all())


if __name__ == '__main__':
    unittest.main()
