import unittest
import numpy as np
from src.motion_reconstruction.comparisons.trend_analysis import trend_metrics


class TrendMetricsTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(0,6,.002)
        self.x=20*np.sin(2*np.pi*self.t)

    def test_offset_scale_and_reverse(self):
        valid=np.ones(len(self.t),bool)
        r=trend_metrics(self.t,self.x,2*self.x+70,valid)
        self.assertAlmostEqual(r['pearson'],1)
        self.assertAlmostEqual(r['rom_ratio'],2)
        self.assertGreater(r['direction_agreement'],.95)
        reverse=trend_metrics(self.t,self.x,-self.x,valid)
        self.assertAlmostEqual(reverse['pearson'],-1)
        self.assertEqual(reverse['direction_agreement'],0)

    def test_gaps_do_not_create_velocity(self):
        mask=np.ones(len(self.t),bool)
        mask[1400:1600]=False
        x=np.where(self.t<3,0.,40.)
        r=trend_metrics(self.t,x,x,mask)
        self.assertEqual(r['moving_frames'],0)
        self.assertIsNone(r['direction_agreement'])

    def test_static_and_short_runs_not_eligible(self):
        r=trend_metrics(self.t,self.x*.01,self.x*.01,np.ones(len(self.t),bool))
        self.assertFalse(r['eligible'])
        mask=np.arange(len(self.t))%20<10
        r=trend_metrics(self.t,self.x,self.x,mask)
        self.assertEqual(r['analyzed_frames'],0)
        self.assertFalse(r['eligible'])


if __name__=='__main__': unittest.main()
