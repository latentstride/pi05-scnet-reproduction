import json,unittest
from pathlib import Path
import numpy as np
from g2_arx_reference import retime_reference

ROOT=Path(__file__).resolve().parents[1]
class ARXReferenceTests(unittest.TestCase):
    def test_lower_acceleration_preserves_every_waypoint_and_common_clock(self):
        z=np.load(ROOT/'research/g2/expert-collision-compare-20261106/workspace-q.npz')
        source=np.load(ROOT/'research/g2/expert-dense-episode0-20261106/trace.npz')
        baseline,_=retime_reference(z['q0'],z['g0'],source['frame_ticks'])
        slower,stats=retime_reference(z['q0'],z['g0'],source['frame_ticks'],arm_acceleration=.5)
        np.testing.assert_array_equal(slower['target'][slower['frame_ticks']],baseline['target'][baseline['frame_ticks']])
        self.assertGreater(len(slower['target']),len(baseline['target']))
        self.assertTrue(np.all(np.array(stats['max_command_acceleration'])[:,:6]<=.5+1e-10))
        self.assertEqual(stats['arm_acceleration_limit_rad_s2'],.5)
        for invalid in (0.,-1.,2.,float('nan')):
            with self.assertRaises(ValueError):retime_reference(z['q0'],z['g0'],source['frame_ticks'],arm_acceleration=invalid)

    def test_all_three_retimed_paths_preserve_targets_and_analytic_command_limits(self):
        z=np.load(ROOT/'research/g2/expert-collision-compare-20261106/workspace-q.npz')
        for ep in range(3):
            source=np.load(ROOT/f'research/g2/expert-dense-episode{ep}-20261106/trace.npz');data,stats=retime_reference(z['q'+str(ep)],z['g'+str(ep)],source['frame_ticks'])
            q=data['target'][data['frame_ticks']];np.testing.assert_allclose(np.stack([q[:,:6],q[:,7:13]],axis=1),z['q'+str(ep)],atol=1e-14,rtol=0)
            np.testing.assert_allclose(q[:,[6,13]],z['g'+str(ep)],atol=1e-14,rtol=0)
            v=np.array(stats['max_command_velocity']);a=np.array(stats['max_command_acceleration'])
            self.assertTrue(np.all(v[:,:6]<=.5+1e-10));self.assertTrue(np.all(a[:,:6]<=1+1e-10))
            self.assertTrue(np.all(v[:,6:]<=.03+1e-10));self.assertTrue(np.all(a[:,6:]<=.1+1e-10))
            self.assertGreaterEqual(stats['retimed_duration_s'],stats['original_source_duration_s'])
            self.assertTrue(np.all(np.diff(data['frame_ticks'])>=0));self.assertFalse(stats['collision_checked'])
    def test_bad_gripper_or_clock_rejected(self):
        q=np.zeros((2,2,6))
        with self.assertRaises(ValueError):retime_reference(q,np.array([[1,1],[2,1]]),[0,15])
        with self.assertRaises(ValueError):retime_reference(q,np.ones((2,2)),[15,0])
