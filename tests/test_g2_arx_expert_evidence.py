import contextlib,hashlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
from g2_verify_arx_expert import verify,first_failure_metrics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research/g2'
class ARXExpertEvidenceTests(unittest.TestCase):
    def test_three_real_replays_keep_slip_failure_separate_from_task_success(self):
        p=BASE/'arx-expert-2N-20261106'
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            report=verify(p,p/'reference-q.npz',BASE,Path(tmp)/'acceptance.json')
        self.assertEqual(report['task_successes'],3);self.assertFalse(report['all_verified'])
        rows=report['episodes'];self.assertEqual([r['verification_passed'] for r in rows],[False,True,True])
        self.assertTrue(all(r['hold_success'] for r in rows))
        self.assertGreater(rows[0]['first_failure_metrics']['translation_mm'],5.)
        self.assertLess(rows[0]['sampled_max_cumulative_slip_mm'],5.)
        self.assertEqual(rows[0]['guard_observed_through_tick'],27463)
        self.assertFalse(any(r['production_collision_gate_exercised'] for r in rows))
    def test_archived_bytes_and_shadow_observer_preserve_physics(self):
        p=BASE/'arx-expert-2N-20261106'
        for rel,r in json.loads((p/'manifest.json').read_text()).items():
            self.assertEqual(hashlib.sha256((p/rel).read_bytes()).hexdigest(),r['sha256'],rel)
        with np.load(p/'episode0/physics-trace.npz') as observed,np.load(p/'unmonitored0/physics-trace.npz') as original:
            for key in ('sample_ticks','actual_native','block_pose','actual_endpose_matrix','drive_native'):
                np.testing.assert_array_equal(observed[key],original[key])
    def test_higher_force_failure_precedes_tray_contact(self):
        p=BASE/'arx-expert-4N-episode0-20261106'
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            report=verify(p,BASE/'arx-expert-2N-20261106/reference-q.npz',BASE,Path(tmp)/'acceptance.json',4.,(0,))
        row=report['episodes'][0];self.assertTrue(row['task_success']);self.assertFalse(row['verification_passed'])
        self.assertEqual(row['first_failure_metrics']['tick'],9155)
        self.assertGreater(row['first_failure_metrics']['translation_mm'],5.)
        with np.load(p/'episode0/physics-trace.npz') as trace:
            first=np.flatnonzero(trace['contacts_4ms'][:,2:4].sum(axis=1)>1e-8)[0]
        self.assertGreater(first,row['first_failure_metrics']['tick'])
        self.assertIsNone(first_failure_metrics({'failure':None}))
