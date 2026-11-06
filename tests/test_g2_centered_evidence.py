import contextlib,hashlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
from g2_verify_arx_expert import verify
from g2_contact import contact_profile
from g2_finger_force import MODE
from g2_make_contact_profile import make_profile,monitor_contract

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'research/g2'
class CenteredEvidenceTests(unittest.TestCase):
    def test_monitor_contract_tracks_thresholds_separately_from_planning_implementation(self):
        source=ROOT/'scripts/g2_contact.py';digest=monitor_contract(source)
        with tempfile.TemporaryDirectory() as tmp:
            candidate=Path(tmp)/'contact.py'
            candidate.write_text(source.read_text().replace('position_slip_m: float = .005','position_slip_m: float = .006'))
            self.assertNotEqual(digest,monitor_contract(candidate))
            candidate.write_text(source.read_text().replace('self.project_actions=project_actions','self.project_actions=False'))
            self.assertEqual(digest,monitor_contract(candidate))
    def test_three_real_cases_verify_force_bounds_task_and_continuous_guard(self):
        p=BASE/'arx-centered-20261106'
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            r=verify(p,p/'reference-q.npz',BASE,Path(tmp)/'acceptance.json')
        self.assertTrue(r['all_verified'],r);self.assertEqual(r['task_successes'],3)
        self.assertTrue(all(v['hold_success'] for v in r['episodes']))
        self.assertTrue(all(v['checks']['all_control_forces_bounded'] for v in r['episodes']))
        for ep in range(3):
            for rel,record in json.loads((p/f'manifest-ep{ep}.json').read_text()).items():
                self.assertEqual(hashlib.sha256((p/rel).read_bytes()).hexdigest(),record['sha256'],rel)
        profile=json.loads((p/'contact-profile.json').read_text())
        cfg={'embodiment':'ARX-X5','contact_profile':profile}
        _,drive=contact_profile(cfg,.009999999776482582)
        self.assertEqual(drive['mode'],MODE)
        with self.assertRaisesRegex(ValueError,'different block mass'):contact_profile(cfg,.72)
        with tempfile.TemporaryDirectory() as tmp:
            selected=make_profile(p/'acceptance.json',Path(tmp)/'verified.json')
            self.assertEqual(selected['drive']['mode'],MODE)
            with self.assertRaises(ValueError):make_profile(p/'acceptance-ep0.json',Path(tmp)/'incomplete.json')
            with self.assertRaises(ValueError):make_profile(BASE/'arx-expert-2N-20261106/acceptance.json',Path(tmp)/'failed.json')
    def test_fix_preserves_commands_and_reduces_measured_exclusive_holder_center_drift(self):
        with np.load(BASE/'arx-centered-20261106/episode0/physics-trace.npz') as current,np.load(BASE/'arx-expert-2N-20261106/episode0/physics-trace.npz') as old:
            for key in ('sample_ticks','frame_ticks','drive_native','normalized_gripper_target'):
                np.testing.assert_array_equal(current[key],old[key])
            sampled=current['sample_ticks'];window=(sampled>=21907)&(sampled<=27463)
            a=(current['actual_native'][window,14]-current['actual_native'][window,15])/2
            b=(old['actual_native'][window,14]-old['actual_native'][window,15])/2
            self.assertLess(np.max(abs(a)),np.max(abs(b))/2)
    def test_failed_force_and_acceleration_attempts_stay_in_the_record(self):
        for rel in ('arx-expert-1N-failed-20261106','arx-slip-comparisons-20261106/side','arx-slip-comparisons-20261106/soft'):
            r=json.loads((BASE/rel/'acceptance.json').read_text())
            self.assertFalse(r['all_verified']);self.assertTrue(r['episodes'][0]['task_success'])
            self.assertIsNotNone(r['episodes'][0]['monitor_failure'])
