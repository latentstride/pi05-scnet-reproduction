import ast,json,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from g2_kinematics import pose_error
from robotwin_control_geometry import pose_from_wxyz

ROOT=Path(__file__).resolve().parents[1]
class NativeReplayTests(unittest.TestCase):
    def setUp(self):
        self.folder=ROOT/'research/g2/native-expert-episode0-20261106'
        self.result=json.loads((self.folder/'replay.json').read_text());self.trace=np.load(self.folder/'physics-trace.npz')
    def test_complete_physics_and_original_success_predicate(self):
        r=self.result;t=self.trace
        self.assertTrue(r['completed']);self.assertEqual(r['physics_steps_replayed'],3945)
        self.assertEqual(len(t['actual_native']),3946+200);self.assertAlmostEqual(r['actual_block_mass_kg'],.01,places=8)
        self.assertFalse(r['gpu_used']);self.assertFalse(r['model_used']);self.assertFalse(r['renderer_created'])
        tree=ast.parse((ROOT/'research/g2/expert-source-20261105/envs/handover_block.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='check_success')
        ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'pinned_success','exec'),ns)
        target=np.array(json.loads((self.folder/'initial.json').read_text())['target_pose']);top=(pose_from_wxyz(target[:3],target[3:])@np.array([0,0,.005,1]))[:3]
        for tick,key in [(3945,'native_success_at_expert_end'),(4145,'native_success_after_diagnostic_hold')]:
            v=t['block_pose'][tick];bottom=(pose_from_wxyz(v[:3],v[3:])@np.array([0,0,-.1,1]))[:3]
            obj=NS(box=NS(get_functional_point=lambda *args:NS(p=bottom)),target_box=NS(get_functional_point=lambda *args:NS(p=top)),is_right_gripper_open=lambda:bool(t['normalized_gripper_target'][tick,1]>.8))
            self.assertEqual(bool(ns['check_success'](obj)),r[key]['success']);self.assertTrue(r[key]['success'])
    def test_native_arm_drive_targets_match_dense_source_and_archived_measured_poses(self):
        z=np.load(ROOT/'research/g2/expert-dense-episode0-20261106/trace.npz');t=self.trace
        np.testing.assert_array_equal(np.c_[t['drive_native'][:3946,:6],t['drive_native'][:3946,8:14]],np.c_[z['target'][:,:6],z['target'][:,7:13]])
        np.testing.assert_array_equal(t['normalized_gripper_target'][:3946],z['target'][:,[6,13]])
        actual=t['actual_endpose_matrix'][t['frame_ticks']];expected=z['expected_endpose']
        errors=np.array([[pose_error(pose_from_wxyz(v[:3],v[3:]),a) for v,a in zip(vs,matrices)] for vs,matrices in zip(expected,actual)])
        self.assertLessEqual(np.linalg.norm(errors[:,:,:3],axis=2).max(),.001)
        self.assertLessEqual(np.rad2deg(np.linalg.norm(errors[:,:,3:],axis=2)).max(),.1)
        self.assertLess(np.min(t['drive_native'][:,[6,7,14,15]]),0.)
        self.assertGreaterEqual(np.min(t['actual_native'][:,[6,7,14,15]]),-.001)
    def test_three_actual_replays_pass_independent_verification(self):
        import tempfile
        from g2_verify_native_expert import verify
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            result=verify(ROOT,Path(tmp)/'acceptance.json')
        self.assertTrue(result['all_verified']);self.assertEqual(result['native_task_successes'],3)
        self.assertEqual([e['physics_steps'] for e in result['episodes']],[3945,3655,3764])
        self.assertTrue(all(e['hold_success'] for e in result['episodes']))
        self.assertTrue(all(e['max_finger_target_actual_penetration_mm']<.02 for e in result['episodes']))
