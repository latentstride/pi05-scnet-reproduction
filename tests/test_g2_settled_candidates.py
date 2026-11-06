import copy,json,unittest
from pathlib import Path
from g2_settled_candidates import build,validate_pair

DATA=Path(__file__).resolve().parents[1]/'research/g2/expert-settle-source-mass-20261106/settling.json'
class SettledCandidateTests(unittest.TestCase):
    def setUp(self):self.data=json.loads(DATA.read_text())
    def test_three_real_pairs_are_stable_but_excluded_from_formal_roster(self):
        roster=build(self.data);self.assertFalse(roster['formal_roster_frozen'])
        self.assertEqual(len({r['seed'] for r in roster['candidates']}),3)
        for row in roster['candidates']:
            self.assertTrue(row['object_pair_check_passed']);self.assertIsNone(row['formal_pair_id'])
            self.assertTrue(row['formal_exclusion_reasons'])
    def test_mismatched_objects_are_rejected(self):
        row=copy.deepcopy(self.data['records'][0]);row['snapshots']['ARX-X5']['initial']['objects_task']['block'][0]+=.002
        self.assertIn('object_block',validate_pair(row))
    def test_moving_object_is_rejected(self):
        row=copy.deepcopy(self.data['records'][0]);row['snapshots']['ARX-X5']['block_velocity_m_s']=[.01,0,0]
        self.assertIn('ARX-X5:unstable_block',validate_pair(row))
    def test_robot_supported_object_is_rejected(self):
        row=copy.deepcopy(self.data['records'][0]);row['snapshots']['ARX-X5']['contacts'].append({'actors':['box','link8']})
        self.assertIn('ARX-X5:unexpected_block_support',validate_pair(row))
    def test_real_fcl_evidence_clears_initial_gate_and_keeps_tray_path_failure(self):
        root=DATA.parents[2]/'g2/expert-path-collision-20261106-r2'
        initial=json.loads((root/'initial-collision.json').read_text())
        paths=[json.loads(p.read_text()) for p in sorted(root.glob('workspace-collision-episode*.json'))]
        result=build(self.data,initial,paths)
        for row in result['candidates']:
            self.assertTrue(row['initial_fcl_geometry_passed'])
            self.assertNotIn('initial_robot_self_contacts_require_review',row['formal_exclusion_reasons'])
            self.assertIn('selected_expert_path_hits_target_tray',row['formal_exclusion_reasons'])
            self.assertGreater(row['known_colliding_path_segments'],0)
        changed=copy.deepcopy(initial);changed[0]['actual_arm_qpos_rad'][0][0]+=.01
        with self.assertRaises(ValueError):build(self.data,changed,paths)
