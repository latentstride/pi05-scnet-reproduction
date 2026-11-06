import json,unittest
from pathlib import Path
import numpy as np
from g2_assets import load_arms
from g2_initial import matrix7
from g2_kinematics import pose_error
from robotwin_control_geometry import inverse_pose

ROOT=Path(__file__).resolve().parents[1]
class ExpertWorkspaceTests(unittest.TestCase):
    def test_all_854_selected_solutions_reproduce_goals_within_frozen_tolerance(self):
        cfg=json.loads((ROOT/'research/g2/expert-settle-source-mass-20261106/config.json').read_text())
        settled=json.loads((ROOT/'research/g2/expert-settle-source-mass-20261106/settling.json').read_text())
        arms=load_arms(ROOT/'research/robotwin/embodiment_assets',ROOT/'research/robotwin/embodiment_config_audit.json','ARX-X5',cfg['calibration']['ARX-X5'])
        count=0
        for ep,record in enumerate(settled['records']):
            bases=record['snapshots']['ARX-X5']['initial']['robot_base_world']
            report=json.loads((ROOT/f'research/g2/expert-workspace-20261106-r2/episode{ep}.json').read_text())
            for frame in report['records']:
                for side,arm,row in zip(('left','right'),arms,frame['arms']):
                    self.assertEqual(row['status'],'ik_solution')
                    q=np.array(row['q'],dtype=np.float32).astype(float)
                    self.assertTrue(np.all(q>=arm.chain.limits[:,0]));self.assertTrue(np.all(q<=arm.chain.limits[:,1]))
                    actual=matrix7(bases[side])@arm.chain.fk(q)@arm.offset()
                    error=pose_error(np.array(row['target_world_tcp']),actual)
                    self.assertLessEqual(np.linalg.norm(error[:3]),.0005)
                    self.assertLessEqual(np.rad2deg(np.linalg.norm(error[3:])),.05)
                count+=1
        self.assertEqual(count,854)
    def test_summary_does_not_claim_collision_or_task_acceptance(self):
        r=json.loads((ROOT/'research/g2/expert-workspace-20261106-r2/summary.json').read_text())
        self.assertFalse(r['collision_checked']);self.assertFalse(r['task_success_evaluated']);self.assertFalse(r['formal_roster_frozen'])

    def _historical_tool_failure(self):
        cfg=json.loads((ROOT/'research/g2/expert-settle-source-mass-20261106/config.json').read_text())
        settled=json.loads((ROOT/'research/g2/expert-settle-source-mass-20261106/settling.json').read_text())
        arms=load_arms(ROOT/'research/robotwin/embodiment_assets',ROOT/'research/robotwin/embodiment_config_audit.json','ARX-X5',cfg['calibration']['ARX-X5'])
        rows=[dict(row,episode=ep) for ep in range(3) for row in json.loads((ROOT/f'research/g2/expert-workspace-20261106/episode{ep}.json').read_text())['records']]
        for r in rows:
            for side,arm,a in zip(('left','right'),arms,r['arms']):
                base=matrix7(settled['records'][r['episode']]['snapshots']['ARX-X5']['initial']['robot_base_world'][side]);q=np.array(a['q'],dtype=np.float32).astype(float);goal=np.array(a['target_world_tcp'])
                err=pose_error(goal,base@arm.chain.fk(q)@arm.offset())
                if np.linalg.norm(err[:3])>.0005:return arm,base,q,goal,a['residual']
        self.fail('historical regression example missing')

    def test_runtime_solver_refines_terminal_solution_that_misses_tool_tolerance(self):
        from g2_adapter import tool_ik,deadline
        arm,base,q,goal,res=self._historical_tool_failure()
        self.assertLessEqual(res['position_m'],.0005)
        solutions=list(tool_ik(arm,base,goal,q,deadline(3.)))
        self.assertTrue(solutions)
        for values,diagnostic in solutions:
            error=pose_error(goal,base@arm.chain.fk(values)@arm.offset())
            self.assertLessEqual(np.linalg.norm(error[:3]),.0005)
            self.assertAlmostEqual(np.linalg.norm(error[:3]),diagnostic['position_m'])

    def test_runtime_solver_independently_rejects_invalid_backend_candidate(self):
        from unittest.mock import patch
        from g2_adapter import tool_ik,deadline
        arm,base,q,goal,res=self._historical_tool_failure()
        with patch.object(arm.chain,'ik',return_value=iter([(q,res)])):
            self.assertEqual(list(tool_ik(arm,base,goal,q,deadline(1.))),[])
