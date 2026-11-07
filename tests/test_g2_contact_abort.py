"""Replay a measured slip through the production execute/episode stop path.

The joint dynamics and collision backend here are synthetic. The failure's
tool/block relation is the retained real CPU episode0 counterexample.
"""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from g2_adapter import Result,Stamp
from g2_capture import capture_arms
from g2_episode import run_episode
from g2_motion import Trajectory
from g2_robotwin_backend import ARXBackend,pose7
from robotwin_control_geometry import Status
from test_g2_robotwin_backend import fake_backend
from test_g2_contact import world,contacts
from test_g2_integration import FixedPolicy,spec

ROOT=Path(__file__).resolve().parents[1]
class ContactAbortTests(unittest.TestCase):
    def test_measured_slip_stops_after_first_step_holds_and_saves_prefix(self):
        monitor=json.loads((ROOT/'research/g2/arx-expert-2N-20261106/episode0/contact-monitor.json').read_text())
        failure=monitor['failure']['diagnostic']['contact_state']
        anchor=np.array(failure['tool_block']['right'])
        slipped=np.array(failure['tool_block_measured']['right'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);b,adapter,states,written,q=fake_backend(root);w=world(adapter.arx)
            w.frames=[s.world_base for s in capture_arms(b.env.robot,Stamp('mock',0,0.))]
            w.refresh=lambda:None
            w.planning_geometry=lambda q,g,previous,remaining:previous.copy()
            measured_checks=[]
            def measured_clear(q,g,*,check_unheld=False):
                measured_checks.append(check_unheld)
                return True
            w.measured_clear=measured_clear
            for i,(_,_,grip) in enumerate(b._joints()):
                for f,joint in enumerate(grip):
                    joint.set_drive_property=lambda **kw:None
                    joint.get_drive_target=lambda i=i,f=f:[states[i]['target'][6+f]]
            for state in states:state['actual'][6:]=.03;state['target'][6:]=.03
            states[1]['target'][6:]=0.
            b.initial['actual_gripper']={'left':.75,'right':.75}
            stepped=[];original_step=b.env.scene.step
            def physics_step():
                original_step();stepped.append(1)
                for state in states:state['actual'][6:]=.03
            b.env.scene.step=physics_step
            def block_pose():
                tool=w.tools(np.array([s['actual'][:6] for s in states]))[1]
                value=pose7(tool@(slipped if stepped else anchor))
                return NS(p=value[:3],q=value[3:])
            b.env.box=NS(get_pose=block_pose)
            b.env.scene.get_contacts=lambda:contacts([3,4])
            w.tracker.holders['right']=anchor.copy()
            backend=ARXBackend(b.env,b.arms,w,b.world_task,b.dt,b.initial,.01,1.,b.video,
                               dict(stiffness=1000.,damping=200.,force_limit_n=2.,source='synthetic_stop_test'))
            drive=np.array([[.03,.03],[0.,0.]])
            plan=Trajectory(.02,np.repeat(q[None],6,axis=0),np.zeros((6,2,6)),
                            np.repeat(drive[None],6,axis=0),(5,),np.full((6,2,2),.03),w.profile_id)
            adapter.actions=lambda *args:Result(Status.OK,'',{},plan)
            closed=[]
            backend.env.close_env=lambda **kw:closed.append(1)
            result=run_episode(spec(1),backend,FixedPolicy(adapter.seed.copy()),root/'episode',adapter=adapter)
            self.assertEqual(result['outcome'],'control_error',result)
            self.assertEqual(result['control_calls'],1);self.assertEqual(result['executed'],0)
            self.assertEqual(len(stepped),1);self.assertEqual(backend.tick,1)
            self.assertEqual(measured_checks,[True])
            self.assertEqual(closed,[1])
            self.assertTrue(all(np.all(state['speed']==0.) for state in states))
            np.testing.assert_array_equal(states[1]['target'][6:],[0.,0.])
            rows=[json.loads(line) for line in (root/'episode/events.jsonl').read_text().splitlines()]
            errors=[row for row in rows if row.get('kind')=='contact_failure']
            self.assertEqual(len(errors),1)
            self.assertIn('slipped',errors[0]['detail'])
            self.assertEqual(errors[0]['diagnostic']['side'],'right')
            self.assertEqual(errors[0]['physics_step_count'],1)
            self.assertTrue((root/'episode/plan-00001.npz').exists())
            self.assertEqual(json.loads((root/'episode/result.json').read_text())['outcome'],'control_error')
