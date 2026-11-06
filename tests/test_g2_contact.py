"""Contact semantics and failure protocols; synthetic geometry is labelled explicitly."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import json
import tempfile
import unittest

import numpy as np

from g2_adapter import G2Error, Stamp, deadline
from g2_contact import ContactLimits, ContactTracker, ContactWorld
from g2_fixtures import fixture
from g2_collision import JointCollisionWorld
from g2_robotwin_backend import ARXBackend
from test_g2_robotwin_backend import fake_backend
from g2_episode import run_episode
from test_g2_integration import MockRobot, FixedPolicy, spec
from g2_task_runtime import validate_config
from g2_contact import CONTACT_MODE
from dataclasses import asdict, replace


def contacts(ids, impulse=1.):
    return [NS(bodies=[NS(entity=NS(per_scene_id=99)),NS(entity=NS(per_scene_id=i))],
               points=[NS(impulse=np.array([impulse,0.,0.]))]) for i in ids]


def tracker():
    return ContactTracker(99,dict(left=[1,2],right=[3,4]),ContactLimits(stable_steps=3))


def world(arms):
    w=ContactWorld.__new__(ContactWorld)
    w.arms=arms
    w.tracker=tracker()
    w.limits=w.tracker.limits
    w.profile_id=w.limits.profile_id
    w.project_actions=True
    w.failure=None
    w.projection_log=[]
    w.initial_fingers=None
    w.initial_drive=None
    w.giver="left"
    w.world_task=np.eye(4)
    # Synthetic obstacle: left fingers meet a 25 mm per-finger opening.
    w.finger_depth=lambda q,g,s,f: max(0.,.025-g[s,f]) if s==0 else 0.
    w.projection_context=lambda q:nullcontext()
    w.base=NS(evaluate=lambda q,g,contact: bool(np.all(g[0]>=.02449)),last_collisions=[])
    return w


class ContactTests(unittest.TestCase):
    def test_projection_context_is_restored_when_deadline_interrupts_scan(self):
        from contextlib import contextmanager
        from robotwin_control_geometry import Status
        adapter,_,q,_=fixture();w=world(adapter.arx);entries=[]
        @contextmanager
        def candidate(q):
            entries.append('enter')
            try:yield
            finally:entries.append('exit')
        w.object_candidate=candidate
        w.projection_context=ContactWorld.projection_context.__get__(w)
        w.set_start(np.full((2,2),.04))
        def expired():raise G2Error(Status.TIMEOUT,'test deadline')
        with self.assertRaises(G2Error):w.project_fingers(q,np.zeros((2,2)),expired)
        self.assertEqual(entries,['enter','exit']);self.assertIsNone(w._projection_q)

    def test_per_side_force_limits_reach_both_native_fingers(self):
        from g2_contact import finger_force_limits
        np.testing.assert_array_equal(finger_force_limits(2.),[2.,2.])
        for invalid in ([2.],[[2.],[1.]],[-1.,2.],[1.,float('nan')]):
            with self.assertRaises((ValueError,G2Error)):finger_force_limits(invalid)
        with tempfile.TemporaryDirectory() as tmp:
            b,adapter,states,written,q=fake_backend(Path(tmp));w=world(adapter.arx);properties=[[],[]]
            for side,(_,_,grip) in enumerate(b._joints()):
                for joint in grip:joint.set_drive_property=lambda side=side,**kw:properties[side].append(kw)
            ARXBackend(b.env,b.arms,w,b.world_task,b.dt,b.initial,.01,1.,b.video,
                dict(stiffness=1000.,damping=200.,force_limit_n=[2.,1.],source='synthetic per-side force test'))
            self.assertEqual([[x['force_limit'] for x in p] for p in properties],[[2.,2.],[1.,1.]])

    def test_force_calibration_checks_actual_mass_not_primitive_density(self):
        from g2_contact import contact_profile
        from test_g2_runtime import configuration
        cfg=configuration();cfg['contact_profile']=dict(status='cpu_diagnostic_only',source='mass binding regression',
            giver='left',limits=asdict(ContactLimits()),validated_block_mass_kg=.01,
            drive=dict(stiffness=1000.,damping=200.,force_limit_n=1.,source='synthetic calibration scope'))
        limits,drive=contact_profile(cfg,.009999999776482582)
        self.assertEqual(drive['force_limit_n'],1.)
        with self.assertRaisesRegex(ValueError,'different block mass'):contact_profile(cfg,.72)
        for invalid in (0.,float('nan')):
            with self.assertRaises((ValueError,G2Error)):contact_profile(cfg,invalid)

    def test_carried_pose_converts_task_to_world_and_checks_both_holders(self):
        adapter,_,q,bases=fixture();w=world(adapter.arx)
        w.frames=bases
        w.world_task=np.eye(4);w.world_task[:3,3]=[1,2,3]
        w.tracker.holders={"left":np.eye(4)}
        expected=w.world_task@w.tools(q)[0]
        np.testing.assert_allclose(w.held_pose_world(q),expected)
        w.tracker.holders["right"]=np.eye(4)
        with self.assertRaises(G2Error):w.held_pose_world(q)

    def test_explicit_diagnostic_config_has_no_formal_or_implicit_default_path(self):
        from test_g2_runtime import configuration
        cfg=configuration();cfg["contact_mode"]=CONTACT_MODE
        cfg["contact_profile"]=dict(status="cpu_diagnostic_only",source="test-only",giver="left",
                                     limits=asdict(ContactLimits()),
                                     drive=dict(source="test-only",stiffness=1000.,damping=200.,force_limit_n=2.))
        validate_config(cfg)
        cfg["data_kind"]="formal"
        with self.assertRaises(ValueError):validate_config(cfg)
        cfg["data_kind"]="diagnostic";del cfg["contact_profile"]["limits"]["penetration_m"]
        with self.assertRaises(ValueError):validate_config(cfg)

    def test_episode_contact_barrier_requeries_instead_of_replaying_suffix(self):
        adapter,qa,qr,bases=fixture()
        original=adapter.actions
        def actions(*args):
            result=original(*args)
            if not result.ok:return result
            plan=result.payload
            end=plan.waypoint_ticks[0]+1
            plan=replace(plan,q=plan.q[:end],velocity=plan.velocity[:end],fingers=plan.fingers[:end],
                         waypoint_ticks=(end-1,),predicted_fingers=plan.fingers[:end].copy(),contact_profile_id="synthetic")
            return replace(result,payload=plan)
        adapter.actions=actions
        backend=MockRobot(adapter,qa,qr,bases)
        with tempfile.TemporaryDirectory() as tmp:
            result=run_episode(spec(4),backend,FixedPolicy(qa),Path(tmp)/"e",adapter=adapter)
            self.assertEqual(result["outcome"],"success",result)
            self.assertEqual(result["queries"],4)
            events=[json.loads(line) for line in (Path(tmp)/"e/events.jsonl").read_text().splitlines()]
            self.assertTrue(any(e.get("kind")=="contact_replan" for e in events))

    def test_execution_interrupt_on_measured_contact_does_not_apply_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            b,adapter,states,written,q=fake_backend(Path(tmp));w=world(adapter.arx)
            w.refresh=lambda:None
            for _,_,grip in b._joints():
                for j in grip:j.set_drive_property=lambda **kw:None
            for i,(_,_,grip) in enumerate(b._joints()):
                for f,j in enumerate(grip):j.get_drive_target=lambda i=i,f=f:[states[i]["target"][6+f]]
            for state in states:state["actual"][6:]=.03;state["target"][6:]=.03
            b.initial["actual_gripper"]={"left":.75,"right":.75}
            b.env.box=NS(get_pose=lambda:NS(p=[0,0,0],q=[1,0,0,0]))
            b.env.scene.get_contacts=lambda:contacts([1,2])
            b=ARXBackend(b.env,b.arms,w,b.world_task,b.dt,b.initial,.01,1.,b.video,
                         dict(stiffness=1000.,damping=200.,force_limit_n=2.,source="synthetic_test"))
            b.reset(dict(episode_id="mock"))
            plan=adapter.motion.plan(q,np.full((2,2),.03),[(q,np.full((2,2),.03)),(q+.005,np.full((2,2),.03))],w,deadline(2))
            logs=[]
            # Use a five-physics-step hold so bilateral stability occurs mid-waypoint.
            plan=replace(plan,period_s=.1)
            done=b.execute(plan,logs.append,lambda:10.)
            self.assertEqual(done,1)
            reason=b.take_replan();self.assertEqual(reason["after"],["left"])
            self.assertTrue(reason["interrupted_waypoint"])
            self.assertIsNone(b.take_replan())
            np.testing.assert_array_equal(np.array([s["target"][:6] for s in states]),q)
    def test_dynamic_lift_finger_asymmetry_is_not_silently_hidden(self):
        # Observed 20 N fixture endpoint: independent fingers differ by 1.455 mm.
        # Existing scalar proxy contract must reject until a new mapping is explicit.
        grip=fixture()[0].arx[0].grip
        with self.assertRaisesRegex(G2Error,"mimic mismatch"):
            grip.read(np.array([.02883804589509964,.030293013900518417]))

    def test_bilateral_stability_clock_and_impulse_required(self):
        t=tracker();tools=np.array([np.eye(4),np.eye(4)])
        for tick,ids in enumerate(([1],[1,2],[1,2],[1,2])):
            state=t.observe(Stamp("e",tick,tick*.01),contacts(ids),tools,np.eye(4))
            self.assertEqual(state["holders"],["left"] if tick==3 else [])
        self.assertEqual(t.observe(Stamp("e",3,.03),[],tools,np.eye(4))["holders"],["left"])
        with self.assertRaises(G2Error):t.observe(Stamp("other",4,.04),[],tools,np.eye(4))
        t=tracker()
        for tick in range(4):t.observe(Stamp("e",tick,tick*.01),contacts([1,2],0.),tools,np.eye(4))
        self.assertFalse(t.holders)

    def test_moving_object_does_not_accumulate_stability_and_gaps_reset_count(self):
        t=tracker();tools=np.array([np.eye(4),np.eye(4)])
        for tick in range(5):
            block=np.eye(4);block[0,3]=tick*.01
            t.observe(Stamp("e",tick,tick*.01),contacts([1,2]),tools,block)
        self.assertFalse(t.holders)
        t=tracker()
        for tick in (0,1,4):t.observe(Stamp("e",tick,tick*.01),contacts([1,2]),tools,np.eye(4))
        self.assertFalse(t.holders)
        self.assertEqual(t.counts["left"],1)

    def test_donor_release_needs_receiver_and_unexpected_loss_fails(self):
        arms=fixture()[0].arx;t=tracker();tools=np.array([np.eye(4),np.eye(4)])
        for tick in range(3):t.observe(Stamp("e",tick,tick*.01),contacts([1,2]),tools,np.eye(4))
        prev=np.zeros((2,2));opened=prev.copy();opened[0]=.04
        with self.assertRaises(G2Error):t.authorize_opening(prev,opened,arms)
        self.assertFalse(t.releasing)
        with self.assertRaises(G2Error):t.observe(Stamp("e",3,.03),contacts([1]),tools,np.eye(4))
        t=tracker()
        for tick in range(3):t.observe(Stamp("e",tick,tick*.01),contacts([1,2,3,4]),tools,np.eye(4))
        t.authorize_opening(prev,opened,arms)
        self.assertEqual(set(t.carrying),{"right"})
        t.observe(Stamp("e",3,.03),contacts([3,4]),tools,np.eye(4))
        self.assertEqual(t.phase,"right_holding")

    def test_donor_release_phase_holds_arms_before_withdrawal(self):
        adapter,_,q,_=fixture();w=world(adapter.arx)
        w.set_start(np.full((2,2),.025),np.zeros((2,2)))
        w.tracker.holders={"left":np.eye(4),"right":np.eye(4)}
        tools=np.array([np.eye(4),np.eye(4)]);goals=tools.copy();goals[0,2,3]=.1
        intent=np.zeros((2,2));intent[0]=.04
        actual,phase=w.prepare_tools(goals,intent,tools)
        np.testing.assert_array_equal(actual,tools)
        np.testing.assert_array_equal(goals[0,:3,3],[0,0,.1])
        self.assertEqual(phase,"donor_release_before_withdrawal")
        self.assertEqual(set(w.tracker.holders),{"left","right"})

    def test_prediction_uses_current_attachment_without_resetting_slip_reference(self):
        t=tracker();tools=np.array([np.eye(4),np.eye(4)])
        for tick in range(3):t.observe(Stamp('e',tick,tick*.01),contacts([1,2]),tools,np.eye(4))
        moved=np.eye(4);moved[0,3]=.003
        t.observe(Stamp('e',3,.03),contacts([1,2]),tools,moved)
        np.testing.assert_array_equal(t.holders['left'],np.eye(4))
        np.testing.assert_allclose(t.carrying['left'],moved)
        moved[0,3]=.006
        with self.assertRaisesRegex(G2Error,'slipped'):
            t.observe(Stamp('e',4,.04),contacts([1,2]),tools,moved)
        np.testing.assert_array_equal(t.holders['left'],np.eye(4))

    def test_confirmed_slip_is_explicit_failure(self):
        t=tracker();tools=np.array([np.eye(4),np.eye(4)])
        for tick in range(3):t.observe(Stamp("e",tick,tick*.01),contacts([1,2]),tools,np.eye(4))
        moved=np.eye(4);moved[2,3]=.02
        with self.assertRaisesRegex(G2Error,"slipped"):
            t.observe(Stamp("e",3,.03),contacts([1,2]),tools,moved)

    def test_prediction_keeps_intent_and_measured_start_separate(self):
        adapter,_,q,_=fixture();w=world(adapter.arx)
        w.set_start(np.full((2,2),.04),np.full((2,2),.04))
        requested=np.zeros((2,2))
        predicted,changes=w.project_fingers(q,requested,deadline(2))
        np.testing.assert_array_equal(requested,np.zeros((2,2)))
        np.testing.assert_allclose(predicted[0],.0245,atol=1e-6)
        np.testing.assert_array_equal(predicted[1],[0.,0.])
        self.assertEqual(len(changes),2)
        self.assertTrue(w.action_barrier(q,requested,deadline(2)))
        # A held gripper retains its zero drive through the next plan.
        measured=np.array([[.025,.025],[0.,0.]])
        w.set_start(measured,np.zeros((2,2)))
        plan=adapter.motion.plan(q,measured,[(q,requested)],w,deadline(2))
        np.testing.assert_array_equal(plan.fingers,np.zeros((len(plan.q),2,2)))
        np.testing.assert_array_equal(plan.predicted_fingers[0],measured)
        self.assertTrue(w(q,plan.predicted_fingers[-1]))
        self.assertFalse(w(q,plan.fingers[-1]))
        self.assertEqual(plan.contact_profile_id,w.profile_id)

    def test_confirmed_holder_preserves_intent_and_full_collision_checks(self):
        adapter,_,q,_=fixture();w=world(adapter.arx)
        measured=np.array([[.025,.026],[.04,.04]])
        w.set_start(measured,np.array([[0.,0.],[.04,.04]]))
        w.tracker.holders['left']=np.eye(4)
        desired=np.array([[0.,0.],[.04,.04]])
        predicted,changes=w.project_fingers(q,desired,deadline(2))
        self.assertTrue(np.all(predicted[0]>=.02449))
        np.testing.assert_array_equal(desired[0],[0.,0.])
        np.testing.assert_array_equal(measured[0],[.025,.026])
        self.assertTrue(w(q,predicted))
        w.base.evaluate=lambda *args,**kw:False
        self.assertFalse(w(q,predicted))

    def test_unknown_depth_is_rejected_and_json_serializable(self):
        adapter,_,q,_=fixture();w=world(adapter.arx)
        w.set_start(np.full((2,2),.04))
        w.finger_depth=lambda *args: None
        with self.assertRaises(G2Error):w.project_fingers(q,np.zeros((2,2)),deadline(2))
        self.assertIsNone(w.depth(NS(get_contacts=lambda:[])))
        w.allowed_pairs={("robot","finger")};w.block_name="box";w.support_names={"table"}
        row=dict(object_name1="box",link_name1="box",object_name2="robot",link_name2="finger",max_penetration_m=None)
        self.assertFalse(w.allows(row));json.dumps(row,allow_nan=False)
        row["max_penetration_m"]=.0001;self.assertTrue(w.allows(row))
        row["max_penetration_m"]=.01;self.assertFalse(w.allows(row))
        row.update(max_penetration_m=.0001,link_name2="palm");self.assertFalse(w.allows(row))

    def test_force_limited_backend_drives_intent_and_logs_actual(self):
        with tempfile.TemporaryDirectory() as tmp:
            b,adapter,states,written,q=fake_backend(Path(tmp));w=world(adapter.arx)
            w.refresh=lambda: None
            props=[]
            for entity,_,grip in b._joints():
                for j in grip:j.set_drive_property=lambda **kw:props.append(kw)
            for i,(_,_,grip) in enumerate(b._joints()):
                for f,j in enumerate(grip):j.get_drive_target=lambda i=i,f=f: [states[i]["target"][6+f]]
            b.env.box=NS(get_pose=lambda:NS(p=[0,0,0],q=[1,0,0,0]))
            b.env.scene.get_contacts=lambda: []
            def step():
                for state in states:
                    state["actual"][:6]=state["target"][:6]
                    state["actual"][6:]=np.maximum(state["target"][6:],.025)
            b.env.scene.step=step
            for state in states:
                state["actual"][6:]=.03;state["target"][6:]=.03
            b.initial["actual_gripper"]={"left":.75,"right":.75}
            b.collision=w
            b=ARXBackend(b.env,b.arms,w,b.world_task,b.dt,b.initial,.01,1.,b.video,
                         dict(stiffness=1000.,damping=200.,force_limit_n=2.,source="synthetic_test"))
            b.reset(dict(episode_id="mock"));self.assertEqual(len(props),4)
            for x in props:self.assertEqual(x["force_limit"],2.)
            plan=adapter.motion.plan(q,np.full((2,2),.03),[(q,np.zeros((2,2)))],w,deadline(2))
            logs=[]
            self.assertEqual(b.execute(plan,logs.append,lambda:10.),1)
            np.testing.assert_array_equal(np.array([s["target"][6:] for s in states]),np.zeros((2,2)))
            np.testing.assert_allclose(logs[-1]["actual_fingers"],.025)
            np.testing.assert_array_equal(logs[-1]["target_fingers"],np.zeros((2,2)))
            self.assertTrue(logs[-1]["finger_drive_properties"]["force_limit_n"]==2.)
            w.tracker.holders["left"]=np.eye(4)
            b.hold()
            np.testing.assert_array_equal(states[0]["target"][6:],[0.,0.])
            np.testing.assert_allclose(states[1]["target"][6:],.025)
