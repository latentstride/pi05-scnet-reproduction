from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import numpy as np

from g2_adapter import Stamp
from g2_capture import capture_snapshot
from g2_fixtures import fixture
from g2_motion import Trajectory
from g2_robotwin_backend import ARXBackend, joint_space_setup, pose7
from g2_adapter import deadline, G2Error
from robotwin_control_geometry import pose_from_wxyz


def fake_backend(root):
    adapter, qa, qr, bases = fixture()
    written, states = [], []
    entities = []
    for side, arm, q, base in zip(("left", "right"), adapter.arx, qr, bases):
        values = np.r_[q, arm.grip.target(.5)]
        names = arm.chain.names + arm.grip.names
        state = dict(actual=values.copy(), target=values.copy(), speed=np.zeros(8))
        states.append(state)
        joints = []
        for index, name in enumerate(names):
            def target(x, index=index, state=state, side=side):
                state["target"][index] = x
                written.append((side, index))
            def speed(x, index=index, state=state):
                state["speed"][index] = x
            joints.append(NS(get_name=lambda name=name: name, set_drive_target=target,
                             set_drive_velocity_target=speed))
        pose = pose7(base)
        entities.append(NS(get_active_joints=lambda joints=joints: joints,
                           get_qpos=lambda state=state: state["actual"].copy(),
                           get_root_pose=lambda pose=pose: NS(p=pose[:3], q=pose[3:]),
                           get_name=lambda instance=arm.instance: instance,
                           compute_passive_force=lambda **kwargs: np.zeros(8), set_qf=lambda x: None))
    def physics_step():
        # All sixteen joint targets must be applied before the first physics step.
        if not any(state.get("stepped") for state in states):
            assert {side for side, _ in written} == {"left", "right"}
            assert len(written) == 16
        for state in states:
            state["actual"] = state["target"].copy()
            state["stepped"] = True
    robot = NS(left_entity=entities[0], right_entity=entities[1])
    env = NS(robot=robot, scene=NS(step=physics_step, get_timestep=lambda: .02), take_action_cnt=0,
             get_obs=lambda: {"observation": {name: {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}
                         for name in ("head_camera", "left_camera", "right_camera")}},
             check_success=lambda: False, close_env=lambda **kwargs: None)
    collision = NS(refresh=lambda: None)
    class Collision:
        def refresh(self):
            pass
        def __call__(self, q, g):
            return True
    class Video:
        path = root / "fake-video"
        def frame(self, obs):
            pass
        def close(self):
            pass
    # Identity names and matrices are collected through the same actual capture code.
    snap, _ = capture_snapshot(env, Stamp("mock", 0, 0.), np.eye(4))
    initial = dict(robot_base_world={side: pose7(row.world_base) for side, row in zip(("left", "right"), snap.arms)},
                   actual_arm_qpos_rad={side: q.tolist() for side, q in zip(("left", "right"), qr)},
                   actual_gripper=dict(left=.5, right=.5))
    backend = ARXBackend(env, adapter.arx, Collision(), np.eye(4), .02, initial, .01, 1., Video())
    return backend, adapter, states, written, qr


class BackendTests(unittest.TestCase):
    def test_changed_scene_rejection_logs_zero_actuation_and_collision_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            backend,adapter,states,written,qr=fake_backend(Path(temp))
            backend.reset(dict(episode_id='mock'))
            plan=adapter.motion.plan(qr,np.full((2,2),.02),[(qr+.005,np.full((2,2),.02))],lambda q,g:True,deadline(2))
            class Blocked:
                last_collisions=[{'object_name1':'arx_left','object_name2':'gate_probe','link_name1':'finger','link_name2':'gate_probe'}]
                def refresh(self):pass
                def __call__(self,q,g):return False
            backend.collision=Blocked();rows=[]
            with self.assertRaisesRegex(G2Error,'scene changed'):backend.execute(plan,rows.append,lambda:5.)
            self.assertEqual(backend.tick,0);self.assertEqual(written,[])
            self.assertEqual(rows[0]['kind'],'planning_gate_rejection')
            self.assertEqual(rows[0]['control_call_count'],0);self.assertEqual(rows[0]['physics_step_count'],0)
            self.assertTrue(rows[0]['collisions'])

    def test_atomic_dual_arm_submission_native_fingers_and_hold(self):
        with tempfile.TemporaryDirectory() as temp:
            backend, adapter, states, written, qr = fake_backend(Path(temp))
            backend.reset(dict(episode_id="mock"))
            target = qr + .005
            plan = adapter.motion.plan(qr, np.full((2, 2), .02),
                                       [(target, np.full((2, 2), .025))], lambda q,g: True, deadline(2))
            evidence = []
            done = backend.execute(plan, evidence.append, lambda: 5.)
            self.assertEqual(done, 1)
            self.assertEqual(len(evidence), len(plan.q) - 1)
            self.assertEqual(backend.env.take_action_cnt, 1)
            np.testing.assert_allclose(np.array([s["actual"][:6] for s in states]), target)
            self.assertAlmostEqual(backend.env.robot.right_gripper_val, .625)
            backend.hold()
            self.assertTrue(all(np.all(s["speed"] == 0) for s in states))

    def test_invalid_plan_or_stale_initial_state_cannot_partially_submit(self):
        with tempfile.TemporaryDirectory() as temp:
            backend, adapter, states, written, qr = fake_backend(Path(temp))
            backend.reset(dict(episode_id="mock"))
            plan = adapter.motion.plan(qr, np.full((2, 2), .02),
                                       [(qr, np.full((2, 2), .02))], lambda q,g: True, deadline(2))
            for bad in (replace(plan, q=plan.q[:, :1]), replace(plan, waypoint_ticks=(500,))):
                with self.assertRaises(ValueError):
                    backend.execute(bad, lambda payload: None, lambda: 5.)
                self.assertEqual(written, [])
            states[0]["actual"][0] += .1
            # Submit must remeasure even when no policy capture was requested.
            with self.assertRaises(G2Error):
                backend.execute(plan, lambda payload: None, lambda: 5.)
            self.assertEqual(written, [])

    def test_scoped_warmup_bypass_restores_class(self):
        class Robot:
            def set_planner(self, scene=None):
                raise RuntimeError("original planner called")
        original = Robot.set_planner
        with joint_space_setup(Robot):
            robot = Robot()
            robot.is_dual_arm = False
            robot.set_planner()
            self.assertEqual(robot.g2_planner_mode, "g2_joint_space_cpu_fcl_v1")
            with self.assertRaises(RuntimeError):
                robot.left_planner.plan_path()
        self.assertIs(Robot.set_planner, original)
