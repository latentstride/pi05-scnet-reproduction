from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import unittest

import numpy as np

from g2_fixtures import fixture, robot_arms, snapshot, ROOT, ASSETS, AUDIT
from g2_adapter import Grip, Status, deadline
from g2_kinematics import ArmChain, pose_error, rotation_vector
from robotwin_control_geometry import inverse_pose, select_joints


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter, self.qa, self.qr, self.bases = fixture()
        self.snap = snapshot(self.adapter, self.qr, self.bases)

    def chunk(self, grip=.5):
        return np.tile(np.column_stack((self.qa, [grip, grip])).ravel(), (50, 1))

    def test_observe_and_actions_roundtrip_with_real_urdfs(self):
        result = self.adapter.observe(self.snap, "handover", 3)
        self.assertTrue(result.ok, (result.detail, result.diagnostic))
        np.testing.assert_allclose(result.payload["state"].reshape(2, 7)[:, :6], self.qa, atol=1e-6)
        self.assertGreater(np.linalg.norm(self.qa - self.qr), .1)
        result = self.adapter.actions(self.chunk(), self.snap, 4, 3)
        self.assertTrue(result.ok, (result.detail, result.diagnostic))
        np.testing.assert_allclose(result.payload.q[-1], self.qr, atol=1e-6)
        self.assertEqual(result.payload.waypoint_ticks, (1, 2, 3, 4))

    def test_task_frame_change_preserves_mapping(self):
        from robotwin_control_geometry import pose_from_wxyz
        task = pose_from_wxyz([.4, -.3, .2], [.9, .1, -.2, .3])
        snap = replace(self.snap, world_task=task,
                       arms=tuple(replace(arm, world_base=task @ arm.world_base) for arm in self.snap.arms))
        result = self.adapter.observe(snap, "x", 2)
        self.assertTrue(result.ok, result.detail)
        np.testing.assert_allclose(result.payload["state"].reshape(2, 7)[:, :6], self.qa, atol=1e-6)

    def test_stale_image_is_rejected(self):
        image = dict(self.snap.images)
        stamp, rgb = image["cam_high"]
        image["cam_high"] = (replace(stamp, tick=1), rgb)
        self.assertEqual(self.adapter.observe(replace(self.snap, images=image), "x", 2).status, Status.UNSYNCED)

    def test_known_pose_ik_with_perturbed_seed(self):
        for robot in ("aloha-agilex", "ARX-X5"):
            for arm in robot_arms(robot):
                q = np.array([.2, .3, -.4, .2, .3, -.2])
                goal = arm.chain.fk(q)
                sols = list(arm.chain.ik(goal, q + .03, deadline(4)))
                self.assertTrue(sols)
                residual = pose_error(goal, arm.chain.fk(sols[0][0]))
                self.assertLess(np.linalg.norm(residual[:3]), .0005)
                self.assertLess(np.linalg.norm(residual[3:]), np.deg2rad(.05))

    def test_fk_matches_independent_sapien_archive(self):
        data = json.loads((ROOT / "research/robotwin/env-build/evidence-20261102/sapien_fk_smoke.json").read_text())
        count = 0
        for frame in data["snapshots"]:
            arms = robot_arms(frame["robot"])
            for item in frame["arms"]:
                side = ("left", "right").index(item["side"])
                arm = arms[side]
                q = select_joints(frame["joint_order"], frame["actual_qpos"], arm.chain.names)
                pose = np.array(frame["root_pose_world"]) @ arm.chain.fk(q)
                target = np.array(item["sapien_child_link_world"])
                self.assertLess(np.linalg.norm(pose[:3, 3] - target[:3, 3]), .001)
                self.assertLess(np.linalg.norm(rotation_vector(target[:3, :3] @ pose[:3, :3].T)), np.deg2rad(.1))
                count += 1
        self.assertEqual(count, 12)

    def test_analytic_jacobian(self):
        for arm in self.adapter.aloha + self.adapter.arx:
            q = self.qa[0].copy()
            pose, jac = arm.chain.fk(q, True)
            for k in range(6):
                new = q.copy()
                new[k] += 1e-6
                fd = pose_error(arm.chain.fk(new), pose) / 1e-6
                np.testing.assert_allclose(fd, jac[:, k], atol=1e-6)

    def test_atomic_failures_and_no_seed_advance(self):
        seed = self.adapter.seed.copy()
        variants = [
            (replace(self.snap, images={}), Status.INVALID),
            (replace(self.snap, arms=(replace(self.snap.arms[0], instance="wrong"), self.snap.arms[1])), Status.INVALID),
            (replace(self.snap, arms=(replace(self.snap.arms[0], stamp=replace(self.snap.stamp, tick=1)), self.snap.arms[1])), Status.UNSYNCED),
        ]
        for value, status in variants:
            result = self.adapter.observe(value, "handover", 2)
            self.assertEqual(result.status, status)
            self.assertIsNone(result.payload)
            np.testing.assert_array_equal(self.adapter.seed, seed)
        self.adapter.aloha = (replace(self.adapter.aloha[0], calibration_id=""), self.adapter.aloha[1])
        self.assertEqual(self.adapter.observe(self.snap, "handover", 2).status, Status.UNCALIBRATED)

    def test_duplicate_joint_and_mimic_mismatch(self):
        arm = self.snap.arms[0]
        duplicate = replace(arm, names=(arm.names[0],) * len(arm.names))
        self.assertEqual(self.adapter.observe(replace(self.snap, arms=(duplicate, self.snap.arms[1])), "x", 2).status, Status.INVALID)
        q = arm.qpos.copy()
        q[0] += .01
        self.assertEqual(self.adapter.observe(replace(self.snap, arms=(replace(arm, qpos=q), self.snap.arms[1])), "x", 2).status, Status.INVALID)

    def test_stale_and_discontinuous_states(self):
        self.assertTrue(self.adapter.observe(self.snap, "x", 2).ok)
        self.assertEqual(self.adapter.observe(self.snap, "x", 2).status, Status.UNSYNCED)
        self.adapter.seed += .05
        self.adapter.source_speed[:] = 1e-8
        result = self.adapter.observe(snapshot(self.adapter, self.qr, self.bases, 1), "x", 5)
        self.assertEqual(result.status, Status.VELOCITY, result.detail)

    def test_gripper_audit_and_illegal_chunk(self):
        self.assertTrue(self.adapter.observe(self.snap, "x", 2).ok)
        result = self.adapter.actions(self.chunk(1.02), self.snap, 1, 3)
        self.assertTrue(result.ok, result.detail)
        self.assertEqual(result.diagnostic["grippers"][0]["raw"], 1.02)
        self.assertEqual(result.diagnostic["grippers"][0]["normalized"], 1.)
        result = self.adapter.actions(self.chunk(1.1), self.snap, 1, 3)
        self.assertEqual(result.status, Status.LIMIT)
        self.assertIsNone(result.payload)
        self.assertIsNone(result.diagnostic["grippers"][0]["normalized"])
        self.assertEqual(self.adapter.actions(self.chunk()[:4], self.snap, 1, 2).status, Status.INVALID)
        bad = self.chunk()
        bad[0, 0] = np.nan
        self.assertEqual(self.adapter.actions(bad, self.snap, 1, 2).status, Status.INVALID)

    def test_no_solution_timeout_and_collision(self):
        unreachable = self.snap.arms[0].world_base.copy()
        unreachable[0, 3] += 10
        bad = replace(self.snap, arms=(replace(self.snap.arms[0], world_base=unreachable), self.snap.arms[1]))
        self.assertEqual(self.adapter.observe(bad, "x", 5).status, Status.IK_FAILED)
        self.assertEqual(self.adapter.observe(self.snap, "x", 1e-12).status, Status.TIMEOUT)
        self.adapter.source_collision = lambda q, g: False
        self.assertEqual(self.adapter.observe(self.snap, "x", 5).status, Status.COLLISION)

    def test_limits_and_calibration_rejected(self):
        with self.assertRaises(ValueError):
            Grip(("a", "b"), np.zeros(2), np.ones(2), np.array([[0, .044]] * 2), "bad")
        row = json.loads(AUDIT.read_text())["robots"][0]
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ArmChain(ASSETS / row["urdf_member"], "0" * 64, row["joint_names"][0], row["ee_joints"][0])

    def test_retiming_and_midpath_collision(self):
        motion = self.adapter.motion
        goal = self.qr + .05
        target = [(goal, np.full((2, 2), .03)), (self.qr, np.full((2, 2), .02))]
        plan = motion.plan(self.qr, np.full((2, 2), .02), target, lambda q, g: True, deadline(2))
        velocity = np.diff(plan.q, axis=0) / plan.period_s
        accel = np.diff(velocity, axis=0) / plan.period_s
        self.assertLessEqual(np.max(abs(velocity)), .5)
        self.assertLessEqual(np.max(abs(accel)), 1. + 1e-9)
        np.testing.assert_allclose(plan.q[plan.waypoint_ticks[0]], goal)
        calls = []
        def collision(q, g):
            calls.append(q.copy())
            return not (.02 < q[0, 0] - self.qr[0, 0] < .03)
        from g2_adapter import G2Error
        with self.assertRaises(G2Error) as ctx:
            motion.plan(self.qr, np.full((2, 2), .02), target, collision, deadline(2))
        self.assertEqual(ctx.exception.status, Status.COLLISION)
        self.assertGreater(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
