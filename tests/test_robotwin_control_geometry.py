"""Run from the repository root with python -m unittest discover -s tests -v."""
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.robotwin_control_geometry import (
    Status, check_joint_segment, gripper_native, inverse_pose, pose_from_wxyz,
    select_joints, split_aloha_action, tool_offsets,
)

DATA = Path(__file__).resolve().parents[1] / "research/robotwin"


class GeometryTests(unittest.TestCase):
    def test_wxyz_and_sign(self):
        q = np.sqrt(0.5) * np.array([1, 0, 0, 1])
        pose = pose_from_wxyz([2, 3, 4], q)
        np.testing.assert_allclose(pose @ [1, 0, 0, 1], [2, 4, 4, 1], atol=1e-12)
        np.testing.assert_allclose(pose_from_wxyz([2, 3, 4], -2*q), pose)
        for invalid in ([0, 0, 0, 0], [np.nan, 0, 0, 1], [1, 0, 0]):
            with self.subTest(q=invalid), self.assertRaises(ValueError):
                pose_from_wxyz([0, 0, 0], invalid)

    def test_composition_and_inverse(self):
        world_base = pose_from_wxyz([2, 3, 4], [1, 0, 0, 1])
        base_child = pose_from_wxyz([1, 0, 0], [0, 1, 0, 0])
        world_child = world_base @ base_child
        np.testing.assert_allclose(world_child @ [0, 1, 0, 1], [3, 4, 4, 1], atol=1e-12)
        np.testing.assert_allclose(inverse_pose(world_base) @ world_child, base_child, atol=1e-12)
        np.testing.assert_allclose(world_child @ inverse_pose(world_child), np.eye(4), atol=1e-12)
        for invalid in (np.diag([-1, 1, 1, 1]), np.diag([2, 1, 1, 1]), np.zeros((4, 4))):
            with self.assertRaises(ValueError):
                inverse_pose(invalid)

    def test_tool_offset_order_and_sign(self):
        # G and D do not commute. A bias below 0.12 puts EE behind J along tool x.
        g = pose_from_wxyz([0, 0, 0], [1, 0, 0, 1])[:3, :3]
        d = pose_from_wxyz([0, 0, 0], [1, 1, 0, 0])[:3, :3]
        ee, tcp = tool_offsets(g, d, 0.10)
        np.testing.assert_allclose(ee[:3, 3], [0, -0.02, 0], atol=1e-12)
        np.testing.assert_allclose(tcp @ [0, 1, 0, 1], [0, 0.1, 1, 1], atol=1e-12)
        np.testing.assert_allclose((inverse_pose(ee) @ tcp)[:3, 3], [0.12, 0, 0], atol=1e-12)

    def test_transfer_preserves_task_tool_pose(self):
        source_world_task = pose_from_wxyz([0.2, 0.1, 0], [1, 0, 0, 1])
        source_world_tcp = pose_from_wxyz([0.3, 0.4, 0.5], [1, 0, 1, 0])
        source_tcp_tool = pose_from_wxyz([0.01, 0, 0], [1, 1, 0, 0])
        target_world_task = pose_from_wxyz([0.6, 0.1, 0], [1, 0, 0, -1])
        target_tcp_tool = pose_from_wxyz([0.02, 0, 0], [1, 0, 0, 0])
        task_tool = inverse_pose(source_world_task) @ source_world_tcp @ source_tcp_tool
        target_world_tcp = target_world_task @ task_tool @ inverse_pose(target_tcp_tool)
        np.testing.assert_allclose(
            target_world_tcp[:3, 3], [0.5, -0.2, 0.51], atol=1e-12
        )
        np.testing.assert_allclose(
            inverse_pose(target_world_task) @ target_world_tcp @ target_tcp_tool,
            task_tool, atol=1e-12
        )

    def test_joint_order_and_action_layout(self):
        action = np.arange(14, dtype=float)
        action[[6, 13]] = [0.2, 0.8]
        left, right = split_aloha_action(action)
        np.testing.assert_array_equal(left[0], np.arange(6))
        np.testing.assert_array_equal(right[0], np.arange(7, 13))
        self.assertEqual((left[1], right[1]), (0.2, 0.8))
        np.testing.assert_array_equal(select_joints(["wheel", "j2", "j1"], [99, 2, 1],
                                                   ["j1", "j2"]), [1, 2])
        for names in (["j1", "j1"], ["j1", "j3"]):
            with self.assertRaises(ValueError):
                select_joints(names, [1, 2], ["j1", "j2"])
        for invalid in (action[:-1], action.reshape(1, 14), np.full(14, np.nan),
                        np.full(14, 1.01)):
            with self.assertRaises(ValueError):
                split_aloha_action(invalid)

    def test_gripper_scale_and_range(self):
        for g, expected in [(0, -0.01), (0.5, 0.0175), (1, 0.045)]:
            self.assertAlmostEqual(gripper_native(g), expected)
        for invalid in (-0.001, 1.001, np.nan, [0.5]):
            with self.assertRaises(ValueError):
                gripper_native(invalid)

    def test_segment_status(self):
        for start, target, duration, status in [
            ([0, 0], [1, -0.5], 1, Status.OK),
            ([0, 0], [1, 0], 0.5, Status.VELOCITY),
            ([0, 0], [0, 1.1], None, Status.LIMIT),
            ([-1.1, 0], [0, 0], 1, Status.LIMIT),
            ([0, 0], [0, 1], None, Status.TIMING_UNKNOWN),
            ([0, 0], [0, 1], 0, Status.INVALID),
            ([0, 0], [0, 1], np.nan, Status.INVALID),
            ([0, 0], [0, np.inf], 1, Status.INVALID),
            ([0, 0], [0], 1, Status.INVALID),
        ]:
            with self.subTest(start=start, target=target, duration=duration):
                result = check_joint_segment(start, target, [[-1, 1]]*2, [1, 1], duration)
                self.assertEqual(result.status, status)
                self.assertEqual(result.ok, status == Status.OK)
        self.assertEqual(check_joint_segment([0], [0], [[-1, 1]], [0], 1).status, Status.INVALID)


class SapienRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = DATA / "env-build/evidence-20261102/sapien_fk_smoke.json"
        cls.evidence = json.loads(path.read_text())
        cls.config = {c["robot"]: c for c in
                      json.loads((DATA / "embodiment_config_audit.json").read_text())["robots"]}

    def assert_pose_close(self, actual, expected):
        expected = np.asarray(expected)
        error = np.linalg.norm(actual[:3, 3] - expected[:3, 3])
        trace = np.trace(actual[:3, :3].T @ expected[:3, :3])
        angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
        self.assertLessEqual(error, 0.001)
        self.assertLessEqual(angle, 0.1)

    def test_same_actual_qpos_fk_and_tools(self):
        self.assertEqual(self.evidence["status"], "passed")
        self.assertEqual(len(self.evidence["snapshots"]), 9)
        count = 0
        for row in self.evidence["snapshots"]:
            cfg = self.config[row["robot"]]
            ee, tcp = tool_offsets(cfg["global_trans_matrix"], cfg["delta_matrix"], cfg["gripper_bias"])
            base_xyz = [0, -0.65, 0] if cfg["dual_arm"] else [
                -0.3 if row["instance"] == 0 else 0.3, -0.35, 0.784]
            base = pose_from_wxyz(base_xyz, [0.707, 0, 0, 0.707])
            self.assert_pose_close(base, row["root_pose_world"])
            for arm in row["arms"]:
                with self.subTest(robot=row["robot"], side=arm["side"], case=row["case"]):
                    count += 1
                    names = cfg["joint_names"][["left", "right"].index(arm["side"])]
                    q = select_joints(row["joint_order"], row["actual_qpos"], names)
                    drive = select_joints(row["joint_order"], row["drive_targets"], names)
                    self.assertGreater(np.max(np.abs(q - drive)), 0.01)
                    joint = next(j for j in row["joints"] if j["name"] == arm["ee_joint"])
                    self.assertEqual(joint["child_link"], arm["ee_child_link"])
                    # Archived independent URDF FK was evaluated at this actual qpos.
                    base_child = inverse_pose(row["root_pose_world"]) @ arm["urdf_child_fk_world"]
                    world_child = base @ base_child
                    world_joint = world_child @ np.asarray(joint["pose_in_child"])
                    self.assert_pose_close(world_child, arm["sapien_child_link_world"])
                    self.assert_pose_close(world_joint, joint["joint_global_world"])
                    self.assert_pose_close(world_joint @ ee, arm["robotwin_ee_world"])
                    world_tcp = world_joint @ tcp
                    self.assert_pose_close(world_tcp, arm["robotwin_tcp_world"])
                    self.assert_pose_close(world_tcp @ inverse_pose(tcp) @ inverse_pose(
                        joint["pose_in_child"]), world_child)
        self.assertEqual(count, 12)

    def test_configured_gripper_endpoints_vs_urdf(self):
        # Per-finger bounds from the URDFs pinned in embodiment_config_audit.json.
        for robot, upper, open_status in [
            ("aloha-agilex", 0.04765, Status.OK),
            ("ARX-X5", 0.044, Status.LIMIT),
        ]:
            with self.subTest(robot=robot):
                bounds = [[0, upper]]
                self.assertEqual(check_joint_segment([0], [gripper_native(0)], bounds, [1], 1).status,
                                 Status.LIMIT)
                self.assertEqual(check_joint_segment([0], [gripper_native(1)], bounds, [1], 1).status,
                                 open_status)
