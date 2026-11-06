import unittest
from types import SimpleNamespace as NS
import numpy as np
from g2_fixtures import fixture, snapshot
from g2_capture import capture_snapshot


class CaptureTests(unittest.TestCase):
    def test_measured_positions_are_read_from_articulation(self):
        adapter, qa, qr, bases = fixture()
        snap = snapshot(adapter, qr, bases)
        def entity(row):
            return NS(get_active_joints=lambda: [NS(get_name=lambda name=name: name) for name in row.names],
                      get_qpos=lambda: row.qpos,
                      get_name=lambda: row.instance,
                      get_root_pose=lambda: NS(p=np.zeros(3), q=np.array([1., 0, 0, 0])))
        images = {name: {"rgb": np.full((8, 8, 3), 7, dtype=np.uint8)}
                  for name in ("head_camera", "left_camera", "right_camera")}
        env = NS(robot=NS(left_entity=entity(snap.arms[0]), right_entity=entity(snap.arms[1])),
                 get_obs=lambda: {"observation": images, "joint_action": {"vector": None}})
        captured, _ = capture_snapshot(env, snap.stamp, np.eye(4))
        np.testing.assert_array_equal(captured.arms[0].qpos, snap.arms[0].qpos)
        self.assertEqual(captured.images["cam_high"][1].shape, (3, 8, 8))
        self.assertEqual(captured.arms[0].names, snap.arms[0].names)
