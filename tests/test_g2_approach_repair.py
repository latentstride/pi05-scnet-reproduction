"""Forward source admission and repeated obstruction handling without safety relaxation."""

from types import SimpleNamespace as NS
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
from g2_adapter import G2Error, Result
from g2_bounded_approach import checked_prefix, source_obstruction
from g2_motion import Trajectory
from g2_remaining_coordinator import RecoveryCoordinator
from robotwin_control_geometry import Status
import test_g2_remaining as remaining_tests
from test_g2_native_backend import native_fixture


def diagnostic(depth=0.000217):
    return dict(
        collision_world="source_collision",
        collision_examples=[
            dict(
                collisions=[
                    dict(
                        collision_type="self",
                        link_name1="fl_link4",
                        link_name2="fl_link6",
                        max_penetration_m=depth,
                    )
                ]
            )
        ],
    )


class ApproachRepairTests(unittest.TestCase):
    def test_canonical_obstruction_uses_pairs_and_numeric_depth(self):
        self.assertEqual(
            source_obstruction(diagnostic()), ((("fl_link4", "fl_link6"),), 0.000217)
        )
        self.assertIsNone(
            source_obstruction(dict(detail="fl_link4/fl_link6 collision"))
        )

    def test_same_worse_source_obstruction_stops_before_another_recovery(self):
        adapter, backend, _ = remaining_tests.CoordinatorTests().setup_pair()
        runner = RecoveryCoordinator(adapter, backend, approach_repair=True)
        fingerprint, _ = source_obstruction(diagnostic())
        runner.source_obstructions[fingerprint] = dict(
            penetration_m=0.000217, attempt=1, policy_actions=140
        )
        runner.recoveries = 1
        adapter.observe = lambda *args: Result(
            Status.COLLISION, "blocked", diagnostic(0.000224), None
        )
        snap, _ = backend.capture()
        with self.assertRaises(G2Error) as caught:
            runner.mapped_observation(snap, "test", lambda: 5, lambda row: None)
        self.assertEqual(
            caught.exception.diagnostic["reason"], "repeated_source_obstruction"
        )
        self.assertEqual(caught.exception.diagnostic["geometry"], "candidate")
        self.assertEqual(backend.actions, 0)
        self.assertEqual(runner.max_recoveries, 4)
        self.assertEqual(runner.recoveries, 1)

    def test_actual_fault_has_precedence_over_exhausted_or_repeated_recovery(self):
        adapter, backend, _ = remaining_tests.CoordinatorTests().setup_pair()
        runner = RecoveryCoordinator(adapter, backend, approach_repair=True)
        runner.recoveries = 4
        adapter.observe = lambda *args: Result(
            Status.COLLISION,
            "actual fault",
            dict(diagnostic(), geometry="actual"),
            None,
        )
        snap, _ = backend.capture()
        with self.assertRaises(G2Error) as caught:
            runner.mapped_observation(snap, "test", lambda: 5, lambda row: None)
        self.assertEqual(str(caught.exception), "actual fault")
        self.assertEqual(caught.exception.diagnostic["geometry"], "actual")
        self.assertEqual(backend.actions, 0)

    def test_actual_action_fault_precedes_bounded_approach_and_exhausted_cap(self):
        adapter, backend, _ = remaining_tests.CoordinatorTests().setup_pair()
        runner = RecoveryCoordinator(adapter, backend, approach_repair=True)
        runner.recoveries = 4
        adapter.actions = lambda *args: Result(
            Status.COLLISION,
            "unsafe measured start",
            dict(collision_world="target_collision", geometry="actual"),
            None,
        )
        snap, _ = backend.capture()
        with self.assertRaises(G2Error) as caught:
            runner.action_plan(np.zeros((50, 14)), snap, 4, lambda: 5, lambda row: None)
        self.assertEqual(caught.exception.diagnostic["geometry"], "actual")
        self.assertEqual(str(caught.exception), "unsafe measured start")
        self.assertEqual(backend.actions, 0)

    def test_new_obstruction_metric_does_not_ignore_improvement(self):
        adapter, backend, _ = remaining_tests.CoordinatorTests().setup_pair()
        runner = RecoveryCoordinator(adapter, backend, approach_repair=True)
        fingerprint, _ = source_obstruction(diagnostic())
        runner.source_obstructions[fingerprint] = dict(
            penetration_m=0.0003, attempt=1, policy_actions=140
        )
        runner.recoveries = 1
        real = adapter.observe
        calls = []

        def observe(*args):
            calls.append(True)
            return (
                Result(Status.COLLISION, "blocked", diagnostic(0.0002), None)
                if len(calls) == 1
                else real(*args)
            )

        adapter.observe = observe
        snap, _ = backend.capture()
        with patch(
            "g2_remaining_coordinator.refined_pair",
            return_value=(backend.q.copy(), [], {}),
        ):
            fresh, request = runner.mapped_observation(
                snap, "test", lambda: 5, lambda row: None
            )
        self.assertGreater(fresh.stamp.tick, snap.stamp.tick)
        self.assertEqual(runner.recoveries, 2)
        self.assertEqual(
            runner.source_obstructions[fingerprint]["penetration_m"], 0.0002
        )

    def test_forward_barrier_retains_original_safe_prefix_and_no_fake_seed(self):
        adapter, backend, _ = remaining_tests.CoordinatorTests().setup_pair()
        snap, _ = backend.capture()
        tracker = NS(
            holders={}, releasing=set(), contact_fingers={"left": set(), "right": set()}
        )
        adapter.target_collision = NS(
            tracker=tracker, action_barrier=lambda *args: False
        )
        q = np.repeat(backend.q[None], 7, axis=0)
        g = np.repeat(
            np.array([a.grip.target(0.5) for a in adapter.arx])[None], 7, axis=0
        )
        plan = Trajectory(0.02, q, np.zeros_like(q), g, (2, 4, 6), g.copy(), "test")
        seed = adapter.seed.copy()
        counter = []

        def check(*args):
            counter.append(True)
            if len(counter) == 2:
                raise G2Error(Status.COLLISION, "source blocked", **diagnostic())
            return {"accepted": True}

        with patch("g2_bounded_approach.source_admissible", side_effect=check):
            prefix, evidence = checked_prefix(adapter, plan, snap, lambda: 1)
        self.assertEqual(prefix.waypoint_ticks, (2,))
        self.assertEqual(len(prefix.q), 3)
        np.testing.assert_array_equal(prefix.q, plan.q[:3])
        np.testing.assert_array_equal(adapter.seed, seed)
        self.assertEqual(evidence["row"], 1)
        self.assertEqual(backend.actions, 0)
        tracker.holders = {"left": np.eye(4)}
        with patch("g2_bounded_approach.source_admissible") as shadow:
            unchanged, evidence = checked_prefix(adapter, plan, snap, lambda: 1)
        self.assertIs(unchanged, plan)
        self.assertIsNone(evidence)
        shadow.assert_not_called()

    def test_native_projection_preserves_closing_and_holding_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            backend, state, _, _ = native_fixture(Path(temp))
            backend.episode = "test"
            tracker = NS(
                holders={"left": np.eye(4)},
                releasing=set(),
                contact_fingers={"left": set(), "right": set()},
            )
            backend.guard = NS(collision=NS(tracker=tracker))
            row = np.zeros(14)
            row[[6, 13]] = 1.0
            with patch("g2_bounded_approach.upward_endpoint") as correction:
                np.testing.assert_array_equal(
                    backend._approach_row(row, lambda: 1, lambda row: None), row
                )
            correction.assert_not_called()
            tracker.holders = {}
            backend.env.robot.left_gripper_scale = (
                backend.env.robot.right_gripper_scale
            ) = [-0.01, 0.045]
            row[6] = 0.0
            with patch("g2_bounded_approach.upward_endpoint") as correction:
                np.testing.assert_array_equal(
                    backend._approach_row(row, lambda: 1, lambda row: None), row
                )
            correction.assert_not_called()


if __name__ == "__main__":
    unittest.main()
