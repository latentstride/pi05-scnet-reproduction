"""Admission boundaries for the opt-in online projection."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from g2_adapter import G2Error, Result, Stamp
from g2_remaining_coordinator import RecoveryCoordinator
from g2_task_tcp_projection import METHOD
from robotwin_control_geometry import Status


class ProjectionBoundaries(unittest.TestCase):
    def setUp(self):
        self.stamp = Stamp("test", 10, .04)
        self.snapshot = SimpleNamespace(stamp=self.stamp, objects_task={"block": 1, "target": 1})
        self.tracker = SimpleNamespace(phase="approach", holders={}, releasing=set(),
                                       contact_fingers={"left": [], "right": []})
        world = SimpleNamespace(tracker=self.tracker)
        self.adapter = SimpleNamespace(actions=Mock(), observe=Mock(), target_collision=world)
        self.backend = Mock()
        self.coordinator = RecoveryCoordinator(
            self.adapter, self.backend, approach_repair=True, task_tcp_projection=METHOD
        )
        self.chunk = object()
        self.log = Mock()

    def plan(self):
        return self.coordinator.action_plan(self.chunk, self.snapshot, 4, lambda: 30., self.log)

    def reject(self, **fields):
        self.adapter.actions.return_value = Result(
            Status.COLLISION, "original rejected", dict(
                collision_world="target_collision", requests=[{"row": 0}], **fields
            )
        )

    def test_original_prefix_passes_through(self):
        original = object()
        self.adapter.actions.return_value = Result(Status.OK, "", {}, original)
        with patch("g2_bounded_approach.checked_prefix", return_value=(original, None)), \
             patch("g2_task_tcp_projection.plan_projection") as projection:
            self.assertEqual(self.plan(), (original, {"repaired": False}))
        projection.assert_not_called()

    def test_contact_and_post_hold_do_not_project(self):
        self.reject()
        with patch("g2_task_tcp_projection.plan_projection") as projection:
            for phase, fingers in (("approach", ["left-finger"]), ("free_after_hold", [])):
                with self.subTest(phase=phase, fingers=fingers):
                    self.tracker.phase = phase
                    self.tracker.contact_fingers["left"] = fingers
                    with self.assertRaises(G2Error) as caught:
                        self.plan()
                    self.assertEqual(caught.exception.diagnostic["reason"], "recovery_unsafe_contact")
        projection.assert_not_called()

    def test_measured_geometry_rejection_does_not_project(self):
        self.reject(cause={"geometry": "actual"})
        with patch("g2_task_tcp_projection.plan_projection") as projection:
            with self.assertRaisesRegex(G2Error, "original rejected"):
                self.plan()
        projection.assert_not_called()

    def test_followup_candidate_failure_does_not_project_again(self):
        self.coordinator.projected_from = self.stamp
        self.reject()
        with patch("g2_task_tcp_projection.plan_projection") as projection:
            with self.assertRaises(G2Error) as caught:
                self.plan()
        self.assertEqual(caught.exception.diagnostic["projection_stop_reason"], "fresh_action_rejected")
        projection.assert_not_called()

    def test_stale_capture_does_not_query_or_move(self):
        self.coordinator.projected_from = self.stamp
        self.backend.capture.return_value = self.snapshot, None
        policy = Mock()
        with self.assertRaises(G2Error) as caught:
            self.coordinator.cycle(policy, "handover", 4, lambda: 30., self.log)
        self.assertEqual(caught.exception.status, Status.UNSYNCED)
        self.adapter.observe.assert_not_called()
        policy.infer_raw.assert_not_called()
        self.backend.execute.assert_not_called()

    def test_source_remapping_failure_after_projection_stops_before_query(self):
        self.coordinator.projected_from = self.stamp
        self.snapshot.stamp = Stamp("test", 15, .06)
        self.backend.capture.return_value = self.snapshot, None
        self.adapter.observe.return_value = Result(
            Status.COLLISION, "source proxy blocked", {"collision_world": "source_collision"}
        )
        policy = Mock()
        with self.assertRaises(G2Error) as caught:
            self.coordinator.cycle(policy, "handover", 4, lambda: 30., self.log)
        self.assertEqual(caught.exception.diagnostic["projection_stop_reason"], "fresh_source_mapping_rejected")
        policy.infer_raw.assert_not_called()
        self.backend.execute.assert_not_called()
