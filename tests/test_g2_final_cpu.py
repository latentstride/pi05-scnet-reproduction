"""Evidence-based native release semantics and unified ARX regression."""

import unittest
from types import SimpleNamespace as NS
import numpy as np
from g2_adapter import G2Error, Stamp
from g2e_native_guard import NativeGuard, NativeReleaseTracker
from test_g2_contact import contacts
from g2_fixtures import fixture


class ReleaseIntentTests(unittest.TestCase):
    def setUp(self):
        self.arms = fixture()[0].arx
        self.tracker = NativeReleaseTracker(99, {"left": [1, 2], "right": [3, 4]})
        self.tools = np.tile(np.eye(4), (2, 1, 1))
        for tick in range(3):
            self.tracker.observe(
                Stamp("release", tick, tick * 0.004),
                contacts([1, 2]),
                self.tools,
                np.eye(4),
            )

    def test_historical_negative_preload_microincrease_does_not_release(self):
        before = np.full((2, 2), -0.01)
        after = before.copy()
        after[0] = -0.01 + 0.055 * 0.0006430061334371567
        anchor = self.tracker.holders["left"].copy()
        self.tracker.authorize_opening(before, after, self.arms)
        self.assertFalse(self.tracker.releasing)
        np.testing.assert_array_equal(self.tracker.holders["left"], anchor)

    def test_preload_entire_closed_domain_needs_no_arbitrary_deadband(self):
        before = np.full((2, 2), -0.01)
        for value in np.linspace(-0.01, 0, 100):
            after = before.copy()
            after[0] = value
            self.tracker.authorize_opening(before, after, self.arms)
            before = after
        self.assertFalse(self.tracker.releasing)
        after = before.copy()
        after[0] = 0.000001
        with self.assertRaisesRegex(G2Error, "donor opening"):
            self.tracker.authorize_opening(before, after, self.arms)

    def test_tiny_positive_opening_accumulates_across_interpolation_steps(self):
        before = np.zeros((2, 2))
        for value in (0.4e-9, 0.8e-9):
            after = before.copy()
            after[0] = value
            self.tracker.authorize_opening(before, after, self.arms)
            before = after
        after = before.copy()
        after[0] = 1.2e-9
        with self.assertRaisesRegex(G2Error, "donor opening"):
            self.tracker.authorize_opening(before, after, self.arms)

    def test_reinitialized_previous_drive_cannot_reset_release_reference(self):
        before = np.zeros((2, 2))
        self.tracker.authorize_opening(before, before, self.arms)
        before[0] = 0.0005
        after = before.copy()
        after[0] += 0.4e-9
        with self.assertRaisesRegex(G2Error, "donor opening"):
            self.tracker.guard_opening(before, after, self.arms)

    def test_real_early_release_still_blocked(self):
        with self.assertRaisesRegex(G2Error, "donor opening"):
            self.tracker.authorize_opening(
                np.zeros((2, 2)), np.full((2, 2), 0.04), self.arms
            )
        self.assertFalse(self.tracker.releasing)

    def test_confirmed_receiver_allows_release_and_contact_loss(self):
        for tick in range(3, 6):
            self.tracker.observe(
                Stamp("release", tick, tick * 0.004),
                contacts([1, 2, 3, 4]),
                self.tools,
                np.eye(4),
            )
        after = np.zeros((2, 2))
        after[0] = 0.04
        self.tracker.authorize_opening(np.zeros((2, 2)), after, self.arms)
        self.tracker.observe(
            Stamp("release", 6, 0.024), contacts([3, 4]), self.tools, np.eye(4)
        )
        self.assertEqual(set(self.tracker.holders), {"right"})

    def test_preload_jitter_cannot_hide_actual_contact_loss(self):
        self.tracker.authorize_opening(
            np.full((2, 2), -0.01), np.full((2, 2), -0.0099), self.arms
        )
        with self.assertRaisesRegex(G2Error, "lost bilateral contact"):
            self.tracker.observe(
                Stamp("release", 3, 0.012), contacts([1]), self.tools, np.eye(4)
            )

    def test_native_guard_lost_drive_history_fails_closed_during_hold(self):
        collision = NS(tracker=self.tracker)
        guard = NativeGuard(collision, self.arms, 0.02, 0.03, 10.0)
        with self.assertRaisesRegex(G2Error, "release history missing") as caught:
            guard.candidate(
                np.zeros((2, 6)),
                np.full((2, 2), 0.03),
                np.zeros((2, 6)),
                np.zeros((2, 2)),
                lambda: 1,
            )
        self.assertIn("drive_arm", caught.exception.diagnostic)
        self.assertIsNone(guard.previous_drive)


class CampaignTests(unittest.TestCase):
    def rehearse(self, *, admission=True, timeout=False, model_exit=False):
        import sys
        import tempfile
        from pathlib import Path
        from g2_final_campaign import supervise
        from g2_remaining_plan import build

        plan = build(Path(__file__).resolve().parents[1])
        for row in plan["trials"]:
            row["hard_timeout_s"] = 0.25
        plan["budget"]["bounded_execution_s"] = (
            sum(plan["budget"]["overhead"].values()) + 3.5
        )
        rows = plan["trials"]
        first, second = (
            rows[0],
            next(row for row in rows if row["embodiment"] == "ARX-X5"),
        )
        evidence = dict.fromkeys(
            ("assets", "reset", "actual_geometry", "input_contract"), True
        )
        command = [
            sys.executable,
            "-c",
            "import time; time.sleep(2)" if timeout or model_exit else "pass",
        ]
        model_duration = 0.42 if model_exit else 3
        runtime = dict(
            mode="cpu_rehearsal",
            timeout_s=3,
            model_timeout_s=1,
            ready_marker="READY",
            model=[
                sys.executable,
                "-u",
                "-c",
                f"import time; print('READY'); time.sleep({model_duration})",
            ],
            commands={
                first["episode_id"]: command,
                second["episode_id"]: [sys.executable, "-c", "pass"],
            },
            admission={
                ("aloha-agilex", first["seed"]): evidence,
                ("ARX-X5", second["seed"]): evidence if admission else {},
            },
            read_result=lambda episode: dict(episode_id=episode, success=False),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PermissionError):
                supervise(plan, runtime, Path(tmp) / "forbidden")
            result = supervise(plan, runtime, Path(tmp) / "run", rehearsal=True)
        self.assertEqual(result["model_server_starts"], 1)
        return result, first["episode_id"], second["episode_id"]

    def test_native_task_failure_preserves_independent_arx_diagnostic(self):
        result, first, second = self.rehearse()
        self.assertEqual(result["rows"][first]["status"], "completed")
        self.assertEqual(result["rows"][second]["status"], "completed")
        self.assertFalse(result["transfer_claim_eligible"])

    def test_missing_arx_admission_is_not_run(self):
        result, _, second = self.rehearse(admission=False)
        self.assertEqual(result["rows"][second]["status"], "not_run")

    def test_timeout_retains_missing_trials_and_cleans_process_group(self):
        result, first, second = self.rehearse(timeout=True)
        self.assertEqual(result["stop_reason"], "trial hard timeout")
        self.assertEqual(result["rows"][first]["status"], "infrastructure_failure")
        self.assertEqual(result["rows"][second]["status"], "not_run")

    def test_model_exit_stops_campaign_without_restart(self):
        result, _, second = self.rehearse(model_exit=True)
        self.assertEqual(result["stop_reason"], "shared model exited")
        self.assertEqual(result["rows"][second]["status"], "not_run")


class FreeSpaceAdmissionTests(unittest.TestCase):
    def test_single_measured_finger_contact_blocks_recovery_and_rrt(self):
        from g2_remaining_coordinator import RecoveryCoordinator
        from g2_remaining_router import JointRouter
        from test_g2_remaining import motion

        live = NS(holders={}, releasing=set(), contact_fingers={"left": {33}})
        world = NS(tracker=live)
        coordinator = RecoveryCoordinator(NS(target_collision=world), None)
        with self.assertRaisesRegex(G2Error, "recovery forbidden"):
            coordinator._free()
        router = JointRouter(motion(), np.tile([-1.0, 1.0], (2, 6, 1)))
        q = np.zeros((2, 6))
        with self.assertRaisesRegex(G2Error, "rerouting forbidden"):
            router.route(q, q.copy(), np.zeros((2, 2)), world, lambda: 1)
