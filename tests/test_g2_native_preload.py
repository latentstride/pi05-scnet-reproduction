"""Candidate drive control, real contact tracker, and native hook regression."""

from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import numpy as np

from g2_adapter import G2Error, Stamp
from g2_contact import ContactLimits
from g2_fixtures import robot_arms
from g2_native_preload import NativePreloadHold, PRELOAD_PROTOCOL
from g2_task_runtime import validate_config
from g2e_native_guard import NativeReleaseTracker
from robotwin_control_geometry import Status
from test_g2_contact import contacts
from test_g2_native_backend import native_fixture


class PreloadTests(unittest.TestCase):
    def setUp(self):
        self.arms = robot_arms("aloha-agilex")
        self.tracker = NativeReleaseTracker(
            99, dict(left=[1, 2], right=[3, 4]), ContactLimits()
        )
        self.control = NativePreloadHold(self.tracker, self.arms)
        self.drive = np.full((2, 2), 0.022)
        self.tools = np.array([np.eye(4), np.eye(4)])

    def observe(self, tick, ids=(1, 2), block=None):
        self.tracker.observe(
            Stamp("test", tick, tick * 0.004),
            contacts(ids),
            self.tools,
            np.eye(4) if block is None else block,
        )
        self.control.observe(tick, self.drive)

    def acquire(self, ids=(1, 2)):
        for tick in range(1, 4):
            self.observe(tick, ids)

    def test_latch_only_after_existing_three_sample_acquisition_and_preserve_anchor(
        self,
    ):
        for tick in (1, 2):
            self.observe(tick)
            self.assertEqual(self.control.references, {})
        self.observe(3)
        anchor = self.tracker.holders["left"].copy()
        desired = self.drive.copy()
        desired[0] = -0.01
        applied, speed, log = self.control.select(
            3, self.drive, desired, np.ones((2, 2))
        )
        np.testing.assert_array_equal(applied, self.drive)
        np.testing.assert_array_equal(speed[0], [0, 0])
        np.testing.assert_array_equal(speed[1], [1, 1])
        np.testing.assert_array_equal(self.tracker.holders["left"], anchor)
        self.assertEqual(log["acquisition_ticks"], {"left": 3})
        self.assertEqual(log["modes"]["left"], "holding_preload")
        # A later lower requested drive cannot ratchet the stored preload.
        self.control.observe(3, self.drive - 0.001)
        np.testing.assert_array_equal(
            self.control.references["left"]["drive"], [0.022, 0.022]
        )

    def test_donor_opening_is_not_hidden_by_hold_and_release_waits_for_receiver(self):
        self.acquire()
        opened = self.drive.copy()
        opened[0] += 0.001
        with self.assertRaisesRegex(G2Error, "donor opening"):
            self.control.select(3, self.drive, opened, np.zeros((2, 2)))
        for tick in (4, 5, 6):
            self.observe(tick, (1, 2, 3, 4))
        applied, speed, log = self.control.select(
            6, self.drive, opened, np.ones((2, 2))
        )
        np.testing.assert_array_equal(applied[0], opened[0])
        self.assertEqual(log["modes"]["left"], "release_requested")
        self.tracker.authorize_opening(self.drive, applied, self.arms)
        self.drive = applied
        self.observe(7, (3, 4))
        self.assertEqual(set(self.tracker.holders), {"right"})
        self.assertNotIn("left", self.control.references)
        # Receiver release to support remains permitted by the existing rule.
        opened = self.drive.copy()
        opened[1] += 0.001
        applied, _, log = self.control.select(7, self.drive, opened, np.ones((2, 2)))
        self.tracker.authorize_opening(self.drive, applied, self.arms)
        self.drive = applied
        self.observe(8, ())
        self.assertFalse(self.control.references)
        self.assertFalse(self.tracker.holders)

    def test_slip_contact_loss_and_stale_state_still_stop(self):
        for kind in ("slip", "contact", "stale"):
            with self.subTest(kind=kind):
                self.setUp()
                self.acquire()
                if kind == "stale":
                    with self.assertRaises(G2Error):
                        self.control.select(4, self.drive, self.drive, np.zeros((2, 2)))
                elif kind == "contact":
                    with self.assertRaisesRegex(G2Error, "lost bilateral"):
                        self.observe(4, (1,))
                else:
                    pose = np.eye(4)
                    pose[0, 3] = 0.00501
                    with self.assertRaisesRegex(G2Error, "object slipped"):
                        self.observe(4, block=pose)
                self.assertEqual(self.tracker.limits, ContactLimits())

    def test_both_fingers_and_reacquisition_get_new_reference(self):
        self.acquire()
        self.tracker.releasing.add("left")
        self.observe(4, ())
        self.drive[0] = [0.019, 0.021]
        for tick in (5, 6, 7):
            self.observe(tick)
        np.testing.assert_array_equal(
            self.control.references["left"]["drive"], [0.019, 0.021]
        )
        self.assertEqual(self.control.references["left"]["tick"], 7)

    def test_negative_preload_relaxes_without_bypassing_physical_release_gate(self):
        self.drive[:] = -0.01
        self.acquire()
        requested = self.drive.copy()
        requested[0] = -0.0045
        applied, velocity, event = self.control.select(
            3, self.drive, requested, np.ones((2, 2))
        )
        np.testing.assert_array_equal(applied[0], requested[0])
        np.testing.assert_array_equal(velocity[0], [1.0, 1.0])
        self.assertEqual(event["modes"]["left"], "relaxing_preload")
        self.tracker.authorize_opening(self.drive, applied, self.arms)
        self.assertFalse(self.tracker.releasing)
        self.drive = applied
        self.observe(4)
        opened = requested.copy()
        opened[0] = 0.001
        with self.assertRaisesRegex(G2Error, "donor opening"):
            self.control.select(4, self.drive, opened, np.ones((2, 2)))
        for tick in (5, 6, 7):
            self.observe(tick, (1, 2, 3, 4))
        applied, _, event = self.control.select(7, self.drive, opened, np.ones((2, 2)))
        np.testing.assert_array_equal(applied[0], opened[0])
        self.assertEqual(event["modes"]["left"], "release_requested")

    def test_candidate_module_is_in_deployable_runtime_list(self):
        from g2_final_prepare_review import RUNTIME_MODULES

        self.assertIn("g2_native_preload", RUNTIME_MODULES)
        self.assertTrue(
            (
                Path(__file__).resolve().parents[1] / "scripts/g2_native_preload.py"
            ).is_file()
        )

    def test_configuration_is_explicit_native_only_and_historical_default_unchanged(
        self,
    ):
        from test_g2_runtime import configuration
        from g2e_protocol import PROTOCOL, CONTACT_MODE_NATIVE

        cfg = configuration()
        cfg.update(
            embodiment="aloha-agilex",
            contact_mode=CONTACT_MODE_NATIVE,
            execution_protocol=PROTOCOL,
            policy_rng_seed=0,
            window=50,
            native_release_semantics="physical_aperture_v1",
        )
        validate_config(cfg, allow_native_h50=True)
        cfg["native_preload_control"] = PRELOAD_PROTOCOL
        validate_config(cfg, allow_native_h50=True)
        for patch in (
            {"embodiment": "ARX-X5"},
            {"native_release_semantics": None},
            {"native_preload_control": "unknown"},
        ):
            bad = dict(cfg, **patch)
            with self.assertRaises(ValueError):
                validate_config(bad, allow_native_h50=True)

    def test_native_hooks_apply_after_candidate_acceptance_and_hold_on_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            backend, state, closed, physical = native_fixture(Path(temp))
            for i, joint in enumerate(
                backend.env.robot.left_entity.get_active_joints()
            ):
                joint.get_drive_velocity_target = lambda i=i: np.array(
                    [state["velocity"][i]]
                )
            tracker = self.tracker
            guard = NS(
                protocol="test",
                previous_drive=None,
                max_physics_steps=25000,
                collision=NS(tracker=tracker),
            )
            reject = [False]
            observed_ids = [(1, 2)]

            def candidate(q, fingers, targets, drive, check):
                if reject[0]:
                    raise G2Error(Status.COLLISION, "injected candidate obstruction")
                before = drive if guard.previous_drive is None else guard.previous_drive
                tracker.authorize_opening(before, drive, self.arms)
                guard.previous_drive = drive.copy()

            def measured(stamp, q, fingers, tools, pose, cs):
                tracker.observe(stamp, contacts(observed_ids[0]), self.tools, np.eye(4))

            guard.candidate, guard.measured = candidate, measured
            guard.tracking = lambda *args: None
            guard.metrics = lambda: {}
            backend.guard = guard
            backend.preload = NativePreloadHold(tracker, backend.arms)
            backend.env.scene.before_step = backend._before_step
            physical.get_contacts = lambda: []
            backend.env.check_success = lambda: False
            backend.episode, backend.window = "test", 4
            logs = []
            action = np.zeros((4, 14))
            action[:, [6, 13]] = 0.55
            action[3, 6] = 0
            backend.execute(action, logs.append, lambda: 1)
            np.testing.assert_allclose(state["target"][[6, 7]], [0.022, 0.022])
            np.testing.assert_array_equal(state["velocity"][[6, 7]], [0, 0])
            self.assertEqual(backend.tick, 4)
            self.assertEqual(
                sum(e.get("kind") == "native_preload_control" for e in logs), 4
            )
            # Synthetic contacts drive the real release state machine through
            # the actual NativeBackend hooks, not a claimed physical handover.
            observed_ids[0] = (1, 2, 3, 4)
            backend.execute(action[:3], logs.append, lambda: 1)
            self.assertEqual(set(tracker.holders), {"left", "right"})
            release = action[:1].copy()
            release[0, 6] = 0.8
            observed_ids[0] = (3, 4)
            backend.execute(release, logs.append, lambda: 1)
            self.assertEqual(set(tracker.holders), {"right"})
            self.assertNotIn("left", backend.preload.references)
            self.assertAlmostEqual(state["target"][6], 0.032)
            release[0, 13] = 0.8
            observed_ids[0] = ()
            backend.execute(release, logs.append, lambda: 1)
            self.assertFalse(tracker.holders)
            self.assertFalse(backend.preload.references)
            np.testing.assert_allclose(state["target"][[14, 15]], [0.032, 0.032])
            stopped_tick = backend.tick
            before = state["actual"].copy()
            reject[0] = True
            with self.assertRaisesRegex(G2Error, "candidate obstruction"):
                backend.execute(action[:1], logs.append, lambda: 1)
            np.testing.assert_array_equal(state["actual"], before)
            self.assertEqual(backend.tick, stopped_tick)
            backend.hold()
            np.testing.assert_array_equal(state["velocity"], np.zeros(16))
            backend.close(Path(temp))
            self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
