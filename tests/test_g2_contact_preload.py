"""Selected 4mm controller preserves measured confirmation and release protection."""

from pathlib import Path
import unittest
import numpy as np
from g2_adapter import G2Error, Stamp
from g2_contact import ContactLimits
from g2e_native_guard import NativeReleaseTracker
from g2_native_contact_preload import ContactPreloadHold
from g2e_protocol import PRELOAD_ESTABLISHMENT, PRELOAD_PROTOCOL
from g2_fixtures import robot_arms
from g2_task_runtime import validate_config
from test_g2_contact import contacts
from test_g2_runtime import configuration
from g2e_protocol import PROTOCOL, CONTACT_MODE_NATIVE


class ContactPreloadTests(unittest.TestCase):
    def setUp(self):
        self.arms = robot_arms("aloha-agilex")
        self.tracker = NativeReleaseTracker(
            99, dict(left=[1, 2], right=[3, 4]), ContactLimits()
        )
        self.control = ContactPreloadHold(self.tracker, self.arms)
        self.drive = np.full((2, 2), 0.022)
        self.fingers = np.full((2, 2), 0.030)

    def step(self, tick, ids=(1, 2)):
        self.tracker.observe(
            Stamp("case", tick, tick * 0.004),
            contacts(ids),
            np.array([np.eye(4), np.eye(4)]),
            np.eye(4),
        )
        self.control.observe(tick, self.drive, self.fingers)

    def test_preload_establishment_does_not_confirm_holder_or_move_anchor(self):
        self.step(1)
        self.assertFalse(self.tracker.holders)
        original = self.tracker.candidates["left"].copy()
        requested = self.drive.copy()
        requested[0] = 0.020
        applied, velocity, event = self.control.select(
            1, self.drive, requested, np.ones((2, 2))
        )
        np.testing.assert_allclose(applied[0], [0.026, 0.026])
        self.assertEqual(event["modes"]["left"], "establishing_preload")
        self.assertEqual(event["establishment_protocol"], PRELOAD_ESTABLISHMENT)
        self.assertFalse(self.tracker.releasing)
        self.drive = applied
        self.step(2)
        self.assertFalse(self.tracker.holders)
        self.step(3)
        self.assertEqual(set(self.tracker.holders), {"left"})
        np.testing.assert_array_equal(self.tracker.holders["left"], original)
        np.testing.assert_array_equal(
            self.control.references["left"]["drive"], self.drive[0]
        )
        self.assertFalse(self.control.pending)
        opened = self.drive.copy()
        opened[0] = 0.032
        with self.assertRaisesRegex(G2Error, "donor opening"):
            self.control.select(3, self.drive, opened, np.zeros((2, 2)))

    def test_unconfirmed_contact_loss_cancels_only_pending_stage(self):
        self.step(1)
        self.assertTrue(self.control.pending)
        self.step(2, (1,))
        self.assertFalse(self.control.pending)
        self.assertFalse(self.tracker.holders)
        requested = self.drive.copy()
        requested[0] = 0.020
        applied, _, event = self.control.select(
            2, self.drive, requested, np.zeros((2, 2))
        )
        np.testing.assert_array_equal(applied, requested)
        self.assertEqual(event["modes"]["left"], "inactive")

    def test_opening_request_can_exit_pending_establishment(self):
        self.step(1)
        requested = self.drive.copy()
        requested[0] = 0.035
        applied, _, _ = self.control.select(1, self.drive, requested, np.zeros((2, 2)))
        np.testing.assert_array_equal(applied, requested)
        self.assertFalse(self.control.pending)

    def test_selected_controller_keeps_measured_transfer_and_release_sequence(self):
        for tick in (1, 2, 3):
            self.step(tick)
        opening = self.drive.copy()
        opening[0] = 0.032
        with self.assertRaisesRegex(G2Error, "donor opening"):
            self.control.select(3, self.drive, opening, np.zeros((2, 2)))
        for tick in (4, 5, 6):
            self.step(tick, (1, 2, 3, 4))
        self.assertEqual(set(self.tracker.holders), {"left", "right"})
        applied, _, event = self.control.select(
            6, self.drive, opening, np.zeros((2, 2))
        )
        self.assertEqual(event["modes"]["left"], "release_requested")
        self.tracker.authorize_opening(self.drive, applied, self.arms)
        self.drive = applied
        self.step(7, (3, 4))
        self.assertEqual(set(self.tracker.holders), {"right"})
        self.assertNotIn("left", self.control.references)
        opening = self.drive.copy()
        opening[1] = 0.032
        applied, _, _ = self.control.select(7, self.drive, opening, np.zeros((2, 2)))
        self.tracker.authorize_opening(self.drive, applied, self.arms)
        self.drive = applied
        self.step(8, ())
        self.assertFalse(self.tracker.holders)
        self.assertFalse(self.control.references)
        self.assertEqual(self.tracker.limits, ContactLimits())

    def test_only_reviewed_profile_can_be_enabled(self):
        cfg = configuration()
        cfg.update(
            embodiment="aloha-agilex",
            contact_mode=CONTACT_MODE_NATIVE,
            execution_protocol=PROTOCOL,
            policy_rng_seed=0,
            window=50,
            native_release_semantics="physical_aperture_v1",
            native_preload_control=PRELOAD_PROTOCOL,
            native_preload_establishment=PRELOAD_ESTABLISHMENT,
        )
        validate_config(cfg, allow_native_h50=True)
        for bad in [None, "bilateral_contact_preload_1mm_v1"]:
            changed = dict(
                cfg,
                native_preload_control=bad if bad is None else PRELOAD_PROTOCOL,
                native_preload_establishment=PRELOAD_ESTABLISHMENT
                if bad is None
                else bad,
            )
            with self.assertRaises(ValueError):
                validate_config(changed, allow_native_h50=True)
        from g2_final_prepare_review import RUNTIME_MODULES

        self.assertIn("g2_native_contact_preload", RUNTIME_MODULES)
        self.assertTrue(
            (
                Path(__file__).resolve().parents[1]
                / "scripts/g2_native_contact_preload.py"
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
