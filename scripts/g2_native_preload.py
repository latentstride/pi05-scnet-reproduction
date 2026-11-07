"""Opt-in native drive intervention; measured contact anchors remain untouched."""

import numpy as np

from g2_adapter import G2Error, SIDES
from g2e_native_guard import NativeReleaseTracker
from robotwin_control_geometry import Status, finite_array


PRELOAD_PROTOCOL = "measured_contact_preload_hold_v1"


class NativePreloadHold:
    """Latch each confirmed holder's drive, never its measured finger position.

    Candidate and measured guards still run every physical step. Opening requests
    pass through the existing release gate before any drive substitution. No
    anchor reset, waiting period, force adjustment or arm command is introduced.
    """

    def __init__(self, tracker, arms):
        if not isinstance(tracker, NativeReleaseTracker):
            raise ValueError(
                "preload hold requires physical-aperture release semantics"
            )
        self.tracker, self.arms = tracker, tuple(arms)
        self.references = {}

    def observe(self, tick, drive):
        """Called only after all measured, tracking and deadline checks pass."""
        drive = finite_array(drive, (2, 2))
        self._clock(tick)
        for i, side in enumerate(SIDES):
            if side not in self.tracker.holders or side in self.tracker.releasing:
                self.references.pop(side, None)
            elif side not in self.references:
                self.references[side] = dict(tick=tick, drive=drive[i].copy())

    def _clock(self, tick):
        stamp = self.tracker.last_stamp
        if stamp is None or stamp.tick != tick:
            raise G2Error(
                Status.UNSYNCED, "preload holder lacks current measured contact"
            )

    def select(self, tick, previous, requested, velocity):
        requested = finite_array(requested, (2, 2))
        velocity = finite_array(velocity, (2, 2))
        if tick:
            self._clock(tick)
        elif self.tracker.holders or self.references:
            raise G2Error(
                Status.UNSYNCED, "preload holder present before first measurement"
            )
        # A hold must never conceal an unsafe donor-opening request.
        self.tracker.guard_opening(previous, requested, self.arms)
        applied, speeds = requested.copy(), velocity.copy()
        modes = {}
        for i, side in enumerate(SIDES):
            ref = self.references.get(side)
            if side in self.tracker.holders and side not in self.tracker.releasing:
                if ref is None:
                    raise G2Error(
                        Status.UNSYNCED, "confirmed holder has no preload reference"
                    )
                before = self.tracker.opening_geometry(ref["drive"], self.arms[i])
                after = self.tracker.opening_geometry(requested[i], self.arms[i])
                if np.any(after - before > 1e-9):
                    modes[side] = "release_requested"
                else:
                    applied[i], speeds[i] = ref["drive"], 0.0
                    modes[side] = "holding_preload"
            else:
                modes[side] = (
                    "releasing" if side in self.tracker.releasing else "inactive"
                )
        return (
            applied,
            speeds,
            dict(
                protocol=PRELOAD_PROTOCOL,
                modes=modes,
                requested_finger_drive_m=requested.copy(),
                applied_finger_drive_m=applied.copy(),
                requested_finger_velocity_m_s=velocity.copy(),
                applied_finger_velocity_m_s=speeds.copy(),
                acquisition_ticks={
                    side: ref["tick"] for side, ref in self.references.items()
                },
            ),
        )
