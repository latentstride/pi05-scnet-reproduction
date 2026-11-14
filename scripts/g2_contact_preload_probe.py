"""CPU candidate for preload establishment before unchanged holder confirmation.

Diagnostic until dynamic comparisons and deployment integration are accepted.
No contact, collision, slip, force-property or anchor thresholds are modified.
"""

import numpy as np
from g2_adapter import SIDES
from g2_native_preload import NativePreloadHold


class ContactPreloadProbe(NativePreloadHold):
    def __init__(self, tracker, arms, measured_fingers, gap_m):
        super().__init__(tracker, arms)
        if gap_m not in (0.001, 0.002, 0.004):
            raise ValueError("diagnostic preload gap outside preregistered grid")
        self.measured_fingers, self.gap_m = measured_fingers, gap_m
        self.pending = {}

    def observe(self, tick, drive, fingers=None):
        super().observe(tick, drive)
        fingers = self.measured_fingers() if fingers is None else fingers
        for i, side in enumerate(SIDES):
            bilateral = self.tracker.contact_fingers[side] == set(
                self.tracker.finger_body_ids[side]
            )
            if (
                side in self.tracker.holders
                or side in self.tracker.releasing
                or not bilateral
            ):
                self.pending.pop(side, None)
            elif side not in self.pending:
                # Preserve the native equal-drive coupling. Drive remains below
                # both measured fingers, rather than opening the measured grasp.
                target = max(
                    float(np.min(fingers[i])) - self.gap_m, float(np.max(drive[i]))
                )
                arm = self.arms[i]
                target = np.clip(
                    target, max(arm.grip.limits[:, 0]), min(arm.grip.limits[:, 1])
                )
                self.pending[side] = dict(tick=tick, drive=np.full(2, target))

    def select(self, tick, previous, requested, velocity):
        applied, speeds, event = super().select(tick, previous, requested, velocity)
        for i, side in enumerate(SIDES):
            if (
                side not in self.pending
                or side in self.tracker.holders
                or side in self.tracker.releasing
            ):
                continue
            target = self.pending[side]["drive"]
            if np.any(np.asarray(requested)[i] - target > 1e-9):
                self.pending.pop(side)
            else:
                applied[i], speeds[i] = target, 0.0
                event["modes"][side] = "establishing_preload"
        event["applied_finger_drive_m"] = applied.copy()
        event["applied_finger_velocity_m_s"] = speeds.copy()
        event["pending_contact_ticks"] = {s: v["tick"] for s, v in self.pending.items()}
        event["preload_gap_m"] = self.gap_m
        return applied, speeds, event
