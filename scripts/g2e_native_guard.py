"""Measured native safety supervisor. No retiming or policy-state substitution.

Collision projections describe finger geometry only. Native force/drive commands
remain exactly those issued by the pinned Aloha qpos controller.
"""

from time import monotonic
import numpy as np
from g2_adapter import G2Error, SIDES
from g2_contact import ContactTracker
from g2_kinematics import pose_error
from g2e_protocol import PROTOCOL
from robotwin_control_geometry import Status, finite_array


class NativeReleaseTracker(ContactTracker):
    """Use physical aperture to classify native Aloha release intent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opening_reference = {}

    def observe(self, *args, **kwargs):
        try:
            return super().observe(*args, **kwargs)
        finally:
            self.opening_reference = {
                side: value
                for side, value in self.opening_reference.items()
                if side in self.holders
            }

    @staticmethod
    def opening_geometry(value, arm):
        """Opening distance from the calibrated physical closed endpoint.

        Aloha uses negative PD targets as closure preload although its physical
        finger calibration ends at zero. Moving within that preload range does
        not request positive aperture. No tolerance or force command is changed.
        """
        closed = np.asarray(arm.grip.closed)
        direction = np.sign(np.asarray(arm.grip.opened) - closed)
        return np.maximum(0.0, (np.asarray(value) - closed) * direction)

    def guard_opening(self, previous, target, arms, giver="left"):
        receiver = "right" if giver == "left" else "left"
        if giver not in self.holders or receiver in self.holders:
            return
        i = SIDES.index(giver)
        before = self.opening_geometry(previous[i], arms[i])
        reference = np.minimum(self.opening_reference.get(giver, before), before)
        opening = self.opening_geometry(target[i], arms[i])
        if np.any(opening - reference > 1e-9):
            raise G2Error(
                Status.BACKEND_ERROR,
                "donor opening before receiver contact confirmation",
                geometry="candidate",
                reason="premature_release",
                contact_state=self.snapshot(),
                opening_geometry_m=opening.tolist(),
                opening_reference_m=reference.tolist(),
            )

    def authorize_opening(self, previous, target, arms, giver="left"):
        self.guard_opening(previous, target, arms, giver)
        for i, side in enumerate(SIDES):
            if side in self.holders:
                before = self.opening_geometry(previous[i], arms[i])
                reference = np.minimum(self.opening_reference.get(side, before), before)
                opening = self.opening_geometry(target[i], arms[i])
                if np.any(opening - reference > 1e-9):
                    self.releasing.add(side)
                self.opening_reference[side] = np.minimum(reference, opening).copy()


class NativeGuard:
    protocol = PROTOCOL

    def __init__(self, collision, arms, joint_step, position_limit, angle_limit):
        self.collision, self.arms = collision, tuple(arms)
        self.joint_step, self.position_limit, self.angle_limit = (
            joint_step,
            position_limit,
            angle_limit,
        )
        if not all(
            np.isfinite(v) and v > 0 for v in (joint_step, position_limit, angle_limit)
        ):
            raise ValueError("invalid native safety bounds")
        self.calls, self.wall_s, self.samples = 0, 0.0, 0
        self.previous_drive = None
        self.max_physics_steps = 25000
        self.max_mean_candidate_s = 0.2
        self.last = {}

    def measured(self, stamp, q, fingers, tools, block_pose, contacts):
        if stamp.tick > self.max_physics_steps:
            raise G2Error(Status.TIMEOUT, "G2-E physical-step ceiling reached")
        for arm, values in zip(self.arms, fingers):
            if np.any(values < arm.grip.limits[:, 0] - arm.grip.tolerance) or np.any(
                values > arm.grip.limits[:, 1] + arm.grip.tolerance
            ):
                raise G2Error(
                    Status.LIMIT,
                    "measured fingers exceed original calibration tolerance",
                    geometry="actual",
                )
        self.collision.tracker.observe(stamp, contacts, tools, block_pose)
        self.collision.refresh()
        if not self.collision.measured_clear(q, fingers):
            raise G2Error(
                Status.COLLISION,
                "measured native geometry blocked",
                geometry="actual",
                collisions=self.collision.last_collisions,
            )
        self.last["contact_state"] = self.collision.tracker.snapshot()

    def candidate(self, q, fingers, drive_q, drive_fingers, check):
        """Check actual-to-intended drive segment before each physical advance."""
        started = monotonic()
        self.calls += 1
        fraction = None
        try:
            check()
            if (
                self.calls >= 200
                and self.wall_s / (self.calls - 1) > self.max_mean_candidate_s
            ):
                raise G2Error(
                    Status.TIMEOUT, "G2-E candidate CPU overhead ceiling reached"
                )
            q, fingers = finite_array(q, (2, 6)), finite_array(fingers, (2, 2))
            drive_q, drive_fingers = (
                finite_array(drive_q, (2, 6)),
                finite_array(drive_fingers, (2, 2)),
            )
            for arm, values in zip(self.arms, drive_q):
                if np.any(values < arm.chain.limits[:, 0]) or np.any(
                    values > arm.chain.limits[:, 1]
                ):
                    raise G2Error(
                        Status.LIMIT, "native interpolated arm drive outside limits"
                    )
            before = (
                drive_fingers if self.previous_drive is None else self.previous_drive
            )
            if self.previous_drive is None and self.collision.tracker.holders:
                raise G2Error(
                    Status.UNSYNCED, "native release history missing during holding"
                )
            self.collision.tracker.guard_opening(before, drive_fingers, self.arms)
            self.collision.refresh()
            self.collision.set_start(fingers, drive_fingers)
            # Negative closure drive is intentional in native Aloha. Project only
            # a bounded collision geometry; never send the projection to PhysX.
            requested = np.array(
                [
                    np.clip(v, arm.grip.limits[:, 0], arm.grip.limits[:, 1])
                    for arm, v in zip(self.arms, drive_fingers)
                ]
            )
            count = max(1, int(np.ceil(np.max(np.abs(drive_q - q)) / self.joint_step)))
            predicted = fingers.copy()
            for fraction in np.linspace(0, 1, count + 1):
                check()
                sample = q + (drive_q - q) * fraction
                # Repeat prediction from actual fingers at each arm pose. This
                # checks the closure sweep, without assuming commanded contact.
                predicted, changes = self.collision.project_fingers(
                    sample, requested, check, previous=fingers
                )
                self.samples += 1
                if not self.collision(sample, predicted):
                    raise G2Error(
                        Status.COLLISION,
                        "native drive candidate blocked",
                        geometry="candidate",
                        segment_fraction=float(fraction),
                        collisions=self.collision.last_collisions,
                    )
            # Authorize release only after candidate acceptance. This influences
            # contact-loss classification, never native action values.
            self.collision.tracker.authorize_opening(before, drive_fingers, self.arms)
            self.previous_drive = drive_fingers.copy()
            self.last.update(
                target_fingers_drive_m=drive_fingers.copy(),
                predicted_fingers=predicted,
                projection_changes=changes,
                guard_samples=self.samples,
            )
        except G2Error as error:
            # A rejected interpolated drive was previously absent from the
            # historical log. Preserve actual, intended and candidate geometry
            # separately so a future checkpoint can be replayed exactly.
            error.diagnostic.setdefault("geometry", "candidate")
            error.diagnostic.update(
                actual_arm=np.asarray(q).tolist(),
                actual_fingers=np.asarray(fingers).tolist(),
                drive_arm=np.asarray(drive_q).tolist(),
                drive_fingers=np.asarray(drive_fingers).tolist(),
                previous_drive=None
                if self.previous_drive is None
                else self.previous_drive.tolist(),
                candidate_segment_fraction=None
                if fraction is None
                else float(fraction),
            )
            raise
        finally:
            self.wall_s += monotonic() - started

    def tracking(self, target_tools, actual_tools):
        errors = [
            pose_error(goal, real) for goal, real in zip(target_tools, actual_tools)
        ]
        if any(
            np.linalg.norm(e[:3]) > self.position_limit
            or np.rad2deg(np.linalg.norm(e[3:])) > self.angle_limit
            for e in errors
        ):
            raise G2Error(
                Status.BACKEND_ERROR,
                "native tracking exceeds existing 30mm/10deg bounds",
                geometry="actual",
                position_error_m=[float(np.linalg.norm(e[:3])) for e in errors],
                orientation_error_deg=[
                    float(np.rad2deg(np.linalg.norm(e[3:]))) for e in errors
                ],
            )

    def metrics(self):
        return dict(
            self.last,
            guard_calls=self.calls,
            guard_wall_s=self.wall_s,
            guard_samples=self.samples,
        )
