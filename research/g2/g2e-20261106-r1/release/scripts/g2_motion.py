"""Synchronized CPU joint-space planning with sampled bimanual collision checks."""
from dataclasses import dataclass

import numpy as np

from g2_adapter import G2Error, clear, positive
from robotwin_control_geometry import Status, finite_array

METHOD = "g2_quintic_stop_at_waypoints_v1"


@dataclass(frozen=True)
class Trajectory:
    period_s: float
    q: np.ndarray
    velocity: np.ndarray
    fingers: np.ndarray
    waypoint_ticks: tuple
    predicted_fingers: np.ndarray = None
    contact_profile_id: str = ""


class QuinticMotion:
    """A straight joint path per waypoint; collision means rejection, not rerouting."""

    method = METHOD

    def __init__(self, period_s, arm_speed, arm_acceleration, finger_speed,
                 finger_acceleration, collision_joint_step, collision_finger_step,
                 limits_source, collision_id, max_ticks=2000):
        self.period = positive(period_s)
        self.speed = np.c_[finite_array(arm_speed, (2, 6)), finite_array(finger_speed, (2, 2))]
        self.accel = np.c_[finite_array(arm_acceleration, (2, 6)), finite_array(finger_acceleration, (2, 2))]
        self.sampling = np.r_[[positive(collision_joint_step)] * 6, [positive(collision_finger_step)] * 2]
        if (np.any(self.speed <= 0) or np.any(self.accel <= 0) or not limits_source or not collision_id
                or type(max_ticks) is not int or max_ticks <= 0):
            raise ValueError("invalid motion limits or provenance")
        self.limits_source, self.collision_id, self.max_ticks = limits_source, collision_id, max_ticks

    def plan(self, q, fingers, targets, collision, check):
        measured=finite_array(fingers,(2,2))
        predictor=getattr(collision,"planning_geometry",None)
        contact=callable(predictor)
        drive=collision.drive_start(measured) if contact else measured
        start = np.c_[finite_array(q, (2, 6)), drive]
        predicted=[measured.copy()]
        geometry=measured.copy()
        path, velocity, events = [start], [np.zeros_like(start)], []
        actual_checker=getattr(collision,"measured_clear",collision)
        if not clear(actual_checker, start[:, :6], measured):
            raise G2Error(Status.COLLISION, "initial pose is in collision", tick=0)
        for row, (q_i, g_i) in enumerate(targets):
            check()
            end = np.c_[finite_array(q_i, (2, 6)), finite_array(g_i, (2, 2))]
            d = end - start
            # Max |s'| = 15/8 and max |s''| = 10/sqrt(3) for s=10u³−15u⁴+6u⁵.
            duration = max(np.max(1.875 * abs(d) / self.speed),
                           np.sqrt(np.max((10 / np.sqrt(3)) * abs(d) / self.accel)))
            ticks = max(1, int(np.ceil(duration / self.period)))
            if len(path) - 1 + ticks > self.max_ticks:
                raise G2Error(Status.PLAN_FAILED, "retimed action prefix exceeds control tick budget", row=row)
            last = start
            for tick in range(1, ticks + 1):
                check()
                u = tick / ticks
                point = start + d * (u**3 * (10 + u * (-15 + 6 * u)))
                subs = max(1, int(np.ceil(np.max(abs(point - last) / self.sampling))))
                for i in range(1, subs + 1):
                    check()
                    candidate = last + (point - last) * (i / subs)
                    geometry=(predictor(candidate[:,:6],candidate[:,6:],geometry,check)
                              if contact else candidate[:,6:])
                    if not clear(collision, candidate[:, :6], geometry):
                        raise G2Error(Status.COLLISION, "retimed segment collides", row=row,
                                      tick=len(path), sample=i, samples=subs)
                last = point
                path.append(point)
                predicted.append(geometry.copy())
                velocity.append(d * (30 * u**2 * (1 - u)**2 / (ticks * self.period)))
            events.append(len(path) - 1)
            start = end
        path = np.array(path)
        return Trajectory(self.period, path[:, :, :6], np.array(velocity)[:, :, :6], path[:, :, 6:], tuple(events),
                          np.array(predicted) if contact else None,
                          collision.profile_id if contact else "")
