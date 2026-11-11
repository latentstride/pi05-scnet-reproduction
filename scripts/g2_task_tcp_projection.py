"""Opt-in bounded task-frame TCP projection for the ARX approach phase.

The coordinator discards the remaining chunk and captures a fresh observation
after executing a corrected prefix.
"""
import numpy as np

from g2_adapter import G2Error, bounds, deadline, read_snapshot, solve_pair
from g2_bounded_approach import checked_prefix, free_space
from g2_kinematics import pose_error
from robotwin_control_geometry import Status, finite_array

METHOD = "bounded_task_tcp_projection_v1"


def plan_projection(adapter, snapshot, chunk, window=4, budget_s=120):
    """Check at most 18 translations; preserve right TCP, orientations and grips.

    The same left task-frame offset applies to every row in this prefix. Full
    target paths and source endpoint observability retain the original gates.
    This is a bounded action intervention, not an assertion of calibration error.
    """
    if snapshot.stamp != adapter.last_stamp:
        raise G2Error(Status.UNSYNCED, "projection needs the mapped observation")
    if type(window) is not int or not 1 <= window <= 4:
        raise ValueError("projection window outside existing 1..4 contract")
    data = finite_array(chunk, (50, 14))
    world = adapter.target_collision
    if not free_space(world) or world.tracker.phase != "approach":
        raise G2Error(Status.COLLISION, "TCP projection requires a contact-free approach",
                      reason="recovery_unsafe_contact")
    q, fingers, frames, _ = read_snapshot(snapshot, adapter.arx)
    for arm, values in zip(adapter.arx, fingers):
        bounds(values, np.asarray(arm.grip.limits), geometry="actual")
    world.refresh()
    if not world.measured_clear(q, fingers):
        raise G2Error(Status.COLLISION, "projection start is not clear", geometry="actual")
    check = deadline(budget_s)
    original, native = [], []
    for row in data[:window].reshape(window, 2, 7):
        tools, grip = [], []
        for arm, target, frame, values in zip(adapter.aloha, adapter.arx, adapter.source_frames, row):
            bounds(values[:6], arm.chain.limits, geometry="candidate")
            raw = float(values[6]); value = float(np.clip(raw, 0, 1))
            if raw != value and (adapter.grip_rule == "reject" or abs(raw-value) > adapter.grip_max_clip):
                raise G2Error(Status.LIMIT, "projection grip outside original envelope")
            tools.append(frame @ arm.chain.fk(values[:6]) @ arm.offset())
            grip.append(target.grip.target(value))
        original.append(np.asarray(tools)); native.append(np.asarray(grip))
    attempts = []
    for radius in (.004, .008, .0115):
        for axis in range(3):
            for sign in (-1, 1):
                check()
                shift = np.eye(3)[axis] * radius * sign
                record = dict(left_task_translation_m=shift.tolist())
                try:
                    seed = q.copy(); goals = []; errors = []
                    for tools, grip in zip(original, native):
                        desired = tools.copy(); desired[0, :3, 3] += shift
                        goal, _ = solve_pair(adapter.arx, frames, desired, seed, grip, world, check)
                        actual = np.array([f @ a.chain.fk(v) @ a.offset() for f, a, v in zip(frames, adapter.arx, goal)])
                        error = [pose_error(t, a) for t, a in zip(tools, actual)]
                        pos = [float(np.linalg.norm(e[:3])) for e in error]
                        ang = [float(np.rad2deg(np.linalg.norm(e[3:]))) for e in error]
                        if max(pos) > .012 or max(ang) > 1.0:
                            raise G2Error(Status.PLAN_FAILED, "projection exceeds original 12mm/1deg envelope")
                        # The unmodified right target must meet ordinary IK residuals.
                        if pos[1] > .0005 or ang[1] > .05:
                            raise G2Error(Status.PLAN_FAILED, "projection altered right TCP")
                        errors.append(dict(position_m=pos, angle_deg=ang))
                        goals.append((goal, grip)); seed = goal
                    plan = adapter.motion.plan(q, fingers, goals, world, check)
                    # The production prefix helper hands predicted contact back to
                    # the normal contact controller. This projection is only
                    # defined for free-space prefixes, so fail closed instead.
                    if any(world.action_barrier(plan.q[t], plan.fingers[t], check)
                           for t in plan.waypoint_ticks):
                        raise G2Error(Status.PLAN_FAILED, "projection reaches a contact or release barrier")
                    prefix, barrier = checked_prefix(adapter, plan, snapshot, check)
                    if barrier is not None or prefix is None or len(prefix.waypoint_ticks) != window:
                        raise G2Error(Status.PLAN_FAILED, "projected prefix remains source-blocked", barrier=barrier)
                    record.update(accepted=True, errors=errors, control_ticks=len(plan.q))
                    attempts.append(record)
                    return plan, dict(method=METHOD, attempts=attempts,
                                      requires_real_reobservation=True, original_goal_reached=False,
                                      expert_actions_used=False, left_task_translation_m=shift.tolist())
                except G2Error as error:
                    if error.status in (Status.TIMEOUT, Status.BACKEND_ERROR, Status.UNSYNCED):
                        raise
                    record.update(accepted=False, status=error.status.value, detail=str(error), diagnostic=error.diagnostic)
                    attempts.append(record)
    raise G2Error(Status.PLAN_FAILED, "18 bounded task TCP projections exhausted", attempts=attempts,
                  geometry="candidate", requires_real_reobservation=True)
