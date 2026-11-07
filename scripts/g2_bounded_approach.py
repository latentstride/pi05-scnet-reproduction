"""Opt-in, recorded free-space endpoint repair within the existing 12mm/1deg envelope."""

from dataclasses import replace
import numpy as np
from g2_adapter import G2Error, solve_pair
from g2_kinematics import pose_error
from g2_remaining_geometry import refined_pair
from robotwin_control_geometry import Status


def free_space(world):
    tracker = getattr(world, "tracker", None)
    return tracker is not None and not (
        tracker.holders or tracker.releasing or any(tracker.contact_fingers.values())
    )


def upward_endpoint(arms, frames, tools, q, fingers, world, check, *, fixed_right=None):
    if not free_space(world):
        raise G2Error(
            Status.COLLISION,
            "bounded approach is forbidden during measured contact",
            geometry="candidate",
        )
    original = np.asarray(tools).copy()
    corrected = original.copy()
    corrected[0, 2, 3] += 0.0115
    goal, residual, meta = refined_pair(
        arms, frames, corrected, q, fingers, world, check
    )
    if fixed_right is not None:
        goal[1] = fixed_right
    executed = np.array(
        [f @ a.chain.fk(v) @ a.offset() for f, a, v in zip(frames, arms, goal)]
    )
    errors = [pose_error(a, b) for a, b in zip(executed, original)]
    position = [float(np.linalg.norm(e[:3])) for e in errors]
    angle = [float(np.rad2deg(np.linalg.norm(e[3:]))) for e in errors]
    if max(position) > 0.012 or max(angle) > 1.0:
        raise G2Error(
            Status.PLAN_FAILED,
            "corrected endpoint exceeds original 12mm/1deg bound",
            geometry="candidate",
        )
    if not world.ik_clear(goal, fingers, check):
        raise G2Error(
            Status.COLLISION,
            "corrected endpoint is still blocked",
            geometry="candidate",
            collisions=world.last_collisions,
        )
    return goal, dict(
        kind="bounded_upward_approach",
        original_goal_reached=False,
        original_tools=original,
        approved_tools=executed,
        pose_change_m=position,
        pose_change_deg=angle,
        goal_q=goal,
        residual=residual,
        refinement=meta,
    )


def source_admissible(adapter, q, fingers, frames, elapsed_s, check):
    """Planning-only shadow IK. Never changes the observed source seed or images."""
    world = adapter.source_collision
    tracker = getattr(world, "tracker", None)
    if tracker is None:
        raise G2Error(Status.INVALID, "source planning has no measured contact tracker")
    tools = np.array(
        [f @ a.chain.fk(v) @ a.offset() for f, a, v in zip(frames, adapter.arx, q)]
    )
    fractions = [a.grip.measure(v)["normalized"] for a, v in zip(adapter.arx, fingers)]
    source_fingers = np.array(
        [a.grip.target(v) for a, v in zip(adapter.aloha, fractions)]
    )
    before = getattr(world, "initial_fingers", None)
    before_drive = getattr(world, "initial_drive", None)
    before = None if before is None else before.copy()
    before_drive = None if before_drive is None else before_drive.copy()
    try:
        world.refresh()
        world.set_start(source_fingers)
        solution, residual = solve_pair(
            adapter.aloha,
            adapter.source_frames,
            tools,
            adapter.seed.copy(),
            source_fingers,
            world,
            check,
            adapter.source_speed * elapsed_s,
        )
        return dict(accepted=True, source_q=solution, residual=residual)
    finally:
        if before is not None:
            world.set_start(before, before_drive)


def checked_prefix(adapter, plan, snapshot, check):
    """Retain only original waypoints that remain source-observable in free space."""
    if not free_space(adapter.target_collision):
        return plan, None
    from g2_adapter import read_snapshot

    _, _, frames, _ = read_snapshot(snapshot, adapter.arx)
    for row, tick in enumerate(plan.waypoint_ticks):
        # Existing contact/release planning retains its own measured phase rules.
        if adapter.target_collision.action_barrier(
            plan.q[tick], plan.fingers[tick], check
        ):
            return plan, None
        predicted = (
            plan.predicted_fingers[tick]
            if plan.predicted_fingers is not None
            else plan.fingers[tick]
        )
        try:
            source_admissible(
                adapter, plan.q[tick], predicted, frames, tick * plan.period_s, check
            )
        except G2Error as error:
            if error.status not in (
                Status.COLLISION,
                Status.IK_FAILED,
                Status.VELOCITY,
            ):
                raise
            evidence = dict(
                kind="source_observability_barrier",
                row=row,
                source_status=error.status.value,
                source_diagnostic=error.diagnostic,
                geometry="candidate",
                original_goal_reached=False,
            )
            if row == 0:
                return None, evidence
            end = plan.waypoint_ticks[row - 1] + 1
            prefix = replace(
                plan,
                q=plan.q[:end],
                velocity=plan.velocity[:end],
                fingers=plan.fingers[:end],
                waypoint_ticks=plan.waypoint_ticks[:row],
                predicted_fingers=None
                if plan.predicted_fingers is None
                else plan.predicted_fingers[:end],
            )
            return prefix, evidence
    return plan, None


def finger_start_blocked(diagnostic):
    stack = [diagnostic]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if (
                item.get("reason") == "predicted finger start intersects task block"
                and item.get("side") == "left"
            ):
                return True
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


def source_obstruction(diagnostic):
    """Canonical collision-pair evidence, with a numerical penetration metric."""
    stack = [diagnostic]
    pairs = set()
    depths = []
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if (
                item.get("collision_type") == "self"
                and item.get("link_name1")
                and item.get("link_name2")
            ):
                pairs.add(tuple(sorted((item["link_name1"], item["link_name2"]))))
                value = item.get("max_penetration_m")
                if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
                    depths.append(float(value))
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return (tuple(sorted(pairs)), min(depths)) if pairs and depths else None
