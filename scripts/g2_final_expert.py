"""Run frozen expert waypoints and live ARX placement in the same scene."""

import json
from pathlib import Path

import numpy as np

from g2_adapter import G2Error, read_snapshot
from g2_initial import matrix7
from g2_native_expert_cpu import unpack_trace
from g2_remaining_geometry import refined_pair
from robotwin_control_geometry import Status


def expert_prefix(trace, metadata):
    data = unpack_trace(trace)
    meta = json.loads(Path(metadata).read_text())
    segments = meta["segments"]
    if (
        len(segments) != 14
        or segments[11]["kind"] != "B"
        or [s["kind"] for s in segments[12:]] != ["R", "RG"]
    ):
        raise ValueError("unknown expert handover/placement boundary")
    boundary = segments[11]["last_tick"]
    ticks = np.asarray(data["frame_ticks"])
    if (
        ticks.ndim != 1
        or not np.issubdtype(ticks.dtype, np.integer)
        or len(ticks) != meta["frames"]
        or np.any(np.diff(ticks) < 0)
        or ticks[0] != 0
        or ticks[-1] != meta["physics_steps"]
        or len(data["target"]) != meta["physics_steps"] + 1
        or not 0 < boundary < len(data["target"])
    ):
        raise ValueError("invalid expert trajectory clock")
    ticks = np.unique(
        np.r_[ticks[ticks < boundary], [s["last_tick"] for s in segments[:12]]]
    )
    rows = data["target"][ticks]
    if rows.shape != (len(ticks), 14) or not np.isfinite(rows).all():
        raise ValueError("invalid expert targets")
    return ticks, rows


def require_holders(tracker, expected, phase):
    if set(tracker.holders) != set(expected):
        raise G2Error(
            Status.PLAN_FAILED, f"{phase} requires measured holders {expected}"
        )


def execute_expert(backend, adapter, ticks, rows, target_pose, log, remaining):
    """Keep the same measured tracker and simulator across all phases."""
    world = adapter.target_collision
    tracker = world.tracker
    for tick, row in zip(ticks, rows):
        snapshot, _ = backend.capture()
        observed = adapter.begin_expert(snapshot, remaining())
        if not observed.ok:
            raise G2Error(observed.status, observed.detail, **observed.diagnostic)
        plan = adapter.actions(np.tile(row, (50, 1)), snapshot, 1, remaining())
        if not plan.ok:
            diagnostic = dict(plan.diagnostic)
            diagnostic.setdefault("geometry", "candidate")
            raise G2Error(plan.status, plan.detail, **diagnostic)
        log(dict(kind="expert_waypoint", source_tick=int(tick)))
        if backend.execute(plan.payload, log, remaining, stop_on_success=False) != 1:
            raise G2Error(Status.BACKEND_ERROR, "expert waypoint execution interrupted")
        transition = backend.take_replan()
        if transition:
            log(dict(kind="expert_measured_transition", transition=transition))
    return execute_placement(backend, adapter, target_pose, log, remaining)


def execute_placement(backend, adapter, target_pose, log, remaining):
    """Finish the expert reference using the live post-handover state."""
    world = adapter.target_collision
    tracker = world.tracker
    # Entry to placement requires an observed transfer, not an injected holder.
    backend.capture()
    require_holders(tracker, ["right"], "placement")
    if not tracker.saw_shared_hold:
        raise G2Error(Status.PLAN_FAILED, "no measured shared hold before placement")
    log(
        dict(
            kind="expert_placement_admitted",
            tick=backend.tick,
            contact_state=tracker.snapshot(),
        )
    )

    tray = matrix7(target_pose)

    def state():
        snapshot, _ = backend.capture()
        q, fingers, frames, _ = read_snapshot(snapshot, backend.arms)
        world.refresh()
        return q, fingers, frames, backend._drive_fingers(), np.asarray(world.tools(q))

    def plan_to(goal_tools, goal_fingers):
        q, fingers, frames, _, _ = state()
        remaining()
        goal = q
        try:
            if goal_tools is not None:
                goal, _, _ = refined_pair(
                    backend.arms, frames, goal_tools, q, goal_fingers, world, remaining
                )
            return adapter.motion.plan(
                q, fingers, [(goal, goal_fingers)], world, remaining
            )
        except G2Error as error:
            error.diagnostic.setdefault("geometry", "candidate")
            raise

    def move(phase, path):
        log(dict(kind="expert_placement_phase", phase=phase, ticks=len(path.q)))
        if backend.execute(path, log, remaining, stop_on_success=False) != 1:
            raise G2Error(Status.BACKEND_ERROR, f"{phase} execution interrupted")
        return backend.take_replan()

    _, _, _, drive, tools = state()
    tools[1, :2, 3] += tray[:2, 3] - tracker.block_pose[:2, 3]
    tools[1, 2, 3] += 0.012
    if move("approach", plan_to(tools, drive)):
        raise G2Error(Status.PLAN_FAILED, "approach interrupted by contact transition")
    require_holders(tracker, ["right"], "descent")
    _, _, _, drive, tools = state()
    block = tracker.block_pose
    lowest = block[2, 3] - np.abs(block[2, :3]) @ np.array([0.03, 0.03, 0.1])
    tools[1, 2, 3] += tray[2, 3] + 0.005 + 0.005 - lowest
    descent = plan_to(tools, drive)
    opening = drive.copy()
    opening[1] = 0.044
    previous = world.initial_fingers.copy(), world.initial_drive.copy()
    try:
        world.set_start(descent.predicted_fingers[-1], descent.fingers[-1])
        adapter.motion.plan(
            descent.q[-1],
            descent.predicted_fingers[-1],
            [(descent.q[-1], opening)],
            world,
            remaining,
        )
    except G2Error as error:
        error.diagnostic.setdefault("geometry", "candidate")
        raise
    finally:
        world.set_start(*previous)
    if move("descent", descent):
        raise G2Error(Status.PLAN_FAILED, "descent interrupted by contact transition")
    require_holders(tracker, ["right"], "opening")
    for attempt in range(6):
        _, _, _, drive, _ = state()
        if not tracker.holders and np.allclose(drive, opening, atol=1e-7, rtol=0):
            break
        if attempt == 5:
            raise G2Error(
                Status.PLAN_FAILED, "release did not reach measured handover endpoint"
            )
        move("opening", plan_to(None, opening))
    backend.capture()
    require_holders(tracker, [], "retreat")
    _, _, _, _, tools = state()
    tools[1, 2, 3] += 0.035
    if move("retreat", plan_to(tools, opening)):
        raise G2Error(Status.PLAN_FAILED, "retreat interrupted by contact transition")
    settle_at = backend.tick + round(0.4 / backend.dt)
    while backend.tick < settle_at:
        if move("settle", plan_to(None, opening)):
            raise G2Error(
                Status.PLAN_FAILED, "settling interrupted by contact transition"
            )
    backend.capture()
    require_holders(tracker, [], "settled placement")
    log(
        dict(
            kind="expert_placement_settled",
            tick=backend.tick,
            contact_state=tracker.snapshot(),
        )
    )
    return backend.success()
