"""Candidate rejection provenance and paused-state proof; no simulator advance."""

import re
import json
from hashlib import sha256


def actual_provenance(value):
    if isinstance(value, dict):
        return value.get("geometry") == "actual" or any(
            actual_provenance(v) for v in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(actual_provenance(v) for v in value)
    return False


def candidate_failure(result):
    failure = result.get("failure") or {}
    diagnostic = failure.get("diagnostic") or {}
    return (
        diagnostic.get("geometry") == "candidate"
        and not actual_provenance(diagnostic)
        and failure.get("code")
        in (
            "collision",
            "ik_failed",
            "unreachable",
            "plan_failed",
            "velocity_limit",
            "position_limit",
        )
    )


def failure_sha256(result):
    from g2_episode import json_value

    failure = result.get("failure") or {}
    return sha256(
        json.dumps(
            dict(code=failure.get("code"), diagnostic=failure.get("diagnostic") or {}),
            default=json_value,
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def verified_candidate_stop(proof, episode=None, result=None):
    return (
        isinstance(proof, dict)
        and proof.get("kind") == "candidate_rejection"
        and isinstance(proof.get("episode"), str)
        and bool(proof["episode"])
        and (episode is None or proof["episode"] == episode)
        and type(proof.get("tick")) is int
        and proof["tick"] >= 0
        and isinstance(proof.get("state_sha256"), str)
        and re.fullmatch("[a-f0-9]{64}", proof["state_sha256"]) is not None
        and isinstance(proof.get("failure_sha256"), str)
        and re.fullmatch("[a-f0-9]{64}", proof["failure_sha256"]) is not None
        and (result is None or proof["failure_sha256"] == failure_sha256(result))
        and all(
            proof.get(k) is True
            for k in (
                "terminal_checks_passed",
                "hold_verified",
                "hold_complete",
                "close_complete",
                "scene_context_closed",
                "no_physics_advance",
            )
        )
    )


def measure_stop(backend, *, held=False):
    """Recheck actual geometry/contact/tracking without resetting the live tracker.

    Physical velocities may be nonzero in a paused simulation. Hold verification
    checks zero *drive velocity*, not fabricated settling or extra physics steps.
    """
    from copy import deepcopy
    from hashlib import sha256
    import json
    import numpy as np
    from g2_adapter import G2Error, Snapshot, Stamp, read_snapshot, SIDES
    from g2_capture import capture_arms, capture_task_objects
    from g2_kinematics import pose_error
    from robotwin_control_geometry import (
        Status,
        inverse_pose,
        pose_from_wxyz,
        finite_array,
    )

    tick = backend.tick
    stamp = Stamp(backend.episode, tick, tick * backend.dt)
    samples = capture_arms(backend.env.robot, stamp)
    q, fingers, frames, tools = read_snapshot(
        Snapshot(stamp, samples, {}, backend.world_task), backend.arms
    )
    for arm, value in zip(backend.arms, fingers):
        arm.grip.measure(value)
    guard = getattr(backend, "guard", None)
    world = guard.collision if guard is not None else backend.collision
    tracker = world.tracker
    if tracker.last_stamp != stamp:
        raise G2Error(
            Status.UNSYNCED,
            "candidate stop lacks current physical contact sample",
            geometry="actual",
        )
    scene = getattr(backend, "actual_scene", backend.env.scene)
    block = getattr(backend.env.box, "actor", backend.env.box).get_pose()
    block_task = inverse_pose(backend.world_task) @ pose_from_wxyz(block.p, block.q)
    # Re-run the unchanged contact protection on a copy, including same-tick
    # contacts. The live tracker's anchors, clock and holder counts never change.
    shadow = deepcopy(tracker)
    shadow.last_stamp = None
    shadow.observe(stamp, scene.get_contacts(), tools, block_task)
    world.verify_contact_stop(q, fingers)
    drive_q, drive_fingers, velocity, measured_velocity = [], [], [], []
    for arm, side in zip(backend.arms, SIDES):
        entity = getattr(backend.env.robot, side + "_entity")
        lookup = {j.get_name(): j for j in entity.get_active_joints()}
        names = arm.chain.names + arm.grip.names
        drive_q.append(
            [float(lookup[n].get_drive_target()[0]) for n in arm.chain.names]
        )
        drive_fingers.append(
            [float(lookup[n].get_drive_target()[0]) for n in arm.grip.names]
        )
        velocity.append(
            [float(lookup[n].get_drive_velocity_target()[0]) for n in names]
        )
        measured_velocity.append(
            finite_array(entity.get_qvel(), (len(entity.get_active_joints()),)).tolist()
        )
        if held:
            if any(float(lookup[n].get_drive_velocity_target()[0]) != 0 for n in names):
                raise G2Error(
                    Status.UNSYNCED,
                    "candidate hold velocity readback failed",
                    geometry="actual",
                )
            i = SIDES.index(side)
            if not np.array_equal(np.array(drive_q[-1]), q[i]):
                raise G2Error(
                    Status.UNSYNCED,
                    "candidate hold position readback failed",
                    geometry="actual",
                )
    expected_fingers = fingers.copy()
    if getattr(backend, "contact_drive", None) is not None:
        for i, side in enumerate(SIDES):
            if side in tracker.holders:
                expected_fingers[i] = drive_fingers[i]
    if held and not np.array_equal(np.asarray(drive_fingers), expected_fingers):
        raise G2Error(
            Status.UNSYNCED, "candidate hold finger readback failed", geometry="actual"
        )
    targets = np.array(
        [
            f @ a.chain.fk(v) @ a.offset()
            for f, a, v in zip(frames, backend.arms, drive_q)
        ]
    )
    if guard is not None:
        guard.tracking(targets, tools)
    else:
        errors = [pose_error(a, b) for a, b in zip(targets, tools)]
        if any(
            np.linalg.norm(e[:3]) > backend.position_tol
            or np.rad2deg(np.linalg.norm(e[3:])) > backend.angle_tol
            for e in errors
        ):
            raise G2Error(
                Status.BACKEND_ERROR,
                "candidate stop actual tracking blocked",
                geometry="actual",
            )
    if backend.tick != tick:
        raise G2Error(
            Status.UNSYNCED, "candidate stop advanced physics", geometry="actual"
        )
    state = dict(
        q=q.tolist(),
        fingers=fingers.tolist(),
        bases=np.asarray(frames).tolist(),
        task_objects={
            name: pose.tolist()
            for name, pose in (
                capture_task_objects(backend.env, backend.world_task) or {}
            ).items()
        },
        block=block_task.tolist(),
        measured_velocity=measured_velocity,
        episode=stamp.episode,
        tick=tick,
    )
    return dict(
        kind="candidate_rejection",
        finger_drive_m=drive_fingers,
        expected_hold_finger_m=expected_fingers.tolist(),
        episode=stamp.episode,
        tick=tick,
        state_sha256=sha256(
            json.dumps(state, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        terminal_checks_passed=True,
        hold_verified=held,
        no_physics_advance=True,
    )
