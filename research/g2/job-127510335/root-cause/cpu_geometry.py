"""Rebuild the author scene on CPU and audit saved poses without advancing them."""

import argparse
import json
import os
from pathlib import Path
from time import monotonic

from mplib.collision_detection.fcl import FCLObject
import numpy as np
from sapien import Pose

from g2_adapter import G2Error
from g2_author_h50_limits import BUDGET, TrialClock
from g2_author_native_h50 import author_scene
from g2_author_native_h50_cpu import Headless
from g2_robotwin_backend import pose7


def archived(run):
    trial = run / "trials/author-native-h50-seed100000"
    receipt = json.loads((trial / "receipt.json").read_bytes())
    chunk = np.load(trial / "chunk-00.npy", allow_pickle=False)
    stop = receipt["physics_steps"]
    start = last = contact = None
    with (trial / "events.jsonl").open() as stream:
        for line in stream:
            if '"tick": 4699,' in line or f'"tick": {stop},' in line:
                row = json.loads(line)
                if row.get("kind") is None:
                    if row["tick"] == 4699:
                        start = row
                    elif row["tick"] == stop:
                        last = row
                elif row.get("kind") == "native_contact_sample" and row["tick"] == stop:
                    contact = row
    if any(value is None for value in (start, last, contact)):
        raise ValueError("required source snapshots missing")
    return receipt, chunk, start, last, contact


def reconstruct(backend, inputs):
    receipt, chunk, start, last, contact = inputs
    d = receipt["result"]["failure"]["diagnostic"]
    world = backend.guard.collision
    block = world.base.world.get_object(world.block_name)
    original_pose = block.pose
    p = np.asarray(pose7(contact["block_pose_task"]))
    world.base.world.add_object(
        FCLObject(world.block_name, Pose(p[:3], p[3:]), block.shapes, block.shape_poses)
    )
    a, q = np.asarray(d["actual_arm"]), np.asarray(d["drive_arm"])
    f, drive = np.asarray(d["actual_fingers"]), np.asarray(d["drive_fingers"])
    projected = np.array(
        [
            np.clip(x, arm.grip.limits[:, 0], arm.grip.limits[:, 1])
            for arm, x in zip(backend.arms, drive)
        ]
    )
    probe = projected.copy()
    probe[0, 0] = f[0, 0]
    desired = chunk[42].reshape(2, 7)[:, :6]
    depths = {}
    try:
        for name, arm, finger in (
            ("actual", a, f),
            ("actual_arm_same_hybrid_fingers", a, probe),
            ("drive_arm_actual_fingers", q, f),
            ("drive_arm_guard_exact_hybrid", q, probe),
            ("drive_arm_requested", q, projected),
            ("raw_action_endpoint_actual_fingers", desired, f),
            ("raw_action_endpoint_requested", desired, projected),
        ):
            depths[name] = world.finger_depth(arm, finger, 0, 0)
        fractions = [
            world.finger_depth(a + (q - a) * x, probe, 0, 0)
            for x in np.linspace(0, 1, 11)
        ]
        sensitivity = []
        for dx in (-0.0001, -0.00005, 0.0, 0.00005, 0.0001):
            for dy in (-0.0001, -0.00005, 0.0, 0.00005, 0.0001):
                varied = p.copy()
                varied[:2] += [dx, dy]
                world.base.world.add_object(
                    FCLObject(
                        world.block_name,
                        Pose(varied[:3], varied[3:]),
                        block.shapes,
                        block.shape_poses,
                    )
                )
                sensitivity.append([dx, dy, world.finger_depth(q, probe, 0, 0)])
    finally:
        world.base.world.add_object(
            FCLObject(world.block_name, original_pose, block.shapes, block.shape_poses)
        )
    topp = {}
    for i, side in enumerate(("left", "right")):
        planner = getattr(backend.env.robot, f"{side}_mplib_planner")
        path = np.stack([start["target_arm"][i], desired[i]])
        times, positions, _, _, duration = planner.TOPP(path, 1 / 250, verbose=False)
        errors = np.max(abs(positions - q[i]), axis=1)
        idx = int(errors.argmin())
        topp[side] = dict(
            samples=len(positions),
            nearest_drive_index=idx,
            nearest_drive_max_error_rad=float(errors[idx]),
            nearest_drive_time_s=float(times[idx]),
            duration_s=float(duration),
        )
    return dict(
        finger_link=world.finger_links[0][0],
        block_actor=backend.env.box.actor.name,
        block_fcl_name=world.block_name,
        finger_links=world.finger_links,
        block_collision_shapes=[type(s).__name__ for s in block.shapes],
        collision_penetration_limit_m=world.limits.penetration_m,
        collision_joint_step_rad=backend.guard.joint_step,
        recorded_depth_m=d["depth_m"],
        reconstructed_depths_m=depths,
        reconstructed_segment_hybrid_depths_m=fractions,
        offline_block_xy_perturbation_m=sensitivity,
        topp_reconstructed_from_action_43_logged_drive_start=topp,
        exact_last_physics_q_matches_stop=np.array_equal(last["actual_arm"], a),
        exact_last_physics_fingers_match_stop=np.array_equal(last["actual_fingers"], f),
        reconstruction_physics_steps=backend.tick,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pace", "release", "run", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("this audit requires hidden CUDA devices")
    args.output.mkdir()
    inputs = archived(args.run)
    began = monotonic()
    clock = TrialClock(BUDGET, began, began + BUDGET.hard_s)
    with author_scene(
        args.pace,
        args.release,
        args.release / "author-h50-reset-reference.json",
        "cpu-author-127510335",
        lambda event: None,
        clock,
        configure_cpu=Headless(args.output),
    ) as (backend, _):
        outcome = reconstruct(backend, inputs)
        (args.output / "geometry.json").write_text(json.dumps(outcome, indent=2) + "\n")
        print("CPU_GEOMETRY_READY", flush=True)
        if args.replay:
            replay = {"kind": "recorded GPU action replay on CPU PhysX; no inference"}
            try:
                for row in inputs[1][:43]:
                    backend.take_action(row)
            except G2Error as error:
                replay.update(
                    code=error.status.value,
                    detail=str(error),
                    diagnostic=error.diagnostic,
                )
            finally:
                backend.hold()
            replay.update(
                completed_actions=backend.completed, physics_steps=backend.tick
            )
            (args.output / "replay.json").write_text(
                json.dumps(replay, indent=2) + "\n"
            )
            print("CPU_REPLAY_READY", flush=True)


if __name__ == "__main__":
    main()
