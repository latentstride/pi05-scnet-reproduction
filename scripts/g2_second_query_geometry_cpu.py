"""Zero-step FCL attribution of saved 127192740 goals; no model or renderer."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from g2_adapter import deadline
from g2_native_expert_cpu import unpack_trace
from g2e_static_guard_cpu import build


def run(args):
    from importlib.metadata import version
    import g2_contact

    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.raw / "trials/unified-seed0-native-w50/episode"
    selected = json.loads((args.raw / "selected-events.json").read_text())
    last = selected["last"]["executed_step"]["event"]
    contact = selected["last"]["native_contact_sample"]["event"]
    cfg_path = args.release / "configs/unified-seed0-native-w50.json"
    config = json.loads(cfg_path.read_text())
    initial = json.loads((raw / "initial-readback.json").read_text())
    snapshot = next(
        r["event"]
        for r in selected["selected"]
        if r["event"].get("kind") == "snapshot" and r["event"]["stamp"]["tick"] == 3575
    )
    for i, side in enumerate(("left", "right")):
        initial["actual_arm_qpos_rad"][side] = last["actual_arm"][i]
        config["reset_fixture"]["robots"]["aloha-agilex"][side]["fingers_m"] = last[
            "actual_fingers"
        ][i]
    from g2_robotwin_backend import pose7

    initial["objects_task"]["block"] = pose7(np.asarray(contact["block_pose_task"]))
    scene, arms, world, actual, fingers = build(
        config, args.release, args.assets, initial, args.output
    )
    drive = np.asarray(contact["target_fingers_drive_m"])
    world.set_start(fingers, drive)
    measured_clear = world.measured_clear(actual, fingers)
    actual_collisions = deepcopy(world.last_collisions)
    result = json.loads((raw / "result.json").read_text())
    nested = result["failure"]["diagnostic"]["attempts"][0]["diagnostic"][
        "collision_examples"
    ][0]
    q2 = np.load(raw / "plan-00002.npy", allow_pickle=False)[0].reshape(2, 7)
    expert = unpack_trace(args.expert_trace)["target"]
    rows = [
        ("model_query2_row0", q2[:, :6], q2[:, 6]),
        ("logged_plus_z_refinement", np.asarray(nested["candidate_q"]), q2[:, 6]),
        (
            "expert_approach_endpoint_576",
            expert[576].reshape(2, 7)[:, :6],
            expert[576, [6, 13]],
        ),
        (
            "expert_pregrasp_endpoint_709",
            expert[709].reshape(2, 7)[:, :6],
            expert[709, [6, 13]],
        ),
    ]
    checks = []
    for label, q, grip in rows:
        requested = np.repeat((-0.01 + 0.055 * grip)[:, None], 2, axis=1)
        requested = np.asarray(
            [
                np.clip(g, arm.grip.limits[:, 0], arm.grip.limits[:, 1])
                for arm, g in zip(arms, requested)
            ]
        )
        world.set_start(fingers, drive)
        clear = world.ik_clear(q, requested, deadline(20))
        checks.append(
            dict(
                label=label,
                clear=bool(clear),
                collisions=deepcopy(world.last_collisions),
                actual_aperture_left_finger_depth_m=[
                    world.finger_depth(q, fingers, 0, i) for i in (0, 1)
                ],
                requested_gripper=grip.tolist(),
                goal_q=q.tolist(),
            )
        )
    assert measured_clear
    assert not checks[0]["clear"] and not checks[1]["clear"]
    assert (
        abs(
            checks[1]["actual_aperture_left_finger_depth_m"][1]
            - nested["collisions"][0]["depth_m"]
        )
        < 1e-6
    )
    report = dict(
        job="127192740",
        tick=3575,
        gpu_used=False,
        model_used=False,
        physics_steps=0,
        renderer_created=False,
        scope="same captured object/robot state; endpoints only, not trajectories or dynamics",
        restored_snapshot_tick=snapshot["stamp"]["tick"],
        measured_clear=bool(measured_clear),
        actual_collisions=actual_collisions,
        checks=checks,
        historical_refinement_depth_m=nested["collisions"][0]["depth_m"],
        collision_reproduced_within_1um=True,
        config_sha256=sha256(cfg_path.read_bytes()).hexdigest(),
        expert_trace_sha256=sha256(args.expert_trace.read_bytes()).hexdigest(),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        contact_source_sha256=sha256(
            Path(g2_contact.__file__).read_bytes()
        ).hexdigest(),
        sapien_version=version("sapien"),
        mplib_version=version("mplib"),
        caveat="Reused CPU scene builder has conservative table legs; the reported finger/block tests use actual audited meshes and captured block pose. No expert continuation from the policy state has been executed.",
    )
    (args.output / "geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                measured_clear=measured_clear,
                checks=[(r["label"], r["clear"]) for r in checks],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "release", "assets", "expert-trace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
