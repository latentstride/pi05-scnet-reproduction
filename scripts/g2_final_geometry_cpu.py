"""Bounded zero-step FCL replay of captured native control failures.

TOPP is reconstructed from the recorded action-start drive and frozen move groups.
Every reconstructed drive is checked against the subsequent recorded steps
before the missing (rejected) drive is evaluated. No physics or model is run.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import json
import signal
import resource
import numpy as np
from g2e_static_guard_cpu import build
from g2e_native_guard import NativeGuard
from g2_adapter import G2Error, deadline
from g2_robotwin_backend import pose7


def drive_sequence(world, audit, start, requested, previous_request, initial_drive):
    from mplib.sapien_utils import SapienPlanner

    positions = []
    for i, group in enumerate(audit["move_group"]):
        planner = SapienPlanner(world.base.world, group)
        try:
            _, pos, _, _, _ = planner.TOPP(
                np.vstack([start[i], requested[i, :6]]), 1 / 250, verbose=False
            )
            if len(pos) == 0:
                raise ValueError("empty TOPP")
        except Exception:
            pos = np.repeat(start[i][None, :], 50, axis=0)
        positions.append(pos)
    lengths = [len(p) for p in positions]
    grippers = [
        np.linspace(previous_request[i], requested[i, 6], lengths[i] + 1)[1:]
        for i in range(2)
    ]
    indices = [0, 0]
    q = start.copy()
    g = initial_drive.copy()
    while any(indices[i] < lengths[i] for i in range(2)):
        for i, j in ((0, 1), (1, 0)):
            if (
                indices[i] < lengths[i]
                and indices[i] / lengths[i] <= indices[j] / lengths[j]
            ):
                q[i] = positions[i][indices[i]]
                value = float(np.clip(grippers[i][indices[i]], 0, 1))
                old = float(np.clip((g[i, 0] + 0.01) / 0.055, 0, 1))
                if value - old > 0.1:
                    value = old + 0.1
                g[i] = float(np.float32(-0.01 + 0.055 * value))
                indices[i] += 1
        yield q.copy(), g.copy()


def run(args):
    import sapien

    resource.setrlimit(resource.RLIMIT_CPU, (150, 155))
    signal.alarm(210)
    args.output.mkdir(parents=True, exist_ok=False)
    audit = next(
        r
        for r in json.loads((args.release / "asset-audit.json").read_text())["robots"]
        if r["robot"] == "aloha-agilex"
    )
    report = {}
    for directory in sorted(args.clips.iterdir()):
        if (
            not directory.is_dir()
            or "seed1-native-w2" in directory.name
            or "seed1-native-w31" in directory.name
        ):
            continue
        data = json.loads((directory / "last-action.json").read_text())
        rows = data["rows"]
        before = rows[0]["previous"]
        request = rows[1]
        laststep = [r for r in rows if r.get("physics_step_count") == 1][-1]
        lastcontact = [r for r in rows if r.get("kind") == "native_contact_sample"][-1]
        historical = [r for r in rows if r.get("kind") == "native_guard_rejection"][-1]
        seed = 0 if "seed0" in directory.name else 1
        cfg = json.loads(
            (args.release / "configs" / f"g2e-seed{seed}-native-w1.json").read_text()
        )
        initial = json.loads((args.raw / directory.name / "initial.json").read_text())
        cfg = deepcopy(cfg)
        initial = deepcopy(initial)
        for i, side in enumerate(("left", "right")):
            initial["actual_arm_qpos_rad"][side] = before["step"]["actual_arm"][i]
            cfg["reset_fixture"]["robots"]["aloha-agilex"][side]["fingers_m"] = before[
                "step"
            ]["actual_fingers"][i]
        initial["objects_task"]["block"] = pose7(
            np.array(before["contact"]["block_pose_task"])
        )
        out = args.output / directory.name
        out.mkdir()
        scene, arms, world, q, g = build(cfg, args.release, args.assets, initial, out)
        # The candidate world is updated from the LAST actual block pose. It
        # must not remain at reset or at policy-query time.
        block = next(a for a in scene.get_all_actors() if a.name == "box")
        p = pose7(np.array(lastcontact["block_pose_task"]))
        block.set_pose(sapien.Pose(p[:3], p[3:]))
        world.refresh()
        requests = json.loads(
            (args.requests / directory.name / "requests.json").read_text()
        )
        previous = (
            requests[-2]["normalized_qpos_request"]
            if len(requests) > 1
            else [*([0.0] * 6), 1.0, *([0.0] * 6), 1.0]
        )
        prior = np.asarray(previous).reshape(2, 7)[:, 6]
        sequence = drive_sequence(
            world,
            audit,
            np.array(before["step"]["target_arm"]),
            np.asarray(request["normalized_qpos_request"]).reshape(2, 7),
            prior,
            np.array(before["contact"]["target_fingers_drive_m"]),
        )
        seen = [r for r in rows if r.get("physics_step_count") == 1]
        contacts = {
            r["tick"]: r for r in rows if r.get("kind") == "native_contact_sample"
        }
        mismatch = 0.0
        gmismatch = 0.0
        for step in seen:
            dq, dg = next(sequence)
            mismatch = max(
                mismatch, float(np.max(abs(dq - np.array(step["target_arm"]))))
            )
            gmismatch = max(
                gmismatch,
                float(
                    np.max(
                        abs(
                            dg
                            - np.array(contacts[step["tick"]]["target_fingers_drive_m"])
                        )
                    )
                ),
            )
        drive, driveg = next(sequence)
        actual = np.asarray(laststep["actual_arm"])
        fingers = np.asarray(laststep["actual_fingers"])
        world.set_start(fingers, driveg)
        clear = world.measured_clear(actual, fingers)
        measured_collisions = world.last_collisions
        guard = NativeGuard(world, arms, 0.02, 0.03, 10.0)
        guard.previous_drive = np.array(lastcontact["target_fingers_drive_m"])
        rejection = None
        try:
            guard.candidate(actual, fingers, drive, driveg, deadline(20))
        except G2Error as error:
            rejection = dict(
                code=error.status.value, detail=str(error), diagnostic=error.diagnostic
            )
        exact = mismatch < 1e-6 and gmismatch < 1e-7
        depths = []
        for fraction in np.linspace(
            0, 1, max(1, int(np.ceil(np.max(abs(drive - actual)) / 0.02))) + 1
        ):
            sample = actual + (drive - actual) * fraction
            depths.append(
                dict(
                    fraction=float(fraction),
                    left_finger1_depth_m=world.finger_depth(sample, fingers, 0, 1),
                )
            )
        # Safe progress is only reported if the original exact request passes;
        # no model action, threshold, collision pair or object pose is altered.
        report[directory.name] = dict(
            physics_steps=0,
            gpu_used=False,
            historical_rejection=historical,
            reconstructed_rejection=rejection,
            recorded_drive_max_error_rad=mismatch,
            recorded_finger_drive_max_error_m=gmismatch,
            reconstructed_drive_validated=exact,
            measured_clear=clear,
            measured_collisions=measured_collisions,
            safe_original_request_proven=bool(exact and clear and rejection is None),
            actual_arm=actual.tolist(),
            actual_fingers=fingers.tolist(),
            drive_arm=drive.tolist(),
            drive_fingers=driveg.tolist(),
            block_pose_world=lastcontact["block_pose_task"],
            depth_samples=depths,
            asset_urdf_sha256=audit["urdf_sha256"],
            scene_world_refreshed=True,
        )
        print(
            directory.name,
            "exact",
            exact,
            "actual_clear",
            clear,
            "rejected",
            rejection and rejection["detail"],
            flush=True,
        )
    (args.output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("release", "assets", "clips", "raw", "requests", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
