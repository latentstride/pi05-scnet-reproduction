"""Three archived right-only-hold checkpoints through the same placement geometry.

No physics. Plans approach, descent, opening and retreat. Hypothetical release is
explicit and cannot establish task success, support impulses or cumulative slip.
"""

import argparse, json
from dataclasses import asdict
from pathlib import Path
from time import monotonic
import zipfile
import numpy as np
from g2_static_failure_replay import build, matrix
from g2_adapter import deadline, G2Error
from g2_motion import QuinticMotion
from g2_remaining_geometry import refined_pair
from g2_robotwin_backend import pose7
from robotwin_control_geometry import inverse_pose, Status


def run(a):
    import sapien

    a.output.mkdir(parents=True, exist_ok=False)
    report = {}
    with zipfile.ZipFile(a.selection) as z:
        for ep in range(3):
            start = monotonic()
            check = deadline(55)
            name = f"seed{min(ep, 1)}-B-arx-w1"
            cfg = json.loads((a.release / "configs" / f"{name}.json").read_text())
            initial = json.loads(z.read(name + "/initial-readback.json"))
            snap = next(
                x
                for x in json.loads(z.read(name + "/events-selected.json"))
                if x.get("kind") == "snapshot"
            )
            directory = a.expert / f"episode{ep}"
            d = np.load(directory / "physics-trace.npz", allow_pickle=False)
            metadata = json.loads((directory / "initial.json").read_text())
            monitor = json.loads((directory / "contact-monitor.json").read_text())
            if monitor["failure"] is not None:
                raise ValueError("centered expert contact monitor did not pass")
            transition = next(e for e in monitor["events"] if e["after"] == ["right"])
            stages = json.loads((directory / "stages.json").read_text())
            checkpoint = next(s["tick"] for s in stages if s["stage"] == 11)
            index = int(np.searchsorted(d["sample_ticks"], checkpoint))
            tick = int(d["sample_ticks"][index])
            state = d["actual_native"][index].reshape(2, 8)
            q, g = state[:, :6], state[:, 6:]
            drive = d["drive_native"][index].reshape(2, 8)[:, 6:]
            for s, b in zip(snap["actual"], metadata["robot_bases"]):
                s["world_base"] = matrix(b)
            initial["objects_task"]["target"] = metadata["target_pose"]
            initial["objects_task"]["block"] = d["block_pose"][index].tolist()
            out = a.output / f"episode{ep}"
            out.mkdir()
            scene, arms, source, frames, bases, world, proxy, _, _ = build(
                cfg,
                a.release / "asset-audit.json",
                a.assets,
                snap,
                initial,
                matrix(initial["objects_task"]["block"]),
                out,
            )
            block = next(x for x in scene.get_all_actors() if x.name == "box")
            block_pose = matrix(initial["objects_task"]["block"])
            tools = world.tools(q)
            relative = inverse_pose(tools[1]) @ block_pose
            world.tracker.holders = {"right": relative.copy()}
            world.tracker.measured_attachments = {"right": relative.copy()}
            world.tracker.block_pose = block_pose.copy()
            limits = cfg["control_limits"]
            motion = QuinticMotion(
                cfg["control_period_s"],
                limits["arm_velocity_rad_s"],
                limits["arm_acceleration_rad_s2"],
                limits["finger_velocity_m_s"],
                limits["finger_acceleration_m_s2"],
                cfg["collision_joint_step_rad"],
                cfg["collision_finger_step_m"],
                limits["source"],
                "archived_right_hold_CPU_FCL",
                max_ticks=10000,
            )
            world.set_start(g, drive)
            phases = []
            attempts = []

            def phase(label, goal_tools, grip):
                nonlocal q, g, drive, block_pose
                goal, res, meta = refined_pair(
                    arms, frames, goal_tools, q, grip, world, check
                )
                path = motion.plan(q, g, [(goal, grip)], world, check)
                np.savez_compressed(
                    out / f"{label}.npz",
                    **{k: v for k, v in asdict(path).items() if v is not None},
                )
                q = path.q[-1].copy()
                g = path.predicted_fingers[-1].copy()
                drive = path.fingers[-1].copy()
                if world.tracker.carrying:
                    block_pose = world.tools(q)[1] @ relative
                    value = pose7(block_pose)
                    block.set_pose(sapien.Pose(value[:3], value[3:]))
                    world.tracker.block_pose = block_pose.copy()
                world.refresh()
                world.set_start(g, drive)
                phases.append(
                    dict(
                        phase=label,
                        control_ticks=len(path.q),
                        max_velocity_rad_s=float(abs(path.velocity).max()),
                    )
                )
                return path

            try:
                tray = matrix(metadata["target_pose"])
                approach = np.asarray(tools).copy()
                approach[1, :2, 3] += tray[:2, 3] - block_pose[:2, 3]
                approach[1, 2, 3] += 0.012
                phase("approach", approach, drive)
                accepted = False
                for height in (0.005, 0.007, 0.009, 0.012, 0.015):
                    check()
                    tools = np.asarray(world.tools(q))
                    low = block_pose[2, 3] - np.abs(block_pose[2, :3]) @ np.array(
                        [0.03, 0.03, 0.1]
                    )
                    goal_tools = tools.copy()
                    goal_tools[1, 2, 3] += tray[2, 3] + 0.005 + height - low
                    try:
                        goal, res, meta = refined_pair(
                            arms, frames, goal_tools, q, drive, world, check
                        )
                        descent = motion.plan(q, g, [(goal, drive)], world, check)
                        old_start, old_drive = (
                            world.initial_fingers.copy(),
                            world.initial_drive.copy(),
                        )
                        try:
                            world.set_start(
                                descent.predicted_fingers[-1], descent.fingers[-1]
                            )
                            opening = descent.fingers[-1].copy()
                            opening[1] = 0.044
                            motion.plan(
                                descent.q[-1],
                                descent.predicted_fingers[-1],
                                [(descent.q[-1], opening)],
                                world,
                                check,
                            )
                        finally:
                            world.set_start(old_start, old_drive)
                        attempts.append(dict(height_m=height, accepted=True))
                        accepted = True
                        break
                    except G2Error as exc:
                        attempts.append(
                            dict(
                                height_m=height,
                                accepted=False,
                                status=exc.status.value,
                                detail=str(exc),
                            )
                        )
                if not accepted:
                    raise G2Error(Status.PLAN_FAILED, "no descent plus opening path")
                phase("descent", goal_tools, drive)
                opening = drive.copy()
                opening[1] = 0.044
                phase("opening", np.asarray(world.tools(q)), opening)
                # Planning branch only: assume physical release is confirmed at
                # runtime. No synthetic ContactTracker.observe event is emitted.
                world.tracker.holders.clear()
                world.tracker.measured_attachments.clear()
                retreat = np.asarray(world.tools(q))
                retreat[1, 2, 3] += 0.035
                phase("retreat", retreat, opening)
                report[str(ep)] = dict(
                    accepted=True,
                    release_height_m=height,
                    checkpoint_tick=tick,
                    original_right_only_transition_tick=transition["tick"],
                    phases=phases,
                    attempts=attempts,
                )
            except G2Error as exc:
                report[str(ep)] = dict(
                    accepted=False,
                    status=exc.status.value,
                    detail=str(exc),
                    diagnostic=exc.diagnostic,
                    checkpoint_tick=tick,
                    phases=phases,
                    attempts=attempts,
                )
            report[str(ep)].update(
                physics_steps=0,
                gpu_used=False,
                task_success=None,
                wall_s=monotonic() - start,
                scope="static attachment geometry; runtime release/contact loss/support and original slip anchor are not dynamically verified",
            )
            (a.output / "placement.json").write_text(
                json.dumps(report, default=lambda v: v.tolist(), indent=2) + "\n"
            )
            print("PLACEMENT", ep, report[str(ep)]["accepted"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("release", "selection", "assets", "expert", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    run(p.parse_args())
