"""Bounded task-space translations of rejected free-space endpoints; CPU only."""

import argparse
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from g2_adapter import G2Error, deadline
from g2_kinematics import pose_error
from g2_final_geometry_sweep_cpu import capture, snapshot
from g2_static_failure_replay import build
from g2_remaining_geometry import refined_pair
from g2_motion import QuinticMotion


def run(a):
    a.output.mkdir(exist_ok=False)
    audit = json.loads((a.release / "asset-audit.json").read_text())
    report = {}
    for robot in ["native", "arx"]:
        if robot == "arx":
            name, folder, cfg, initial, events = capture(a, 0, "arx-full-w4")
            failure = [
                r for r in events if r.get("kind") == "original_action_rejection"
            ][-1]
            at = next(
                r
                for r in events
                if r.get("kind") == "executed_step"
                and r["tick"] == failure["diagnostic"]["stamp"]["tick"]
            )
            q, g = np.array(at["actual_arm"]), np.array(at["actual_fingers"])
            block = np.array(at["contact_state"]["block_pose_task"])
            request = failure["diagnostic"]["requests"][-1]
            tools = np.array(request["task_tools"])
            fingers = np.array(request["fingers"])
            drive = np.array(at["target_fingers"])
            sample = snapshot(cfg, initial, q, g, audit)
        else:
            name, folder, ncfg, initial_native, events = capture(a, 0, "native-w50")
            request = next(
                r
                for r in events
                if r.get("kind") == "native_action_request"
                and r["action_index_in_episode"] == 26
            )
            at = next(
                r
                for r in events
                if r.get("kind") == "executed_step" and r["tick"] == request["tick"]
            )
            contact = next(
                r
                for r in events
                if r.get("kind") == "native_contact_sample"
                and r["tick"] == request["tick"]
            )
            q, g = np.array(at["actual_arm"]), np.array(at["actual_fingers"])
            block = np.array(contact["block_pose_task"])
            drive = np.array(contact["target_fingers_drive_m"])
            _, _, cfg, initial, arx_events = capture(a, 0, "arx-full-w4")
            dummy = next(r for r in arx_events if r.get("kind") == "executed_step")
            sample = snapshot(
                cfg,
                initial,
                np.array(dummy["actual_arm"]),
                np.array(dummy["actual_fingers"]),
                audit,
            )
        out = a.output / robot
        out.mkdir()
        scene, arms, source, frames, bases, world, proxy, _, _ = build(
            cfg, a.release / "asset-audit.json", a.assets, sample, initial, block, out
        )
        if robot == "native":
            arms, frames, world, cfg = source, bases, proxy, ncfg
            action = np.load(folder / "action-00001.npy", allow_pickle=False)[
                26
            ].reshape(2, 7)
            tools = np.array(
                [
                    f @ arm.chain.fk(v[:6]) @ arm.offset()
                    for f, arm, v in zip(frames, arms, action)
                ]
            )
            fingers = np.repeat(
                (-0.01 + 0.055 * np.clip(action[:, 6], 0, 1))[:, None], 2, axis=1
            )
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
            "bounded_endpoint_CPU",
            max_ticks=cfg["max_plan_ticks"],
        )
        world.set_start(g, drive)
        cases = []
        accepted = None
        check = deadline(140)
        for radius in [0.004, 0.008, 0.0115]:
            for axis in range(3):
                for sign in [-1, 1]:
                    candidate = tools.copy()
                    candidate[0, axis, 3] += sign * radius
                    case = dict(
                        offset_m=(candidate[0, :3, 3] - tools[0, :3, 3]).tolist()
                    )
                    try:
                        goal, res, meta = refined_pair(
                            arms, frames, candidate, q, fingers, world, check
                        )
                        executed = np.array(
                            [
                                f @ arm.chain.fk(v) @ arm.offset()
                                for f, arm, v in zip(frames, arms, goal)
                            ]
                        )
                        errors = [pose_error(t, s) for t, s in zip(executed, tools)]
                        changes = [float(np.linalg.norm(e[:3])) for e in errors]
                        if (
                            max(changes) > 0.012
                            or max(
                                float(np.rad2deg(np.linalg.norm(e[3:]))) for e in errors
                            )
                            > 1.0
                        ):
                            raise G2Error(
                                __import__(
                                    "robotwin_control_geometry"
                                ).Status.PLAN_FAILED,
                                "executed pose outside original 12mm/1deg bound",
                            )
                        path = motion.plan(q, g, [(goal, fingers)], world, check)
                        case.update(
                            accepted=True,
                            goal_q=goal,
                            residual=res,
                            executed_tools=executed,
                            pose_change_m=changes,
                            control_ticks=len(path.q),
                            original_goal_reached=False,
                        )
                        np.savez_compressed(
                            out / "path.npz",
                            **{k: v for k, v in asdict(path).items() if v is not None},
                        )
                        accepted = case
                    except G2Error as e:
                        case.update(
                            accepted=False,
                            status=e.status.value,
                            detail=str(e),
                            diagnostic=e.diagnostic,
                        )
                    cases.append(case)
                    if accepted:
                        break
                if accepted:
                    break
            if accepted:
                break
        report[robot] = dict(
            accepted=accepted,
            attempts=cases,
            physics_steps=0,
            gpu_used=False,
            scope="bounded changed endpoint, not original model target attainment",
        )
        (a.output / "projection.json").write_text(
            json.dumps(report, indent=2, default=lambda x: x.tolist()) + "\n"
        )
        print("BOUNDED_PROJECTION", robot, accepted is not None, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ["raw", "release", "assets", "output"]:
        p.add_argument("--" + n, type=Path, required=True)
    run(p.parse_args())
