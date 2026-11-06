"""Recheck four real failure snapshots, exact refinements and explicit safe progress; zero steps."""

import argparse, json
from pathlib import Path
from dataclasses import asdict
from time import monotonic
import zipfile
import numpy as np
from g2_static_failure_replay import build, matrix
from g2_adapter import G2Error, deadline
from g2_motion import QuinticMotion
from g2_remaining_geometry import refined_pair, bounded_progress


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    report = {}
    with zipfile.ZipFile(a.selection) as z:
        for name in sorted(
            n
            for n in json.loads((a.release / "run_lock.json").read_text())["bindings"]
            if "B-arx" in n
        ):
            out = a.output / name
            out.mkdir()
            events = json.loads(z.read(name + "/events-selected.json"))
            steps = json.loads(z.read(name + "/control-samples.json"))
            failure = [e for e in events if e.get("status") == "collision"][-1]
            snaps = [e for e in events if e.get("kind") == "snapshot"]
            snap = snaps[-1]
            cfg = json.loads((a.release / "configs" / f"{name}.json").read_text())
            initial = json.loads(z.read(name + "/initial-readback.json"))
            state = steps[-1]["contact_state"] if steps else None
            block = (
                np.array(state["block_pose_task"])
                if state
                else matrix(initial["objects_task"]["block"])
            )
            scene, arms, source, frames, bases, target, proxy, q, g = build(
                cfg, a.release / "asset-audit.json", a.assets, snap, initial, block, out
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
                "remaining_CPU_real_FCL",
                max_ticks=cfg["max_plan_ticks"],
            )
            check = deadline(45)
            started = monotonic()
            case = dict(
                original_phase=failure["phase"],
                physics_steps=0,
                model_started=False,
                gpu_used=False,
            )
            if failure["phase"] == "observation":
                tools = np.array(
                    [
                        f @ arm.chain.fk(v) @ arm.offset()
                        for arm, f, v in zip(arms, frames, q)
                    ]
                )
                fractions = [
                    arm.grip.measure(v)["normalized"] for arm, v in zip(arms, g)
                ]
                fingers = np.array(
                    [arm.grip.target(v) for arm, v in zip(source, fractions)]
                )
                proxy.set_start(fingers)
                seed = np.asarray(
                    [e for e in events if e.get("kind") == "policy_input"][-1]["state"]
                ).reshape(2, 7)[:, :6]
                delta = np.asarray(cfg["source_proxy_velocity_rad_s"]) * (
                    snap["stamp"]["sim_s"] - snaps[-2]["stamp"]["sim_s"]
                )
                active = (source, bases, tools, seed, fingers, proxy)
            else:
                req = failure["requests"][0]
                tools = np.asarray(req["task_tools"])
                fingers = np.asarray(req["fingers"])
                seed = np.asarray(req["ik_start"])
                delta = None
                active = (arms, frames, tools, seed, fingers, target)
            try:
                solved, residual, meta = refined_pair(*active, check, delta)
                case["exact_refinement"] = dict(
                    accepted=True, q=solved, residual=residual, metadata=meta
                )
            except G2Error as exc:
                case["exact_refinement"] = dict(
                    accepted=False,
                    status=exc.status.value,
                    detail=str(exc),
                    diagnostic=exc.diagnostic,
                )
            if failure["phase"] == "observation":
                # Recover the actual ARX towards the last accepted source pose;
                # never invent an observation at an unachieved virtual pose.
                recovery_tools = np.array(
                    [
                        b @ arm.chain.fk(v) @ arm.offset()
                        for arm, b, v in zip(source, bases, seed)
                    ]
                )
                target.set_start(g)
                try:
                    goal, residual, meta = refined_pair(
                        arms, frames, recovery_tools, q, g, target, check
                    )
                    path = motion.plan(q, g, [(goal, g)], target, check)
                    case["recovery"] = dict(
                        accepted=True,
                        mode="physical_return_to_last_source_pose",
                        goal_q=goal,
                        control_ticks=len(path.q),
                        residual=residual,
                    )
                    np.savez_compressed(
                        out / "recovery-path.npz",
                        **{k: v for k, v in asdict(path).items() if v is not None},
                    )
                except G2Error as exc:
                    case["recovery"] = dict(
                        accepted=False,
                        status=exc.status.value,
                        detail=str(exc),
                        diagnostic=exc.diagnostic,
                    )
            else:
                target.set_start(g)
                try:
                    result = bounded_progress(
                        arms, frames, tools, q, g, fingers, target, motion, check
                    )
                    path = result.pop("path")
                    np.savez_compressed(
                        out / "recovery-path.npz",
                        **{k: v for k, v in asdict(path).items() if v is not None},
                    )
                    case["recovery"] = dict(
                        accepted=True,
                        mode="explicit_changed_command",
                        control_ticks=len(path.q),
                        **result,
                    )
                except G2Error as exc:
                    case["recovery"] = dict(
                        accepted=False,
                        status=exc.status.value,
                        detail=str(exc),
                        diagnostic=exc.diagnostic,
                    )
            if case["recovery"]["accepted"]:
                end_q = np.asarray(case["recovery"]["goal_q"])
                end_tools = np.array(
                    [
                        f @ arm.chain.fk(v) @ arm.offset()
                        for arm, f, v in zip(arms, frames, end_q)
                    ]
                )
                end_grip = (
                    path.predicted_fingers[-1]
                    if path.predicted_fingers is not None
                    else path.fingers[-1]
                )
                normalized = [
                    arm.grip.measure(v)["normalized"] for arm, v in zip(arms, end_grip)
                ]
                source_fingers = np.array(
                    [arm.grip.target(v) for arm, v in zip(source, normalized)]
                )
                proxy.set_start(source_fingers)
                source_seed = np.asarray(
                    [e for e in events if e.get("kind") == "policy_input"][-1]["state"]
                ).reshape(2, 7)[:, :6]
                try:
                    mapped, residual, meta = refined_pair(
                        source,
                        bases,
                        end_tools,
                        source_seed,
                        source_fingers,
                        proxy,
                        check,
                        np.asarray(cfg["source_proxy_velocity_rad_s"])
                        * ((len(path.q) - 1) * path.period_s),
                    )
                    case["recovery"]["source_roundtrip"] = dict(
                        accepted=True, q=mapped, residual=residual
                    )
                except G2Error as exc:
                    case["recovery"]["source_roundtrip"] = dict(
                        accepted=False, status=exc.status.value, detail=str(exc)
                    )
            case["wall_s"] = monotonic() - started
            report[name] = case
            (a.output / "replay.json").write_text(
                json.dumps(report, default=lambda v: v.tolist(), indent=2) + "\n"
            )
            print(
                name,
                "exact",
                case["exact_refinement"]["accepted"],
                "recovery",
                case["recovery"]["accepted"],
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("release", "selection", "assets", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    run(p.parse_args())
