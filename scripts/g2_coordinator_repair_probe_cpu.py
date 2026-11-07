"""Exact opt-in coordinator at captured ARX seed0 rejection, CPU geometry only."""

import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import json
import numpy as np
from g2_adapter import Adapter, Snapshot, Stamp, G2Error
from g2_capture import capture_arms
from g2_final_geometry_sweep_cpu import capture, snapshot
from g2_static_failure_replay import build, matrix
from g2_motion import QuinticMotion
from g2_remaining_router import JointRouter
from g2_remaining_coordinator import RecoveryCoordinator


def run(a):
    a.output.mkdir(exist_ok=False)
    audit = json.loads((a.release / "asset-audit.json").read_text())
    name, folder, cfg, initial, events = capture(a, 0, "arx-full-w4")
    failure = [r for r in events if r.get("kind") == "original_action_rejection"][
        a.checkpoint_index
    ]
    stamp = Stamp(**failure["diagnostic"]["stamp"])
    at = next(
        r
        for r in events
        if r.get("kind") == "executed_step" and r["tick"] == stamp.tick
    )
    requests = [
        r
        for r in events
        if r.get("kind") == "policy_request" and r["stamp"]["tick"] <= stamp.tick
    ]
    req = requests[-1]
    q, g = np.array(at["actual_arm"]), np.array(at["actual_fingers"])
    scene, arms, source, frames, bases, world, proxy, _, _ = build(
        cfg,
        a.release / "asset-audit.json",
        a.assets,
        snapshot(cfg, initial, q, g, audit),
        initial,
        np.array(at["contact_state"]["block_pose_task"]),
        a.output,
    )
    world.set_start(g, np.array(at["target_fingers"]))
    entities = {e.name: e for e in scene.get_all_articulations()}
    samples = capture_arms(
        NS(left_entity=entities["arx_left"], right_entity=entities["arx_right"]), stamp
    )
    arms = tuple(
        replace(arm, instance=sample.instance) for arm, sample in zip(arms, samples)
    )
    snap = Snapshot(stamp, samples, {}, matrix(cfg["world_task"]))
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
        "captured_checkpoint",
        max_ticks=cfg["max_plan_ticks"],
    )
    motion = JointRouter(motion, np.array([arm.chain.limits for arm in arms]))
    seed = np.array(req["state"]).reshape(2, 7)[:, :6].copy()
    adapter = Adapter(
        source,
        arms,
        bases,
        matrix(cfg["source_world_task"]),
        seed,
        np.array(cfg["source_proxy_velocity_rad_s"]),
        proxy,
        world,
        motion,
        grip_rule="clip",
        grip_max_clip=cfg["native_grip_max_clip"],
    )
    adapter.last_stamp = stamp
    proxy.set_start(
        np.array(
            [
                arm.grip.target(arm2.grip.measure(v)["normalized"])
                for arm, arm2, v in zip(source, arms, g)
            ]
        )
    )
    backend = NS(_drive_fingers=lambda: np.array(at["target_fingers"]))
    chunk = np.load(folder / f"action-{len(requests):05}.npy", allow_pickle=False)
    report = dict(physics_steps=0, model_inference=False, stamp=vars(stamp), cases={})
    for enabled in (False, True):
        coordinator = RecoveryCoordinator(adapter, backend, approach_repair=enabled)
        rows = []
        try:
            plan, receipt = coordinator.action_plan(
                chunk, snap, 4, lambda: 120, rows.append
            )
            case = dict(
                accepted=True,
                receipt=receipt,
                actions=len(plan.waypoint_ticks),
                control_ticks=len(plan.q),
                recoveries=coordinator.recoveries,
                source_seed_unchanged=np.array_equal(seed, adapter.seed),
                records=rows,
            )
        except G2Error as e:
            case = dict(
                accepted=False,
                status=e.status.value,
                detail=str(e),
                diagnostic=e.diagnostic,
                recoveries=coordinator.recoveries,
                records=rows,
            )
        report["cases"]["candidate" if enabled else "baseline"] = case
        print(
            "COORDINATOR_CPU", enabled, case["accepted"], case.get("detail"), flush=True
        )
        (a.output / "coordinator.json").write_text(
            json.dumps(report, indent=2, default=lambda x: x.tolist()) + "\n"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ["raw", "release", "assets", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--checkpoint-index", type=int, default=-1)
    run(p.parse_args())
