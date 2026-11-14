"""Planning admission against actual saved ARX prefixes; no policy or physical advance."""

import argparse
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
from dataclasses import replace
import json
import numpy as np
from g2_adapter import Adapter, Snapshot, Stamp, G2Error, deadline
from g2_capture import capture_arms
from g2_static_failure_replay import build, matrix
from g2_final_geometry_sweep_cpu import capture, snapshot
from g2_bounded_approach import checked_prefix, source_admissible
from g2_motion import Trajectory, QuinticMotion
from g2_remaining_router import JointRouter
from g2_remaining_coordinator import RecoveryCoordinator


def run(a):
    a.output.mkdir(exist_ok=False)
    audit = json.loads((a.release / "asset-audit.json").read_text())
    name, folder, cfg, initial, events = capture(a, 1, "arx-full-w4")
    requests = [r for r in events if r.get("kind") == "policy_request"]
    steps = {r["tick"]: r for r in events if r.get("kind") == "executed_step"}
    reports = []
    for query in [35, 39]:
        req = requests[query - 1]
        at = steps[req["stamp"]["tick"]]
        q, g = np.array(at["actual_arm"]), np.array(at["actual_fingers"])
        out = a.output / f"query{query}"
        out.mkdir()
        scene, arms, source, frames, bases, world, proxy, _, _ = build(
            cfg,
            a.release / "asset-audit.json",
            a.assets,
            snapshot(cfg, initial, q, g, audit),
            initial,
            np.array(at["contact_state"]["block_pose_task"]),
            out,
        )
        world.set_start(g, np.array(at["target_fingers"]))
        entities = {e.name: e for e in scene.get_all_articulations()}
        stamp = Stamp(name, req["stamp"]["tick"], req["stamp"]["sim_s"])
        samples = capture_arms(
            NS(left_entity=entities["arx_left"], right_entity=entities["arx_right"]),
            stamp,
        )
        arms = tuple(
            replace(arm, instance=sample.instance) for arm, sample in zip(arms, samples)
        )
        snap = Snapshot(stamp, samples, {}, matrix(cfg["world_task"]))
        adapter = NS(
            arx=arms,
            aloha=source,
            source_frames=bases,
            target_collision=world,
            source_collision=proxy,
            source_speed=np.array(cfg["source_proxy_velocity_rad_s"]),
            seed=np.array(req["state"]).reshape(2, 7)[:, :6].copy(),
        )
        fractions = [arm.grip.measure(v)["normalized"] for arm, v in zip(arms, g)]
        proxy.set_start(
            np.array([arm.grip.target(v) for arm, v in zip(source, fractions)])
        )
        original_seed = adapter.seed.copy()
        data = np.load(folder / f"plan-{query:05}.npz", allow_pickle=False)
        plan = Trajectory(
            float(data["period_s"]),
            data["q"],
            data["velocity"],
            data["fingers"],
            tuple(int(x) for x in data["waypoint_ticks"]),
            data["predicted_fingers"],
            str(data["contact_profile_id"]),
        )
        check = deadline(100)
        report = dict(
            query=query,
            stamp=req["stamp"],
            original_actions=len(plan.waypoint_ticks),
            waypoints=[],
        )
        for row, tick in enumerate(plan.waypoint_ticks):
            elapsed = tick * plan.period_s
            physical_tick = stamp.tick + int(round(elapsed / at["period_s"]))
            measured = steps[physical_tick]
            assert (
                np.max(np.abs(np.array(measured["target_arm"]) - plan.q[tick])) < 1e-6
            )
            point = dict(row=row, physical_tick=physical_tick)
            for label, target, finger in [
                ("planned", plan.q[tick], plan.predicted_fingers[tick]),
                (
                    "recorded_actual",
                    np.array(measured["actual_arm"]),
                    np.array(measured["actual_fingers"]),
                ),
            ]:
                try:
                    point[label] = source_admissible(
                        adapter, target, finger, frames, elapsed, check
                    )
                except G2Error as e:
                    point[label] = dict(
                        accepted=False,
                        status=e.status.value,
                        detail=str(e),
                        diagnostic=e.diagnostic,
                    )
            report["waypoints"].append(point)
        prefix, receipt = checked_prefix(adapter, plan, snap, check)
        assert np.array_equal(adapter.seed, original_seed)
        report.update(
            approved_actions=0 if prefix is None else len(prefix.waypoint_ticks),
            barrier=receipt,
            source_seed_unchanged=True,
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
            "recorded_checkpoint",
            max_ticks=cfg["max_plan_ticks"],
        )
        adapter.motion = JointRouter(
            motion, np.array([arm.chain.limits for arm in arms])
        )
        adapter.grip_rule = "clip"
        adapter.grip_max_clip = cfg["native_grip_max_clip"]
        adapter.last_stamp = stamp
        adapter.actions = MethodType(Adapter.actions, adapter)
        runner = RecoveryCoordinator(
            adapter,
            NS(_drive_fingers=lambda: np.array(at["target_fingers"])),
            approach_repair=True,
        )
        records = []
        approved, receipt = runner.action_plan(
            np.load(folder / f"action-{query:05}.npy", allow_pickle=False),
            snap,
            4,
            lambda: 100,
            records.append,
        )
        assert np.array_equal(adapter.seed, original_seed)
        report["production_coordinator"] = dict(
            approved_actions=len(approved.waypoint_ticks),
            recoveries=runner.recoveries,
            source_seed_unchanged=True,
            receipt=receipt,
            records=records,
        )
        reports.append(report)
        (a.output / "admission.json").write_text(
            json.dumps(
                dict(physics_steps=0, model_inference=False, reports=reports),
                indent=2,
                default=lambda x: x.tolist(),
            )
            + "\n"
        )
        print("SOURCE_ADMISSION", query, report["approved_actions"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ["raw", "release", "assets", "output"]:
        p.add_argument("--" + n, type=Path, required=True)
    run(p.parse_args())
