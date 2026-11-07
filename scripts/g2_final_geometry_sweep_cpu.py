"""Real job checkpoints, CPU FCL/IK only; no physics, renderer or policy."""

import argparse
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from g2_adapter import G2Error, deadline, solve_pair
from g2_static_failure_replay import build, matrix, alternatives, pose
from g2_remaining_geometry import refined_pair
from g2_remaining_router import JointRouter
from g2_motion import QuinticMotion
from g2e_native_guard import NativeGuard


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist()) + "\n")


def capture(args, seed, robot):
    name = f"unified-seed{seed}-{robot}"
    folder = args.raw / "trials" / name / "episode"
    cfg = json.loads((args.release / "configs" / f"{name}.json").read_text())
    initial = json.loads((folder / "initial-readback.json").read_text())
    rows = [json.loads(line) for line in (folder / "events.jsonl").open()]
    return name, folder, cfg, initial, rows


def snapshot(cfg, initial, q, g, audit):
    row = next(r for r in audit["robots"] if r["robot"] == "ARX-X5")
    return dict(
        actual=[
            dict(
                names=row["joint_names"][i]
                + cfg["calibration"]["ARX-X5"][side]["finger_names"],
                qpos=list(q[i]) + list(g[i]),
                world_base=matrix(initial["robot_base_world"][side]),
            )
            for i, side in enumerate(("left", "right"))
        ]
    )


def detour(arms, q, fingers, drive, goal, requested, world, cfg, output):
    check = deadline(60)
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
        "frozen_checkpoint_detour",
        max_ticks=cfg["max_plan_ticks"],
    )
    world.set_start(fingers, drive)

    class FixedDriveGeometry:
        tracker = world.tracker

        def __call__(self, sample, actual):
            try:
                projected = world.planning_geometry(sample, drive, fingers, check)
                return world(sample, projected)
            except G2Error as exc:
                if exc.status.value not in ("collision", "plan_failed"):
                    raise
                return False

    router = JointRouter(motion, np.array([a.chain.limits for a in arms]))
    try:
        vertices = router.route(q, goal, fingers, FixedDriveGeometry(), check)
        # Finger intent changes only after the fixed-aperture detour reaches q.
        targets = [(v, drive) for v in vertices[1:]] + [(goal, requested)]
        path = motion.plan(q, fingers, targets, world, check)
        np.savez_compressed(
            output, **{k: v for k, v in asdict(path).items() if v is not None}
        )
        return dict(
            accepted=True,
            vertices=len(vertices),
            control_ticks=len(path.q),
            endpoint_q_error=float(np.max(np.abs(path.q[-1] - goal))),
            scope="static complete dense geometry; not dynamic execution",
        )
    except G2Error as exc:
        return dict(
            accepted=False,
            status=exc.status.value,
            detail=str(exc),
            diagnostic=exc.diagnostic,
        )


def run(args):
    args.output.mkdir(exist_ok=False)
    audit = json.loads((args.release / "asset-audit.json").read_text())
    report = dict(
        cpu_only=True, physics_steps=0, renderer=False, model=False, arx=[], native={}
    )
    # All proxy failures are checked against their own actual ARX state and block pose.
    for seed in (0, 1):
        name, folder, cfg, initial, events = capture(args, seed, "arx-full-w4")
        actual = None
        for event in events:
            if event.get("kind") == "executed_step":
                actual = event
            if event.get("kind") not in (
                "original_proxy_rejection",
                "original_action_rejection",
            ):
                continue
            d = event["diagnostic"]
            tick = d["stamp"]["tick"]
            out = args.output / f"{name}-tick{tick}"
            out.mkdir()
            q = np.array(actual["actual_arm"])
            g = np.array(actual["actual_fingers"])
            block = np.array(actual["contact_state"]["block_pose_task"])
            scene, arms, source, frames, bases, world, proxy, _, _ = build(
                cfg,
                args.release / "asset-audit.json",
                args.assets,
                snapshot(cfg, initial, q, g, audit),
                initial,
                block,
                out,
            )
            world.set_start(g, np.array(actual["target_fingers"]))
            check = deadline(90)
            case = dict(
                episode=name,
                tick=tick,
                kind=event["kind"],
                actual_q=q,
                actual_fingers=g,
                actual_clear=world.measured_clear(q, g),
                actual_contacts=actual["contact_state"],
            )
            if event["kind"] == "original_proxy_rejection":
                tools = np.array(d["arx_tools_task"])
                start = np.array(d["ik_seed"])
                native = np.array(d["source_fingers"])
                delta = (
                    None
                    if d["continuity_delta"] is None
                    else np.array(d["continuity_delta"])
                )
                proxy.set_start(native)
                active = (source, bases, tools, start, native, proxy)
                case["original_source_seed_clear"] = bool(proxy(start, native))
            else:
                request = d["requests"][-1]
                tools = np.array(request["task_tools"])
                start = np.array(request["ik_start"])
                native = np.array(request["fingers"])
                delta = None
                active = (arms, frames, tools, start, native, world)
            for label, operation in [
                ("baseline", lambda: solve_pair(*active, check, delta)),
                ("refined", lambda: refined_pair(*active, check, delta)),
            ]:
                try:
                    solved = operation()
                    case[label] = dict(accepted=True, q=solved[0], residual=solved[1])
                    if event["kind"] == "original_action_rejection":
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
                            "actual_job_geometry_CPU",
                            max_ticks=cfg["max_plan_ticks"],
                        )
                        try:
                            path = motion.plan(
                                q, g, [(solved[0], native)], world, check
                            )
                            case[label]["full_path"] = dict(
                                accepted=True, ticks=len(path.q)
                            )
                        except G2Error as e:
                            case[label]["full_path"] = dict(
                                accepted=False, detail=str(e), diagnostic=e.diagnostic
                            )
                            case[label]["detour"] = detour(
                                arms,
                                q,
                                g,
                                np.array(actual["target_fingers"]),
                                solved[0],
                                native,
                                world,
                                cfg,
                                out / (label + "-detour.npz"),
                            )
                except G2Error as e:
                    case[label] = dict(
                        accepted=False,
                        status=e.status.value,
                        detail=str(e),
                        diagnostic=e.diagnostic,
                        last_collisions=active[-1].last_collisions,
                    )
            if not case["refined"]["accepted"]:
                try:
                    case["extra_seeds"] = alternatives(*active, delta)
                except G2Error as e:
                    case["extra_seeds"] = dict(
                        found=False, status=e.status.value, detail=str(e)
                    )
            report["arx"].append(case)
            save(args.output / "geometry.json", report)
            print(
                "CPU_GEOMETRY",
                name,
                tick,
                case["baseline"]["accepted"],
                case["refined"]["accepted"],
                flush=True,
            )
    # Native query uses the identical source URDF/actors in the CPU proxy world.
    name, folder, ncfg, ninitial, events = capture(args, 0, "native-w50")
    rejection = next(r for r in events if r.get("kind") == "native_guard_rejection")
    block = np.array(
        [r for r in events if r.get("kind") == "native_contact_sample"][-1][
            "block_pose_task"
        ]
    )
    _, afolder, cfg, initial, aevents = capture(args, 0, "arx-full-w4")
    last = next(r for r in aevents if r.get("kind") == "executed_step")
    out = args.output / "native-seed0"
    out.mkdir()
    scene, _, source, _, bases, _, world, _, _ = build(
        cfg,
        args.release / "asset-audit.json",
        args.assets,
        snapshot(
            cfg,
            initial,
            np.array(last["actual_arm"]),
            np.array(last["actual_fingers"]),
            audit,
        ),
        initial,
        block,
        out,
    )
    q = np.array(rejection["actual_arm"])
    g = np.array(rejection["actual_fingers"])
    drive = np.array(rejection["drive_fingers"])
    target = np.array(rejection["drive_arm"])
    world.set_start(g, np.array(rejection["previous_drive"]))
    guard = NativeGuard(
        world,
        source,
        ncfg["collision_joint_step_rad"],
        ncfg["tracking_position_limit_m"],
        ncfg["tracking_angle_limit_deg"],
    )
    guard.previous_drive = np.array(rejection["previous_drive"])
    check = deadline(90)
    case = dict(
        actual_clear=world.measured_clear(q, g),
        q=q,
        fingers=g,
        first_rejected_tick=rejection["tick"],
    )
    for label, goal, grip in [
        ("rejected_interpolated_drive", target, drive),
        (
            "complete_policy_goal",
            np.load(folder / "action-00001.npy", allow_pickle=False)[26].reshape(2, 7)[
                :, :6
            ],
            -0.01
            + 0.055
            * np.clip(
                np.load(folder / "action-00001.npy", allow_pickle=False)[26].reshape(
                    2, 7
                )[:, 6, None],
                0,
                1,
            )
            * np.ones((2, 2)),
        ),
    ]:
        try:
            guard.candidate(q, g, goal, grip, check)
            case[label] = dict(accepted=True)
        except G2Error as e:
            case[label] = dict(
                accepted=False,
                status=e.status.value,
                detail=str(e),
                diagnostic=e.diagnostic,
            )
        # Distinguish a blocked endpoint from a blocked route to a clear endpoint.
        world.set_start(g, grip)
        try:
            endpoint, changes = world.project_fingers(
                goal,
                np.array(
                    [
                        np.clip(v, a.grip.limits[:, 0], a.grip.limits[:, 1])
                        for a, v in zip(source, grip)
                    ]
                ),
                check,
                previous=g,
            )
            clear = bool(world(goal, endpoint))
            case[label]["endpoint"] = dict(
                clear=clear, fingers=endpoint, collisions=world.last_collisions
            )
        except G2Error as e:
            case[label]["endpoint"] = dict(
                clear=False, detail=str(e), diagnostic=e.diagnostic
            )
    contact = [r for r in events if r.get("kind") == "native_guard_state"][-1][
        "contact_state"
    ]
    case["actual_contact_fingers"] = contact["contact_fingers"]
    if any(contact["contact_fingers"].values()):
        case["same_policy_endpoint_detour"] = dict(
            accepted=False,
            reason="actual contact forbids free-space rerouting at rejection",
        )
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
    obj = next(
        r
        for r in events
        if r.get("kind") == "native_contact_sample" and r["tick"] == request["tick"]
    )
    state = next(
        r
        for r in events
        if r.get("kind") == "native_guard_state" and r["tick"] == request["tick"]
    )
    actor = next(a for a in scene.get_all_actors() if a.name == "box")
    actor.set_pose(pose(np.array(obj["block_pose_task"])))
    world.refresh()
    policy = np.load(folder / "action-00001.npy", allow_pickle=False)[26].reshape(2, 7)
    goal = policy[:, :6]
    requested = np.repeat(
        (-0.01 + 0.055 * np.clip(policy[:, 6], 0, 1))[:, None], 2, axis=1
    )
    q0, g0 = np.array(at["actual_arm"]), np.array(at["actual_fingers"])
    original_drive = np.array(obj["target_fingers_drive_m"])
    world.set_start(g0, original_drive)
    early = dict(
        tick=request["tick"],
        actual_clear=world.measured_clear(q0, g0),
        contact_fingers=state["contact_state"]["contact_fingers"],
    )
    try:
        geometry, changes = world.project_fingers(
            goal,
            np.array(
                [
                    np.clip(v, a.grip.limits[:, 0], a.grip.limits[:, 1])
                    for a, v in zip(source, requested)
                ]
            ),
            deadline(20),
            previous=g0,
        )
        early["goal_clear"] = bool(world(goal, geometry))
        early["goal_collisions"] = world.last_collisions
    except G2Error as exc:
        early.update(goal_clear=False, detail=str(exc), diagnostic=exc.diagnostic)
    if early["goal_clear"] and not any(early["contact_fingers"].values()):
        early["same_goal_detour"] = detour(
            source,
            q0,
            g0,
            original_drive,
            goal,
            requested,
            world,
            ncfg,
            out / "native-early-detour.npz",
        )
    case["pre_contact_action_start"] = early
    report["native"] = case
    save(args.output / "geometry.json", report)
    print(
        "CPU_NATIVE",
        case["actual_clear"],
        case["complete_policy_goal"]["endpoint"]["clear"],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ["raw", "release", "assets", "output"]:
        p.add_argument("--" + n, type=Path, required=True)
    run(p.parse_args())
