"""Real-mesh obstacle negative control and joint-detour validation; zero physics."""

import argparse, json
from dataclasses import asdict
from pathlib import Path
import zipfile
import numpy as np
from g2_static_failure_replay import build, matrix
from g2_adapter import deadline, G2Error
from g2_motion import QuinticMotion
from g2_remaining_router import JointRouter
from g2_robotwin_backend import pose7


def run(a):
    import sapien

    a.output.mkdir(parents=True, exist_ok=False)
    check = deadline(65)
    name = "seed0-B-arx-w1"
    with zipfile.ZipFile(a.selection) as z:
        events = json.loads(z.read(name + "/events-selected.json"))
        snap = next(e for e in events if e.get("kind") == "snapshot")
        cfg = json.loads((a.release / "configs" / f"{name}.json").read_text())
        initial = json.loads(z.read(name + "/initial-readback.json"))
    scene, arms, source, frames, bases, world, proxy, q, g = build(
        cfg,
        a.release / "asset-audit.json",
        a.assets,
        snap,
        initial,
        matrix(initial["objects_task"]["block"]),
        a.output,
    )
    from g2_collision import JointCollisionWorld, audited_arx_self_profile

    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.008] * 3)
    tray = builder.build_static(name="intentional_route_probe")
    tray.set_pose(sapien.Pose([3, 3, 3]))
    entities = [scene.get_all_articulations()[i] for i in range(2)]
    world.base = JointCollisionWorld.from_scene(
        scene,
        entities,
        [arm.chain.names for arm in arms],
        [arm.grip.names for arm in arms],
    )
    world.base.use_self_profile(audited_arx_self_profile(a.assets))
    original = tray.pose
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
        "real_mesh_obstacle_CPU",
        max_ticks=10000,
    )
    router = JointRouter(
        motion, np.array([arm.chain.limits for arm in arms]), max_nodes=384
    )
    cases = []
    for displacement in (0.9, -0.9, 1.2, -1.2, 1.8, -1.8, 2.2, -2.2):
        goal = q.copy()
        goal[0, 0] += displacement
        mid = (q + goal) / 2
        with world.base.candidate(mid, g):
            finger = next(
                obj
                for obj in world.base.models[0].get_fcl_model().get_collision_objects()
                if obj.name == world.finger_links[0][0]
            )
            obstacle_pose = finger.pose
        tray.set_pose(sapien.Pose(obstacle_pose.p, obstacle_pose.q))
        world.refresh()
        world.set_start(g)
        if not world.measured_clear(q, g) or not world.measured_clear(goal, g):
            cases.append(
                dict(
                    displacement=displacement,
                    scope="endpoint blocked; excluded from detour test",
                )
            )
            continue
        try:
            motion.plan(q, g, [(goal, g)], world, check)
            cases.append(dict(displacement=displacement, direct_clear=True))
            continue
        except G2Error as exc:
            if exc.status.value != "collision":
                raise
        try:
            path = router.plan(q, g, [(goal, g)], world, check)
            np.savez_compressed(
                a.output / "detour.npz",
                **{k: v for k, v in asdict(path).items() if v is not None},
            )
            cases.append(
                dict(
                    displacement=displacement,
                    direct_clear=False,
                    reroute_accepted=True,
                    metrics=router.last,
                    control_ticks=len(path.q),
                    policy_actions=len(path.waypoint_ticks),
                    max_velocity_rad_s=float(abs(path.velocity).max()),
                    endpoint_q_max_difference=float(abs(path.q[-1] - goal).max()),
                )
            )
            break
        except G2Error as exc:
            cases.append(
                dict(
                    displacement=displacement,
                    direct_clear=False,
                    reroute_accepted=False,
                    status=exc.status.value,
                    detail=str(exc),
                )
            )
    tray.set_pose(original)
    world.refresh()
    (a.output / "route.json").write_text(
        json.dumps(
            dict(
                cases=cases,
                physics_steps=0,
                gpu_used=False,
                scope="intentional 8mm physical box obstacle with real robot meshes; not a policy endpoint fix",
            ),
            indent=2,
        )
        + "\n"
    )
    print("ROUTE_DONE", cases[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("release", "selection", "assets", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    run(p.parse_args())
