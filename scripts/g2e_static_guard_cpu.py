"""Zero-step real SAPIEN/FCL checks and native-guard cost measurement.

Uses actual locked native geometry. Frozen-scene prefix checks are feasibility
probes, not a dynamics replay or proof of policy success.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace as NS
import numpy as np
from g2_adapter import G2Error, Stamp, deadline
from g2_assets import load_arms
from g2_capture import capture_arms
from g2_collision import JointCollisionWorld
from g2_contact import ContactLimits, ContactTracker, ContactWorld
from g2_initial import matrix7
from g2_proxy_collision import collision_only_urdf
from g2_robotwin_backend import pose7
from g2e_native_guard import NativeGuard


def build(config, release, assets, initial, output):
    import sapien
    from mplib.sapien_utils.conversion import convert_object_name

    (row,) = [
        x
        for x in json.loads((release / "asset-audit.json").read_text())["robots"]
        if x["robot"] == "aloha-agilex"
    ]
    arms = load_arms(
        assets,
        release / "asset-audit.json",
        "aloha-agilex",
        config["calibration"]["aloha-agilex"],
    )
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(0.004)
    scene.default_physical_material = scene.create_physical_material(0.5, 0.5, 0.0)
    scene.add_ground(0, render=False)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    entity = loader.load(
        str(
            collision_only_urdf(
                assets / row["urdf_member"], row["urdf_sha256"], output / "native.urdf"
            )
        )
    )
    entity.set_name("native_aloha")
    p = initial["robot_base_world"]["left"]
    entity.set_root_pose(sapien.Pose(p[:3], p[3:]))
    q = np.asarray([initial["actual_arm_qpos_rad"][s] for s in ("left", "right")])
    # Read exact initial fingers from reset fixture, not the scalar observation.
    values = {}
    for side, arm, qa in zip(("left", "right"), arms, q):
        fixture = config["reset_fixture"]["robots"]["aloha-agilex"][side]
        for n, v in zip(arm.chain.names, qa):
            values[n] = v
        for n, v in zip(arm.grip.names, fixture["fingers_m"]):
            values[n] = v
    entity.set_qpos(
        np.array([values.get(j.name, 0.0) for j in entity.get_active_joints()])
    )
    fingers = np.array([[values[n] for n in a.grip.names] for a in arms])
    objects = {}
    for name, half, pose_name in [
        ("table", [0.6, 0.35, 0.025], "table"),
        ("target_box", [0.05, 0.05, 0.005], "target"),
        ("box", [0.03, 0.03, 0.1], "block"),
    ]:
        b = scene.create_actor_builder()
        b.add_box_collision(
            pose=sapien.Pose([0, 0, -0.025]) if name == "table" else sapien.Pose(),
            half_size=half,
        )
        if name == "table":
            for x in (-0.55, 0.55):
                for y in (-0.30, 0.30):
                    b.add_box_collision(
                        pose=sapien.Pose([x, y, -0.372]), half_size=[0.05, 0.05, 0.368]
                    )
        actor = b.build_static(name=name)
        p = initial["objects_task"][pose_name]
        actor.set_pose(sapien.Pose(p[:3], p[3:]))
        objects[name] = actor
    b = scene.create_actor_builder()
    b.add_box_collision(half_size=[3, 0.6, 1.5])
    wall = b.build_static(name="wall")
    wall.set_pose(sapien.Pose([0, 1, 1.5]))
    robot = NS(left_entity=entity, right_entity=entity)
    samples = capture_arms(robot, Stamp("static", 0, 0.0))
    arms = tuple(replace(a, instance=s.instance) for a, s in zip(arms, samples))
    tracker = ContactTracker.from_scene(robot, objects["box"], arms, ContactLimits())
    checker = JointCollisionWorld.from_scene(
        scene,
        [entity, entity],
        [a.chain.names for a in arms],
        [a.grip.names for a in arms],
    )
    frames = [matrix7(initial["robot_base_world"][s]) for s in ("left", "right")]
    world = ContactWorld(
        checker,
        tracker,
        arms,
        frames,
        convert_object_name(objects["box"]),
        [convert_object_name(a) for a in scene.get_all_actors()],
        [
            [entity.find_joint_by_name(n).child_link.name for n in a.grip.names]
            for a in arms
        ],
        [convert_object_name(objects[n]) for n in ("table", "target_box")],
    )
    return scene, arms, world, q, fingers


def run(a):
    import sapien

    a.output.mkdir(parents=True, exist_ok=False)
    report = {}
    for seed in (0, 1):
        name = f"seed{seed}-A-native-w1"
        out = a.output / name
        out.mkdir()
        cfg = json.loads((a.release / "configs" / f"{name}.json").read_text())
        directory = a.raw / "trials" / name
        initial = json.loads((directory / "initial-readback.json").read_text())
        scene, arms, world, q, g = build(cfg, a.release, a.assets, initial, out)
        guard = NativeGuard(world, arms, 0.02, 0.03, 10.0)
        clear = world.measured_clear(q, g)
        if not clear:
            raise ValueError(f"initial native FCL blocked: {world.last_collisions}")
        calls = []
        for _ in range(10):
            start = monotonic()
            guard.candidate(q, g, q, g, deadline(10))
            calls.append(monotonic() - start)
        chunk = np.load(directory / "action-00001.npy", allow_pickle=False)
        windows = {}
        for window in (1, 4, 26, 31):
            qa = q.copy()
            fingers = g.copy()
            guard.previous_drive = None
            start = monotonic()
            check = deadline(25)
            completed = 0
            rejection = None
            for index, row in enumerate(chunk[:window].reshape(-1, 2, 7)):
                drive = np.repeat((-0.01 + row[:, 6] * 0.055)[:, None], 2, axis=1)
                try:
                    guard.candidate(qa, fingers, row[:, :6], drive, check)
                except G2Error as exc:
                    rejection = dict(
                        index=index,
                        code=exc.status.value,
                        detail=str(exc),
                        diagnostic=exc.diagnostic,
                    )
                    break
                qa = row[:, :6].copy()
                fingers = guard.last["predicted_fingers"].copy()
                completed += 1
            windows[str(window)] = dict(
                accepted_actions=completed,
                rejection=rejection,
                wall_s=monotonic() - start,
            )
        # Put a real tray collision shape directly at the actual right finger.
        # This is an intentional negative control; block contact permission must
        # not exempt it. No real scene.step or model call occurs.
        entity = scene.get_all_articulations()[0]
        finger = entity.find_joint_by_name(arms[1].grip.names[0]).child_link
        tray = next(a for a in scene.get_all_actors() if a.name == "target_box")
        old = tray.pose
        tray.set_pose(finger.pose)
        world.refresh()
        tray_rejected = not world.measured_clear(q, g)
        collisions = world.last_collisions
        tray.set_pose(old)
        world.refresh()
        if not tray_rejected:
            raise AssertionError("real tray intrusion was not rejected")
        report[name] = dict(
            initial_fcl_clear=clear,
            physics_steps=0,
            gpu_used=False,
            static_candidate_wall_median_s=float(np.median(calls)),
            static_candidate_wall_max_s=max(calls),
            prefix_checks=windows,
            tray_intrusion_rejected=tray_rejected,
            tray_collisions=collisions,
            scope="exact locked native roots/arms/fingers and rigid shapes, zero-step frozen scene; no contact impulses or native TOPP dynamics",
            guard_calls=guard.calls,
            guard_samples=guard.samples,
        )
    (a.output / "acceptance.json").write_text(
        json.dumps(report, default=lambda v: v.tolist(), indent=2) + "\n"
    )
    print("G2E_STATIC_PASS", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("release", "assets", "raw", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    run(p.parse_args())
