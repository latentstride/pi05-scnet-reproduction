#!/usr/bin/env python3
"""Actual mesh collision checks on CPU; no renderer, physics stepping, CUDA or model."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from g2_collision import JointCollisionWorld
from g2_adapter import deadline
from g2_motion import QuinticMotion


def run(assets, output):
    import sapien
    output.mkdir(parents=True, exist_ok=False)
    urdf = assets / "X5A.urdf"
    original = urdf.read_bytes()
    if sha256(original).hexdigest() != "7aaa308a1a5b6a213d914db86ba09ddd6252f25cb498ac9a148bfa6c7166a423":
        raise ValueError("ARX URDF differs from audited asset")
    tree = ET.fromstring(original)
    visual_count = 0
    for parent in tree.iter():
        for child in list(parent):
            if child.tag in ("visual", "material"):
                parent.remove(child)
                visual_count += 1
    mesh_hashes = {}
    for node in tree.iter("mesh"):
        path = (urdf.parent / node.get("filename")).resolve()
        if not path.is_file():
            raise ValueError(f"missing collision mesh: {path.name}")
        node.set("filename", str(path))
        mesh_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    cpu_urdf = output / "ARX-collision-only.urdf"
    cpu_urdf.write_bytes(ET.tostring(tree))
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    entities = []
    for side, x in zip(("left", "right"), (-.3, .3)):
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        entity = loader.load(str(cpu_urdf))
        entity.set_name("g2_" + side)
        entity.set_root_pose(sapien.Pose([x, -.35, .784], [.70710678, 0, 0, .70710678]))
        q = np.zeros(entity.get_dof(), dtype=np.float32)
        for i, joint in enumerate(entity.get_active_joints()):
            if joint.get_name() in ("joint7", "joint8"):
                q[i] = .02
        entity.set_qpos(q)
        entities.append(entity)
    names = [tuple(f"joint{i}" for i in range(1, 7))] * 2
    fingers = [("joint7", "joint8")] * 2
    checker = JointCollisionWorld.from_scene(scene, entities, names, fingers)
    q = np.zeros((2, 6))
    g = np.full((2, 2), .02)
    neutral = checker(q, g)
    neutral_details = checker.last_collisions.copy()
    motion = QuinticMotion(.02, np.full((2, 6), .5), np.ones((2, 6)),
                           np.full((2, 2), .03), np.full((2, 2), .1), .02, .001,
                           "synthetic_CPU_command_limits_only", "sapien3.0.0b1_mplib0.2.1_FCL")
    plan = motion.plan(q, g, [(q + .002, g)], checker, deadline(10))
    velocity = np.diff(plan.q, axis=0) / plan.period_s
    acceleration = np.diff(velocity, axis=0) / plan.period_s
    path_limits_ok = bool(np.max(abs(velocity)) <= .5 + 1e-9 and
                          np.max(abs(acceleration)) <= 1. + 1e-9)
    before = [entity.get_qpos().copy() for entity in entities]
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[2., 2., 2.])
    blocker = builder.build_static(name="g2_test_obstacle")
    blocker.set_pose(sapien.Pose([0, 0, 1]))
    blocked_checker = JointCollisionWorld.from_scene(scene, entities, names, fingers)
    blocked = blocked_checker(q, g)
    blocked_details = blocked_checker.last_collisions.copy()
    scene.remove_actor(blocker)
    # Identical root poses must produce an inter-arm collision.
    entities[1].set_root_pose(entities[0].get_root_pose())
    overlap_checker = JointCollisionWorld.from_scene(scene, entities, names, fingers)
    overlap = overlap_checker(q, g)
    restored_actual = all(np.array_equal(entity.get_qpos(), state) for entity, state in zip(entities, before))
    result = dict(data_kind="diagnostic", gpu_used=False, renderer_created=False, physics_stepped=False,
                  versions={"sapien": version("sapien"), "mplib": version("mplib")},
                  urdf_sha256=sha256(original).hexdigest(), mesh_sha256=mesh_hashes,
                  removed_visual_material_count=visual_count,
                  retained_collision_elements=len(list(tree.iter("collision"))),
                  neutral_clear=neutral, neutral_collisions=neutral_details,
                  obstacle_collision_detected=not blocked, obstacle_collisions=blocked_details,
                  overlap_collision_detected=not overlap,
                  actual_qpos_unchanged=restored_actual,
                  synchronized_joint_path_ticks=len(plan.q)-1, collision_check_calls=checker.calls,
                  cpu_path_velocity_acceleration_checked=path_limits_ok,
                  test_motion_limits_scope="imposed synthetic CPU test bounds, not operational robot ratings",
                  passed=not blocked and not overlap and restored_actual and path_limits_ok,
                  scope="CPU synthetic scene with real ARX collision meshes; no task/contact/control acceptance")
    (output / "collision-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.assets, args.output) else 1)
