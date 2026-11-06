"""Reconstruct captured ARX endpoints in CPU FCL with zero physics advancement."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np

from g2_adapter import G2Error, deadline, solve_pair, tool_ik
from g2_assets import load_arms
from g2_collision import JointCollisionWorld, audited_arx_self_profile
from g2_contact import ContactTracker, ContactWorld, contact_profile
from g2_motion import QuinticMotion
from g2_proxy_collision import ProxyCollisionWorld, collision_only_urdf
from robotwin_control_geometry import inverse_pose, pose_from_wxyz


def matrix(value):
    return pose_from_wxyz(value[:3], value[3:])


def pose(value):
    import sapien
    from g2_robotwin_backend import pose7

    v = pose7(value)
    return sapien.Pose(v[:3], v[3:])


def build(config, audit_path, assets, snapshot, initial, block_pose, output):
    import sapien
    from mplib.sapien_utils.conversion import convert_object_name

    audit = json.loads(audit_path.read_text())
    (row,) = [r for r in audit["robots"] if r["robot"] == "ARX-X5"]
    arms = load_arms(assets, audit_path, "ARX-X5", config["calibration"]["ARX-X5"])
    source = load_arms(
        assets, audit_path, "aloha-agilex", config["calibration"]["aloha-agilex"]
    )
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(0.004)
    scene.default_physical_material = scene.create_physical_material(0.5, 0.5, 0.0)
    scene.add_ground(0, render=False)
    table_builder = scene.create_actor_builder()
    table_builder.add_box_collision(
        pose=sapien.Pose([0, 0, -0.025]), half_size=[0.6, 0.35, 0.025]
    )
    for x in (-0.55, 0.55):
        for y in (-0.30, 0.30):
            table_builder.add_box_collision(
                pose=sapien.Pose([x, y, -0.372]), half_size=[0.05, 0.05, 0.368]
            )
    table = table_builder.build_static(name="table")
    table.set_pose(pose(matrix(initial["objects_task"]["table"])))
    wall_builder = scene.create_actor_builder()
    wall_builder.add_box_collision(half_size=[3, 0.6, 1.5])
    wall = wall_builder.build_static(name="wall")
    wall.set_pose(sapien.Pose([0, 1, 1.5]))
    entities = []
    urdf = collision_only_urdf(
        assets / row["urdf_member"], row["urdf_sha256"], output / "arx.urdf"
    )
    for side, sample in zip(("left", "right"), snapshot["actual"]):
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        entity = loader.load(str(urdf))
        entity.set_name("arx_" + side)
        entity.set_root_pose(pose(np.asarray(sample["world_base"])))
        values = dict(zip(sample["names"], sample["qpos"]))
        entity.set_qpos(np.array([values[j.name] for j in entity.get_active_joints()]))
        entities.append(entity)
    objects = {}
    for name, half in [("box", [0.03, 0.03, 0.1]), ("target_box", [0.05, 0.05, 0.005])]:
        actor = sapien.Entity()
        actor.name = name
        body = (
            sapien.physx.PhysxRigidDynamicComponent()
            if name == "box"
            else sapien.physx.PhysxRigidStaticComponent()
        )
        body.attach(
            sapien.physx.PhysxCollisionShapeBox(
                half_size=half, material=scene.default_physical_material
            )
        )
        actor.add_component(body)
        actor.set_pose(
            pose(
                block_pose
                if name == "box"
                else matrix(initial["objects_task"]["target"])
            )
        )
        scene.add_entity(actor)
        if name == "box":
            body.mass = 0.01
        objects[name] = actor
    robot = SimpleNamespace(left_entity=entities[0], right_entity=entities[1])
    limits, _ = contact_profile(config, 0.01)
    tracker = ContactTracker.from_scene(robot, objects["box"], arms, limits)
    tracker.block_pose = block_pose
    checker = JointCollisionWorld.from_scene(
        scene, entities, [a.chain.names for a in arms], [a.grip.names for a in arms]
    )
    checker.use_self_profile(audited_arx_self_profile(assets))
    frames = [
        inverse_pose(matrix(config["world_task"])) @ np.asarray(s["world_base"])
        for s in snapshot["actual"]
    ]
    names = [convert_object_name(x) for x in scene.get_all_actors()]
    supports = [convert_object_name(x) for x in (table, objects["target_box"])]
    target = ContactWorld(
        checker,
        tracker,
        arms,
        frames,
        convert_object_name(objects["box"]),
        names,
        [
            [e.find_joint_by_name(n).child_link.name for n in a.grip.names]
            for e, a in zip(entities, arms)
        ],
        supports,
        world_task=matrix(config["world_task"]),
    )
    bases = [matrix(v) for v in config["source_base_world"]]
    (src,) = [r for r in audit["robots"] if r["robot"] == "aloha-agilex"]
    proxy = ProxyCollisionWorld(
        scene,
        assets / src["urdf_member"],
        src["urdf_sha256"],
        pose(bases[0]),
        source,
        output / "proxy.urdf",
    )
    source_contact = ContactWorld(
        proxy.checker,
        tracker,
        source,
        bases,
        convert_object_name(objects["box"]),
        names,
        [
            [proxy.entity.find_joint_by_name(n).child_link.name for n in a.grip.names]
            for a in source
        ],
        supports,
        world_task=matrix(config["world_task"]),
    )
    q, g = [], []
    for a, s in zip(arms, snapshot["actual"]):
        values = dict(zip(s["names"], s["qpos"]))
        q.append([values[n] for n in a.chain.names])
        g.append([values[n] for n in a.grip.names])
    q, g = np.asarray(q), np.asarray(g)
    target.set_start(g)
    return scene, arms, source, frames, bases, target, source_contact, q, g


def evaluate(arms, frames, tools, seed, fingers, world, max_delta=None):
    try:
        q, residual = solve_pair(
            arms, frames, tools, seed, fingers, world, deadline(20), max_delta
        )
        return dict(accepted=True, q=q.tolist(), residual=residual)
    except G2Error as error:
        return dict(
            accepted=False,
            status=error.status.value,
            detail=str(error),
            diagnostic=error.diagnostic,
        )


def alternatives(arms, frames, tools, seed, fingers, world, max_delta=None):
    # Search additional seeds under the original pose, joint and continuity bounds.
    check = deadline(30)
    choices = []
    for arm, frame, tool, original in zip(arms, frames, tools, seed):
        found = []
        seeds = [np.asarray(original).copy()]
        for joint in range(6):
            for change in (-1.0, 1.0):
                shifted = np.asarray(original).copy()
                shifted[joint] += change * np.pi / 2
                seeds.append(shifted)
        for value in seeds:
            for q, residual in tool_ik(arm, frame, tool, value, check):
                q = q - 2 * np.pi * np.round((q - original) / (2 * np.pi))
                if np.any(q < arm.chain.limits[:, 0]) or np.any(
                    q > arm.chain.limits[:, 1]
                ):
                    continue
                if max_delta is not None and np.any(
                    np.abs(q - original) > max_delta[len(choices)] + 1e-9
                ):
                    continue
                if not any(np.max(np.abs(q - old)) < 1e-4 for old in found):
                    found.append(q)
            if len(found) >= 16:
                break
        choices.append(found)
    from itertools import product

    pairs = sorted(
        product(*choices), key=lambda x: np.linalg.norm(np.asarray(x) - seed)
    )
    for i, pair in enumerate(pairs):
        check()
        values = np.asarray(pair)
        if world.ik_clear(values, fingers, check):
            return dict(
                found=True,
                counts=[len(x) for x in choices],
                checked_pairs=i + 1,
                q=values.tolist(),
            )
    return dict(
        found=False,
        counts=[len(x) for x in choices],
        checked_pairs=len(pairs),
        scope="bounded extra-seed search; no global infeasibility proof",
    )


def run(args):
    output = args.output.resolve()
    for source in (args.release, args.assets, args.selection.parent):
        source = source.resolve()
        if output == source or source in output.parents:
            raise ValueError("analysis output must be outside the input directories")
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    with zipfile.ZipFile(args.selection) as bundle:
        for name in sorted(
            json.loads((args.release / "run_lock.json").read_text())["bindings"]
        ):
            if "B-arx" not in name:
                continue
            seed_number = int(name.split("-")[0][4:])
            expert = np.load(
                args.expert_trace / f"episode{seed_number}.npz", allow_pickle=False
            )["target"]
            output = args.output / name
            output.mkdir()
            events = json.loads(bundle.read(name + "/events-selected.json"))
            steps = json.loads(bundle.read(name + "/control-samples.json"))
            failure = [e for e in events if e.get("status") == "collision"][-1]
            snaps = [e for e in events if e.get("kind") == "snapshot"]
            snapshot = snaps[-1]
            config = json.loads((args.release / "configs" / f"{name}.json").read_text())
            initial = json.loads(bundle.read(name + "/initial-readback.json"))
            contact = steps[-1]["contact_state"] if steps else None
            if contact and (contact["holders"] or contact["phase"] != "approach"):
                raise ValueError(
                    "reconstruction supports only recorded approach states without a holder"
                )
            block = (
                np.array(contact["block_pose_task"])
                if contact
                else matrix(initial["objects_task"]["block"])
            )
            scene, arms, source, frames, bases, target, proxy, q, g = build(
                config,
                args.release / "asset-audit.json",
                args.assets,
                snapshot,
                initial,
                block,
                output,
            )
            source_tools = np.array(
                [f @ a.chain.fk(v) @ a.offset() for a, f, v in zip(arms, frames, q)]
            )
            if failure["phase"] == "observation":
                world = proxy
                fractions = [a.grip.measure(v)["normalized"] for a, v in zip(arms, g)]
                fingers = np.array(
                    [a.grip.target(v) for a, v in zip(source, fractions)]
                )
                proxy.set_start(fingers)
                previous = [e for e in events if e.get("kind") == "policy_input"][-1]
                seed = np.asarray(previous["state"]).reshape(2, 7)[:, :6]
                delta = np.asarray(config["source_proxy_velocity_rad_s"]) * (
                    snapshot["stamp"]["sim_s"] - snaps[-2]["stamp"]["sim_s"]
                )
                actual = evaluate(
                    source, bases, source_tools, seed, fingers, proxy, delta
                )
                world_kind = "source_collision"
            else:
                world = target
                req = failure["requests"][0]
                action_name = sorted(
                    n
                    for n in bundle.namelist()
                    if n.startswith(name + "/action-") and n.endswith(".npy")
                )[-1]
                action_bytes = bundle.read(action_name)
                action = np.load(BytesIO(action_bytes), allow_pickle=False)[
                    req["row"]
                ].reshape(2, 7)
                policy_tools = np.array(
                    [
                        f @ a.chain.fk(v[:6]) @ a.offset()
                        for a, f, v in zip(source, bases, action)
                    ]
                )
                difference = float(
                    np.max(np.abs(policy_tools - np.asarray(req["task_tools"])))
                )
                if difference > 1e-10:
                    raise ValueError(
                        "raw policy FK differs from the original requested tool poses"
                    )
                actual = evaluate(
                    arms,
                    frames,
                    policy_tools,
                    np.asarray(req["ik_start"]),
                    np.asarray(req["fingers"]),
                    target,
                )
                actual.update(
                    raw_action_file=action_name,
                    raw_action_sha256=sha256(action_bytes).hexdigest(),
                    source_fk_max_abs_difference=difference,
                )
                world_kind = "target_collision"
            expanded = None
            if args.search_alternatives:
                try:
                    expanded = (
                        alternatives(
                            source, bases, source_tools, seed, fingers, proxy, delta
                        )
                        if failure["phase"] == "observation"
                        else alternatives(
                            arms,
                            frames,
                            np.asarray(req["task_tools"]),
                            np.asarray(req["ik_start"]),
                            np.asarray(req["fingers"]),
                            target,
                        )
                    )
                except G2Error as error:
                    expanded = dict(
                        found=False,
                        status=error.status.value,
                        detail=str(error),
                        scope="bounded search incomplete",
                    )
            examples = []
            for candidate in failure["collision_examples"]:
                values = np.asarray(candidate["candidate_q"])
                fingers = (
                    np.asarray(failure["requests"][0]["fingers"])
                    if failure["phase"] != "observation"
                    else fingers
                )
                verdict = world.ik_clear(values, fingers, deadline(10))
                examples.append(
                    dict(
                        accepted=bool(verdict),
                        q=values.tolist(),
                        collisions=world.last_collisions,
                    )
                )
            references = []
            for tick in (0, 25, 100):
                row = expert[tick]
                source_q = np.array([row[:6], row[7:13]])
                tools = np.array(
                    [
                        f @ a.chain.fk(v) @ a.offset()
                        for a, f, v in zip(source, bases, source_q)
                    ]
                )
                endpoint = evaluate(arms, frames, tools, q, g, target)
                path_result = dict(checked=False, reason="endpoint rejected")
                if endpoint["accepted"]:
                    limits = config["control_limits"]
                    motion = QuinticMotion(
                        config["control_period_s"],
                        limits["arm_velocity_rad_s"],
                        limits["arm_acceleration_rad_s2"],
                        limits["finger_velocity_m_s"],
                        limits["finger_acceleration_m_s2"],
                        config["collision_joint_step_rad"],
                        config["collision_finger_step_m"],
                        limits["source"],
                        "captured_static_CPU_FCL",
                        max_ticks=config["max_plan_ticks"],
                    )
                    try:
                        trajectory = motion.plan(
                            q, g, [(np.asarray(endpoint["q"]), g)], target, deadline(20)
                        )
                        path_result = dict(
                            checked=True,
                            accepted=True,
                            control_ticks=len(trajectory.q),
                            max_abs_velocity_rad_s=float(
                                np.abs(trajectory.velocity).max()
                            ),
                        )
                    except G2Error as error:
                        path_result = dict(
                            checked=True,
                            accepted=False,
                            status=error.status.value,
                            detail=str(error),
                            diagnostic=error.diagnostic,
                        )
                references.append(
                    dict(
                        source="official Aloha expert target FK in task frame",
                        physics_tick=tick,
                        result=endpoint,
                        path=path_result,
                    )
                )
            results[name] = dict(
                world=world_kind,
                original=failure,
                reconstructed=actual,
                expanded_search=expanded,
                recorded_candidate_checks=examples,
                expert_task_space_controls=references,
                scene_physics_steps=0,
                model_started=False,
                gpu_used=False,
                scope="captured q/fingers and approach block pose; audited URDF and primitive task geometry; no dynamic replay",
                block_pose_task=block.tolist(),
                last_measured_fingers=g.tolist(),
            )
            (args.output / "replay.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(
                name, world_kind, actual["accepted"], actual.get("detail"), flush=True
            )
            del world, target, proxy, scene


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "assets", "selection", "expert-trace", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--search-alternatives", action="store_true")
    run(p.parse_args())
