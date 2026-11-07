"""Headless native CPU replay with guarded arm or gripper drive interventions."""

import argparse
import ast
from contextlib import redirect_stdout
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Literal
import time

import numpy as np

from g2_adapter import G2Error, Stamp, deadline
from g2_assets import load_arms
from g2_capture import capture_arms
from g2_collision import JointCollisionWorld
from g2_contact import ContactLimits, ContactWorld
from g2_initial import matrix7
from g2_kinematics import pose_error
from g2_proxy_collision import collision_only_urdf
from g2e_native_guard import NativeGuard, NativeReleaseTracker


class EndReplay(Exception):
    pass


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def methods(path, class_name, names):
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    selected = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in selected} == set(names)
    namespace = dict(np=np, Literal=Literal)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return {n.name: namespace[n.name] for n in selected}


def run(args):
    import sapien
    from mplib.sapien_utils import SapienPlanner
    from mplib.sapien_utils.conversion import convert_object_name

    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    name = f"unified-seed1-native-w{args.window}"
    config = json.loads((args.release / f"configs/{name}.json").read_text())
    source_audit = json.loads((args.release / "source-audit.json").read_text())
    for relative in ("envs/_base_task.py", "envs/robot/robot.py"):
        assert (
            sha256((args.pace / relative).read_bytes()).hexdigest()
            == source_audit[relative]
        )
    raw = args.raw / "trials" / name / "episode"
    audit = json.loads((args.release / "asset-audit.json").read_text())
    robot_audit = next(r for r in audit["robots"] if r["robot"] == "aloha-agilex")
    arms = load_arms(
        args.assets,
        args.release / "asset-audit.json",
        "aloha-agilex",
        config["calibration"]["aloha-agilex"],
    )
    initial = json.loads((raw / "initial-readback.json").read_text())
    steps, contacts, requests = {}, {}, []
    first_holder = None
    for line in (raw / "events.jsonl").open():
        row = json.loads(line)
        kind, tick = row.get("kind"), row.get("tick")
        if kind == "executed_step":
            steps[tick] = row
        if kind == "native_contact_sample":
            contacts[tick] = row
        if kind == "native_action_request":
            requests.append(row["normalized_qpos_request"])
        if (
            kind == "native_guard_state"
            and row["contact_state"]["holders"]
            and first_holder is None
        ):
            first_holder = tick
    last_tick = max(steps)
    assert first_holder in (4568, 5088) and last_tick in (4777, 5139)
    # Headless scene. Primitive sizes and table thickness follow the pinned task.
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(0.004)
    scene.add_ground(0, render=False)
    scene.default_physical_material = scene.create_physical_material(0.5, 0.5, 0.0)

    def box(name, half, pose, dynamic=False):
        entity = sapien.Entity()
        entity.name = name
        body = (
            sapien.physx.PhysxRigidDynamicComponent()
            if dynamic
            else sapien.physx.PhysxRigidStaticComponent()
        )
        body.attach(
            sapien.physx.PhysxCollisionShapeBox(half, scene.default_physical_material)
        )
        entity.add_component(body)
        entity.set_pose(sapien.Pose(pose[:3], pose[3:]))
        scene.add_entity(entity)
        return entity, body

    box("wall", [3, 0.6, 1.5], [0, 1, 1.5, 1, 0, 0, 0])
    builder = scene.create_actor_builder()
    builder.add_box_collision(
        pose=sapien.Pose([0, 0, -0.025]),
        half_size=[0.6, 0.35, 0.025],
        material=scene.default_physical_material,
    )
    for x in (-0.55, 0.55):
        for y in (-0.30, 0.30):
            builder.add_box_collision(
                pose=sapien.Pose([x, y, -0.372]), half_size=[0.025, 0.025, 0.368]
            )
    table = builder.build_static(name="table")
    table.set_pose(sapien.Pose([0, 0, 0.74]))
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf = collision_only_urdf(
        args.assets / robot_audit["urdf_member"],
        robot_audit["urdf_sha256"],
        args.output / "robot.urdf",
    )
    entity = loader.load(str(urdf))
    entity.name = "native_aloha"
    base = initial["robot_base_world"]["left"]
    entity.set_root_pose(sapien.Pose(base[:3], base[3:]))
    joints = entity.get_active_joints()
    lookup = {j.name: j for j in joints}
    for link in entity.get_links():
        link.set_mass(1.0)
    for joint in joints:
        joint.set_drive_property(stiffness=1000, damping=200, mode="force")
    block, body = box(
        "box",
        [0.03, 0.03, 0.1],
        config["reset_fixture"]["objects_world"]["block"],
        True,
    )
    box(
        "target_box",
        [0.05, 0.05, 0.005],
        config["reset_fixture"]["objects_world"]["target"],
    )
    body.mass = 0.01
    assert abs(body.mass - 0.01) < 1e-8
    robot_functions = methods(
        args.pace / "envs/robot/robot.py",
        "Robot",
        [
            "_entity_qf",
            "set_arm_joints",
            "set_gripper",
            "get_normal_real_gripper_val",
            "get_left_arm_jointState",
            "get_right_arm_jointState",
            "get_left_gripper_val",
            "get_right_gripper_val",
        ],
    )
    Robot = type("PinnedNativeRobot", (), robot_functions)
    robot = Robot()
    robot.left_entity = robot.right_entity = entity
    for side, arm in zip(("left", "right"), arms):
        setattr(robot, side + "_arm_joints", [lookup[n] for n in arm.chain.names])
        setattr(
            robot, side + "_gripper", [(lookup[n], 1.0, 0.0) for n in arm.grip.names]
        )
        setattr(robot, side + "_gripper_scale", [-0.01, 0.045])

    def reset():
        values = {j.name: 0.0 for j in joints}
        for side, arm in zip(("left", "right"), arms):
            fixture = config["reset_fixture"]["robots"]["aloha-agilex"][side]
            values.update(
                zip(
                    arm.chain.names + arm.grip.names,
                    fixture["q_rad"] + fixture["fingers_m"],
                )
            )
            setattr(robot, side + "_gripper_val", initial["actual_gripper"][side])
        entity.set_qpos([values[j.name] for j in joints])
        entity.set_qvel(np.zeros(len(joints)))
        for j in joints:
            j.set_drive_target(values[j.name])
            j.set_drive_velocity_target(0.0)

    reset()
    checker = JointCollisionWorld.from_scene(
        scene,
        [entity, entity],
        [a.chain.names for a in arms],
        [a.grip.names for a in arms],
    )
    for side, group in zip(("left", "right"), robot_audit["move_group"]):
        setattr(robot, side + "_mplib_planner", SapienPlanner(checker.world, group))
    take_action = methods(
        args.pace / "envs/_base_task.py", "Base_Task", ["take_action"]
    )["take_action"]
    env = NS(
        robot=robot,
        take_action_cnt=0,
        step_lim=10000,
        eval_success=False,
        eval_video_path=None,
        render_freq=0,
        _update_render=lambda: None,
        check_success=lambda: False,
    )
    controls = []

    def capture_control():
        controls.append(
            [
                [
                    float(lookup[n].get_drive_target()[0]),
                    float(lookup[n].get_drive_velocity_target()[0]),
                ]
                for a in arms
                for n in a.chain.names + a.grip.names
            ]
        )
        if len(controls) == last_tick:
            raise EndReplay()

    env.scene = NS(step=capture_control)
    with (args.output / "topp.log").open("w") as log, redirect_stdout(log):
        try:
            for action in requests:
                take_action(env, action, "qpos")
        except EndReplay:
            pass
    controls = np.asarray(controls).reshape(-1, 2, 8, 2)
    assert len(controls) == last_tick
    drive_error = max(
        float(np.max(np.abs(controls[t - 1, :, :6, 0] - steps[t]["target_arm"])))
        for t in steps
    )
    finger_error = max(
        float(
            np.max(
                np.abs(
                    controls[t - 1, :, 6:, 0] - contacts[t]["target_fingers_drive_m"]
                )
            )
        )
        for t in contacts
    )
    save(
        args.output / "control-reconstruction.json",
        dict(
            position_drive_max_rad=drive_error,
            finger_drive_max_m=finger_error,
            physics_ticks=last_tick,
        ),
    )
    assert drive_error < 2e-7 and finger_error < 2e-7, (
        "native drive reconstruction differs"
    )
    reset()
    actual = capture_arms(robot, Stamp(name, 0, 0.0))
    arms = tuple(replace(a, instance=s.instance) for a, s in zip(arms, actual))
    tracker = NativeReleaseTracker.from_scene(robot, block, arms, ContactLimits())
    frames = [matrix7(initial["robot_base_world"][s]) for s in ("left", "right")]
    world = ContactWorld(
        checker,
        tracker,
        arms,
        frames,
        convert_object_name(block),
        [convert_object_name(a) for a in scene.get_all_actors()],
        [[lookup[n].child_link.name for n in a.grip.names] for a in arms],
        [
            convert_object_name(a)
            for a in scene.get_all_actors()
            if a.name in ("table", "target_box")
        ],
    )
    guard = NativeGuard(
        world,
        arms,
        config["collision_joint_step_rad"],
        config["tracking_position_limit_m"],
        config["tracking_angle_limit_deg"],
    )
    check = deadline(1500)
    index = {j.name: i for i, j in enumerate(joints)}

    def measured():
        q = entity.get_qpos()
        return np.array([[q[index[n]] for n in a.chain.names] for a in arms]), np.array(
            [[q[index[n]] for n in a.grip.names] for a in arms]
        )

    original_stop_tick = last_tick
    frozen_grip = controls[first_holder - 1, 0, 6:, 0].copy()
    horizon = max(last_tick, first_holder + args.settle_ticks)
    assert horizon <= 25000
    if horizon > last_tick:
        tail = np.repeat(controls[-1][None, :], horizon - last_tick, axis=0)
        tail[:, :, :, 1] = 0.0
        controls = np.concatenate([controls, tail])
    records = []
    acquired = None
    frozen = None
    failure = None
    maxq = maxg = maxblock = 0.0
    for tick, command in enumerate(controls, 1):
        check()
        command = command.copy()
        if args.mode == "fixed-wrist" and tick > first_holder:
            if frozen is None:
                frozen = controls[first_holder - 1, 0, :6, 0].copy()
            command[0, :6, 0] = frozen
            command[0, :6, 1] = 0.0
        if args.mode == "hold-gripper" and tick > first_holder:
            command[0, 6:, 0] = frozen_grip
            command[0, 6:, 1] = 0.0
        entity.set_qf(
            entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        )
        for side, a in enumerate(arms):
            for j, n in enumerate(a.chain.names + a.grip.names):
                lookup[n].set_drive_target(float(command[side, j, 0]))
                lookup[n].set_drive_velocity_target(float(command[side, j, 1]))
        q, g = measured()
        try:
            guard.candidate(q, g, command[:, :6, 0], command[:, 6:, 0], check)
            scene.step()
            q, g = measured()
            tools = np.array(
                [
                    frame @ a.chain.fk(v) @ a.offset()
                    for frame, a, v in zip(frames, arms, q)
                ]
            )
            pose = block.pose.to_transformation_matrix()
            physical_contacts = scene.get_contacts()
            try:
                guard.measured(
                    Stamp(name, tick, tick * 0.004),
                    q,
                    g,
                    tools,
                    pose,
                    physical_contacts,
                )
            finally:
                if tracker.holders and acquired is None:
                    acquired = tick
                if tick <= original_stop_tick:
                    maxq = max(
                        maxq,
                        float(
                            np.max(np.abs(q - np.asarray(steps[tick]["actual_arm"])))
                        ),
                    )
                    maxg = max(
                        maxg,
                        float(
                            np.max(
                                np.abs(g - np.asarray(steps[tick]["actual_fingers"]))
                            )
                        ),
                    )
                    maxblock = max(
                        maxblock,
                        float(
                            np.linalg.norm(
                                pose[:3, 3]
                                - np.asarray(contacts[tick]["block_pose_task"])[:3, 3]
                            )
                        ),
                    )
                anchor = tracker.holders.get("left")
                err = (
                    pose_error(pose, tools[0] @ anchor)
                    if anchor is not None
                    else np.zeros(6)
                )
                records.append(
                    dict(
                        tick=tick,
                        q=q.tolist(),
                        arm_velocity_rad_s=[
                            [float(entity.get_qvel()[index[n]]) for n in a.chain.names]
                            for a in arms
                        ],
                        block_linear_velocity_m_s=body.linear_velocity.tolist(),
                        block_angular_velocity_rad_s=body.angular_velocity.tolist(),
                        drive_finger_velocity_m_s=command[:, 6:, 1].tolist(),
                        contact_state=tracker.snapshot(),
                        maintenance_tail=tick > original_stop_tick,
                        fingers=g.tolist(),
                        drive=command[:, 6:, 0].tolist(),
                        tool=tools[0].tolist(),
                        block=pose.tolist(),
                        slip_m=float(np.linalg.norm(err[:3])),
                        slip_deg=float(np.rad2deg(np.linalg.norm(err[3:]))),
                        holders=sorted(tracker.holders),
                        forces=[
                            dict(
                                names=[b.entity.name for b in c.bodies],
                                impulses=[p.impulse.tolist() for p in c.points],
                                normals=[p.normal.tolist() for p in c.points],
                                positions=[p.position.tolist() for p in c.points],
                                separations_m=[float(p.separation) for p in c.points],
                            )
                            for c in physical_contacts
                            if any(
                                b.entity.per_scene_id == block.per_scene_id
                                for b in c.bodies
                            )
                        ],
                    )
                )
            guard.tracking(
                np.array(
                    [
                        f @ a.chain.fk(v) @ a.offset()
                        for f, a, v in zip(frames, arms, command[:, :6, 0])
                    ]
                ),
                tools,
            )
        except G2Error as exc:
            failure = dict(tick=tick, code=exc.status.value, detail=str(exc))
            break
        if tick % 1000 == 0:
            print("CPU_REPLAY", args.window, args.mode, tick, flush=True)
    save(args.output / "trace.json", records)
    save(
        args.output / "result.json",
        dict(
            cpu_only=True,
            renderer_created=False,
            model_used=False,
            window=args.window,
            mode=args.mode,
            first_holder_tick=acquired,
            historical_holder_tick=first_holder,
            historical_stop_tick=original_stop_tick,
            horizon_tick=horizon,
            settle_ticks_requested=args.settle_ticks,
            grip_freeze_target_m=frozen_grip.tolist()
            if args.mode == "hold-gripper"
            else None,
            safety_anchor_mode="unchanged live tracker; delayed anchors are offline observers only",
            failure=failure,
            steps=len(records),
            max_joint_difference_rad=maxq,
            max_finger_difference_m=maxg,
            max_block_difference_m=maxblock,
            elapsed_s=time.monotonic() - start,
            block_mass_kg=float(body.mass),
            control_reconstruction=dict(
                arm_max_rad=drive_error, finger_max_m=finger_error
            ),
            freeze_semantics={
                "original": "original reconstructed control schedule; optional protected maintenance tail",
                "fixed-wrist": "left six arm position drive targets at historical first-holder tick, zero arm velocity drive, unchanged gripper and right arm schedule",
                "hold-gripper": "left finger position drives at historical first-holder tick with zero finger velocity drive; arm and right gripper follow original schedule until protected maintenance tail",
            }[args.mode],
            source_hashes={
                str(p.relative_to(args.pace)): sha256(p.read_bytes()).hexdigest()
                for p in [
                    args.pace / "envs/_base_task.py",
                    args.pace / "envs/robot/robot.py",
                ]
            },
        ),
    )
    print("CPU_RESULT", (args.output / "result.json").read_text(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ["raw", "release", "assets", "pace", "output"]:
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--window", type=int, choices=[31, 50], required=True)
    p.add_argument(
        "--mode", choices=["original", "fixed-wrist", "hold-gripper"], required=True
    )
    p.add_argument("--settle-ticks", type=int, default=0, choices=range(301))
    run(p.parse_args())
