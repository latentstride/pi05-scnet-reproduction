"""Actual initial snapshots and explicit reset fixtures; seeds never imply matching."""
from hashlib import sha256
import json

import numpy as np

from g2_adapter import SIDES, Snapshot, Stamp, read_snapshot, G2Error
from g2_capture import capture_arms
from g2_kinematics import pose_error
from g2_robotwin_backend import pose7
from robotwin_control_geometry import checked_pose, inverse_pose, pose_from_wxyz, finite_array, Status

OBJECTS = {"block": "box", "target": "target_box", "table": "table"}


def digest_json(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def matrix7(values):
    values = finite_array(values, (7,))
    return pose_from_wxyz(values[:3], values[3:])


def raw_actor(actor):
    # RoboTwin Actor exposes get_pose but not set_pose; mutations use its Entity.
    return getattr(actor, "actor", actor)


def capture_initial(env, arms, world_task, episode, fixture_sha256, workspace_check_id, workspace_pass=False):
    stamp = Stamp(episode, 0, 0.)
    snapshot = Snapshot(stamp, capture_arms(env.robot, stamp), {}, checked_pose(world_task))
    q, native, _, _ = read_snapshot(snapshot, arms)
    task_world = inverse_pose(world_task)
    objects = {}
    for name, attribute in OBJECTS.items():
        p = getattr(env, attribute).get_pose()
        objects[name] = pose7(task_world @ pose_from_wxyz(p.p, p.q))
    return dict(robot_base_world={side: pose7(sample.world_base) for side, sample in zip(SIDES, snapshot.arms)},
                task_frame_world=pose7(world_task),
                actual_arm_qpos_rad={side: row.tolist() for side, row in zip(SIDES, q)},
                actual_gripper={side: arm.grip.read(value) for side, arm, value in zip(SIDES, arms, native)},
                objects_task=objects, reset_fixture_sha256=fixture_sha256,
                workspace_check_id=workspace_check_id, common_workspace_pass=bool(workspace_pass))


def compare_initial(actual, expected, *, position_m=.001, angle_deg=.1, joint_rad=.001, grip_fraction=.01):
    failures = []
    for key, a, b in [("task_frame_world", actual["task_frame_world"], expected["task_frame_world"])] + [
        ("base_" + side, actual["robot_base_world"][side], expected["robot_base_world"][side]) for side in SIDES
    ]:
        error = pose_error(matrix7(a), matrix7(b))
        if np.linalg.norm(error[:3]) > position_m or np.rad2deg(np.linalg.norm(error[3:])) > angle_deg:
            failures.append(key)
    if set(actual["objects_task"]) != set(expected["objects_task"]):
        failures.append("object_names")
    else:
        for name in actual["objects_task"]:
            error = pose_error(matrix7(actual["objects_task"][name]), matrix7(expected["objects_task"][name]))
            if np.linalg.norm(error[:3]) > position_m or np.rad2deg(np.linalg.norm(error[3:])) > angle_deg:
                failures.append("object_" + name)
    for side in SIDES:
        a = finite_array(actual["actual_arm_qpos_rad"][side], (6,))
        b = finite_array(expected["actual_arm_qpos_rad"][side], (6,))
        if np.max(abs(a-b)) > joint_rad:
            failures.append(side + "_q")
        if abs(actual["actual_gripper"][side] - expected["actual_gripper"][side]) > grip_fraction:
            failures.append(side + "_gripper")
    return failures


def compare_objects(a, b, *, position_m=.001, angle_deg=.1):
    """Compare two independent task-frame snapshots across embodiments."""
    if set(a["objects_task"]) != set(b["objects_task"]):
        return ["object_names"]
    failures = []
    for name in a["objects_task"]:
        error = pose_error(matrix7(a["objects_task"][name]), matrix7(b["objects_task"][name]))
        if np.linalg.norm(error[:3]) > position_m or np.rad2deg(np.linalg.norm(error[3:])) > angle_deg:
            failures.append(name)
    return failures


def apply_fixture(env, arms, fixture, embodiment, make_pose):
    """Apply the same recorded object poses with per-embodiment named joint states.

    No physics step is inserted; capture_initial must verify the result before
    policy use. This is an explicit reset intervention, not a recreated seed.
    """
    if fixture["task"] != "handover_block" or set(fixture["objects_world"]) != set(OBJECTS):
        raise ValueError("fixture must contain the three fixed task actors")
    robot_fixture = fixture["robots"][embodiment]
    # Validate the complete request before modifying either arm or any actor.
    updates = {}
    for side, arm in zip(SIDES, arms):
        entity = getattr(env.robot, side + "_entity")
        names = [joint.get_name() for joint in entity.get_active_joints()]
        if len(set(names)) != len(names):
            raise ValueError("duplicate articulation joint name")
        row = robot_fixture[side]
        q = finite_array(row["q_rad"], (6,))
        g = finite_array(row["fingers_m"], (2,))
        if np.any((q < arm.chain.limits[:, 0]) | (q > arm.chain.limits[:, 1])):
            raise G2Error(Status.LIMIT, "fixture arm state exceeds limits")
        arm.grip.read(g)
        for values, limits in ((g, arm.grip.limits),):
            if np.any((values < limits[:, 0]) | (values > limits[:, 1])):
                raise G2Error(Status.LIMIT, "fixture finger state exceeds limits")
        base = matrix7(row["base_world"])
        key = id(entity)
        if key not in updates:
            updates[key] = dict(entity=entity, names=names, q=np.asarray(entity.get_qpos()).copy(), base=base)
        elif not np.allclose(updates[key]["base"], base, atol=1e-9, rtol=0):
            raise ValueError("one shared articulation cannot receive two base poses")
        for name, value in zip(arm.chain.names + arm.grip.names, np.r_[q, g]):
            updates[key]["q"][names.index(name)] = value
    poses = {name: matrix7(pose) for name, pose in fixture["objects_world"].items()}
    for state in updates.values():
        entity = state["entity"]
        base = pose7(state["base"])
        entity.set_root_pose(make_pose(base[:3], base[3:]))
        entity.set_qpos(state["q"])
        entity.set_qvel(np.zeros_like(state["q"]))
        for joint, value in zip(entity.get_active_joints(), state["q"]):
            joint.set_drive_target(float(value))
            joint.set_drive_velocity_target(0.)
    for name, matrix in poses.items():
        actor = raw_actor(getattr(env, OBJECTS[name]))
        p = pose7(matrix)
        actor.set_pose(make_pose(p[:3], p[3:]))
        # SAPIEN 3 exposes dynamic velocities through its physical component.
        import sapien
        component = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if component is not None:
            component.linear_velocity = np.zeros(3)
            component.angular_velocity = np.zeros(3)
    for side, arm in zip(SIDES, arms):
        native = robot_fixture[side]["fingers_m"]
        setattr(env.robot, side + "_gripper_val", arm.grip.read(native))
