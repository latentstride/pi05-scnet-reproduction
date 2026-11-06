"""Capture actual named articulation state without using RoboTwin drive-target getters."""
from g2_adapter import ArmSample, Snapshot, Stamp, SIDES
from robotwin_control_geometry import pose_from_wxyz, checked_pose, inverse_pose
import numpy as np


def articulation_id(entity):
    root_entity = getattr(getattr(entity, "root", None), "entity", None)
    scene_id = getattr(root_entity, "per_scene_id", None)
    return str(entity.get_name()) + (":" + str(scene_id) if scene_id is not None else "")


def capture_arms(robot, stamp):
    if not isinstance(stamp, Stamp):
        raise ValueError("capture requires authoritative episode/physics stamp")
    samples = []
    for side in SIDES:
        entity = getattr(robot, side + "_entity")
        joints = entity.get_active_joints()
        names = tuple(joint.get_name() for joint in joints)
        pose = entity.get_root_pose()
        samples.append(ArmSample(stamp, articulation_id(entity), names,
                                 np.asarray(entity.get_qpos()).copy(),
                                 pose_from_wxyz(pose.p, pose.q)))
    return tuple(samples)


def capture_task_objects(env, world_task):
    task_world = inverse_pose(checked_pose(world_task))
    objects = {}
    for name, attribute in (("block", "box"), ("target", "target_box"), ("table", "table")):
        actor = getattr(env, attribute, None)
        if actor is not None:
            value = actor.get_pose()
            objects[name] = task_world @ pose_from_wxyz(value.p, value.q)
    return objects or None


def capture_snapshot(env, stamp, world_task):
    """Capture synchronous RGB and named actual state while physics is paused."""
    samples = capture_arms(env.robot, stamp)
    observation = env.get_obs()
    images = {}
    for name, source in zip(("cam_high", "cam_left_wrist", "cam_right_wrist"),
                             ("head_camera", "left_camera", "right_camera")):
        rgb = np.asarray(observation["observation"][source]["rgb"])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"invalid RGB capture: {source}")
        images[name] = (stamp, np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return Snapshot(stamp, tuple(samples), images, checked_pose(world_task).copy(),
                    capture_task_objects(env, world_task)), observation
