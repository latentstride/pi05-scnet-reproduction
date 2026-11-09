"""CPU geometry and validation for the RoboTwin control adapter contract."""
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Status(str, Enum):
    OK = "ok"
    INVALID = "invalid_input"
    UNSYNCED = "unsynced"
    UNCALIBRATED = "uncalibrated"
    LIMIT = "position_limit"
    VELOCITY = "velocity_limit"
    TIMING_UNKNOWN = "timing_unknown"
    UNREACHABLE = "unreachable"
    IK_FAILED = "ik_failed"
    COLLISION = "collision"
    PLAN_FAILED = "plan_failed"
    TIMEOUT = "timeout"
    BACKEND_ERROR = "backend_error"


@dataclass(frozen=True)
class Check:
    status: Status
    detail: str = ""

    @property
    def ok(self):
        return self.status == Status.OK


def finite_array(value, shape):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"expected finite array of shape {shape}")
    return array


def checked_pose(matrix):
    matrix = finite_array(matrix, (4, 4))
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError("expected a proper homogeneous rigid transform")
    return matrix


def pose_from_wxyz(xyz, quaternion):
    xyz = finite_array(xyz, (3,))
    q = finite_array(quaternion, (4,))
    scale = np.max(np.abs(q))
    if scale < 1e-12:
        raise ValueError("degenerate quaternion")
    q = q / scale
    w, x, y, z = q / np.linalg.norm(q)
    pose = np.eye(4)
    pose[:3, :3] = [
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ]
    pose[:3, 3] = xyz
    return pose


def inverse_pose(matrix):
    matrix = checked_pose(matrix)
    inverse = np.eye(4)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ matrix[:3, 3]
    return inverse


def tool_offsets(global_rotation, delta_rotation, bias_m):
    """Return T_joint_EE and T_joint_TCP for RoboTwin's two pose conventions."""
    global_rotation = finite_array(global_rotation, (3, 3))
    delta_rotation = finite_array(delta_rotation, (3, 3))
    bias_m = finite_array(bias_m, ()).item()
    result = []
    for distance in (bias_m - 0.12, bias_m):
        offset = np.eye(4)
        offset[:3, :3] = global_rotation @ delta_rotation
        offset[:3, 3] = offset[:3, 0] * distance
        result.append(checked_pose(offset))
    return tuple(result)


def select_joints(actual_order, actual_qpos, requested_order):
    """Select by name within one articulation instance."""
    qpos = finite_array(actual_qpos, (len(actual_order),))
    if len(set(actual_order)) != len(actual_order):
        raise ValueError("duplicate articulation joint name")
    if len(set(requested_order)) != len(requested_order):
        raise ValueError("duplicate requested joint name")
    by_name = dict(zip(actual_order, qpos))
    try:
        return np.array([by_name[name] for name in requested_order])
    except KeyError as exc:
        raise ValueError(f"missing joint {exc.args[0]}") from exc


def gripper_native(g):
    """Configured base drive coordinate in m, before per-finger limit checks."""
    g = finite_array(g, ()).item()
    if not 0 <= g <= 1:
        raise ValueError("gripper outside [0, 1]")
    return -0.01 + 0.055*g


def split_aloha_action(action):
    """Split an absolute 14-vector into left and right (q6, normalized grip)."""
    action = finite_array(action, (14,))
    left_grip, right_grip = float(action[6]), float(action[13])
    gripper_native(left_grip)
    gripper_native(right_grip)
    return (action[:6].copy(), left_grip), (action[7:13].copy(), right_grip)


def check_joint_segment(start, target, position_limits, max_speed, duration_s):
    """Check endpoint bounds and minimum average speed for one segment.

    Acceleration, interpolation extrema and collisions require separate checks.
    None represents unknown timing. Speeds must be supplied by the caller.
    """
    try:
        count = len(position_limits)
        if count == 0:
            raise ValueError("empty joint limit table")
        limits = finite_array(position_limits, (count, 2))
        start = finite_array(start, (count,))
        target = finite_array(target, (count,))
        speed = finite_array(max_speed, (count,))
        if np.any(limits[:, 0] > limits[:, 1]) or np.any(speed <= 0):
            raise ValueError("invalid position or velocity limits")
        if duration_s is not None:
            duration_s = finite_array(duration_s, ()).item()
            if duration_s <= 0:
                raise ValueError("duration must be positive")
    except (TypeError, ValueError, OverflowError) as exc:
        return Check(Status.INVALID, str(exc))

    outside = (np.minimum(start, target) < limits[:, 0]) | (
        np.maximum(start, target) > limits[:, 1])
    if outside.any():
        return Check(Status.LIMIT, f"joint indices {np.flatnonzero(outside).tolist()}")
    if duration_s is None:
        return Check(Status.TIMING_UNKNOWN, "segment duration is unknown")
    with np.errstate(over="ignore"):
        too_fast = np.abs(target - start) / duration_s > speed
    if too_fast.any():
        return Check(Status.VELOCITY, f"joint indices {np.flatnonzero(too_fast).tolist()}")
    return Check(Status.OK)
