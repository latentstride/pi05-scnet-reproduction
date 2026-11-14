"""Versioned, CPU-only position-bounded IK for the audited six-joint URDF chains."""
from hashlib import sha256

import numpy as np

from robotwin_control_geometry import checked_pose, finite_array
from verify_robotwin_fk import Chain, rot, transform

BACKEND = "g2_urdf_damped_jacobian_v1"


def rotation_vector(r):
    v = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    angle = np.arctan2(np.linalg.norm(v), np.clip((np.trace(r) - 1) / 2, -1, 1))
    if angle < 1e-8:
        return v
    if np.pi - angle < 1e-5:
        axis = np.linalg.eigh((r + r.T) / 2)[1][:, -1]
        return (-axis if axis @ v < 0 else axis) * angle
    return v * (angle / np.linalg.norm(v))


def pose_error(goal, actual):
    """Translation and rotation residuals in the goal/actual common frame."""
    return np.r_[goal[:3, 3] - actual[:3, 3], rotation_vector(goal[:3, :3] @ actual[:3, :3].T)]


class ArmChain:
    def __init__(self, urdf, expected_sha256, names, terminal, position_limits=None):
        if sha256(urdf.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(f"URDF hash mismatch: {urdf}")
        self.asset_sha256 = expected_sha256
        self.names = tuple(names)
        chain = Chain(urdf, terminal)
        self.joints = []
        active, limits = [], []
        for node in chain.chain:
            origin = node.find("origin")
            xyz = [0, 0, 0] if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            r, p, y = [0, 0, 0] if origin is None else np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            frame = checked_pose(transform(xyz, rot([0, 0, 1], y) @ rot([0, 1, 0], p) @ rot([1, 0, 0], r)))
            kind = node.get("type")
            axis = None
            if kind == "revolute":
                active.append(node.get("name"))
                bounds = node.find("limit")
                limits.append([float(bounds.get(k)) for k in ("lower", "upper")])
                tag = node.find("axis")
                axis = finite_array([1, 0, 0] if tag is None else np.fromstring(tag.get("xyz"), sep=" "), (3,))
                if not np.linalg.norm(axis):
                    raise ValueError("zero axis")
                axis /= np.linalg.norm(axis)
            elif kind != "fixed":
                raise ValueError(f"unsupported arm joint: {kind}")
            self.joints.append((frame, axis))
        if tuple(active) != self.names or len(active) != 6 or len(set(active)) != 6:
            raise ValueError(f"URDF chain order mismatch: {active} vs {self.names}")
        self.limits = finite_array(limits, (6, 2))
        if position_limits is not None:
            extra = finite_array(position_limits, (6, 2))
            self.limits = np.column_stack((np.maximum(extra[:, 0], self.limits[:, 0]),
                                           np.minimum(extra[:, 1], self.limits[:, 1])))
        if np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("empty arm joint interval")

    def fk(self, q, jacobian=False):
        q = iter(finite_array(q, (6,)))
        t = np.eye(4)
        axes, origins = [], []
        for frame, axis in self.joints:
            t = t @ frame
            if axis is not None:
                if jacobian:
                    axes.append(t[:3, :3] @ axis)
                    origins.append(t[:3, 3].copy())
                t[:3, :3] = t[:3, :3] @ rot(axis, next(q))
        if not jacobian:
            return t
        return t, np.vstack((np.cross(axes, t[:3, 3] - np.asarray(origins)).T, np.asarray(axes).T))

    def ik(self, goal, seed, check_deadline, *, pos_tol=0.0005, ang_tol=np.deg2rad(0.05)):
        """Generate residual-checked candidates in deterministic seed order."""
        goal = checked_pose(goal)
        lo, hi = self.limits.T
        midpoint = (lo + hi) / 2
        best = None
        weight = np.array([1, 1, 1, .25, .25, .25])
        seeds = [finite_array(seed, (6,)), midpoint,
                 lo + (hi - lo) * .25, lo + (hi - lo) * .75]
        for index, initial in enumerate(seeds):
            q = np.clip(initial, lo, hi)
            for _ in range(90):
                check_deadline()
                current, j = self.fk(q, jacobian=True)
                error = pose_error(goal, current)
                p, a = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
                residual = np.linalg.norm(error * weight)
                if best is None or residual < best["weighted_error"]:
                    best = dict(weighted_error=float(residual), position_m=float(p), angle_deg=float(np.rad2deg(a)))
                if p <= pos_tol and a <= ang_tol:
                    sigma = np.linalg.svd(j * weight[:, None], compute_uv=False)[-1]
                    if sigma >= 1e-5:
                        yield q.copy(), dict(seed=index, position_m=float(p),
                                            angle_deg=float(np.rad2deg(a)), sigma_min=float(sigma))
                    break
                j *= weight[:, None]
                delta = j.T @ np.linalg.solve(j @ j.T + 1e-5 * np.eye(6), error * weight)
                delta *= min(1., .35 / max(np.linalg.norm(delta), 1e-12))
                for scale in (1., .5, .25, .125):
                    candidate = np.clip(q + scale * delta, lo, hi)
                    if np.linalg.norm(pose_error(goal, self.fk(candidate)) * weight) < residual - 1e-12:
                        q = candidate
                        break
                else:
                    break
        return best
