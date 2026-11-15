"""Opt-in geometry and feedback controller for the first left ARX grasp.

The caller supplies block pose and provenance explicitly. Uses existing target
IK, motion planning, contact tracker and ARXBackend. No model or scene writes.
"""

from dataclasses import dataclass
from hashlib import sha256
from xml.etree import ElementTree

import numpy as np

from g2_adapter import G2Error, bounds, deadline, read_snapshot, solve_pair
from robotwin_control_geometry import Status, checked_pose, inverse_pose
from verify_robotwin_fk import rot, transform


@dataclass(frozen=True)
class JawGeometry:
    face_extents_tcp_m: np.ndarray
    full_vertices_tcp_m: np.ndarray
    directions: np.ndarray
    travel_m: np.ndarray
    meshes_sha256: tuple

    def faces(self, opening):
        return self.face_extents_tcp_m + (self.directions * self.travel_m[:, None] *
                                          np.asarray(opening).reshape(2, 1))[:, None, :]

    def center(self, opening=(0., 0.)):
        return self.faces(opening).mean(axis=(0, 1))


def urdf_origin(node):
    if node is None:
        return np.eye(4)
    xyz = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
    roll, pitch, yaw = np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
    return transform(xyz, rot([0, 0, 1], yaw) @ rot([0, 1, 0], pitch) @ rot([1, 0, 0], roll))


def load_jaws(urdf, arm):
    """Audited opposed jaws with inward pad planes from the collision STL."""
    if sha256(urdf.read_bytes()).hexdigest() != arm.chain.asset_sha256:
        raise ValueError("grasp geometry URDF identity differs")
    root = ElementTree.parse(urdf).getroot()
    joints = {j.attrib["name"]: j for j in root.findall("joint")}
    links = {j.attrib["name"]: j for j in root.findall("link")}
    stl_triangle = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    faces, meshes, directions, all_vertices, parents = [], [], [], [], []
    for name, closed in zip(arm.grip.names, arm.grip.closed):
        joint = joints[name]
        parents.append(joint.find("parent").get("link"))
        joint_tcp = inverse_pose(arm.offset()) @ urdf_origin(joint.find("origin"))
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        axis_tcp = joint_tcp[:3, :3] @ axis
        link = links[joint.find("child").get("link")]
        collision, = link.findall("collision")
        mesh = collision.find("geometry/mesh")
        path = urdf.parent / mesh.get("filename")
        raw = path.read_bytes()
        vertices = np.frombuffer(raw, stl_triangle, offset=84)["vertices"].reshape(-1, 3)
        scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        frame = joint_tcp @ urdf_origin(collision.find("origin"))
        points = (vertices * scale) @ frame[:3, :3].T + frame[:3, 3] + closed * axis_tcp
        projections = points @ axis_tcp
        pad = points[np.isclose(projections, projections.min(), atol=1e-6, rtol=0)]
        faces.append(np.stack((pad.min(axis=0), pad.max(axis=0))))
        directions.append(axis_tcp)
        all_vertices.append(points)
        meshes.append(sha256(raw).hexdigest())
    if len(set(parents)) != 1 or not np.allclose(directions, [[0, -1, 0], [0, 1, 0]], atol=1e-8):
        raise ValueError("unsupported jaw coordinate convention")
    return JawGeometry(np.array(faces), np.concatenate(all_vertices), np.array(directions),
                       np.asarray(arm.grip.opened) - np.asarray(arm.grip.closed), tuple(meshes))


class FirstLeftGrasp:
    def __init__(self, adapter, source, target, block_half_size_m, *, pose_source):
        if pose_source != "oracle_object_pose":
            raise ValueError("the evaluated pose provider is the CPU oracle")
        self.adapter, self.source, self.target = adapter, source, target
        self.half = np.asarray(block_half_size_m)
        if self.half.shape != (3,) or not np.all(self.half > 0):
            raise ValueError("missing audited box dimensions")

    def raw_targets(self, row):
        return np.array([base @ arm.chain.fk(raw[:6]) @ arm.offset()
                         for base, arm, raw in zip(self.adapter.source_frames, self.adapter.aloha, row)])

    def center_target(self, nominal, block, *, standoff=False):
        """Center the left closing planes on the object with policy orientation and height."""
        result = checked_pose(nominal).copy()
        block = checked_pose(block)
        tcp_block = inverse_pose(result) @ block
        half = np.abs(tcp_block[:3, :3]) @ self.half
        center = self.target.center()
        # Keep the policy's approach distance in pad coordinates. Do not drive
        # the center planes beyond the object on an open-gripper approach.
        approach = max(0., tcp_block[0, 3] - self.source.center()[0])
        if standoff:
            approach = max(approach, half[0] + self.target.full_vertices_tcp_m[:, 0].max()
                           - center[0] + .006)
        correction = np.array([tcp_block[0, 3] - center[0] - approach,
                               tcp_block[1, 3] - center[1],
                               self.source.center()[2] - center[2]])
        result[:3, 3] += result[:3, :3] @ correction
        return result

    def open_clearance(self, tool, block, normalized):
        tool_block = inverse_pose(tool) @ block
        half = np.abs(tool_block[:3, :3]) @ self.half
        faces = self.target.faces(normalized).mean(axis=1)
        center = self.target.center()
        offset = tool_block[:3, 3] - center
        clear = np.min([tool_block[1, 3] - half[1] - faces[0, 1],
                        faces[1, 1] - tool_block[1, 3] - half[1]])
        pad_bounds = self.target.face_extents_tcp_m
        floor, ceiling = tool_block[2, 3] - half[2], tool_block[2, 3] + half[2]
        overlap_z = min(ceiling, pad_bounds[:, 1, 2].min()) - max(floor, pad_bounds[:, 0, 2].max())
        # The jaw center must be inside the object's longitudinal support and
        # both jaws must clear it before moving the drive toward closure.
        ready = bool(abs(offset[0]) < .004 and abs(offset[1]) < .004 and clear > .001
                     and overlap_z > .004)
        return dict(ready=ready, offset_m=offset, side_clearance_m=float(clear),
                    half_width_m=float(half[1]), pad_height_overlap_m=float(overlap_z))

    def plan(self, snapshot, tools, grip, remaining):
        q, measured, frames, actual = read_snapshot(snapshot, self.adapter.arx)
        world = self.adapter.target_collision
        check = deadline(remaining())
        world.refresh()
        if not world.measured_clear(q, measured):
            raise G2Error(Status.COLLISION, "measured ARX state blocks adapted plan",
                          geometry="actual", collisions=world.last_collisions)
        native = np.array([arm.grip.target(float(amount)) for arm, amount in zip(self.adapter.arx, grip)])
        for arm, goal in zip(self.adapter.arx, native):
            bounds(goal, arm.grip.limits, geometry="candidate")
        tools, phase = world.prepare_tools(tools, native, actual)
        target, residual = solve_pair(self.adapter.arx, frames, tools, q, native, world, check)
        world.action_barrier(target, native, check)
        return self.adapter.motion.plan(q, measured, [(target, native)], world, check), residual
