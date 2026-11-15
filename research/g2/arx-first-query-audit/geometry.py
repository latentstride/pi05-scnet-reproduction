"""Signed grasp geometry for archived absolute joint targets, in SI and mm."""
import numpy as np

from robotwin_control_geometry import inverse_pose


def fk(arms, frames, joints):
    return np.array([frame @ arm.chain.fk(q) @ arm.offset()
                     for arm, frame, q in zip(arms, frames, joints)])


def relative(tools, block, centers):
    block_in_tcp = np.array([inverse_pose(tool) @ block for tool in tools])
    center_in_block = np.array([(inverse_pose(block) @ tool @ np.r_[center, 1.])[:3]
                                for tool, center in zip(tools, centers)])
    return dict(block_pose_in_tcp=block_in_tcp,
                block_xyz_tcp_mm=1000 * block_in_tcp[:, :3, 3],
                block_xyz_jaw_mm=1000 * (block_in_tcp[:, :3, 3] - centers),
                tcp_in_block=[inverse_pose(block) @ tool for tool in tools],
                jaw_xyz_block_mm=1000 * center_in_block)


def intersects_box(triangles, half):
    """Triangle–box separating axes. True means surface intersection, not penetration depth."""
    edges = np.roll(triangles, -1, axis=1) - triangles
    unit = np.eye(3)
    normal = np.cross(edges[:, 0], edges[:, 1])
    crossed = np.cross(edges[:, :, None, :], unit[None, None, :, :]).reshape(-1, 9, 3)
    axes = np.concatenate((np.broadcast_to(unit, (len(edges), 3, 3)), normal[:, None], crossed), axis=1)
    vertex_projections = np.einsum("nvi,nai->nva", triangles, axes)
    reach = np.abs(axes) @ half
    return bool(np.any(np.all((vertex_projections.min(axis=1) <= reach + 1e-12)
                              & (vertex_projections.max(axis=1) >= -reach - 1e-12), axis=1)))


def finger_hits(jaw, tool, block, opening, half):
    triangles = jaw.full_vertices_tcp_m.reshape(2, -1, 3, 3)
    shifts = jaw.directions * jaw.travel_m[:, None] * np.asarray(opening)[:, None]
    transform = inverse_pose(block) @ tool
    vertices = (triangles + shifts[:, None, None, :]) @ transform[:3, :3].T + transform[:3, 3]
    return [intersects_box(finger, half) for finger in vertices]


def closure_rows(chunk):
    return {str(limit): next((row for row, action in enumerate(chunk, 1) if action[6] <= limit), None)
            for limit in (.95, .5, .1)}
