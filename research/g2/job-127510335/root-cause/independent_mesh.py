"""Cross-check frozen poses using URDF FK and convex-mesh/box separating axes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
from xml.etree import ElementTree

import numpy as np
from scipy.spatial import ConvexHull

from robotwin_control_geometry import pose_from_wxyz
from analyze_trace import load_frozen_arms


def vertices(stl):
    content = stl.read_bytes()
    count = struct.unpack_from("<I", content, 80)[0]
    if len(content) != 84 + 50 * count:
        raise ValueError("expected a binary STL")
    rows = np.frombuffer(
        content,
        dtype=np.dtype([("v", "<f4", (12,)), ("attr", "<u2")]),
        offset=84,
        count=count,
    )
    return np.unique(rows["v"][:, 3:].reshape(-1, 3), axis=0).astype(float)


def sat(finger, hull, half_extent):
    basis = np.eye(3)
    edges = np.vstack(
        [
            finger[hull.simplices[:, (i + 1) % 3]] - finger[hull.simplices[:, i]]
            for i in range(3)
        ]
    )
    normals = np.cross(
        finger[hull.simplices[:, 1]] - finger[hull.simplices[:, 0]],
        finger[hull.simplices[:, 2]] - finger[hull.simplices[:, 0]],
    )
    axes = np.vstack(
        (basis, normals, np.cross(edges[:, None, :], basis).reshape(-1, 3))
    )
    length = np.linalg.norm(axes, axis=1)
    axes = axes[length > 1e-12] / length[length > 1e-12, None]
    support = np.abs(axes) @ half_extent
    projected = finger @ axes.T
    gap = np.maximum(projected.min(axis=0) - support, -support - projected.max(axis=0))
    i = int(np.argmax(gap))
    return dict(signed_separating_axis_gap_m=float(gap[i]), axis_box=axes[i].tolist())


def collect(run, release, assets, mesh, publication):
    trial = run / "trials/author-native-h50-seed100000"
    stop = json.loads((trial / "receipt.json").read_bytes())
    stage = json.loads((publication / "result.json").read_bytes())
    diagnostic = stop["result"]["failure"]["diagnostic"]
    chunk = np.load(trial / "chunk-00.npy", allow_pickle=False)
    tick = stop["physics_steps"]
    box = None
    with (trial / "events.jsonl").open() as stream:
        for line in stream:
            if f'"tick": {tick},' in line and '"kind": "native_contact_sample"' in line:
                box = np.asarray(json.loads(line)["block_pose_task"])
    if box is None:
        raise ValueError("terminal block pose missing")
    left, _ = load_frozen_arms(release, assets)
    origin = stage["scene"]["initial"]["robot_base_world"]["left"]
    base = pose_from_wxyz(origin[:3], origin[3:])
    urdf = assets / next(
        x["urdf_member"]
        for x in json.loads((release / "asset-audit.json").read_bytes())["robots"]
        if x["robot"] == "aloha-agilex"
    )
    tree = ElementTree.parse(urdf)
    joint = tree.find(".//joint[@name='fl_joint7']")
    link = tree.find(".//link[@name='fl_link7']/collision")
    if (
        link.find("origin").attrib != {"xyz": "0 0 0", "rpy": "0 0 0"}
        or link.find("geometry/mesh").get("filename").split("/")[-1]
        != mesh.name.split("-")[-1]
        or joint.find("origin").get("rpy") != "0 0 0"
    ):
        raise ValueError("unverified collision transform")
    offset = np.fromstring(joint.find("origin").get("xyz"), sep=" ")
    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
    vertices_local = vertices(mesh)
    hull = ConvexHull(vertices_local)

    def gap(arm_q, gripper_q, box_xy_offset=(0.0, 0.0)):
        pose = base @ left.chain.fk(arm_q)
        shift = offset + axis * gripper_q
        world = (vertices_local + shift) @ pose[:3, :3].T + pose[:3, 3]
        block_center = box[:3, 3] + np.r_[box_xy_offset, 0.0]
        in_box = (world - block_center) @ box[:3, :3]
        return sat(in_box, hull, np.asarray((0.03, 0.03, 0.1)))

    actual_q, drive_q = (
        np.array(diagnostic[k][0]) for k in ("actual_arm", "drive_arm")
    )
    f = diagnostic["start_native_m"]
    return {
        "method": "Convex hull of URDF finger STL and oriented task box. Positive gap means separated. Negative gap means overlap.",
        "mesh_sha256": sha256(mesh.read_bytes()).hexdigest(),
        "urdf_sha256": sha256(urdf.read_bytes()).hexdigest(),
        "block_pose_task": box.tolist(),
        "actual": gap(actual_q, f),
        "next_drive": gap(drive_q, f),
        "raw_action_endpoint": gap(chunk[42, :6], f),
        "raw_action_endpoint_with_rejected_drive_finger": gap(
            chunk[42, :6], diagnostic["requested_native_m"]
        ),
        "next_drive_with_rejected_drive_finger": gap(
            drive_q, diagnostic["requested_native_m"]
        ),
        "offline_block_xy_offsets": [
            dict(dx_m=dx, dy_m=dy, **gap(drive_q, f, (dx, dy)))
            for dx in (-0.0001, -0.00005, 0.0, 0.00005, 0.0001)
            for dy in (-0.0001, -0.00005, 0.0, 0.00005, 0.0001)
        ],
        "actual_to_drive_segment": [
            dict(fraction=float(x), **gap(actual_q + (drive_q - actual_q) * x, f))
            for x in np.linspace(0, 1, 11)
        ],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "release", "assets", "mesh", "publication"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    print(
        json.dumps(collect(a.run, a.release, a.assets, a.mesh, a.publication), indent=2)
    )
