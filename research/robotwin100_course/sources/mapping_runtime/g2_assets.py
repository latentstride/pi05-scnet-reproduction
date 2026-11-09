"""Construct adapters from hash-audited kinematics and explicit tool/finger calibration."""
from hashlib import sha256
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from g2_adapter import Arm, Grip
from g2_kinematics import ArmChain
from robotwin_control_geometry import tool_offsets


def load_arms(assets, audit_path, robot, calibration):
    audit = json.loads(Path(audit_path).read_text())
    cfg, = [item for item in audit["robots"] if item["robot"] == robot]
    urdf = Path(assets) / cfg["urdf_member"]
    config = Path(assets) / cfg["archive_config_member"]
    if sha256(config.read_bytes()).hexdigest() != cfg["config_sha256"]:
        raise ValueError(f"asset config hash mismatch: {config}")
    _, joint_tcp = tool_offsets(cfg["global_trans_matrix"], cfg["delta_matrix"], cfg["gripper_bias"])
    joints = {node.get("name"): node for node in ElementTree.parse(urdf).getroot().findall("joint")}
    arms = []
    for side, names, terminal in zip(("left", "right"), cfg["joint_names"], cfg["ee_joints"]):
        row = calibration[side]
        finger_names = tuple(row["finger_names"])
        limits = []
        for name in finger_names:
            node = joints[name]
            if node.get("type") != "prismatic":
                raise ValueError(f"{name} is not a prismatic gripper joint")
            limits.append([float(node.find("limit").get(k)) for k in ("lower", "upper")])
        grip = Grip(finger_names, np.array(row["closed_m"]), np.array(row["opened_m"]),
                    np.array(limits), row["grip_source"],
                    state_mode=row.get("grip_state_mode","strict_mimic"),state_source=row.get("grip_state_source",""))
        if grip.state_mode=="independent_aperture_v1":
            a,b=[joints[name] for name in finger_names]
            axes=[np.fromstring(node.find("axis").get("xyz"),sep=" ") for node in (a,b)]
            rotations=[node.find("origin").get("rpy","0 0 0") for node in (a,b)]
            if (any(node.find("mimic") is not None for node in (a,b)) or
                    a.find("parent").get("link")!=b.find("parent").get("link") or
                    rotations[0]!=rotations[1] or not np.allclose(axes[0],-axes[1],atol=1e-9,rtol=0) or
                    not np.isclose(np.linalg.norm(axes[0]),1.)):
                raise ValueError("independent aperture requires opposed unit axes, common parent and no mimic")
        chain = ArmChain(urdf, cfg["urdf_sha256"], names, terminal, row.get("position_limits"))
        arms.append(Arm(chain, row["instance"], np.array(row["child_joint"]), joint_tcp,
                        np.array(row["tcp_tool"]), grip, row["id"]))
    return tuple(arms)
