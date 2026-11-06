#!/usr/bin/env python3
"""Derive diagnostic runtime configurations from audited URDF/config and G1 snapshots."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from g2_episode import save_json
from g2_initial import digest_json

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, asset_root, audit_path, reference):
    output.mkdir(parents=True, exist_ok=False)
    audit = json.loads(audit_path.read_text())
    g1 = json.loads(reference.read_text())
    calibration, endpoints = {}, {}
    identity = np.eye(4).tolist()
    for row in audit["robots"]:
        robot = row["robot"]
        urdf = asset_root / row["urdf_member"]
        from hashlib import sha256
        if sha256(urdf.read_bytes()).hexdigest() != row["urdf_sha256"]:
            raise ValueError("URDF differs from audit")
        joints = {node.get("name"): node for node in ET.parse(urdf).getroot().findall("joint")}
        calibration[robot], endpoints[robot] = {}, {}
        for side, prefix in zip(("left", "right"), ("fl_", "fr_") if robot == "aloha-agilex" else ("", "")):
            names = [prefix + "joint7", prefix + "joint8"]
            limits = [[float(joints[name].find("limit").get(key)) for key in ("lower", "upper")] for name in names]
            raw_closed, raw_opened = row["gripper_scale"]
            closed = [max(lo, min(hi, raw_closed)) for lo, hi in limits]
            opened = [max(lo, min(hi, raw_opened)) for lo, hi in limits]
            evidence = dict(urdf_sha256=row["urdf_sha256"], configured_closed_m=raw_closed,
                            configured_open_m=raw_opened, urdf_finger_limits_m=limits,
                            rule="intersect configured endpoint with each URDF prismatic interval",
                            final_closed_m=closed, final_opened_m=opened,
                            calibration_scope="nominal simulation geometry; contact dynamics unverified")
            endpoints[robot][side] = evidence
            calibration[robot][side] = dict(
                finger_names=names, closed_m=closed, opened_m=opened,
                grip_source="g2_URDF_endpoint_intersection_v1:" + digest_json(evidence),
                instance="bound_to_actual_articulation_after_setup", child_joint=identity,
                tcp_tool=identity, id="nominal_RoboTwin_TCP_v1:" + row["urdf_sha256"],
                geometry_only=True, contact_dynamics_verified=False)
    for side in ("left","right"):
        calibration["ARX-X5"][side].update(grip_state_mode="independent_aperture_v1",
            grip_state_source="audited ARX URDF: opposed prismatic axes, common link6 parent, no mimic; aperture sum v1")
    base_a = [0, -.65, 0, .7071067811865476, 0, 0, .7071067811865476]
    fixture = dict(task="handover_block", data_kind="diagnostic",
                   origin="explicit reset from G1 job 126905534 object snapshots; robot q set to declared home",
                   source_reference_sha256=digest_json(g1), candidate_id="g1-126905534-scene100000",
                   formal_pair_id=None, objects_world={}, robots={})
    for key in ("block", "target", "table"):
        p = g1["initial_poses"][key]
        fixture["objects_world"][key] = p["position"] + p["quaternion_wxyz"]
    for robot in calibration:
        fixture["robots"][robot] = {}
        for side, x in zip(("left", "right"), (-.3, .3)):
            base = base_a if robot == "aloha-agilex" else [x, -.35, .784, *base_a[3:]]
            fixture["robots"][robot][side] = dict(base_world=base, q_rad=[0]*6,
                                                  fingers_m=calibration[robot][side]["opened_m"])
    control = dict(source="G2 experiment-imposed simulation command bounds v1; not manufacturer ratings",
                   arm_velocity_rad_s=np.full((2, 6), .5).tolist(),
                   arm_acceleration_rad_s2=np.ones((2, 6)).tolist(),
                   finger_velocity_m_s=np.full((2, 2), .03).tolist(),
                   finger_acceleration_m_s2=np.full((2, 2), .1).tolist())
    common = dict(data_kind="diagnostic", task="handover_block", scene_seed=g1["scene_seed"],
                  episode_info={}, prompt=g1["prompt"], window=4, max_actions=500, timeout_s=600,
                  budget_scope="per-episode diagnostic ceiling; not an approved/frozen experiment budget",
                  calibration=calibration, endpoint_evidence=endpoints,
                  world_task=[0,0,0,1,0,0,0], source_world_task=[0,0,0,1,0,0,0],
                  source_base_world=[base_a, base_a], source_proxy_seed_rad=[[0]*6]*2,
                  source_proxy_velocity_rad_s=np.ones((2,6)).tolist(), arx_base_distance_m=.6,
                  control_period_s=.02, collision_joint_step_rad=.02, collision_finger_step_m=.001,
                  max_plan_ticks=2000, control_limits=control, tracking_position_limit_m=.03,
                  tracking_angle_limit_deg=10, grip_rule="clip", grip_max_clip=.01,
                  native_grip_max_clip=.01, contact_mode="strict_no_contact_exemptions",
                  reset_fixture=fixture, formal_release_ready=False)
    for robot in calibration:
        for window in (1,4):
            config = deepcopy(common)
            config.update(embodiment=robot, window=window,
                          condition=("A-native" if robot == "aloha-agilex" else "B-arx") + f"-w{window}")
            save_json(output / (config["condition"] + ".json"), config)
    save_json(output / "reset-fixture.json", fixture)
    save_json(output / "endpoint-evidence.json", endpoints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "research/robotwin/embodiment_assets")
    parser.add_argument("--audit", type=Path, default=ROOT / "research/robotwin/embodiment_config_audit.json")
    parser.add_argument("--g1-reference", type=Path, default=ROOT / "research/g2/g1-reference/aloha-summary.json")
    args = parser.parse_args()
    prepare(args.output, args.assets, args.audit, args.g1_reference)
