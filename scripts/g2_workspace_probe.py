#!/usr/bin/env python3
"""Kinematic screening of archived G1 targets against nominal ARX bases.

Uses nominal RoboTwin TCP, not contact calibration. Not a paired scene roster.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from g2_assets import load_arms
from g2_adapter import deadline
from g2_episode import save_json
from g2_kinematics import BACKEND
from robotwin_control_geometry import inverse_pose, pose_from_wxyz

ROOT = Path(__file__).resolve().parents[1]


def nominal_arms(robot):
    rows = {}
    for side, prefix in zip(("left", "right"), ("fl_", "fr_") if robot == "aloha-agilex" else ("", "")):
        rows[side] = dict(finger_names=[prefix + "joint7", prefix + "joint8"],
                         closed_m=[0., 0.], opened_m=[.04, .04],
                         grip_source="geometry_only_no_contact_claim",
                         instance=robot + "_" + side, child_joint=np.eye(4).tolist(),
                         tcp_tool=np.eye(4).tolist(), id="nominal_tcp_geometry_only")
    return load_arms(ROOT / "research/robotwin/embodiment_assets",
                     ROOT / "research/robotwin/embodiment_config_audit.json", robot, rows)


def probe(reference, output):
    source = nominal_arms("aloha-agilex")
    target = nominal_arms("ARX-X5")
    summary = json.loads((reference / "aloha-summary.json").read_text())
    base_a = [pose_from_wxyz(summary["initial_poses"][side + "_robot_base"]["position"],
                              summary["initial_poses"][side + "_robot_base"]["quaternion_wxyz"])
              for side in ("left", "right")]
    base_r = [pose_from_wxyz([x, -.35, .784], [.707, 0, 0, .707]) for x in (-.3, .3)]
    measurements = [json.loads(line) for line in (reference / "observations.jsonl").read_text().splitlines()]
    seed = np.array([[.2, .3, -.4, .2, .3, -.2]] * 2)
    records = []
    for entry in measurements:
        state = np.array(entry["measured_qpos"]).reshape(2, 7)
        pair = []
        for side, a, r, ba, br, q, q0 in zip(("left", "right"), source, target, base_a, base_r, state[:, :6], seed):
            world_tool = ba @ a.chain.fk(q) @ a.offset()
            link_goal = inverse_pose(br) @ world_tool @ inverse_pose(r.offset())
            candidates = []
            search = r.chain.ik(link_goal, q0, deadline(3.))
            while True:
                try:
                    candidates.append(next(search))
                except StopIteration as done:
                    best = done.value
                    break
            row = dict(side=side, target_world_tcp=world_tool.tolist(),
                       status="kinematic_solution" if candidates else "ik_failed",
                       best_residual=best, accepted_candidates=len(candidates))
            if candidates:
                selected = min(candidates, key=lambda item: np.linalg.norm(item[0] - q0))
                row.update(arx_q=selected[0].tolist(), residual=selected[1])
                seed[("left", "right").index(side)] = selected[0]
            pair.append(row)
        records.append(dict(g1_step=entry["step"], arms=pair,
                            both_have_kinematic_solution=all(x["accepted_candidates"] for x in pair)))
    output.mkdir(parents=True, exist_ok=True)
    result = dict(data_kind="diagnostic", gpu_used=False, simulator_run=False, backend=BACKEND,
                  reference_job="126905534", source_scene_seed=summary["scene_seed"],
                  nominal_arx_bases=[x.tolist() for x in base_r],
                  contact_calibration_verified=False, collision_checked=False,
                  actual_arx_snapshot_available=False, formal_pairs_frozen=False,
                  note="Finite IK search failure does not prove geometric unreachability.",
                  input_sha256={p.name: sha256(p.read_bytes()).hexdigest()
                                for p in reference.iterdir() if p.is_file()},
                  records=records)
    save_json(output / "workspace-probe.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=ROOT / "research/g2/g1-reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.reference, args.output)
    print(json.dumps({"samples": len(result["records"]),
                      "both_arms_solved": sum(x["both_have_kinematic_solution"] for x in result["records"]),
                      "formal_pairs_frozen": False}))
