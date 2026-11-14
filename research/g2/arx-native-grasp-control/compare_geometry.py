"""Measure the native and archived grasp in one audited TCP convention."""
import argparse
import json
from pathlib import Path

import numpy as np

from g2_adapter import deadline, solve_pair
from g2_arx_direct import kinematic_pair
from g2_assets import load_arms
from g2_kinematics import pose_error
from robotwin_control_geometry import pose_from_wxyz


def matrix(pose):
    return pose_from_wxyz(pose[:3], pose[3:])


def tools(arms, frames, state):
    return np.array([frame @ arm.chain.fk(q) @ arm.offset()
                     for frame, arm, q in zip(frames, arms, state)])


def block_local(tool, center):
    return (1000 * (tool[:3, :3].T @ (center - tool[:3, 3]))).tolist()


def native_case(spec, assets, audit, trace):
    config = spec["config"]
    arx = load_arms(assets, audit, "ARX-X5", config["calibration"]["ARX-X5"])
    aloha = load_arms(assets, audit, "aloha-agilex", config["calibration"]["aloha-agilex"])
    target_frames = [matrix(p) for p in spec["initial"]["robot_base_world"].values()]
    source_frames = [matrix(p) for p in config["source_base_world"]]
    source_seed = np.array(config["source_proxy_seed_rad"])
    target_seed = np.array(spec["at"]["actual_arm"])
    residuals = []
    for row in spec["reference"]["native_rows"]:
        row = np.array(row).reshape(2, 7)
        original = tools(arx, target_frames, row[:, :6])
        pending = [a.grip.target(g) for a, g in zip(aloha, row[:, 6])]
        source_seed, _ = solve_pair(aloha, source_frames, original, source_seed, pending,
                                    kinematic_pair, deadline(5))
        mapped = tools(aloha, source_frames, source_seed)
        pending = [a.grip.target(g) for a, g in zip(arx, row[:, 6])]
        target_seed, _ = solve_pair(arx, target_frames, mapped, target_seed, pending,
                                    kinematic_pair, deadline(5))
        final = tools(arx, target_frames, target_seed)
        residual = np.array([pose_error(reference, output) for reference, output in zip(original, final)])
        residuals.append(np.column_stack((1000*np.linalg.norm(residual[:, :3], axis=1),
            np.rad2deg(np.linalg.norm(residual[:, 3:], axis=1)))))
    with np.load(trace, allow_pickle=False) as data:
        tick = spec["reference"]["first_left_contact_tick"]
        instances = data["actual_qpos"][tick].reshape(2, 8)
        tcp = tools(arx, target_frames, instances[:, :6])
        native = data["block_pose"][tick, :3]
        valid = data["contacts"][:, 0].astype(int)
        bilateral_tick = int(np.flatnonzero(valid & 3 == 3)[0])
    peak = np.max(residuals, axis=0)
    return dict(native_episode=spec["source"]["episode"], input_waypoints=len(residuals),
                roundtrip_max_mm=peak[:, 0].tolist(), roundtrip_max_deg=peak[:, 1].tolist(),
                first_finger_tick=tick, first_bilateral_tick=bilateral_tick,
                first_finger_native_drive_request=spec["reference"]["normalized_left_request_at_first_contact"],
                first_contact_block_in_true_tcp_mm=block_local(tcp[0], native))


def policy_case(spec):
    event = spec["reference"]["physics_at_first_contact"]
    approved = spec["reference"]["approved"]["requests"][0]
    center = np.array(event["contact_state"]["block_pose_task"])[:3, 3]
    start = np.array(spec["at"]["contact_state"]["block_pose_task"])[:3, 3]
    target = np.array(approved["task_tools"][0])
    measured = np.array(event["actual_tools_task"][0])
    closure_rows = np.flatnonzero(np.array(spec["chunk"])[:, 6] <= .5) + 1
    delta = pose_error(target, measured)
    return dict(job=spec["job"], query=spec["query"], row=spec["row"], tick=event["tick"],
                event_line=spec["reference"]["first_contact_line"],
                action_sha256=spec["action_sha256"], events_sha256=spec["events_sha256"],
                raw_grip=spec["chunk"][spec["row"]-1][6], first_half_closed_row=(
                    int(closure_rows[0]) if closure_rows.size else None),
                source_target_block_in_tcp_mm=block_local(target, center),
                actual_block_in_tcp_mm=block_local(measured, center),
                block_displacement_mm=(1000*(center-start)).tolist(),
                ik_residual_mm=1000*approved["residual"][0]["position_m"],
                interpolated_target_to_actual_mm=1000*event["position_error_m"][0],
                row_endpoint_to_actual_mm=1000*float(np.linalg.norm(delta[:3])),
                row_endpoint_to_actual_deg=float(np.rad2deg(np.linalg.norm(delta[3:]))),
                actual_fingers_mm=(1000*np.array(event["actual_fingers"][0])).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("inputs", "assets", "audit", "native_runs", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    cases = [native_case(json.loads((args.inputs / f"{name}.json").read_text()),
                         args.assets, args.audit, args.native_runs / name / "attempt/physics-trace.npz")
             for name in ("episode01", "episode07")]
    cases.append(policy_case(json.loads((args.inputs / "first-push.json").read_text())))
    args.output.write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
