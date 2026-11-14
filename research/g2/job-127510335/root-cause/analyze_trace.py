"""Extract per-action TCP and raw PhysX contact evidence from the archived trial."""

import argparse
from collections import Counter
import csv
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from g2_assets import load_arms
from robotwin_control_geometry import pose_from_wxyz


def check_file(path, expected):
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"artifact checksum mismatch: {path.name}")


def position(poses, side):
    return np.asarray(poses[side])[:3, 3]


def block_distance(tool, box):
    return float(np.linalg.norm(tool - box))


def load_frozen_arms(release, assets):
    config = release / "configs/unified-seed0-native-w50.json"
    return load_arms(
        assets,
        release / "asset-audit.json",
        "aloha-agilex",
        json.loads(config.read_text())["calibration"]["aloha-agilex"],
    )


def physical_contacts(sample, finger_links):
    for contact in sample["contacts"]:
        names = set(contact["body_names"])
        if "box" not in names or not names.intersection(finger_links):
            continue
        for point in contact["points"]:
            yield dict(pair=contact["body_names"], **point)


def collect(run, publication, release, assets):
    result = json.loads((publication / "result.json").read_text())
    trial = Path("trials/author-native-h50-seed100000")
    for leaf in ("receipt.json", "events.jsonl", "chunk-00.npy"):
        name = trial / leaf
        check_file(run / name, result["source_sha256"][str(name)])
    arms = load_frozen_arms(release, assets)
    finger_links = {"fl_link7", "fl_link8", "fr_link7", "fr_link8"}
    chunk = np.load(run / trial / "chunk-00.npy", allow_pickle=False)
    rows, contact_counts, phase_counts = [], Counter(), Counter()
    actual, guard = None, None
    max_errors = np.zeros((2, 2))
    closest_tcp_mm = np.full(2, np.inf)
    final_raw = []
    with (run / trial / "events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            kind = event.get("kind")
            if kind == "author_reset_admitted":
                init = event["initial"]
                bases = [
                    pose_from_wxyz(v[:3], v[3:])
                    for v in init["robot_base_world"].values()
                ]
            elif kind == "author_action":
                index = event["index"]
                row = dict(action=index + 1, first_tick=event["tick"], completed=False)
                for i, side in enumerate(("left", "right")):
                    q = chunk[index, i * 7 : i * 7 + 6]
                    goal = bases[i] @ arms[i].chain.fk(q) @ arms[i].offset()
                    row.update(
                        {
                            f"{side}_model_goal_{a}_m": float(v)
                            for a, v in zip("xyz", goal[:3, 3])
                        }
                    )
                    row[f"{side}_model_gripper"] = float(chunk[index, 7 * i + 6])
                rows.append(row)
            elif kind == "native_contact_sample":
                raw = list(physical_contacts(event, finger_links))
                contact_counts["tick_samples_with_finger_block_manifold"] += bool(raw)
                contact_counts[
                    "tick_samples_with_nonpositive_finger_block_separation"
                ] += any(p["separation"] <= 0 for p in raw)
                contact_counts["tick_samples_with_finger_block_impulse_ge_1e-8"] += any(
                    np.linalg.norm(p["impulse"]) >= 1e-8 for p in raw
                )
                block = np.asarray(event["block_pose_task"])[:3, 3]
                final_raw = raw
            elif kind == "native_guard_state":
                guard = event["contact_state"]
                phase_counts[guard["phase"]] += 1
                for i in range(2):
                    dist = 1000 * block_distance(
                        position(actual["actual_tools_task"], i), block
                    )
                    closest_tcp_mm[i] = min(closest_tcp_mm[i], dist)
            elif kind is None and event.get("physics_step_count") == 1:
                actual = event
                max_errors[0] = np.maximum(max_errors[0], event["position_error_m"])
                max_errors[1] = np.maximum(
                    max_errors[1], event["orientation_error_deg"]
                )
            elif kind in ("author_action_complete", "author_terminal"):
                row = rows[-1]
                row.update(
                    last_tick=actual["tick"],
                    completed=kind == "author_action_complete",
                    contact_phase=guard["phase"],
                    physics_steps=actual["tick"] - row["first_tick"],
                )
                for i, side in enumerate(("left", "right")):
                    for label, poses in (
                        ("actual_tcp", actual["actual_tools_task"]),
                        ("applied_drive_tcp", actual["target_tools_task"]),
                    ):
                        row.update(
                            {
                                f"{side}_{label}_{a}_m": float(v)
                                for a, v in zip("xyz", position(poses, i))
                            }
                        )
                    row[f"{side}_tcp_to_block_center_mm"] = 1000 * block_distance(
                        position(actual["actual_tools_task"], i), block
                    )
    if len(rows) != 43 or sum(r["completed"] for r in rows) != 42:
        raise ValueError("unexpected archived trajectory")
    diagnostic = json.loads((publication / "trial-receipt.json").read_text())["result"][
        "failure"
    ]["diagnostic"]
    # Advect the measured near-contact points by the arm rigid transform.
    # This first-order check uses archived PhysX normals, not FCL.
    a = bases[0] @ arms[0].chain.fk(diagnostic["actual_arm"][0])
    endpoints = {
        "next_drive": diagnostic["drive_arm"][0],
        "raw_action_endpoint": chunk[42, :6],
    }
    projected_gaps = {}
    for label, q in endpoints.items():
        move = (bases[0] @ arms[0].chain.fk(q)) @ np.linalg.inv(a)
        projected_gaps[label] = [
            float(
                1000
                * (
                    point["separation"]
                    + np.dot(
                        (move @ np.r_[point["position"], 1])[:3] - point["position"],
                        point["normal"],
                    )
                )
            )
            for point in final_raw
        ]
    limit_mm = diagnostic["allowed_penetration_m"] * 1000
    depth_mm = diagnostic["depth_m"] * 1000
    return rows, dict(
        actual_contact_phases_by_tick=phase_counts,
        frozen_contact_normal_projected_gaps_mm=projected_gaps,
        raw_physx_finger_block_contacts=contact_counts,
        final_tick_finger_block_physx_points=final_raw,
        maximum_tracking_position_error_mm=(max_errors[0] * 1000).tolist(),
        maximum_tracking_angle_error_deg=max_errors[1].tolist(),
        closest_measured_tcp_to_block_center_mm=closest_tcp_mm.tolist(),
        production_candidate_penetration_mm=depth_mm,
        frozen_penetration_limit_mm=limit_mm,
        single_predicate_offline_only=[
            dict(counterfactual_limit_mm=t, rejected=depth_mm > t)
            for t in (1.9, 2.0, 2.025, 2.05, 2.051, 2.052, 2.075, 2.1, 2.2)
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "publication", "release", "assets", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    rows, summary = collect(a.run, a.publication, a.release, a.assets)
    with (a.output / "trajectory-43.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (a.output / "trace.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
