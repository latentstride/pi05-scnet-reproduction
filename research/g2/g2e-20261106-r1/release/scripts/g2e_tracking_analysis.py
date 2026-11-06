"""Analyze all original Aloha physics steps without running a model or simulator."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(raw, output):
    raw = raw.resolve(strict=True)
    output = output.resolve()
    if output == raw or raw in output.parents:
        raise ValueError("analysis output must be outside the original run")
    output.mkdir(parents=True, exist_ok=False)
    report = {}
    for directory in sorted((raw / "trials").glob("*A-native*")):
        actual, target, errors, times, fingers, labels = [], [], [], [], [], []
        queries, requested, window_goals = 0, [], []
        first_bad = None
        with (directory / "events.jsonl").open() as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                if row.get("kind") == "snapshot":
                    queries += 1
                    window_goals = []
                elif row.get("kind") == "gripper_target" and row["side"] == "left":
                    window_goals.append(row["normalized"])
                    requested.append(row["normalized"])
                elif row.get("kind") == "executed_step":
                    actual.append(row["actual_arm"])
                    target.append(row["target_arm"])
                    errors.append(
                        [row["position_error_m"], row["orientation_error_deg"]]
                    )
                    times.append(row["sim_s"])
                    fingers.append(row["actual_fingers"])
                    goals = np.asarray(window_goals)
                    label = (
                        "all_open"
                        if np.all(goals > 0.8)
                        else "all_close"
                        if np.all(goals < 0.5)
                        else "mixed"
                    )
                    labels.append(label)
                    if first_bad is None and (
                        max(row["position_error_m"]) > 0.03
                        or max(row["orientation_error_deg"]) > 10
                    ):
                        first_bad = dict(query=queries, line=line_number, event=row)
        actual, target, errors, fingers = (
            np.array(actual),
            np.array(target),
            np.array(errors),
            np.array(fingers),
        )
        delta = target - actual
        wrapped = (delta + np.pi) % (2 * np.pi) - np.pi
        per_joint = []
        for side in range(2):
            for joint in range(6):
                value = delta[:, side, joint]
                per_joint.append(
                    dict(
                        side=["left", "right"][side],
                        joint=joint + 1,
                        rmse_rad=float(np.sqrt(np.mean(value**2))),
                        abs_p95_rad=float(np.percentile(np.abs(value), 95)),
                        abs_max_rad=float(np.abs(value).max()),
                        wrapped_rmse_rad=float(
                            np.sqrt(np.mean(wrapped[:, side, joint] ** 2))
                        ),
                    )
                )
        masks = np.array(labels)
        phases = {}
        for label in ("all_open", "all_close", "mixed"):
            selected = masks == label
            phases[label] = dict(
                physics_steps=int(selected.sum()),
                actual_left_finger_mean_m=np.mean(fingers[selected, 0], axis=0).tolist()
                if selected.any()
                else None,
                actual_left_aperture_sum_range_m=[
                    float(fingers[selected, 0].sum(axis=1).min()),
                    float(fingers[selected, 0].sum(axis=1).max()),
                ]
                if selected.any()
                else None,
            )
        report[directory.name] = dict(
            physics_steps=len(actual),
            joint_metrics=per_joint,
            position_error_rmse_m=np.sqrt(np.mean(errors[:, 0] ** 2, axis=0)).tolist(),
            position_error_p95_m=np.percentile(errors[:, 0], 95, axis=0).tolist(),
            position_error_max_m=errors[:, 0].max(axis=0).tolist(),
            position_error_over_30mm_fraction=np.mean(
                errors[:, 0] > 0.03, axis=0
            ).tolist(),
            angle_error_over_10deg_fraction=np.mean(errors[:, 1] > 10, axis=0).tolist(),
            first_existing_tracking_limit_exceeded=first_bad,
            gripper_command_windows=phases,
            total_sim_s=float(times[-1]),
            command_left_min=float(min(requested)),
            scope="every recorded physics step; gripper window category can include interpolated transitions; no original per-step finger drive or object contacts available",
        )
    (output / "tracking.json").write_text(json.dumps(report, indent=2) + "\n")
    print("TRACKING_STEPS", sum(r["physics_steps"] for r in report.values()))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.raw, args.output)
