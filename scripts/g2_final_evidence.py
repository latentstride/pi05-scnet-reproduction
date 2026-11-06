"""Analyze the immutable G2-E prediction and physical event excerpts."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from g2_adapter import G2Error, Stamp
from g2_contact import ContactTracker
from g2_kinematics import pose_error


def read(path):
    return json.loads(path.read_text())


def contact_objects(contacts):
    return [
        SimpleNamespace(
            bodies=[
                SimpleNamespace(entity=SimpleNamespace(per_scene_id=i))
                for i in contact["body_ids"]
            ],
            points=[SimpleNamespace(impulse=p["impulse"]) for p in contact["points"]],
        )
        for contact in contacts
    ]


def replay_slip(rows, output):
    steps = {row["tick"]: row for row in rows if row.get("physics_step_count") == 1}
    tracker = ContactTracker(68, {"left": [33, 34], "right": [43, 44]})
    first = failure = None
    series = []
    previous = None
    for sample in rows:
        if sample.get("kind") != "native_contact_sample":
            continue
        tick = sample["tick"]
        step = steps[tick]
        tools = np.asarray(step["actual_tools_task"])
        block = np.asarray(sample["block_pose_task"])
        contacts = sample["contacts"]
        speed = (
            None
            if previous is None
            else np.linalg.norm(tools[0, :3, 3] - previous[0])
            / (step["sim_s"] - previous[1])
        )
        previous = (tools[0, :3, 3], step["sim_s"])
        try:
            tracker.observe(
                Stamp("replay", tick, step["sim_s"]),
                contact_objects(contacts),
                tools,
                block,
            )
        except G2Error as error:
            if str(error) != "object slipped relative to confirmed holder":
                raise
            failure = tick
        if "left" not in tracker.holders:
            continue
        first = tick if first is None else first
        residual = pose_error(block, tools[0] @ tracker.holders["left"])

        def impulse(body):
            return sum(
                np.linalg.norm(p["impulse"])
                for c in contacts
                if 68 in c["body_ids"] and body in c["body_names"]
                for p in c["points"]
            )

        series.append(
            dict(
                tick=tick,
                drift_mm=1000 * np.linalg.norm(residual[:3]),
                drift_deg=np.rad2deg(np.linalg.norm(residual[3:])),
                block_z_m=block[2, 3],
                table_impulse_ns=impulse("table"),
                left_finger1_impulse_ns=impulse("fl_link7"),
                left_finger2_impulse_ns=impulse("fl_link8"),
                left_drive_m=sample["target_fingers_drive_m"][0][0],
                left_finger1_m=step["actual_fingers"][0][0],
                left_finger2_m=step["actual_fingers"][0][1],
                left_tcp_error_mm=1000 * step["position_error_m"][0],
                left_tcp_speed_m_s=speed,
            )
        )
        if failure is not None:
            break
    assert first == 4802 and failure == 4845
    with output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(series[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(series)
    return dict(
        first_confirmation=first,
        anchor_tick=first - 2,
        failure_tick=failure,
        samples=len(series),
        support_samples=sum(bool(r["table_impulse_ns"] > 0) for r in series),
        first=series[0],
        final=series[-1],
    )


def run(raw, clips, output):
    result = {}
    for trial in sorted((raw / "trials").iterdir()):
        row = read(trial / "result.json")
        requests = read(raw / "derived" / trial.name / "requests.json")
        arrays = [
            np.load(p, allow_pickle=False) for p in sorted(trial.glob("action-*.npy"))
        ]
        assert len(arrays) == row["queries"] and all(
            a.shape == (50, 14) and np.isfinite(a).all() for a in arrays
        )
        hands = {}
        for side, col in (("left", 6), ("right", 13)):
            first = [
                next((i for i, v in enumerate(a[:, col]) if v < 0.5), None)
                for a in arrays
            ]
            issued = [
                r["action_index_in_episode"]
                for r in requests
                if r["normalized_qpos_request"][col] < 0.5
            ]
            hands[side] = dict(
                first_raw_indices=first,
                predictions=sum(x is not None for x in first),
                in_prefix=sum(x is not None and x < row["window"] for x in first),
                request_indices=issued,
                completed_indices=[i for i in issued if i < row["executed"]],
            )
        clip = read(clips / trial.name / "last-action.json")
        rejection = next(
            r for r in clip["rows"] if r.get("kind") == "native_guard_rejection"
        )
        result[trial.name] = dict(
            hands=hands, rejection=rejection, historical_success=row["success"]
        )
    assert len(result) == 8 and not any(
        r["historical_success"] for r in result.values()
    )
    band = read(clips / "g2e-seed1-native-w26/hold-band.json")
    slip = replay_slip(band, output / "k26-slip.csv")
    clip = read(clips / "g2e-seed1-native-w31" / "last-action.json")
    before = clip["rows"][0]["previous"]
    final = clip["rows"][1]["normalized_qpos_request"][6]
    # The rejected substep itself was never recorded. Its monotone interpolation
    # stays between the previous command (zero) and this next request.
    k31 = dict(
        previous_drive_m=before["contact"]["target_fingers_drive_m"][0],
        measured_fingers_m=before["step"]["actual_fingers"][0],
        next_normalized_request=final,
        max_positive_drive_delta_mm=55 * final,
        first_rejected_substep_drive="NOT_RECORDED",
        physical_closed_m=0.0,
        drive_end_m=-0.01 + 0.055 * final,
    )
    (output / "failure-evidence.json").write_text(
        json.dumps(dict(trials=result, slip=slip, k31=k31), indent=2) + "\n"
    )
    print(json.dumps(slip, indent=2))
    print(
        "Right-hand predictions:",
        [
            (n, r["hands"]["right"]["predictions"], r["hands"]["right"]["in_prefix"])
            for n, r in result.items()
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "clips", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.raw, args.clips, args.output)
