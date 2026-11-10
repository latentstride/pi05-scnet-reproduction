"""Compare archived chunk windows and initial-state-matched ARX expert evidence."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


POLICY = "unified-seed1-arx-full-w4"
EXPERT = "unified-seed1-arx-expert-repaired"


def hash_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def window_scan(folder):
    rows = []
    for path in sorted(folder.glob("action-*.npy")):
        actions = np.load(path, allow_pickle=False)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError("unexpected model output")
        positions = []
        for col in (6, 13):
            idx = np.flatnonzero(actions[:, col] <= 0.5)
            positions.append(int(idx[0]) + 1 if len(idx) else None)
        rows.append(
            dict(
                query=int(path.stem.split("-")[1]),
                first_close=positions,
                source_sha256=hash_file(path),
            )
        )
    return dict(
        chunks=len(rows),
        rows=rows,
        retaining_at_least_one_close={
            w: [
                sum(
                    r["first_close"][i] is not None and r["first_close"][i] <= w
                    for r in rows
                )
                for i in (0, 1)
            ]
            for w in (4, 26, 31, 50)
        },
    )


def read(path):
    return json.loads(path.read_bytes())


def compare(current, old):
    policy = current / "trials" / POLICY / "episode"
    expert = old / "trials" / EXPERT / "episode"
    historical_policy = old / "trials" / POLICY / "episode"
    fields = (
        "robot_base_world",
        "task_frame_world",
        "objects_task",
        "reset_fixture_sha256",
        "actual_arm_qpos_rad",
        "actual_gripper",
    )
    left = read(policy / "result.json")["initial"]
    right = read(expert / "initial-readback.json")
    initial = {k: left[k] == right[k] for k in fields}
    state = None
    with (policy / "events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "executed_step":
                state = row
    if state["tick"] != 16160:
        raise ValueError("unexpected terminal state")

    def relative(row):
        return np.linalg.inv(
            np.asarray(row["contact_state"]["block_pose_task"])
        ) @ np.asarray(row["actual_tools_task"])

    policy_relative = relative(state)
    experts = Counter()
    nearest = None
    first = {}
    with (expert / "events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if "contact_state" not in row or not row.get("physics_step_count"):
                continue
            phase = row["contact_state"]["phase"]
            experts[phase] += 1
            first.setdefault(phase, row["tick"])
            if phase != "approach":
                continue
            candidate = relative(row)
            error = np.linalg.norm(
                candidate[:, :3, 3] - policy_relative[:, :3, 3], axis=1
            )
            rotations = np.einsum(
                "nji,njk->nik", candidate[:, :3, :3], policy_relative[:, :3, :3]
            )
            angle = np.rad2deg(
                np.arccos(
                    np.clip((np.trace(rotations, axis1=1, axis2=2) - 1) / 2, -1, 1)
                )
            )
            rank = float(np.linalg.norm(error))
            if nearest is None or rank < nearest["combined_tcp_position_error_m"]:
                nearest = dict(
                    tick=row["tick"],
                    combined_tcp_position_error_m=rank,
                    position_error_m=error.tolist(),
                    angle_error_deg=angle.tolist(),
                    joint_error_max_rad=float(
                        np.max(
                            np.abs(np.asarray(row["actual_arm"]) - state["actual_arm"])
                        )
                    ),
                )
    if not experts or nearest is None:
        raise ValueError("missing actual expert phase evidence")
    return dict(
        same_seed_expert=dict(
            initial_exact_by_field=initial,
            first_tick_by_phase=first,
            measured_phase_counts=experts,
            policy_stop_tick=state["tick"],
            closest_expert_approach_by_object_relative_two_tcp_position=nearest,
            expert_initial_sha256=hash_file(expert / "initial-readback.json"),
            expert_events_sha256=hash_file(expert / "events.jsonl"),
            expert_succeeded=read(expert.parent / "receipt.json")["success"],
        ),
        historical_127147943_policy_windows=window_scan(historical_policy),
        terminal_127510363_windows=window_scan(policy),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("current", "historical"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    print(json.dumps(compare(a.current, a.historical), indent=2))
