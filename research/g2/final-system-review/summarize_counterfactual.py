"""Compute measured outcomes from the two raw CPU PhysX streams."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np


FINGERS = {"fl_link7", "fl_link8", "fr_link7", "fr_link8"}


def read(path):
    return json.loads(path.read_bytes())


def summarize(root):
    modes = ("author_native", "protected")
    cases = {}
    for mode in modes:
        where = root / mode
        result = read(where / "result.json")
        stats = Counter()
        first, last = None, None
        min_separation, max_impulse, max_speed = None, 0.0, 0.0
        max_angular_speed, max_displacement, max_angle = 0.0, 0.0, 0.0
        phase, actions, milestones, contacts = {}, {}, {}, {}
        hash_stream = sha256()
        with (where / "physics.jsonl").open("rb") as stream:
            for line in stream:
                hash_stream.update(line)
                row = json.loads(line)
                tick, action = row["tick"], row["action"]
                if tick == 0:
                    first = row
                    names = row["joint_names"]
                    finger_indices = [
                        names.index(name)
                        for name in ("fl_joint7", "fl_joint8", "fr_joint7", "fr_joint8")
                    ]
                    home = np.asarray(row["block_pose"])
                last = row
                pose = np.asarray(row["block_pose"])
                shift = float(np.linalg.norm(pose[:3, 3] - home[:3, 3]))
                angular = float(
                    np.rad2deg(
                        np.arccos(
                            np.clip(
                                (np.trace(home[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1
                            )
                        )
                    )
                )
                speed = float(np.linalg.norm(row["block_linear_velocity"]))
                angular_speed = float(np.linalg.norm(row["block_angular_velocity"]))
                max_speed = max(max_speed, speed)
                max_angular_speed = max(max_angular_speed, angular_speed)
                max_displacement = max(max_displacement, shift)
                max_angle = max(max_angle, angular)
                loaded = set()
                finger_points = []
                for pair in row["contacts"]:
                    links = set(pair["pair"]).intersection(FINGERS)
                    if not links:
                        continue
                    for point in pair["points"]:
                        impulse = float(np.linalg.norm(point["impulse"]))
                        sep = point["separation"]
                        if min_separation is None or sep < min_separation:
                            min_separation = sep
                            contacts["minimum_separation"] = dict(
                                tick=tick, action=action, **point, pair=pair["pair"]
                            )
                        if impulse > max_impulse:
                            max_impulse = impulse
                            contacts["maximum_impulse"] = dict(
                                tick=tick, action=action, **point, pair=pair["pair"]
                            )
                        if impulse >= 1e-8:
                            loaded.update(links)
                        finger_points.append(point)
                for flag, cond in (
                    ("near_manifold_ticks", bool(finger_points)),
                    (
                        "negative_separation_ticks",
                        any(x["separation"] < 0 for x in finger_points),
                    ),
                    ("load_bearing_finger_ticks", bool(loaded)),
                    ("left_bilateral_loaded_ticks", {"fl_link7", "fl_link8"} <= loaded),
                    (
                        "right_bilateral_loaded_ticks",
                        {"fr_link7", "fr_link8"} <= loaded,
                    ),
                ):
                    stats[flag] += bool(cond)
                    if cond and flag not in phase:
                        phase[flag] = dict(
                            tick=tick, action=action, loaded=sorted(loaded)
                        )
                compact = dict(
                    tick=tick,
                    action=action,
                    block_pose=row["block_pose"],
                    block_displacement_m=shift,
                    block_rotation_deg=angular,
                    block_linear_velocity=row["block_linear_velocity"],
                    block_angular_velocity=row["block_angular_velocity"],
                    finger_qpos_m=[row["qpos"][i] for i in finger_indices],
                    finger_relative_to_block=row["finger_relative_to_block"],
                    finger_contact_count=len(finger_points),
                    loaded=sorted(loaded),
                )
                actions[action] = compact
                if tick in (0, 4699, 4746, 4747):
                    milestones[tick] = compact
        if last is None or last["tick"] != result["physics_steps"]:
            raise ValueError("physics log does not end at receipt tick")
        cases[mode] = dict(
            result=result,
            physics_log_sha256=hash_stream.hexdigest(),
            first_physical_state=first,
            contact_counts=stats,
            first_events=phase,
            measured_peak=dict(
                finger_block_impulse_N_s=max_impulse,
                finger_block_minimum_separation_m=min_separation,
                block_speed_m_s=max_speed,
                block_angular_speed_rad_s=max_angular_speed,
                block_displacement_m=max_displacement,
                block_rotation_deg=max_angle,
            ),
            extreme_contacts=contacts,
            checkpoints=milestones,
            action_end_states=[
                actions[n] for n in actions if n in (42, 43, 44, 45, 46, 47, 48, 49, 50)
            ],
        )
    max_state_delta = Counter()
    shared = 0
    with (
        (root / modes[0] / "physics.jsonl").open() as a,
        (root / modes[1] / "physics.jsonl").open() as b,
    ):
        for la, lb in zip_longest(a, b):
            if not la or not lb:
                break
            x, y = json.loads(la), json.loads(lb)
            if (x["tick"], x["action"]) != (y["tick"], y["action"]):
                raise ValueError("shared replay control clocks differ")
            for field in (
                "qpos",
                "qvel",
                "block_pose",
                "block_linear_velocity",
                "block_angular_velocity",
            ):
                delta = float(
                    np.max(np.abs(np.asarray(x[field]) - np.asarray(y[field])))
                )
                max_state_delta[field] = max(max_state_delta[field], delta)
            shared = x["tick"]
    return dict(
        cases=cases,
        last_shared_tick=shared,
        shared_prefix_max_absolute_difference=max_state_delta,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    a = p.parse_args()
    print(json.dumps(summarize(a.run), indent=2, allow_nan=False))
