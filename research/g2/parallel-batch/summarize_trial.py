"""Extract phase witnesses from a real trial without changing its outcome."""

import argparse
import json
from pathlib import Path

import numpy as np


def rows(path):
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                if line.endswith("\n"):
                    yield json.loads(line)


def summarize(trial):
    names = ("left", "right")
    prediction = dict.fromkeys(names, "NOT_REACHED")
    completed_close = dict.fromkeys(names, "NOT_REACHED")
    contact = dict.fromkeys(names, "NOT_REACHED")
    stages = dict.fromkeys(
        (
            "left_bilateral",
            "both_bilateral",
            "right_bilateral",
            "release_after_contact",
            "target_support_after_contact",
        ),
        "NOT_REACHED",
    )
    chunks = sorted(trial.glob("chunk-*.npy"), key=lambda p: int(p.stem.split("-")[1]))
    for chunk in chunks:
        values = np.load(chunk, allow_pickle=False)
        for side, column in zip(names, (6, 13)):
            found = np.flatnonzero(values[:, column] < 0.2)
            if len(found) and prediction[side] == "NOT_REACHED":
                prediction[side] = dict(chunk=chunk.name, row=int(found[0]) + 1)
    attempted = {}
    horizons = []
    inference = []
    for row in rows(trial / "events.jsonl"):
        kind = row.get("kind")
        if (
            kind in ("response", "author_chunk")
            and row.get("service_inference_s") is not None
        ):
            inference.append(row["service_inference_s"])
        if kind == "author_chunk":
            horizons.append(dict(start=row["query"] * 50, selected=50))
        if kind == "block":
            horizons.append(dict(start=row["step"], selected=row["horizon"]))
        if kind == "author_action":
            attempted[row["index"]] = [row["row"][6], row["row"][13]]
        if kind in ("action_complete", "author_action_complete"):
            grips = (
                row["gripper_request"]
                if kind == "action_complete"
                else attempted[row["index"]]
            )
            for side, value in zip(names, grips):
                if value < 0.2 and completed_close[side] == "NOT_REACHED":
                    completed_close[side] = dict(tick=row["tick"], value=value)

    physics = trial / "physics.jsonl"
    sampled = (
        rows(physics)
        if physics.is_file()
        else (
            r
            for r in rows(trial / "events.jsonl")
            if r.get("kind") == "native_contact_sample"
        )
    )
    previous = False
    first_xyz = None
    max_shift = 0.0
    last_tick = None
    for row in sampled:
        tick = row["tick"]
        last_tick = tick
        pose = row.get("block_pose", row.get("block_pose_task"))
        xyz = np.asarray(pose)[:3, 3]
        if first_xyz is None:
            first_xyz = xyz
        max_shift = max(max_shift, float(np.linalg.norm(xyz - first_xyz)))
        touching = set()
        block_name = row.get("block_actor_name", "box")
        for item in row["contacts"]:
            bodies = item.get("pair", item.get("body_names"))
            # Native physics.jsonl already filters to the block. Protected
            # samples include every manifold, so require a matching actor name.
            if not item["points"] or (
                not physics.is_file() and block_name not in bodies
            ):
                continue
            touching.update(bodies)
        for side, prefix in zip(names, ("fl", "fr")):
            if contact[side] == "NOT_REACHED" and touching.intersection(
                {prefix + "_link7", prefix + "_link8"}
            ):
                contact[side] = tick
        left = {"fl_link7", "fl_link8"} <= touching
        right = {"fr_link7", "fr_link8"} <= touching
        candidates = dict(
            left_bilateral=left and not right,
            both_bilateral=left and right,
            right_bilateral=right and not left,
            release_after_contact=previous and not (left or right),
            target_support_after_contact=previous and "target_box" in touching,
        )
        for key, present in candidates.items():
            if present and stages[key] == "NOT_REACHED":
                stages[key] = tick
        previous = previous or left or right
    receipt_path = trial / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    return dict(
        closure_definition="raw normalized gripper < 0.2 (author helper)",
        contact_definition="observed manifold proxy; not a stable grasp or a slip diagnosis",
        predicted_close=prediction,
        completed_close=completed_close,
        first_finger_contact=contact,
        contact_proxies=stages,
        completed_actions=receipt.get("completed_actions"),
        physics_steps=receipt.get("physics_steps", last_tick),
        chunk_count=len(chunks),
        selected_windows=horizons,
        first_inference_s=inference[0] if inference else None,
        mean_inference_s=sum(inference) / len(inference) if inference else None,
        maximum_block_displacement_m=max_shift if last_tick is not None else None,
        native_success=receipt.get("success"),
        outcome=receipt.get("outcome", "NOT_REACHED"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trial), indent=2))
