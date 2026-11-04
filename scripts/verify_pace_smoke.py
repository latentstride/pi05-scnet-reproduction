#!/usr/bin/env python3
"""Read back Fake Policy smoke artifacts and verify they describe one consistent run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from imageio_ffmpeg import count_frames_and_secs
import numpy as np

from pace_runtime import CAMERAS, PACE_REVISION, validate_request, validate_response


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root, native=False):
    def rows(name):
        with (root / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    summary = json.loads((root / "summary.json").read_text())
    steps, queries = rows("steps.csv"), rows("queries.csv")
    observations = [json.loads(line) for line in (root / "observations.jsonl").read_text().splitlines()]
    require(summary["status"] == "complete" and summary["backend"] == "fake", "incomplete fake run")
    if native:
        require(summary["environment"] == "robotwin" and summary["pace_revision"] == PACE_REVISION
                and len(summary["initial_poses"]) == 5, "missing native scene evidence")
        require(summary["cache_split_seed"] == summary["cache_index"] == 0, "unexpected scene selection")
    require(summary["control_steps"] == len(steps) <= summary["max_steps"]
            and summary["take_action_cnt"] - summary["initial_take_action_cnt"] == len(steps)
            and len(steps) > 0, "inconsistent control counter")
    require(summary["queries_attempted"] == summary["queries_received"] == len(queries)
            <= summary["max_queries"] and len(queries) > 0, "inconsistent query counter")
    require(summary["observations"] == len(observations) == len(steps) + 1, "missing fresh observations")
    hashes = {name: set() for name in CAMERAS}
    for index, obs in enumerate(observations):
        require(obs["step"] == index, "observation sequence is out of order")
        for name in ("state", "measured_qpos"):
            values = np.asarray(obs[name])
            require(values.shape == (14,) and np.isfinite(values).all(), f"invalid {name}")
        require(set(obs["cameras"]) == set(CAMERAS), "missing camera evidence")
        for name, record in obs["cameras"].items():
            require(len(record["shape"]) == 3 and record["shape"][0] == 3
                    and min(record["shape"]) > 0 and record["dtype"] == "uint8"
                    and 0 <= record["min"] <= record["max"] <= 255, f"invalid {name} format")
            require(not native or record["min"] < record["max"], f"uniform {name}")
            hashes[name].add(record["sha256"])
    require(summary["distinct_frames"] == {name: len(values) for name, values in hashes.items()},
            "inconsistent camera history")
    if native:
        require(any(len(values) > 1 for values in hashes.values()),
                "all three views are byte-identical across the entire rollout")

    with np.load(root / "initial_observation.npz", allow_pickle=False) as data:
        images = {name: data[name] for name in CAMERAS}
        validate_request({"state": data["state"], "images": images, "prompt": summary["prompt"]})
        require(data["state"].dtype == np.float32
                and np.array_equal(data["state"], observations[0]["state"]), "initial state mismatch")
        for name, image in images.items():
            require(hashlib.sha256(image).hexdigest() == observations[0]["cameras"][name]["sha256"],
                    f"initial frame mismatch for {name}")

    step_index = 0
    for index, query in enumerate(queries, start=1):
        name = f"chunk_{index:03d}.npy"
        require(query["status"] == "ok" and int(query["query"]) == index
                and int(query["step_before"]) == step_index and query["chunk_file"] == name,
                "query does not reference the latest observation")
        actions = validate_response({"actions": np.load(root / name, allow_pickle=False)})
        expected = np.array(observations[step_index]["state"], dtype=np.float32)
        expected[[6, 13]] = expected[[6, 13]].clip(0, 1)
        require(np.all(actions == expected), "chunk does not match queried state")
        used = int(query["used"])
        planned = min(summary["prefix"], summary["effective_step_limit"] - step_index)
        early_success = index == len(queries) and summary["stop_reason"] == "task_predicate"
        require(0 < used <= planned and (used == planned or early_success), "invalid chunk prefix")
        for row_index in range(used):
            row = steps[step_index]
            step_index += 1
            require((int(row["step"]), int(row["query"]), int(row["chunk_index"]))
                    == (step_index, index, row_index), "step does not reference the correct action")
            require(np.array_equal([float(row[f"qpos_{i}"]) for i in range(14)], actions[row_index]),
                    "step values differ from the saved action")
    require(step_index == len(steps), "steps not accounted for")
    reason = summary["stop_reason"]
    require(reason == "task_predicate"
            or (reason == "control_step_limit" and step_index == summary["effective_step_limit"])
            or (reason == "inference_call_limit" and len(queries) == summary["max_queries"]
                and step_index < summary["effective_step_limit"]), "inconsistent stop reason")
    count, _ = count_frames_and_secs(root / "head.mp4")
    require(count == len(observations), "video frame count does not match observations")
    return {"passed": True, "environment": summary["environment"], "queries": len(queries),
            "steps": len(steps), "frames": count, "stop_reason": reason}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.output, native=args.native), indent=2))
