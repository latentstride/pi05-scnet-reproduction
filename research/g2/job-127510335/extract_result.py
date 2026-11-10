"""Audit the standalone author run from its original, locally available artifacts."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        h = sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
        return h.hexdigest()


def array_digest(array):
    value = np.ascontiguousarray(array)
    meta = json.dumps([value.dtype.str, value.shape], separators=(",", ":")).encode()
    return sha256(meta + b"\0" + value.tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect(root, submission):
    trial = Path("trials") / submission["episode"]
    source = {}

    def artifact(path):
        source[str(path)] = digest(root / path)
        return root / path

    auth = read(artifact("authorization-receipt.json"))
    for key in ("job_id", "archive_sha256", "configuration_sha256", "commit"):
        require(auth[key] == submission[key], f"submission binding differs: {key}")
    require(
        auth["candidate_sha256"] == submission["manifest_sha256"], "manifest differs"
    )
    campaign = read(artifact("supervisor/campaign.json"))
    row = campaign["rows"][submission["episode"]]
    standalone = read(artifact("standalone-result.json"))
    receipt = read(artifact(trial / "receipt.json"))
    require(
        len(campaign["rows"]) == 1
        and row == standalone["row"]
        and receipt == row["result"]
        and standalone["job_id"] == auth["job_id"],
        "supervisor or standalone receipt differs",
    )
    model = read(artifact("model/startup.json"))
    artifact("model/phases.jsonl")
    graphics = artifact("graphics/check_status.tsv")
    first = artifact(trial / "request-00.npz")
    chunk = np.load(artifact(trial / "chunk-00.npy"), allow_pickle=False)
    require(chunk.shape == (50, 14), "unexpected first chunk shape")
    with np.load(first, allow_pickle=False) as request:
        camera = {k: request[k] for k in request.files if k.startswith("cam_")}
        require(
            len(camera) == 3
            and all(
                x.shape == (3, 240, 320) and x.max() > x.min() for x in camera.values()
            ),
            "invalid camera evidence",
        )
        observations = dict(
            state=request["state"].tolist(),
            prompt=request["prompt"].item(),
            cameras={
                k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=array_digest(v))
                for k, v in camera.items()
            },
        )
    require(
        observations["prompt"] == receipt["prompt"]
        and len(observations["state"]) == 14,
        "first request differs",
    )
    counters, holders, fingers, phases = Counter(), Counter(), Counter(), Counter()
    started, finished, metadata = [], [], {}
    with artifact(trial / "events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            kind = event.get("kind", "physics_step")
            counters[kind] += 1
            if kind == "native_guard_state":
                state = event["contact_state"]
                phases[state["phase"]] += 1
                holders.update(state["holders"])
                for side, contact in state["contact_fingers"].items():
                    fingers[side] += bool(contact)
            elif kind == "author_action":
                started.append(event)
            elif kind == "author_action_complete":
                finished.append(event)
            elif kind in (
                "author_query",
                "author_chunk",
                "author_model_rng_reset",
                "author_reset_admitted",
                "author_terminal",
            ):
                metadata[kind] = event
    require(
        metadata["author_chunk"]["sha256"] == array_digest(chunk)
        and counters["author_query"] == counters["author_chunk"] == 1
        and [q["index"] for q in started] == list(range(receipt["native_action_count"]))
        and [q["index"] for q in finished] == list(range(receipt["completed_actions"]))
        and all(
            np.array_equal(q["row"], chunk[q["index"]])
            and q["row_sha256"] == array_digest(chunk[q["index"]])
            for q in started
        )
        and counters["physics_step"] == receipt["physics_steps"]
        and {k: v for k, v in metadata["author_terminal"].items() if k != "kind"}
        == receipt,
        "action, physics or final evidence differs",
    )
    steps = [q["physics_steps"] for q in finished]
    require(
        finished[-1]["tick"] == started[-1]["tick"] == sum(steps),
        "partial action boundary differs",
    )
    return {
        "job_id": auth["job_id"],
        "runtime_binding": {
            k: auth[k]
            for k in (
                "archive_sha256",
                "configuration_sha256",
                "commit",
                "candidate_sha256",
            )
        },
        "model": {k: model[k] for k in ("checkpoint_revision", "status", "devices")},
        "graphics": dict(
            line.split("\t") for line in graphics.read_text().splitlines()
        ),
        "model_server_starts": campaign["model_server_starts"],
        "peak_gpu_mib": campaign["observed_gpu_peak_mib"],
        "campaign_stop_reason": campaign["stop_reason"],
        "trial_status": row["status"],
        "process_group_reaped": row["process_group_reaped"],
        "scene": metadata["author_reset_admitted"],
        "first_observation": observations,
        "first_query": metadata["author_query"],
        "first_chunk": metadata["author_chunk"],
        "model_rng_reset": metadata["author_model_rng_reset"],
        "event_counts": counters,
        "contact_phases_by_tick": phases,
        "holder_ticks": holders,
        "tracked_finger_contact_ticks": fingers,
        "first_chunk_gripper_ranges": {
            side: [float(chunk[:, i].min()), float(chunk[:, i].max())]
            for side, i in (("left", 6), ("right", 13))
        },
        "last_attempted_row": started[-1],
        "per_completed_action_physics_steps": steps,
        "last_attempted_action_partial_physics_steps": receipt["physics_steps"]
        - started[-1]["tick"],
        "unattempted_one_based_actions_in_first_chunk": list(
            range(len(started) + 1, len(chunk) + 1)
        ),
        "receipt_sha256": source[str(trial / "receipt.json")],
        "source_sha256": source,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("submission_receipt", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(args.run_directory, read(args.submission_receipt)),
            ensure_ascii=False,
            indent=2,
        )
    )
