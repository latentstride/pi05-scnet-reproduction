"""Locate push displacement and later closure in the existing offline Q2 trace."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def analyze(folder):
    spec = json.loads((folder / "inputs/q2-block8.json").read_text())
    dynamic = folder / "dynamic/q2-final-tail"
    raw = gzip.decompress((dynamic / "events.jsonl.gz").read_bytes())
    receipt = json.loads((dynamic / "receipt.json").read_text())
    original_hash = json.loads((folder / "dynamic/physical-summary.json").read_text())[
        "q2-final-tail"]["full_events_sha256"]
    if sha256(raw).hexdigest() != original_hash:
        raise ValueError("archived physical events digest differs")
    block_at_start = np.array(spec["at"]["contact_state"]["block_pose_task"])[:3, 3]
    first_crossing, rows = {}, {}
    for line in raw.splitlines():
        event = json.loads(line)
        if not event.get("physics_step_count"):
            continue
        row, contact = event["original_policy_row"], event["contact_state"]
        pose = np.array(contact["block_pose_task"])[:3, 3]
        movement = 1000 * float(np.linalg.norm((pose - block_at_start)[:2]))
        observed = np.array(event["actual_fingers"][0])
        record = dict(row=row, tick=event["tick"], block_planar_displacement_mm=movement,
                      raw_left_grip=spec["chunk"][row - 1][6],
                      drive_left_fingers_mm=(1000*np.array(event["target_fingers"][0])).tolist(),
                      observed_left_fingers_mm=(1000*observed).tolist(),
                      observed_left_grip=float(np.mean(observed)/.044),
                      left_contacts=len(contact["contact_fingers"]["left"]), holds=contact["holders"])
        for threshold in (1, 5, 10):
            if movement >= threshold:
                first_crossing.setdefault(threshold, record)
        if row in (8, 21, 27, 50):
            boundaries = rows.setdefault(row, {"first": record})
            boundaries["last"] = record
    return dict(job=spec["job"], query=spec["query"], source_action_sha256=spec["action_sha256"],
                source_events_sha256=original_hash, first_crossing_mm=first_crossing, rows=rows,
                contact_steps=receipt["contact_steps"], max_contact_fingers=receipt["max_contact_fingers"],
                stable_grasp=receipt["stable_grasp"], no_new_model_queries=True,
                counterfactual="Starts at reconstructed query 2 row 7; archived tail execution is offline only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.archive), indent=2, allow_nan=False) + "\n")
