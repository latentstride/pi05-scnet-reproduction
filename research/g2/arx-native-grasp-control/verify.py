"""Recompute the recorded outcomes from measured per-tick contact and block state."""
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def confirm(folder):
    root = folder / "evidence"
    hashes = json.loads((root / "hashes.json").read_text())
    for name, expected_sha in hashes.items():
        path = root / name
        data = path.read_bytes() if path.is_file() else gzip.decompress(path.with_suffix(".jsonl.gz").read_bytes())
        if sha256(data).hexdigest() != expected_sha:
            raise ValueError(f"evidence file differs: {name}")
    for case, source in (
        ("policy-at-first-contact", "first-push"),
        ("verify-episode01-native", "episode01"),
        ("verify-episode01-roundtrip", "episode01"),
        ("verify-episode07-native", "episode07"),
        ("verify-episode07-roundtrip", "episode07"),
    ):
        case_path = root / case
        raw = (folder / "inputs" / f"{source}.json").read_bytes()
        spec = json.loads(raw)
        record = json.loads((case_path / "receipt.json").read_text())
        if record["input_sha256"] != sha256(raw).hexdigest():
            raise ValueError(f"input differs in {case}")
        with gzip.open(case_path / "events.jsonl.gz", "rt") as stream:
            events = [e for line in stream if (e := json.loads(line)).get("physics_step_count")]
        if len(events) != record["physical_steps"] or not record["hold_same_tick"]:
            raise ValueError(f"missing physical ticks in {case}")
        finger_counts = np.array([len(e["contact_state"]["contact_fingers"]["left"]) for e in events])
        heights = np.array([e["contact_state"]["block_pose_task"][2][3] for e in events])
        z0 = spec["at"]["contact_state"]["block_pose_task"][2][3]
        rise_mm = 1000*max(0, heights.max() - z0)
        if ((finger_counts == 2).sum() != record["both_left_steps"]
                or not np.isclose(rise_mm, record["max_block_rise_mm"])):
            raise ValueError(f"physical metrics differ in {case}")
        first = events[int(np.flatnonzero(finger_counts)[0])]
        if first["tick"] != record["first_contact"]["tick"]:
            raise ValueError(f"first contact tick differs in {case}")
        if case == "policy-at-first-contact":
            if first["tick"] != spec["reference"]["physics_at_first_contact"]["tick"]:
                raise ValueError("CPU and GPU first contact differ")
            if finger_counts.max() != 1 or record["first_holder"] is not None:
                raise ValueError("first policy contact differs")
            if record["original_row"] != spec["chunk"][spec["row"]-1]:
                raise ValueError("original model row was not preserved")
        elif not (record["first_holder"] and rise_mm >= 20 and finger_counts.max() == 2):
            raise ValueError(f"grasp and lift were not verified in {case}")
        print(case, len(events), int((finger_counts == 2).sum()), f"{rise_mm:.2f} mm")


if __name__ == "__main__":
    confirm(Path(__file__).resolve().parent)
