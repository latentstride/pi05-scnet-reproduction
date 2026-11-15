"""Summarize completed CPU PhysX receipts and bind them to their raw logs."""
import argparse
import gzip
from hashlib import file_digest
import json
from math import hypot
from pathlib import Path

METRICS = (
    "physical_steps", "before_close_max_xy_mm", "bilateral_steps", "single_finger_steps",
    "maximum_simultaneous_left_fingers", "stable_hold_s", "peak_held_lift_mm",
    "first_close_tick", "first_holder_tick", "local_grasp_pass", "pose_source",
    "input_sha256", "source_action_sha256", "threshold_m", "failure",
)


def record(directory):
    with (directory / "receipt.json").open() as stream:
        return json.load(stream)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def legacy_baseline(receipt, measurements, spec, input_digest):
    if receipt["mode"] != "archived_tail" or receipt["policy_source_sha256"] != spec["action_sha256"]:
        raise ValueError("production baseline input differs")
    if receipt["min_measured_aperture"][0] <= .95 or receipt["max_contact_fingers"][0] > 1:
        raise ValueError("production baseline requires a pre-closure, pre-grasp record")
    initial = spec["at"]["contact_state"]["block_pose_task"]
    max_shift = max(hypot(*[1000 * (event["contact_state"]["block_pose_task"][i][3] - initial[i][3])
                           for i in (0, 1)]) for event in measurements if event.get("contact_state"))
    return dict(mode="A", online_inference=False, **{
        "physical_steps": receipt["physical_steps"], "before_close_max_xy_mm": max_shift,
        "bilateral_steps": 0, "single_finger_steps": receipt["contact_steps"][0],
        "maximum_simultaneous_left_fingers": receipt["max_contact_fingers"][0],
        "stable_hold_s": 0., "peak_held_lift_mm": 0., "local_grasp_pass": False,
        "source_action_sha256": receipt["policy_source_sha256"], "input_sha256": input_digest,
        "threshold_m": spec["config"]["contact_profile"]["limits"]["penetration_m"],
        "failure": {k: receipt["error"][k] for k in ("detail", "row")},
    })


def summarize(root, input_path):
    spec = json.loads(input_path.read_text())
    cases = {}
    for name in ("A", "B", "C"):
        receipt = record(root / name)
        events = root / name / "events.jsonl.gz"
        with gzip.open(events, "rt") as stream:
            measurements = [e for e in map(json.loads, stream) if e.get("physics_step_count")]
        if name == "A" and receipt["mode"] == "archived_tail":
            receipt = legacy_baseline(receipt, measurements, spec, digest(input_path))
        if receipt["mode"] != name or receipt["online_inference"] or receipt["threshold_m"] != .002:
            raise ValueError(f"{name}: invalid receipt")
        if len({e["tick"] for e in measurements}) != receipt["physical_steps"]:
            raise ValueError(f"{name}: physical evidence does not match step count")
        cases[name] = {field: receipt.get(field) for field in METRICS}
    if len({(r["input_sha256"], r["source_action_sha256"]) for r in cases.values()}) != 1:
        raise ValueError("A/B/C replay inputs differ")
    expert = record(root / "expert")
    return dict(runs=cases, native_positive={"same_initial_scene": False, **{
        field: expert.get(field) for field in ("both_left_steps", "first_holder", "max_block_rise_mm",
                                              "physical_steps", "error")}},
        files_sha256={str(p.relative_to(root)): digest(p) for p in sorted(root.glob("**/*")) if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.evidence, args.input), indent=2, allow_nan=False) + "\n")
