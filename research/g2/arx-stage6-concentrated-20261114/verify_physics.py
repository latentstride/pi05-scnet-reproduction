"""Verify the archived CPU receipts against the original physics logs."""

import gzip
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def verify():
    dynamic = HERE / "dynamic"
    summary = read_json(dynamic / "physical-summary.json")
    for case, expected in summary.items():
        receipt = read_json(dynamic / case / "receipt.json")
        with gzip.open(dynamic / case / "events.jsonl.gz", "rb") as stream:
            data = stream.read()
        assert sha256(data).hexdigest() == expected["full_events_sha256"]
        assert len(data) == expected["bytes"]
        events = [json.loads(line) for line in data.splitlines()]
        steps = sum(event.get("physics_step_count", 0) for event in events)
        receipt_steps = receipt.get("physical_steps", receipt.get("physics_steps"))
        assert steps == expected["physical_steps"] == receipt_steps
        assert receipt["hold_no_advance"]
        if case in {"q1-final-interface", "q2-final-interface"}:
            assert receipt["fresh_request"]["validated"]
            assert receipt["fresh_request"]["stamp"]["tick"] == receipt["end_tick"]
            assert receipt["prefix"]["reason"] == "measured_first_contact"
            assert any(event.get("receipt", {}).get("contact_motion")
                       == "measured_safe_prefix_v1" for event in events)
        elif case == "q2-final-tail":
            assert receipt["completed_rows"] == list(range(8, 51))
            assert receipt["safe_segments"] == 101
            assert steps == 6330 and receipt["contact_steps"] == [1099, 0]
            assert receipt["min_measured_aperture"][0] < .005
            assert receipt["max_contact_fingers"] == [1, 0] and not receipt["stable_grasp"]
        elif case == "q2-final-negative":
            assert steps == 0 and receipt["real_arx_collision_rejected"]
            assert receipt["collision_diagnostic"]["geometry"] == "actual"
    for case, row_number, source_depth, target_depth in (
            ("q1-endpoints", 36, .494, 2.490), ("q2-endpoints", 8, 2.493, 4.484)):
        receipt = read_json(HERE / "endpoints" / case / "receipt.json")
        row, = [entry for entry in receipt["records"] if entry["row"] == row_number]
        assert abs(max(row["source_start_finger_depth_mm"][0]) - source_depth) < .001
        assert abs(max(row["target_start_finger_depth_mm"][0]) - target_depth) < .001
        assert row["production_ik_clear"] is False
    print("PASS: 5 original CPU logs, 2 mesh endpoints, 2 reobservations and the collision negative")


if __name__ == "__main__":
    verify()
