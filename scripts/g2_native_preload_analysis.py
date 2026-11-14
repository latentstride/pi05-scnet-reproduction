"""Audit integrated CPU preload/lift records without inferring policy success."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def audit(run, baseline, output):
    output.mkdir(parents=True, exist_ok=False)
    result = json.loads((run / "result.json").read_text())
    rows = json.loads((run / "trace.json").read_text())
    old = json.loads((baseline / "trace.json").read_text())
    commands = json.loads((run / "preload-controls.json").read_text())
    assert (
        result["cpu_only"]
        and not result["model_used"]
        and not result["renderer_created"]
    )
    assert result["integrated_native_backend"] and result["mode"] == "candidate-preload"
    acquired = result["first_holder_tick"]
    assert acquired == 5088 and [r["tick"] for r in rows] == list(
        range(1, len(rows) + 1)
    )
    assert len(commands) >= len(rows)
    prefix = rows[:acquired]
    original = old[:acquired]
    errors = {}
    for key in ("q", "fingers", "block", "tool"):
        errors[key] = float(
            np.max(
                np.abs(
                    np.array([r[key] for r in prefix])
                    - np.array([r[key] for r in original])
                )
            )
        )
        assert errors[key] < 1e-7, (
            "prefix differs before intervention",
            key,
            errors[key],
        )
    for key in (
        "arm_velocity_rad_s",
        "block_linear_velocity_m_s",
        "block_angular_velocity_rad_s",
        "drive_finger_velocity_m_s",
        "drive",
    ):
        errors[key] = float(
            np.max(
                np.abs(
                    np.array([r[key] for r in prefix])
                    - np.array([r[key] for r in original])
                )
            )
        )
        assert errors[key] == 0.0, (key, errors[key])

    def canonical_contacts(r):
        return sorted(json.dumps(c, sort_keys=True) for c in r["forces"])

    assert all(
        canonical_contacts(a) == canonical_contacts(b) for a, b in zip(prefix, original)
    )
    rotation_error = max(
        float(
            np.max(
                np.abs(
                    np.array(a["contact_state"]["block_pose_task"])
                    - np.array(b["contact_state"]["block_pose_task"])
                )
            )
        )
        for a, b in zip(prefix, original)
    )
    assert rotation_error < 1e-7
    physical_keys = (
        "tick",
        "q",
        "fingers",
        "block",
        "tool",
        "arm_velocity_rad_s",
        "block_linear_velocity_m_s",
        "block_angular_velocity_rad_s",
        "drive_finger_velocity_m_s",
        "drive",
    )

    def physical_fingerprint(values):
        return sha256(
            json.dumps(
                [
                    dict(
                        **{k: r[k] for k in physical_keys},
                        contacts=canonical_contacts(r),
                    )
                    for r in values
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    physical_hash = physical_fingerprint(prefix)
    assert physical_hash == physical_fingerprint(original)
    anchor = prefix[-1]["contact_state"]["tool_block"]["left"]
    after = rows[acquired:]
    assert all(r["contact_state"]["tool_block"].get("left") == anchor for r in after)
    held = [r for r in commands if r["modes"]["left"] == "holding_preload"]
    assert held and held[0]["tick"] == acquired
    drive = held[0]["applied_finger_drive_m"][0]
    assert all(r["applied_finger_drive_m"][0] == drive for r in held)
    assert all(r["applied_finger_velocity_m_s"][0] == [0.0, 0.0] for r in held)
    assert all(r["acquisition_ticks"]["left"] == acquired for r in held)
    lift = result["lift_diagnostic"]
    lift_start = lift["start_tick"] if lift else None

    def compact(row):
        return dict(
            tick=row["tick"],
            block_z_m=row["block"][2][3],
            tool_z_m=row["tool"][2][3],
            slip_mm=row["slip_m"] * 1000,
            slip_deg=row["slip_deg"],
            actual_left_fingers_mm=(np.array(row["fingers"][0]) * 1000).tolist(),
            drive_left_fingers_mm=(np.array(row["drive"][0]) * 1000).tolist(),
            holders=row["holders"],
        )

    result_summary = dict(
        scope="CPU physics and original native hooks; no model continuation, no full task success claim",
        source_job="127147943",
        prefix_max_absolute_errors=errors,
        prefix_records_identical=prefix == original,
        physical_prefix_identical=True,
        physical_prefix_sha256=physical_hash,
        contact_geometry_impulse_multisets_identical=True,
        tracker_rotation_representation_max_abs=rotation_error,
        prefix_metadata_difference="NativeBackend normalizes quaternion rotations in float64; old helper used float32 transform; contact list order can differ; measured physics is identical",
        prefix_fingerprint=sha256(
            json.dumps(prefix, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        baseline_prefix_fingerprint=sha256(
            json.dumps(original, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        native_backend_hooks=True,
        first_holder_tick=acquired,
        preload_drive_m=drive,
        original_anchor_unchanged=True,
        maintenance_end=compact(rows[min(5388, len(rows)) - 1]),
        final=compact(rows[-1]),
        failure=result["failure"],
        physical_lift_completed=bool(
            lift and not result["failure"] and len(rows) == result["horizon_tick"]
        ),
        physical_release_validated=False,
        release_validation="real contact state machine and NativeBackend with synthetic contact receipts; not physical transfer",
        GPU_operations=0,
        checkpoint_modified=False,
        thresholds_modified=False,
        source_hashes={p.name: digest(p) for p in run.iterdir() if p.is_file()},
    )
    if lift_start and len(rows) > lift_start:
        start, end = rows[lift_start - 1], rows[-1]
        result_summary["measured_lift"] = dict(
            requested_mm=lift["requested_m"] * 1000,
            tool_delta_mm=(
                (np.array(end["tool"])[:3, 3] - np.array(start["tool"])[:3, 3]) * 1000
            ).tolist(),
            block_delta_mm=(
                (np.array(end["block"])[:3, 3] - np.array(start["block"])[:3, 3]) * 1000
            ).tolist(),
            tool_delta_m=(
                np.array(end["tool"])[:3, 3] - np.array(start["tool"])[:3, 3]
            ).tolist(),
            block_delta_m=(
                np.array(end["block"])[:3, 3] - np.array(start["block"])[:3, 3]
            ).tolist(),
            max_slip_mm=max(r["slip_m"] * 1000 for r in rows[lift_start:]),
            max_slip_deg=max(r["slip_deg"] for r in rows[lift_start:]),
            bilateral_holder_all_samples=all(
                "left" in r["holders"] for r in rows[lift_start:]
            ),
        )
    (output / "analysis.json").write_text(
        json.dumps(result_summary, indent=2, allow_nan=False) + "\n"
    )
    with ZipFile(output / "key-excerpts.zip", "w", ZIP_DEFLATED) as z:
        z.writestr(
            "candidate-contact-lift.json",
            json.dumps(rows[5082:], separators=(",", ":")),
        )
        z.writestr(
            "candidate-controls.json",
            json.dumps(commands[5082:], separators=(",", ":")),
        )
        z.writestr("result.json", json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result_summary.items() if k != "source_hashes"}, indent=2
        )
    )
    return result_summary


def verify_published(folder):
    from g2_kinematics import pose_error

    summary = json.loads((folder / "analysis.json").read_text())
    with ZipFile(folder / "key-excerpts.zip") as z:
        rows = json.loads(z.read("candidate-contact-lift.json"))
        commands = json.loads(z.read("candidate-controls.json"))
        result = json.loads(z.read("result.json"))
    assert [r["tick"] for r in rows] == list(
        range(rows[0]["tick"], rows[-1]["tick"] + 1)
    )
    for r in rows:
        if "left" not in r["holders"]:
            continue
        anchor = np.array(r["contact_state"]["tool_block"]["left"])
        err = pose_error(np.array(r["block"]), np.array(r["tool"]) @ anchor)
        assert abs(np.linalg.norm(err[:3]) - r["slip_m"]) < 1e-10
        assert abs(np.rad2deg(np.linalg.norm(err[3:])) - r["slip_deg"]) < 1e-8
    held = [r for r in commands if r["modes"]["left"] == "holding_preload"]
    assert all(
        r["applied_finger_drive_m"][0] == summary["preload_drive_m"] for r in held
    )
    assert all(r["applied_finger_velocity_m_s"][0] == [0.0, 0.0] for r in held)
    if summary["physical_lift_completed"]:
        assert result["failure"] is None and rows[-1]["tick"] == result["horizon_tick"]
        assert all(r["slip_m"] <= 0.005 and r["slip_deg"] <= 5.0 for r in rows)
    report = dict(
        verified=True,
        records=len(rows),
        controls=len(commands),
        physical_lift_completed=summary["physical_lift_completed"],
        physical_release_validated=False,
        policy_task_success_validated=False,
        scope="published measured relative drift and applied drives; full prefix requires private raw archive",
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify-published", type=Path)
    p.add_argument("run", type=Path, nargs="?")
    p.add_argument("baseline", type=Path, nargs="?")
    p.add_argument("output", type=Path, nargs="?")
    a = p.parse_args()
    if a.verify_published:
        verify_published(a.verify_published)
    else:
        if not all((a.run, a.baseline, a.output)):
            p.error("run, baseline and a new output directory are required")
        audit(a.run, a.baseline, a.output)
