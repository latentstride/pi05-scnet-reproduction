"""Analyze one guarded H50 closure intervention and offline anchor observers."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from g2_adapter import G2Error, Stamp
from g2_contact import ContactLimits
from g2e_native_guard import NativeReleaseTracker
from g2_kinematics import pose_error


def read(path):
    return json.loads(path.read_text())


def error(anchor, actual):
    delta = pose_error(np.asarray(anchor), np.asarray(actual))
    return dict(
        position_mm=float(np.linalg.norm(delta[:3]) * 1000),
        angle_deg=float(np.rad2deg(np.linalg.norm(delta[3:]))),
    )


def bilateral(row):
    seen = set()
    for contact in row["forces"]:
        if "box" not in contact["names"]:
            continue
        if any(
            np.linalg.norm(v) >= ContactLimits().minimum_impulse
            for v in contact["impulses"]
        ):
            seen.update(contact["names"])
    return {"fl_link7", "fl_link8"}.issubset(seen)


def observer(rows, delay):
    start = next(r["tick"] for r in rows if bilateral(r)) + delay
    names = sorted({n for r in rows for c in r["forces"] for n in c["names"]})
    ids = {n: i for i, n in enumerate(names)}
    ids.setdefault("fr_link7", len(ids))
    ids.setdefault("fr_link8", len(ids))
    tracker = NativeReleaseTracker(
        ids["box"],
        dict(
            left=[ids["fl_link7"], ids["fl_link8"]],
            right=[ids["fr_link7"], ids["fr_link8"]],
        ),
    )
    acquisition = stop = None
    for row in rows:
        if row["tick"] < start:
            continue
        contacts = [
            NS(
                bodies=[NS(entity=NS(per_scene_id=ids[n])) for n in c["names"]],
                points=[NS(impulse=np.array(v)) for v in c["impulses"]],
            )
            for c in row["forces"]
        ]
        try:
            tracker.observe(
                Stamp("offline", row["tick"], row["tick"] * 0.004),
                contacts,
                np.array([row["tool"], np.eye(4)]),
                np.array(row["block"]),
            )
        except G2Error as exc:
            stop = dict(tick=row["tick"], detail=str(exc))
            break
        if tracker.holders and acquisition is None:
            acquisition = row["tick"]
    reference = tracker.holders.get("left")
    final = rows[-1]
    drift = (
        error(reference, np.linalg.inv(final["tool"]) @ final["block"])
        if reference is not None
        else None
    )
    return dict(
        delay_ticks=delay,
        start_tick=start,
        acquisition_tick=acquisition,
        observer_stop=stop,
        drift_at_physical_end=drift,
        physical_end_tick=final["tick"],
        can_authorize_physics=False,
    )


def contact_forces(row):
    result = {}
    for contact in row["forces"]:
        if "box" not in contact["names"]:
            continue
        normal = tangent = 0.0
        for impulse, n in zip(contact["impulses"], contact["normals"], strict=True):
            impulse = np.asarray(impulse)
            n = np.asarray(n)
            projection = float(impulse @ n)
            normal += abs(projection) / 0.004
            tangent += float(np.linalg.norm(impulse - projection * n)) / 0.004
        result["|".join(contact["names"])] = dict(
            normal_impulse_over_dt_n=normal, tangent_impulse_over_dt_n=tangent
        )
    return result


def summarize(rows, acquired):
    selected = [r for r in rows if r["tick"] >= acquired]
    final = selected[-1]
    relative = [np.linalg.inv(r["tool"]) @ r["block"] for r in selected]
    last_interval = relative[-min(51, len(relative)) :]
    return dict(
        last_tick=final["tick"],
        duration_after_acquisition_s=(final["tick"] - acquired) * 0.004,
        last_original_anchor_slip_mm=final["slip_m"] * 1000,
        last_original_anchor_slip_deg=final["slip_deg"],
        last_block_linear_velocity_m_s=final["block_linear_velocity_m_s"],
        last_block_angular_velocity_rad_s=final["block_angular_velocity_rad_s"],
        last_actual_fingers_m=final["fingers"][0],
        last_drive_m=final["drive"][0],
        terminal_contact_forces=contact_forces(final),
        last_02s_table_impulse_over_dt_peak_n=max(
            (
                sum(
                    float(np.linalg.norm(v))
                    for c in r["forces"]
                    if "table" in c["names"]
                    for v in c["impulses"]
                )
                / 0.004
                for r in selected[-min(51, len(selected)) :]
            ),
            default=0.0,
        ),
        last_02s_relative_change=error(last_interval[0], last_interval[-1]),
        max_original_anchor_slip_mm=max(r["slip_m"] for r in selected) * 1000,
        bilateral_contact_all_samples=all(bilateral(r) for r in selected),
        wrist_change=error(selected[0]["tool"], final["tool"]),
        final_holders=final["holders"],
    )


def analyze(runs, plan_path, baseline_plan, output):
    output.mkdir(parents=True, exist_ok=False)
    plan = read(plan_path)
    modes = plan["physical_conditions"]
    results = {m: read(runs / m / "result.json") for m in modes}
    traces = {m: read(runs / m / "trace.json") for m in modes}
    limits = read(baseline_plan)["baseline_acceptance"]
    base = results["original"]
    assert base["first_holder_tick"] == 5088 and base["failure"]["tick"] == 5139
    assert base["failure"]["detail"] == "object slipped relative to confirmed holder"
    assert (
        base["max_joint_difference_rad"] <= limits["trajectory_arm_max_difference_rad"]
    )
    assert (
        base["max_finger_difference_m"] <= limits["trajectory_finger_max_difference_m"]
    )
    assert base["max_block_difference_m"] <= limits["trajectory_block_max_difference_m"]
    for r in results.values():
        assert (
            r["control_reconstruction"]["arm_max_rad"]
            <= limits["arm_drive_reconstruction_max_rad"]
        )
        assert (
            r["control_reconstruction"]["finger_max_m"]
            <= limits["finger_drive_reconstruction_max_m"]
        )
    prefixes = {m: [r for r in traces[m] if r["tick"] <= 5088] for m in modes}
    assert prefixes["original"] == prefixes["hold-gripper"]
    frozen = np.asarray(plan["physical_intervention"]["left_gripper_position_drive_m"])
    for row in traces["hold-gripper"]:
        if row["tick"] > 5088:
            np.testing.assert_array_equal(row["drive"][0], frozen)
            np.testing.assert_array_equal(
                row["drive_finger_velocity_m_s"][0], [0.0, 0.0]
            )
    prefix_fingerprints = {
        mode: sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for mode, rows in prefixes.items()
    }
    conditions = {}
    for mode in modes:
        rows = traces[mode]
        assert not any(
            "fr_link7" in c["names"] or "fr_link8" in c["names"]
            for r in rows
            for c in r["forces"]
        )
        conditions[mode] = dict(
            result={k: v for k, v in results[mode].items() if k != "freeze_semantics"},
            intervention_semantics=(
                "original arm and gripper control schedule"
                if mode == "original"
                else "freeze left finger position drive at first acquisition and zero finger velocity drive"
            ),
            summary=summarize(rows, 5088),
            offline_observers=[
                observer(rows, delay)
                for delay in plan["offline_anchor_observers"][
                    "delays_from_first_bilateral_impulse_ticks"
                ]
            ],
        )
        (output / (mode + "-excerpts.json")).write_text(
            json.dumps([r for r in rows if r["tick"] >= 5083], indent=2) + "\n"
        )
    common = min(rows[-1]["tick"] for rows in traces.values())
    common_conditions = {
        mode: summarize([r for r in rows if r["tick"] <= common], 5088)
        for mode, rows in traces.items()
    }
    report = dict(
        cpu_only=True,
        baseline_accepted=True,
        prefix_identical=True,
        prefix_fingerprints=prefix_fingerprints,
        live_anchor_unchanged=True,
        source_plan_sha256=sha256(plan_path.read_bytes()).hexdigest(),
        conditions=conditions,
        common_tick=common,
        common_conditions=common_conditions,
        original_guard_hold_horizon_completed=(
            results["hold-gripper"]["failure"] is None
            and traces["hold-gripper"][-1]["tick"] == 5388
        ),
        task_success_evaluated=False,
        late_anchor_passing_proves_stability=False,
    )
    manifest = {}
    for mode in modes:
        for name in (
            "result.json",
            "trace.json",
            "control-reconstruction.json",
            "topp.log",
        ):
            f = runs / mode / name
            manifest[mode + "/" + name] = dict(
                bytes=f.stat().st_size, sha256=sha256(f.read_bytes()).hexdigest()
            )
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "raw-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def assert_same(actual, expected):
    """Keep discrete outcomes exact and allow cross-platform floating-point noise."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_same(a, b)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)
    else:
        assert actual == expected


def verify_published(directory):
    import zipfile

    report = read(directory / "analysis.json")
    plan = read(directory / "plan.json")
    checks = {}
    with zipfile.ZipFile(directory / "key-excerpts.zip") as archive:
        for mode in plan["physical_conditions"]:
            rows = json.loads(archive.read(mode + "-excerpts.json"))
            actual = [
                observer(rows, delay)
                for delay in plan["offline_anchor_observers"][
                    "delays_from_first_bilateral_impulse_ticks"
                ]
            ]
            expected = report["conditions"][mode]["offline_observers"]
            assert_same(actual, expected)
            summary = summarize(rows, 5088)
            assert_same(summary, report["conditions"][mode]["summary"])
            checks[mode] = dict(
                samples=len(rows), observers=len(actual), final_tick=rows[-1]["tick"]
            )
    result = dict(
        passed=True,
        scope="published original-threshold contact observers, relative motion, velocities and contact force decomposition",
        conditions=checks,
        full_prefix_verification_requires_raw_cpu_archive=True,
        gpu_used=False,
        simulation_run=False,
    )
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("runs", "plan", "baseline_plan", "output"):
        parser.add_argument(arg, nargs="?", type=Path)
    parser.add_argument("--verify-published", type=Path)
    args = parser.parse_args()
    if args.verify_published:
        verify_published(args.verify_published)
    elif all((args.runs, args.plan, args.baseline_plan, args.output)):
        analyze(args.runs, args.plan, args.baseline_plan, args.output)
    else:
        parser.error(
            "supply runs plan baseline_plan output, or --verify-published directory"
        )
