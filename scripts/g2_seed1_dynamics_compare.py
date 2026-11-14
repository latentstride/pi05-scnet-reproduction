"""Compare actual CPU dynamics branches after validating historical reproduction."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def motion(rows, tick):
    selected = [r for r in rows if r["tick"] >= tick]
    first = np.asarray(selected[0]["tool"])
    tool_mm = max(
        float(np.linalg.norm(np.asarray(r["tool"])[:3, 3] - first[:3, 3]) * 1000)
        for r in selected
    )
    block = np.asarray(selected[0]["block"])
    block_mm = max(
        float(np.linalg.norm(np.asarray(r["block"])[:3, 3] - block[:3, 3]) * 1000)
        for r in selected
    )
    last = selected[-1]
    forces = {}
    for contact in last["forces"]:
        forces["|".join(contact["names"])] = (
            sum(float(np.linalg.norm(p)) for p in contact["impulses"]) / 0.004
        )
    return dict(
        last_tick=last["tick"],
        last_slip_mm=last["slip_m"] * 1000,
        last_slip_deg=last["slip_deg"],
        peak_slip_mm=max(r["slip_m"] for r in selected) * 1000,
        peak_tool_displacement_mm=tool_mm,
        peak_block_displacement_mm=block_mm,
        final_fingers_m=last["fingers"][0],
        final_drive_m=last["drive"][0],
        final_contact_impulse_over_dt_n=forces,
    )


def compare(runs, plan, output):
    output.mkdir(parents=True, exist_ok=False)
    thresholds = read(plan)["baseline_acceptance"]
    reports = []
    manifest = {}
    for window in (50, 31):
        paths = {
            mode: runs / f"h{window}-{mode}-r3" for mode in ("original", "fixed-wrist")
        }
        results = {"original": read(paths["original"] / "result.json")}
        traces = {"original": read(paths["original"] / "trace.json")}
        base = results["original"]
        checks = dict(
            reconstructed_arm=base["control_reconstruction"]["arm_max_rad"]
            <= thresholds["arm_drive_reconstruction_max_rad"],
            reconstructed_fingers=base["control_reconstruction"]["finger_max_m"]
            <= thresholds["finger_drive_reconstruction_max_m"],
            acquisition=base["first_holder_tick"] is not None
            and abs(base["first_holder_tick"] - base["historical_holder_tick"])
            <= thresholds["first_holder_tick_difference_max"],
            stop=base["failure"] is not None
            and base["failure"]["detail"]
            == "object slipped relative to confirmed holder"
            and abs(base["failure"]["tick"] - base["historical_stop_tick"])
            <= thresholds["slip_stop_tick_difference_max"],
            arm_trajectory=base["max_joint_difference_rad"]
            <= thresholds["trajectory_arm_max_difference_rad"],
            finger_trajectory=base["max_finger_difference_m"]
            <= thresholds["trajectory_finger_max_difference_m"],
            block_trajectory=base["max_block_difference_m"]
            <= thresholds["trajectory_block_max_difference_m"],
        )
        if not all(checks.values()):
            reports.append(
                dict(
                    window=window,
                    baseline_checks=checks,
                    baseline_accepted=False,
                    original_result=base,
                    counterfactual_run=False,
                    reason="historical baseline not reproduced within preregistered limits",
                )
            )
            for name in (
                "result.json",
                "control-reconstruction.json",
                "trace.json",
                "topp.log",
            ):
                f = paths["original"] / name
                manifest[f"h{window}-original-r3/{name}"] = dict(
                    sha256=sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size
                )
            continue
        results["fixed-wrist"] = read(paths["fixed-wrist"] / "result.json")
        traces["fixed-wrist"] = read(paths["fixed-wrist"] / "trace.json")
        tick = base["historical_holder_tick"]
        prefixes = {
            mode: [r for r in trace if r["tick"] <= tick]
            for mode, trace in traces.items()
        }
        exact_prefix = prefixes["original"] == prefixes["fixed-wrist"]
        assert exact_prefix, "branches differ before the intervention"
        row = dict(
            window=window,
            baseline_accepted=True,
            baseline_checks=checks,
            prefix_identical=True,
            first_holder_tick=tick,
            original_result=base,
            fixed_result=results["fixed-wrist"],
            original=motion(traces["original"], tick),
            fixed_wrist=motion(traces["fixed-wrist"], tick),
        )
        common_tick = min(row["original"]["last_tick"], row["fixed_wrist"]["last_tick"])
        row["common_tick_comparison"] = dict(tick=common_tick)
        for mode, trace in traces.items():
            prefix = [r for r in trace if r["tick"] <= common_tick]
            row["common_tick_comparison"][mode] = motion(prefix, tick)
        reports.append(row)
        excerpt = {
            mode: [r for r in trace if r["tick"] >= tick - 3]
            for mode, trace in traces.items()
        }
        (output / f"h{window}-branch-excerpts.json").write_text(
            json.dumps(excerpt, indent=2) + "\n"
        )
        for mode, path in paths.items():
            for name in (
                "result.json",
                "control-reconstruction.json",
                "trace.json",
                "topp.log",
            ):
                f = path / name
                manifest[f"h{window}-{mode}-r3/{name}"] = dict(
                    sha256=sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size
                )
    result = dict(
        cpu_only=True,
        model_used=False,
        renderer_created=False,
        windows=reports,
        scope="fixed original left-arm drive targets and zero arm velocity feedforward, not a physical wrist weld; common historical-stop horizon",
        physical_success_evaluated=False,
        all_baselines_accepted=all(r.get("baseline_accepted", False) for r in reports),
        all_compared_prefixes_identical=all(
            r.get("prefix_identical", True) for r in reports
        ),
    )
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "raw-cpu-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def verify_excerpts(directory):
    from types import SimpleNamespace as NS
    from g2_adapter import G2Error, Stamp
    from g2e_native_guard import NativeReleaseTracker

    data = read(directory / "h50-branch-excerpts.json")
    comparison = read(directory / "comparison.json")["windows"][0]
    verified = {}
    for mode, rows in data.items():
        names = sorted(
            {
                name
                for row in rows
                for contact in row["forces"]
                for name in contact["names"]
            }
        )
        ids = {name: index for index, name in enumerate(names)}
        # Right hand has no block contacts in this diagnostic interval.
        ids.setdefault("fr_link7", len(ids))
        ids.setdefault("fr_link8", len(ids))
        tracker = NativeReleaseTracker(
            ids["box"],
            {
                "left": [ids["fl_link7"], ids["fl_link8"]],
                "right": [ids["fr_link7"], ids["fr_link8"]],
            },
        )
        first = stopped = None
        for row in rows:
            contacts = [
                NS(
                    bodies=[
                        NS(entity=NS(per_scene_id=ids[name]))
                        for name in contact["names"]
                    ],
                    points=[
                        NS(impulse=np.array(value)) for value in contact["impulses"]
                    ],
                )
                for contact in row["forces"]
            ]
            try:
                tracker.observe(
                    Stamp(mode, row["tick"], row["tick"] * 0.004),
                    contacts,
                    np.array([row["tool"], np.eye(4)]),
                    np.array(row["block"]),
                )
            except G2Error as error:
                assert str(error) == "object slipped relative to confirmed holder"
                stopped = row["tick"]
                break
            if tracker.holders and first is None:
                first = row["tick"]
        expected = comparison[
            "original_result" if mode == "original" else "fixed_result"
        ]
        assert (
            first == expected["first_holder_tick"]
            and stopped == expected["failure"]["tick"]
        )
        verified[mode] = dict(first_holder_tick=first, slip_stop_tick=stopped)
    result = dict(
        passed=True,
        scope="unchanged left contact tracker replay from published impulses and matrices; no physics simulation",
        branches=verified,
    )
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="?", type=Path)
    parser.add_argument("plan", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--verify-excerpts", type=Path)
    args = parser.parse_args()
    if args.verify_excerpts:
        verify_excerpts(args.verify_excerpts)
    elif args.runs and args.plan and args.output:
        compare(args.runs, args.plan, args.output)
    else:
        parser.error("supply runs plan output, or --verify-excerpts directory")
