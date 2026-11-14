"""Recompute concentrated CPU evidence from preserved replay outputs, not model inference."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from g2_kinematics import pose_error


def read(path):
    return json.loads(path.read_text())


def persist(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def signature(path, method=None):
    tree = ast.parse(path.read_text())
    if method:
        tree = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == method
        )
    return sha256(ast.dump(tree).encode()).hexdigest()


def trace_metrics(folder):
    result = read(folder / "result.json")
    trace = read(folder / "trace.json")
    held = [r for r in trace if "left" in r["holders"]]
    maximum = max(r["slip_m"] for r in held)
    value = dict(
        steps=result["steps"],
        holder_tick=result["first_holder_tick"],
        failure=result["failure"],
        peak_slip_mm=1000 * maximum,
        minimum_position_margin_mm=1000 * (0.005 - maximum),
        peak_slip_deg=max(r["slip_deg"] for r in held),
        final_left_fingers_m=trace[-1]["fingers"][0],
        final_left_drive_m=trace[-1]["drive"][0],
        final_holders=trace[-1]["holders"],
        elapsed_s=result["elapsed_s"],
    )
    if result["failure"] is None:
        start = next(
            r for r in trace if r["tick"] == result["lift_diagnostic"]["start_tick"]
        )
        value["requested_lift_mm"] = 1000 * result["lift_diagnostic"]["requested_m"]
        value["measured_block_lift_mm"] = 1000 * (
            trace[-1]["block"][2][3] - start["block"][2][3]
        )
        value["last_second_slip_change_mm"] = 1000 * (
            trace[-1]["slip_m"] - trace[-251]["slip_m"]
        )
    return value, trace


def run(a):
    a.output.mkdir(parents=True, exist_ok=True)
    root = a.evidence
    analysis = dict(
        scope="CPU geometry, saved-command dynamics and regression only; no fresh policy inference or full-task success",
        source_job="127147943",
        gpu_operations=0,
        thresholds_modified=False,
        seed1={},
    )
    clips = {}
    for label, relative in [
        ("gap1", "dynamics/gap1"),
        ("gap2", "dynamics/gap2"),
        ("gap4", "dynamics/gap4"),
        ("production_lift100", "production-contact/lift100"),
    ]:
        metrics, trace = trace_metrics(root / relative)
        analysis["seed1"][label] = metrics
        clips[label] = [
            {
                k: r[k]
                for k in [
                    "tick",
                    "fingers",
                    "drive",
                    "tool",
                    "block",
                    "slip_m",
                    "slip_deg",
                    "holders",
                    "contact_state",
                ]
            }
            for r in trace
            if r["tick"]
            in {5085, 5086, 5087, 5088, 5090, 5139, 5388, 5888, 6138, 6388, 6888}
        ]
    assert analysis["seed1"]["gap1"]["failure"]["tick"] == 5090
    assert all(
        analysis["seed1"][k]["failure"] is None
        for k in ["gap2", "gap4", "production_lift100"]
    )
    baseline = read(a.baseline / "trace.json")
    selected = read(root / "production-contact/lift100/trace.json")
    fields = [
        "q",
        "fingers",
        "block",
        "tool",
        "arm_velocity_rad_s",
        "block_linear_velocity_m_s",
        "block_angular_velocity_rad_s",
        "drive_finger_velocity_m_s",
        "drive",
    ]
    errors = {
        k: max(
            float(np.max(np.abs(np.array(r[k]) - b[k])))
            for r, b in zip(selected[:5086], baseline[:5086])
        )
        for k in fields
    }
    assert max(errors.values()) == 0
    anchor_a = selected[5088 - 1]["contact_state"]["tool_block"]["left"]
    anchor_b = baseline[5088 - 1]["contact_state"]["tool_block"]["left"]
    anchor_error = float(np.max(np.abs(np.array(anchor_a) - anchor_b)))
    assert anchor_error == 0
    analysis["seed1"]["physical_prefix_through_first_bilateral"] = dict(
        tick=5086, max_absolute_errors=errors
    )
    analysis["seed1"]["confirmed_anchor_at_5088_max_error"] = anchor_error
    analysis["seed1"]["old_minimum_margin_mm"] = 0.141793
    saved = {}
    for name in ["native0-baseline", "native0-repair", "native1-saved100"]:
        report = read(root / "saved-actions" / name / "saved-action-result.json")
        saved[name] = report
        if name == "native1-saved100":
            slips = []
            first = None
            for line in (
                root / "saved-actions" / name / "saved-action-events.jsonl"
            ).open():
                event = json.loads(line)
                state = event.get("contact_state") or {}
                if "left" in state.get("holders", []):
                    first = event["tick"] if first is None else first
                    error = pose_error(
                        np.array(state["tool_block"]["left"]),
                        np.array(state["tool_block_measured"]["left"]),
                    )
                    slips.append(
                        (
                            event["tick"],
                            float(np.linalg.norm(error[:3])),
                            float(np.rad2deg(np.linalg.norm(error[3:]))),
                        )
                    )
            report["peak_slip_mm"] = 1000 * max(v[1] for v in slips)
            report["minimum_position_margin_mm"] = 5 - report["peak_slip_mm"]
            report["peak_slip_deg"] = max(v[2] for v in slips)
            report["first_holder_tick"] = first
            report["measured_rise_since_holder_mm"] = 1000 * (
                report["final_contact_state"]["block_pose_task"][2][3]
                - selected[first - 1]["block"][2][3]
            )
    assert (
        saved["native0-baseline"]["physical_steps"] == 3528
        and saved["native0-baseline"]["completed_actions"] == 26
    )
    assert (
        saved["native0-repair"]["completed_actions"] == 27
        and saved["native0-repair"]["failure"] is None
    )
    assert (
        saved["native1-saved100"]["completed_actions"] == 100
        and saved["native1-saved100"]["failure"] is None
    )
    analysis["saved_command_replay"] = saved
    geometry = read(root / "geometry-r3/geometry.json")
    analysis["geometry"] = dict(native=geometry["native"], arx=[])
    for case in geometry["arx"]:
        collisions = case["baseline"]["diagnostic"].get("collision_examples", [])
        first = collisions[0]["collisions"][0] if collisions else {}
        analysis["geometry"]["arx"].append(
            dict(
                episode=case["episode"],
                tick=case["tick"],
                actual_clear=case["actual_clear"],
                baseline_accepted=case["baseline"]["accepted"],
                refined_accepted=case["refined"]["accepted"],
                extra_seeds=case.get("extra_seeds"),
                first_collision=first,
            )
        )
    coordinator = read(a.arx / "coordinator-r1/result/coordinator.json")
    candidate = coordinator["cases"]["candidate"]
    assert not coordinator["cases"]["baseline"]["accepted"] and candidate["accepted"]
    analysis["arx_seed0_coordinator"] = dict(
        baseline_accepted=False,
        candidate_accepted=True,
        actions=candidate["actions"],
        control_ticks=candidate["control_ticks"],
        recoveries=candidate["recoveries"],
        source_seed_unchanged=candidate["source_seed_unchanged"],
        receipt=candidate["receipt"],
    )
    admission = read(a.arx / "source-admission-r2/result/admission.json")
    for r in admission["reports"]:
        assert (
            r["approved_actions"] == 2
            and r["production_coordinator"]["approved_actions"] == 2
        )
        assert (
            r["production_coordinator"]["recoveries"] == 1
            and r["source_seed_unchanged"]
        )
    analysis["arx_seed1_source_admission"] = admission
    analysis["source_ast_equivalence"] = {}
    for stage in ["production-contact", "saved-actions"]:
        methods = {
            "g2_native_backend": ["_before_step", "_stepped"]
            + (["_approach_row"] if stage == "saved-actions" else []),
            "g2_native_preload": [None],
            "g2_native_contact_preload": [None],
        }
        for name, items in methods.items():
            for method in items:
                before = root / stage / "scripts" / f"{name}.py"
                now = a.project / "scripts" / f"{name}.py"
                key = f"{stage}/{name}:{method or 'module'}"
                same = signature(before, method) == signature(now, method)
                assert same, key
                analysis["source_ast_equivalence"][key] = same
    if a.additional is not None:
        old = read(a.additional / "native0-full50-r2/result/saved-action-result.json")
        final = read(
            a.additional / "native0-barrier-final/result/saved-action-result.json"
        )
        assert old["completed_actions"] == 27 and old["failure"] is not None
        assert final["completed_actions"] == 27 and final["failure"] is None
        assert final["physical_steps"] == 3568
        assert final["requires_fresh_model_request"] is True
        assert final["approach_barrier"]["consumed_actions"] == 27
        first = read(a.additional / "coordinator-first-r2/result/coordinator.json")
        assert all(c["accepted"] and c["actions"] == 1 for c in first["cases"].values())
        analysis["native_stale_tail_regression"] = dict(
            old_completed_actions=27,
            old_stop_tick=old["physical_steps"],
            old_failure_status=old["failure"]["status"],
            old_failure_detail=old["failure"]["detail"],
            final_receipt=final,
            original_remaining_chunk_not_executed=True,
            fresh_inference_not_simulated=True,
        )
        analysis["arx_first_checkpoint"] = dict(
            tick=first["stamp"]["tick"],
            legacy_safe_prefix_retained=True,
            baseline_actions=first["cases"]["baseline"]["actions"],
            candidate_actions=first["cases"]["candidate"]["actions"],
        )
    manifest = {}
    for label, folder in [
        ("raw-r2", root),
        ("raw-r3", a.arx),
        ("baseline-preload", a.baseline),
    ]:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                manifest[f"{label}/{p.relative_to(folder)}"] = sha256(
                    p.read_bytes()
                ).hexdigest()
    if a.additional is not None:
        for p in sorted(a.additional.rglob("*")):
            if p.is_file():
                manifest[f"raw-r4/{p.relative_to(a.additional)}"] = sha256(
                    p.read_bytes()
                ).hexdigest()
    persist(a.output / "analysis.json", analysis)
    persist(a.output / "contact-evidence-excerpts.json", clips)
    persist(a.output / "raw-evidence-hashes.json", manifest)
    print(
        json.dumps(
            dict(
                cpu_only=True,
                analysis_assertions="passed",
                raw_files=len(manifest),
                production_lift_margin_mm=analysis["seed1"]["production_lift100"][
                    "minimum_position_margin_mm"
                ],
                saved100_margin_mm=saved["native1-saved100"][
                    "minimum_position_margin_mm"
                ],
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["evidence", "arx", "baseline", "project", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--additional", type=Path)
    run(p.parse_args())
