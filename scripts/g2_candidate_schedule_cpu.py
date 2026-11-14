"""Original ten-row supervisor with explicitly synthetic stop/cleanup proofs."""

from copy import deepcopy
from functools import partial
from types import SimpleNamespace as NS
from unittest.mock import patch
from g2_candidate_stop import failure_sha256
from g2_final_campaign import supervise
from g2_final_trial import classify
from g2_unified_plan import build, validate_native_startup


def proof_for(result, episode):
    return dict(
        kind="candidate_rejection",
        episode=episode,
        tick=3575,
        state_sha256="a" * 64,
        failure_sha256=failure_sha256(result),
        terminal_checks_passed=True,
        hold_verified=True,
        hold_complete=True,
        close_complete=True,
        scene_context_closed=True,
        no_physics_advance=True,
    )


def simulate(project, output, archived_result, fault="none", position=0):
    plan = build(project)
    ids = [r["episode_id"] for r in plan["trials"]]
    # No parameters of the frozen roster/budget are edited.
    probes = {}
    for r in plan["trials"]:
        key = (r["embodiment"], r["seed"])
        if key[0] == "aloha-agilex" and r["window"] == 4:
            continue
        if r["window"] == 50:
            key = (*key, 50)
        probes.setdefault(key, (r["episode_id"], ["probe", r["episode_id"]]))
    clock = NS(now=0.0)
    clock.monotonic = lambda: clock.now
    clock.sleep = lambda s: setattr(clock, "now", clock.now + s)
    processes = []
    launches = []
    admissions = []

    def spawn(command, stdout, **kwargs):
        stage = command[0]
        ep = command[1] if len(command) > 1 else None
        started = clock.now
        if stage == "monitor":
            stdout.write(b"time, CPU-double, 14358, 24576\n")
        if stage == "trial":
            launches.append(ep)
        if stage == "probe":
            admissions.append(ep)
        proc = NS(pid=len(processes) + 1, returncode=0)

        def poll():
            active = ids.index(launches[-1]) if launches else -1
            if stage in ("monitor", "model"):
                return 1 if fault == "service_exit" and active == position else None
            if stage == "trial" and ep == ids[position] and fault == "timeout":
                return None
            return 0 if clock.now > started else None

        proc.poll = poll
        processes.append((stage, ep, proc))
        return proc

    def receipt(ep, probe=False):
        cleanup = dict(
            scene_opened=True,
            hold_complete=True,
            close_complete=True,
            scene_context_closed=True,
        )
        row = dict(
            episode_id=ep, fresh_reset_verified=True, cleanup=cleanup, success=False
        )
        if probe:
            return dict(
                row,
                category="admitted",
                physical_admitted=True,
                initial_sha256="i",
                first_exchange=dict(request_sha256="r", chunk_sha256="c"),
            )
        if ep != ids[position]:
            return dict(
                row,
                category="diagnostic_complete",
                result=dict(success=False, outcome="task_unsuccessful", failure={}),
            )
        result = deepcopy(archived_result)
        # Counterfactual result for CPU injection only; original receipt is not edited.
        result["failure"]["diagnostic"]["geometry"] = "candidate"
        proof = proof_for(result, ep)
        if fault == "actual":
            result["failure"]["diagnostic"]["geometry"] = "actual"
        if fault == "nested_actual":
            result["failure"]["diagnostic"]["nested"] = {"geometry": "actual"}
        if fault == "unknown":
            result["failure"]["code"] = "RuntimeError"
        if fault in [
            "terminal_checks_passed",
            "hold_verified",
            "no_physics_advance",
            "close_complete",
            "scene_context_closed",
        ]:
            proof[fault] = False
        if fault == "missing_proof":
            proof = {}
        if fault == "wrong_episode":
            proof["episode"] = "old-episode"
        if fault == "wrong_failure":
            proof["failure_sha256"] = "b" * 64
        if fault == "cleanup":
            cleanup["close_complete"] = False
        if fault == "reset":
            row["fresh_reset_verified"] = False
        category = classify(result, candidate_stop=proof)
        if fault in ("forged_controlled_actual", "forged_controlled_unknown"):
            if fault == "forged_controlled_actual":
                result["failure"]["diagnostic"]["nested"] = {"geometry": "actual"}
            else:
                result["failure"]["code"] = "RuntimeError"
            proof = proof_for(result, ep)
            category = "controlled_candidate_stop"
        if fault == "forged_diagnostic":
            category = "diagnostic_complete"
            proof = {}
        return dict(row, category=category, result=result, candidate_stop=proof)

    runtime = dict(
        mode="allocated",
        timeout_s=25200,
        model_timeout_s=960,
        startup_cap_s=3600,
        model=["model"],
        monitor=["monitor"],
        ready_marker="READY",
        probes=probes,
        admission={},
        validate_startup=partial(validate_native_startup, probes=probes),
        commands={ep: ["trial", ep] for ep in ids},
        read_result=receipt,
    )

    def gone(pid):
        stage, ep, _ = processes[pid - 1]
        return not (fault == "group_alive" and stage == "trial" and ep == ids[position])

    with (
        patch("g2_final_campaign.time", clock),
        patch("g2_final_campaign.subprocess.Popen", side_effect=spawn),
        patch("g2_final_campaign.wait_ready"),
        patch("g2_final_campaign.stop_group"),
        patch("g2_final_campaign.process_group_gone", side_effect=gone),
    ):
        summary = supervise(plan, runtime, output, allocated=True)
    return dict(
        data_kind="synthetic_schedule",
        physical_stop_validated=False,
        roster=ids,
        launched=launches,
        startup_admissions=admissions,
        summary=summary,
        fault=fault,
        position=position,
    )


def main():
    """Replay only exception data and mocked scheduling; never construct a scene."""
    import argparse
    import json
    from hashlib import sha256
    from pathlib import Path

    from g2_adapter import G2Error
    from g2_fixtures import robot_arms
    from g2_remaining_geometry import refined_pair
    from robotwin_control_geometry import Status

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    fixture = project / "research/g2/candidate-stop-127192740/historical-fixture.json"
    original = json.loads(fixture.read_text())
    historical = original["receipt"]["result"]
    attempts = historical["failure"]["diagnostic"]["attempts"]
    errors = [
        G2Error(Status.COLLISION, a["detail"], **a["diagnostic"]) for a in attempts
    ]
    reproduced = None
    with patch("g2_remaining_geometry.solve_pair", side_effect=errors):
        try:
            refined_pair(
                robot_arms("aloha-agilex"), None, None, None, None, None, lambda: 1
            )
        except G2Error as error:
            reproduced = error.diagnostic
    assert reproduced["geometry"] == "candidate"
    assert reproduced["attempts"] == attempts
    assert classify(historical) == "actual_safety_stop"
    args.output.mkdir(parents=True, exist_ok=False)
    normal = simulate(project, args.output / "schedule", historical)
    assert normal["launched"] == normal["roster"] and len(normal["roster"]) == 10
    assert normal["summary"]["stop_reason"] is None
    assert normal["summary"]["model_server_starts"] == 1
    faults = []
    for i, fault in enumerate(
        (
            "actual",
            "nested_actual",
            "unknown",
            "terminal_checks_passed",
            "hold_verified",
            "no_physics_advance",
            "close_complete",
            "scene_context_closed",
            "missing_proof",
            "wrong_episode",
            "wrong_failure",
            "cleanup",
            "reset",
            "group_alive",
            "timeout",
            "service_exit",
            "forged_diagnostic",
            "forged_controlled_actual",
            "forged_controlled_unknown",
        )
    ):
        pos = (0, 3, 6, 9)[i % 4]
        run = simulate(project, args.output / f"fault-{fault}", historical, fault, pos)
        assert run["launched"] == run["roster"][: pos + 1]
        assert run["summary"]["stop_reason"] is not None
        faults.append(
            dict(
                fault=fault,
                position=pos,
                launched=run["launched"],
                stop_reason=run["summary"]["stop_reason"],
                passed=True,
            )
        )
    summary = dict(
        source_job=original["source_job"],
        source_receipt_sha256=original["source_receipt_sha256"],
        public_fixture_sha256=sha256(fixture.read_bytes()).hexdigest(),
        historical_category=classify(historical),
        historical_live_stop_proof_present=False,
        reproduced_geometry=reproduced["geometry"],
        all_nested_attempts_preserved=True,
        candidate_marker_without_stop_proof_allows_continuation=False,
        synthetic_schedule_only=True,
        physical_stop_validated=False,
        historical_result_modified=False,
        original_ten_in_order=normal["launched"],
        shared_model_starts=normal["summary"]["model_server_starts"],
        startup_admissions=len(normal["startup_admissions"]),
        simulated_completed=len(normal["launched"]),
        faults=faults,
        gpu_used=False,
    )
    (args.output / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(completed=10, fault_cases=len(faults), gpu_used=False)))


if __name__ == "__main__":
    main()
