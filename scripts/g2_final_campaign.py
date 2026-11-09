"""Supervise one shared model and independently admitted diagnostic trials."""

import os
from pathlib import Path
import subprocess
import time

from g2_candidate_stop import (
    actual_provenance,
    candidate_failure,
    verified_candidate_stop,
)
from g2e_host_supervisor import argv, interrupt_signals, stop_group
from pace_g1_orchestrate import persist, wait_ready


def verified_contact_stop(proof):
    return (
        isinstance(proof, dict)
        and proof.get("kind") in ("slip", "bilateral_loss")
        and type(proof.get("tick")) is int
        and proof["tick"] > 0
        and all(
            proof.get(key) is True
            for key in (
                "terminal_checks_passed",
                "hold_complete",
                "close_complete",
                "scene_context_closed",
            )
        )
    )


def process_group_gone(pid):
    """A reaped leader alone does not prove the simulator group is gone."""
    end = time.monotonic() + 2
    while True:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(0.025)


def reap_process_group(proc):
    stop_group(proc)
    if not process_group_gone(proc.pid):
        raise RuntimeError("process group cleanup unverified")


def verified_scene_cleanup(receipt, *, probe=False):
    cleanup = receipt.get("cleanup")
    result = receipt.get("result") or {}
    if (
        not isinstance(cleanup, dict)
        or receipt.get("cleanup_failures")
        or result.get("cleanup_failures")
    ):
        return False
    if (
        probe
        and receipt.get("category") == "not_admitted"
        and receipt.get("admission_stage") == "assets"
    ):
        return cleanup.get("scene_opened") is False and not receipt.get(
            "physical_admitted"
        )
    if not all(
        cleanup.get(key) is True
        for key in ("scene_opened", "close_complete", "scene_context_closed")
    ):
        return False
    if probe:
        if (
            receipt.get("category") == "admitted"
            or receipt.get("admission_stage") == "input_contract"
        ):
            return (
                receipt.get("physical_admitted") is True
                and receipt.get("fresh_reset_verified") is True
            )
        return (
            receipt.get("admission_stage") == "initial_state"
            and receipt.get("physical_admitted") is False
            and receipt.get("fresh_reset_verified") is False
        )
    return (
        receipt.get("fresh_reset_verified") is True
        and type(receipt.get("success")) is bool
        and (receipt["success"] or cleanup.get("hold_complete") is True)
    )


def verified_recovery_budget(result, expected_limit=None):
    failure = result.get("failure") or {}
    diagnostic = failure.get("diagnostic") or {}
    used = diagnostic.get("recoveries_used")
    limit = diagnostic.get("recovery_limit")
    return (
        result.get("outcome") == "recovery_budget_exhausted"
        and result.get("success") is False
        and failure.get("code") == "plan_failed"
        and failure.get("phase") in ("observation", "planning")
        and diagnostic.get("reason") == result["outcome"]
        and type(used) is int
        and type(limit) is int
        and 0 <= used == limit <= 4
        and (expected_limit is None or limit == expected_limit)
    )


def gpu_samples(path):
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) == 4 and all(x.strip().isdigit() for x in fields[2:]):
            yield tuple(map(int, fields[2:]))


def supervise(plan, runtime, output, *, rehearsal=False, allocated=False):
    roster = plan["trials"]
    ids = [row["episode_id"] for row in roster]
    if len(ids) != len(set(ids)) or not 0 < len(ids) <= 14:
        raise ValueError("invalid locked roster")
    budget = plan["budget"]
    if (
        plan["allocations_planned"] != 1
        or plan["model_server_starts"] != 1
        or budget["gpu_count"] != 1
        or budget["proposed_allocation_s"] > 34200
        or budget["proposed_fee_cap_yuan"] > 34.20
        or sum(row["hard_timeout_s"] for row in roster)
        + sum(budget["overhead"].values())
        != budget["bounded_execution_s"]
        or budget["bounded_execution_s"] > budget["proposed_allocation_s"]
    ):
        raise ValueError("invalid campaign budget")
    if rehearsal == allocated:
        raise PermissionError("review-only package has no executable GPU admission")
    if runtime.get("mode") != ("cpu_rehearsal" if rehearsal else "allocated"):
        raise ValueError("campaign runtime mode mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = {episode: {"status": "not_run", "reason": "not reached"} for episode in ids}
    summary = dict(
        rows=rows,
        model_server_starts=0,
        formal_score_eligible=False,
        transfer_claim_eligible=False,
    )
    overhead = budget["overhead"]
    supervisor_cap = budget["proposed_allocation_s"] - sum(
        overhead[key] for key in ("graphics_s", "statistics_s", "cleanup_s")
    )
    deadline = time.monotonic() + min(runtime["timeout_s"], supervisor_cap)
    model = trial = monitor = None

    def save():
        persist(output / "campaign.json", summary)

    def launch(command, log):
        with log.open("wb") as stream:
            return subprocess.Popen(
                argv(command),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def admitted(row):
        key = (row["embodiment"], row["seed"])
        evidence = runtime["admission"].get(key, {})
        required = ["assets", "reset", "actual_geometry", "input_contract"]
        if "expert" in row["method"]:
            required.append("expert_phase_transition")
        return all(evidence.get(item) is True for item in required)

    def wait_trial(proc, stop_at):
        while proc.poll() is None:
            failure = service_failure()
            if failure:
                return failure
            if time.monotonic() >= stop_at:
                return "trial hard timeout"
            time.sleep(0.025)
        failure = service_failure()
        if failure:
            return failure
        if time.monotonic() >= stop_at:
            return "trial hard timeout"
        if proc.returncode:
            return "trial process failure"
        return None

    reason = None
    active_entry = None

    def service_failure():
        if model.poll() is not None:
            return "shared model exited"
        if monitor is not None and monitor.poll() is not None:
            return "GPU monitor exited"
        return None

    save()
    with interrupt_signals():
        try:
            if allocated:
                monitor = launch(runtime["monitor"], output / "gpu-memory.csv")
            # Use spare allocation time for startup, while protecting every
            # declared trial cap. Model and probes share this one deadline.
            trial_reserve = sum(
                row["hard_timeout_s"]
                for row in roster
                if row["episode_id"] in runtime["commands"]
            )
            startup_deadline = deadline - trial_reserve
            if "startup_cap_s" in runtime:
                startup_deadline = min(
                    startup_deadline, time.monotonic() + runtime["startup_cap_s"]
                )
            if startup_deadline <= time.monotonic():
                raise TimeoutError("no startup time remains after reserving trials")
            model_log = output / "model.log"
            model = launch(runtime["model"], model_log)
            summary["model_server_starts"] = 1
            save()
            wait_ready(
                model,
                model_log,
                runtime["ready_marker"],
                min(runtime["model_timeout_s"], startup_deadline - time.monotonic()),
                runtime.get("model_phases_file"),
                runtime.get("model_phase_timeouts"),
            )
            if allocated:
                summary["startup"] = {}
                probes = runtime["probes"]
                for index, (key, (episode, command)) in enumerate(probes.items()):
                    began = time.monotonic()
                    allowance = (startup_deadline - began) / (len(probes) - index)
                    if allowance <= 0:
                        raise TimeoutError("dynamic admission deadline")
                    entry = dict(status="running", allowance_s=allowance)
                    active_entry = entry
                    summary["startup"][episode] = entry
                    save()
                    trial = launch(command, output / f"probe-{episode}.log")
                    reason = wait_trial(trial, began + allowance)
                    reap_process_group(trial)
                    trial = None
                    entry.update(wall_s=time.monotonic() - began)
                    if reason:
                        entry.update(status="failed", reason=reason)
                        save()
                        raise RuntimeError(f"dynamic admission {episode}: {reason}")
                    receipt = runtime["read_result"](episode, probe=True)
                    category = receipt.get("category")
                    if (
                        receipt.get("episode_id") != episode
                        or category
                        not in (
                            "admitted",
                            "not_admitted",
                        )
                        or receipt.get("stop_campaign") is True
                    ):
                        entry["result"] = receipt
                        raise RuntimeError(f"dynamic admission {episode}: {category}")
                    if not verified_scene_cleanup(receipt, probe=True):
                        entry["result"] = receipt
                        raise RuntimeError(
                            f"dynamic admission {episode}: cleanup or reset unverified"
                        )
                    entry.update(receipt, status=category, process_group_reaped=True)
                    active_entry = None
                    save()
                    runtime["admission"][key] = receipt
                if "validate_startup" in runtime:
                    runtime["validate_startup"](
                        summary["startup"], runtime["admission"]
                    )
                    save()
                samples = list(gpu_samples(output / "gpu-memory.csv"))
                if (
                    monitor.poll() is not None
                    or not samples
                    or min(total - used for used, total in samples) < 1024
                ):
                    raise RuntimeError(
                        "startup GPU headroom below 1024 MiB or missing measurement"
                    )
            for row in roster:
                episode = row["episode_id"]
                reason = service_failure()
                if reason:
                    break
                key = row["embodiment"], row["seed"]
                live = runtime["admission"].get(key, {})
                if allocated:
                    entry_ok = live.get("category") == "admitted" or (
                        "expert" in row["method"]
                        and live.get("physical_admitted") is True
                    )
                else:
                    entry_ok = admitted(row)
                if not entry_ok or episode not in runtime["commands"]:
                    rows[episode]["reason"] = live.get(
                        "reason", "independent admission or executable unavailable"
                    )
                    continue
                remaining = deadline - time.monotonic()
                if remaining < row["hard_timeout_s"]:
                    reason = "insufficient remaining time for full trial cap"
                    break
                rows[episode] = {"status": "running"}
                active_entry = rows[episode]
                save()
                trial = launch(runtime["commands"][episode], output / f"{episode}.log")
                reason = wait_trial(trial, time.monotonic() + row["hard_timeout_s"])
                reap_process_group(trial)
                trial = None
                if reason:
                    rows[episode] = {
                        "status": "infrastructure_failure",
                        "reason": reason,
                    }
                    break
                receipt = runtime["read_result"](episode)
                if receipt.get("episode_id") != episode:
                    rows[episode] = {
                        "status": "infrastructure_failure",
                        "reason": "trial receipt mismatch",
                    }
                    reason = "trial receipt mismatch"
                    break
                active_entry.update(result=receipt, process_group_reaped=True)
                category = receipt.get("category")
                if actual_provenance(
                    ((receipt.get("result") or {}).get("failure") or {}).get(
                        "diagnostic"
                    )
                ):
                    category = "actual_safety_stop"
                if receipt.get("stop_campaign") is True or category in (
                    "infrastructure_failure",
                    "actual_safety_stop",
                ):
                    reason = (
                        category
                        if category in ("infrastructure_failure", "actual_safety_stop")
                        else "trial requested campaign stop"
                    )
                    active_entry["status"] = (
                        category
                        if category in ("actual_safety_stop", "infrastructure_failure")
                        else "infrastructure_failure"
                    )
                    active_entry["reason"] = reason
                    break
                if allocated and not verified_scene_cleanup(receipt):
                    reason = "trial completion, independent reset or scene cleanup unverified"
                    break
                result = receipt.get("result") or {}
                if (
                    allocated
                    and result.get("outcome") == "recovery_budget_exhausted"
                    and not (
                        receipt["success"] is False
                        and verified_recovery_budget(result, row["recovery_cap"])
                    )
                ):
                    reason = "recovery budget receipt differs from locked trial"
                    break
                if category == "controlled_candidate_stop" or candidate_failure(result):
                    if not (
                        category == "controlled_candidate_stop"
                        and candidate_failure(result)
                        and receipt.get("success") is False
                        and verified_candidate_stop(
                            receipt.get("candidate_stop"), episode, result
                        )
                    ):
                        reason = "candidate stop safety or cleanup unverified"
                        break
                elif category == "controlled_contact_stop":
                    if not (
                        receipt.get("fresh_reset_verified") is True
                        and receipt.get("success") is False
                        and verified_contact_stop(receipt.get("contact_stop"))
                    ):
                        reason = "contact stop safety or cleanup unverified"
                        break
                elif allocated and category != "diagnostic_complete":
                    reason = "trial completion or independent reset unverified"
                    break
                active_entry["status"] = "completed"
                active_entry = None
                save()
            if reason is None:
                reason = service_failure()
        except (Exception, KeyboardInterrupt, SystemExit) as error:
            reason = str(error) or type(error).__name__
        finally:
            cleanup_failures = []
            for label, proc in (
                ("trial", trial),
                ("model", model),
                ("monitor", monitor),
            ):
                if proc is not None:
                    try:
                        reap_process_group(proc)
                    except (Exception, KeyboardInterrupt, SystemExit) as error:
                        cleanup_failures.append(f"{label}: {error}")
            if monitor is not None:
                try:
                    summary["observed_gpu_peak_mib"] = max(
                        (used for used, _ in gpu_samples(output / "gpu-memory.csv")),
                        default=None,
                    )
                except OSError as error:
                    cleanup_failures.append(f"GPU monitor log: {error}")
            if cleanup_failures:
                summary["cleanup_failures"] = cleanup_failures
                reason = reason or "campaign cleanup unverified"
            if active_entry is not None and active_entry["status"] == "running":
                active_entry.update(status="infrastructure_failure", reason=reason)
            for state in rows.values():
                if state["status"] == "not_run" and state["reason"] == "not reached":
                    state["reason"] = reason or "no trial receipt"
            summary["stop_reason"] = reason
            save()
    return summary
