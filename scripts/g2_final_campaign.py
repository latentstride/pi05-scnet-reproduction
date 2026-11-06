"""Supervise one shared model and independently admitted diagnostic trials."""

import os
from pathlib import Path
import subprocess
import time

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
    deadline = time.monotonic() + min(
        runtime["timeout_s"], budget["bounded_execution_s"]
    )
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
        while (
            proc.poll() is None and model.poll() is None and time.monotonic() < stop_at
        ):
            time.sleep(0.025)
        if model.poll() is not None:
            return "shared model exited"
        if proc.poll() is None:
            return "trial hard timeout"
        if proc.returncode:
            return "trial process failure"
        return None

    reason = None
    save()
    with interrupt_signals():
        try:
            if allocated:
                monitor = launch(runtime["monitor"], output / "gpu-memory.csv")
            startup_deadline = min(
                deadline,
                time.monotonic()
                + budget["overhead"]["shared_model_and_dynamic_entry_s"],
            )
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
                for index, (key, (episode, command)) in enumerate(
                    runtime["probes"].items()
                ):
                    if time.monotonic() >= startup_deadline:
                        raise TimeoutError("dynamic admission deadline")
                    trial = launch(command, output / f"probe-{episode}.log")
                    reason = wait_trial(
                        trial,
                        min(
                            startup_deadline,
                            time.monotonic() + (480 if index == 0 else 145),
                        ),
                    )
                    stop_group(trial)
                    trial = None
                    if reason:
                        raise RuntimeError(f"dynamic admission {episode}: {reason}")
                    receipt = runtime["read_result"](episode, probe=True)
                    category = receipt.get("category")
                    summary["startup"][episode] = receipt
                    save()
                    if receipt.get("episode_id") != episode or category not in (
                        "admitted",
                        "not_admitted",
                    ):
                        raise RuntimeError(f"dynamic admission {episode}: {category}")
                    runtime["admission"][key] = receipt
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
                if model.poll() is not None:
                    reason = "shared model exited"
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
                    rows[episode]["reason"] = (
                        "independent admission or executable unavailable"
                    )
                    continue
                remaining = deadline - time.monotonic()
                if remaining < row["hard_timeout_s"]:
                    rows[episode]["reason"] = (
                        "insufficient remaining time for full trial cap"
                    )
                    continue
                rows[episode] = {"status": "running"}
                save()
                trial = launch(runtime["commands"][episode], output / f"{episode}.log")
                reason = wait_trial(trial, time.monotonic() + row["hard_timeout_s"])
                trial_pid = trial.pid
                stop_group(trial)
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
                rows[episode] = {"status": "completed", "result": receipt}
                save()
                category = receipt.get("category")
                if receipt.get("stop_campaign") is True or category in (
                    "infrastructure_failure",
                    "actual_safety_stop",
                ):
                    reason = (
                        receipt.get("category")
                        or "actual safety or input contract failure"
                    )
                    break
                if category == "controlled_contact_stop":
                    if not (
                        receipt.get("fresh_reset_verified") is True
                        and verified_contact_stop(receipt.get("contact_stop"))
                        and process_group_gone(trial_pid)
                    ):
                        reason = "contact stop safety or cleanup unverified"
                        break
                elif allocated and (
                    category != "diagnostic_complete"
                    or receipt.get("fresh_reset_verified") is not True
                ):
                    reason = "trial completion or independent reset unverified"
                    break
        except (Exception, KeyboardInterrupt) as error:
            reason = str(error)
            for state in rows.values():
                if state["status"] == "running":
                    state.update(status="infrastructure_failure", reason=reason)
        finally:
            stop_group(trial)
            stop_group(model)
            stop_group(monitor)
            if monitor is not None:
                summary["observed_gpu_peak_mib"] = max(
                    (used for used, _ in gpu_samples(output / "gpu-memory.csv")),
                    default=None,
                )
            for state in rows.values():
                if state["status"] == "not_run" and state["reason"] == "not reached":
                    state["reason"] = reason or "no trial receipt"
            summary["stop_reason"] = reason
            save()
    return summary
