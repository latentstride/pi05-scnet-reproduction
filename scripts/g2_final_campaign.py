"""Supervisor contract for one already allocated G2 campaign.

Runs a shared process and bounded trial commands. The review package supplies
no production commands or authorization. Tests inject CPU-only subprocesses.
"""

from pathlib import Path
import subprocess
import time

from g2e_host_supervisor import argv, interrupt_signals, stop_group
from pace_g1_orchestrate import persist, wait_ready


def supervise(plan, runtime, output, *, rehearsal=False):
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
    if not rehearsal:
        raise PermissionError("review-only package has no executable GPU admission")
    if runtime.get("mode") != "cpu_rehearsal":
        raise ValueError("CPU rehearsal runtime required")
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
    model = trial = None

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

    reason = None
    save()
    with interrupt_signals():
        try:
            model_log = output / "model.log"
            model = launch(runtime["model"], model_log)
            summary["model_server_starts"] = 1
            save()
            wait_ready(
                model,
                model_log,
                runtime["ready_marker"],
                min(runtime["model_timeout_s"], deadline - time.monotonic()),
            )
            for row in roster:
                episode = row["episode_id"]
                if model.poll() is not None:
                    reason = "shared model exited"
                    break
                if not admitted(row) or episode not in runtime["commands"]:
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
                stop_at = time.monotonic() + row["hard_timeout_s"]
                while (
                    trial.poll() is None
                    and model.poll() is None
                    and time.monotonic() < stop_at
                ):
                    time.sleep(0.025)
                if model.poll() is not None:
                    reason = "shared model exited"
                elif trial.poll() is None:
                    reason = "trial hard timeout"
                elif trial.returncode:
                    reason = "trial process failure"
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
                if receipt.get("stop_campaign") is True:
                    reason = "actual safety or input contract failure"
                    break
        except (Exception, KeyboardInterrupt) as error:
            reason = str(error)
            for state in rows.values():
                if state["status"] == "running":
                    state.update(status="infrastructure_failure", reason=reason)
        finally:
            stop_group(trial)
            stop_group(model)
            for state in rows.values():
                if state["status"] == "not_run" and state["reason"] == "not reached":
                    state["reason"] = reason or "no trial receipt"
            summary["stop_reason"] = reason
            save()
    return summary
