#!/usr/bin/env python3
"""Supervise bounded G2 stages in one existing allocation. Never allocate or submit."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
import os
import math
import signal
import subprocess
import time

from pace_g1_orchestrate import persist as save_json
from pace_g1_orchestrate import StageError, utc, wait_ready, memory_samples

STAGES = ("dynamic", "aloha", "statistics")


def stop_group(proc, grace_s=2.0):
    """Reap the leader and stop surviving descendants even if it has exited."""
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def argv(value):
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(x, str) or not x for x in value)
    ):
        raise ValueError("expected a nonempty argument vector")
    if Path(value[0]).name in ("sbatch", "srun", "salloc"):
        raise ValueError("orchestrator cannot allocate resources")
    return value


def validate_plan(plan):
    if plan["data_kind"] not in ("synthetic", "diagnostic"):
        raise ValueError("formal G2 requires the future frozen release gate")
    if any(
        not math.isfinite(plan[k]) or plan[k] <= 0
        for k in ("total_timeout_s", "model_ready_timeout_s")
    ):
        raise ValueError("invalid orchestration budget")
    argv(plan["model_server"])
    if not isinstance(plan["ready_line"], str) or not plan["ready_line"]:
        raise ValueError("missing model readiness marker")
    if set(plan["stages"]) != set(STAGES):
        raise ValueError("all three native G2-E stages must be represented explicitly")
    seen = set()
    for name in STAGES:
        row = plan["stages"][name]
        if row.get("argv") is None:
            if not row.get("skip_reason"):
                raise ValueError(f"{name} requires an explicit skip reason")
        else:
            argv(row["argv"])
            if not math.isfinite(row["timeout_s"]) or row["timeout_s"] <= 0:
                raise ValueError("nonpositive stage timeout")
        if not set(row["requires"]).issubset(seen):
            raise ValueError("stage prerequisites must name earlier stages")
        if name == "arx" and "dynamic" not in row["requires"]:
            raise ValueError("ARX requires the dynamic control gate")
        if name in ("ablation", "expert") and "dynamic" not in row["requires"]:
            raise ValueError("retargeted controls require the dynamic gate")
        seen.add(name)


@contextmanager
def interrupt_signals():
    previous = {}

    def handler(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, handler)
    try:
        yield
    finally:
        for number, old in previous.items():
            signal.signal(number, old)


def orchestrate(plan, output):
    validate_plan(plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        data_kind=plan["data_kind"],
        formal_score_eligible=False,
        passed=False,
        started_at=utc(),
        model_server_starts=0,
        execution_complete=False,
        stages={name: dict(status="skipped", reason="not reached") for name in STAGES},
    )
    result_path = output / "g2_acceptance.json"
    end = time.monotonic() + plan["total_timeout_s"]
    if "deadline_epoch" in plan:
        end = min(end, time.monotonic() + max(0, plan["deadline_epoch"] - time.time()))
    model, monitor, stage = None, None, None
    streams = []
    model_error = None

    def save():
        save_json(result_path, report)

    def budget(seconds):
        value = min(seconds, end - time.monotonic())
        if value <= 0:
            raise StageError("shared execution deadline exhausted", 124)
        return value

    def launch(command, filename):
        stream = (output / filename).open("wb")
        streams.append(stream)
        return subprocess.Popen(
            argv(command),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def require_model():
        nonlocal model, model_error
        if model_error:
            raise StageError("model startup previously failed: " + model_error)
        if model is None:
            model = launch(plan["model_server"], "model-server.log")
            report.update(model_server_starts=1, model_server_pid=model.pid)
            save()
            try:
                wait_ready(
                    model,
                    output / "model-server.log",
                    plan["ready_line"],
                    budget(plan["model_ready_timeout_s"]),
                    plan.get("model_phases_file"),
                    plan.get("model_phase_timeouts"),
                )
            except BaseException as exc:
                model_error = str(exc)
                report["model_start_error"] = model_error
                raise
        if model.poll() is not None:
            model_error = "shared model process exited"
            raise StageError(model_error)

    save()
    try:
        if plan.get("monitor"):
            monitor = launch(plan["monitor"], "gpu-memory.csv")
        for name in STAGES:
            config = plan["stages"][name]
            entry = report["stages"][name]
            if config.get("argv") is None:
                entry["reason"] = config["skip_reason"]
                save()
                continue
            missing = [
                dep
                for dep in config["requires"]
                if report["stages"][dep]["status"] != "passed"
            ]
            if missing:
                entry["reason"] = "prerequisite failed or skipped: " + ", ".join(
                    missing
                )
                save()
                continue
            entry.clear()
            entry.update(status="running", started_at=utc())
            start = time.monotonic()
            stage = None
            save()
            try:
                stage_end = time.monotonic() + budget(config["timeout_s"])
                if config.get("needs_model"):
                    require_model()
                stage_timeout = min(stage_end, end) - time.monotonic()
                if stage_timeout <= 0:
                    raise StageError(
                        "stage budget exhausted during model readiness", 124
                    )
                stage = launch(config["argv"], name + ".log")
                # Poll to detect a dead policy service while its client is still running.
                while stage.poll() is None:
                    if time.monotonic() >= min(stage_end, end):
                        raise StageError("stage hard timeout", 124)
                    if config.get("needs_model") and model.poll() is not None:
                        raise StageError("shared model died during stage")
                    time.sleep(0.03)
                if stage.returncode:
                    raise StageError(
                        f"stage exited {stage.returncode}", stage.returncode
                    )
                entry.update(status="passed", exit_code=0)
            except KeyboardInterrupt as exc:
                entry.update(status="failed", reason=str(exc), exit_code=130)
                raise
            except Exception as exc:
                entry.update(
                    status="failed", reason=str(exc), exit_code=getattr(exc, "code", 1)
                )
            finally:
                stop_group(stage)
                stage = None
                entry.update(ended_at=utc(), wall_s=time.monotonic() - start)
                save()
        summary = (
            Path(plan["evaluation_summary_file"])
            if plan.get("evaluation_summary_file")
            else None
        )
        if summary is not None and summary.is_file():
            report["evaluation"] = json.loads(summary.read_text())
            if report["evaluation"].get("formal_score_eligible") is not False:
                raise ValueError("G2-E diagnostic eligibility mismatch")
        report["passed"] = all(
            row["status"] in ("passed", "skipped") for row in report["stages"].values()
        )
        report["complete"] = all(
            row["status"] == "passed" for row in report["stages"].values()
        )
        report["execution_complete"] = report["complete"] and report.get(
            "evaluation", {}
        ).get("all_trials_complete", False)
    except KeyboardInterrupt as exc:
        report.update(interrupted=True, error=str(exc))
        for entry in report["stages"].values():
            if entry.get("reason") == "not reached":
                entry["reason"] = "orchestrator interrupted"
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        stop_group(stage)
        stop_group(model)
        stop_group(monitor)
        for stream in streams:
            stream.close()
        samples = memory_samples(output / "gpu-memory.csv")
        report["observed_gpu_peak_mib"] = max(samples) if samples else None
        report["ended_at"] = utc()
        save()
    return 0 if report["passed"] else 1


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with interrupt_signals():
        raise SystemExit(orchestrate(json.loads(args.plan.read_text()), args.output))


if __name__ == "__main__":
    main()
