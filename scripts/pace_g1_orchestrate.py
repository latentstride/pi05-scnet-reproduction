#!/usr/bin/env python3
"""Supervise all three G1 stages in one allocation; never submit a job."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


STAGES = ("G1.1", "G1.2", "G1.3")


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def persist(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def terminate(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class StageError(RuntimeError):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def command(argv, log, seconds):
    with log.open("w") as stream:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(seconds)
            if rc:
                raise StageError("command exited {}; see {}".format(rc, log.name), rc)
        except subprocess.TimeoutExpired:
            raise StageError("command timed out after {}s; see {}".format(seconds, log.name), 124)
        finally:
            terminate(proc)


def wait_ready(proc, log, marker, seconds):
    deadline = time.monotonic() + seconds
    offset = 0
    marker = marker.encode("utf-8")
    with log.open("rb") as stream:
        while time.monotonic() < deadline:
            rc = proc.poll()
            if rc is not None:
                raise StageError("checkpoint server exited before ready; see model-server.log", rc or 1)
            stream.seek(offset)
            while True:
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    break
                if line.rstrip() == marker:
                    return
                offset = stream.tell()
            time.sleep(0.25)
    raise StageError("checkpoint server did not reach readiness within {}s".format(seconds), 124)


def orchestrate(plan, output):
    """The one model child remains alive from the G1.2 readiness gate through G1.3."""
    output.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "job_id": plan["job_id"], "release_commit": plan["release_commit"],
              "started_at": utc(), "stages": {name: {"status": "skipped", "reason": "not reached"}
                                              for name in STAGES}, "model_server_starts": 0}
    acceptance = output / "g1_acceptance.json"
    persist(acceptance, report)
    model = monitor = None
    monitor_file = None
    model_file = None

    def run_stage(name, fn):
        entry = report["stages"][name]
        entry.clear()
        entry.update(status="running", started_at=utc())
        begin = time.monotonic()
        persist(acceptance, report)
        try:
            fn()
        except BaseException as exc:
            entry.update(status="failed", exit_code=getattr(exc, "code", 130 if isinstance(exc, KeyboardInterrupt)
                                                            else 1), error=str(exc))
            with (output / (name + "-exception.log")).open("w") as stream:
                traceback.print_exc(file=stream)
            if isinstance(exc, KeyboardInterrupt):
                raise
            return False
        else:
            entry.update(status="passed", exit_code=0)
            return True
        finally:
            entry.update(ended_at=utc(), elapsed_s=time.monotonic() - begin)
            persist(acceptance, report)

    def one():
        command(plan["g1_1"], output / "G1.1.log", plan["timeouts"]["G1.1"])

    def two():
        nonlocal model, model_file
        model_log = output / "model-server.log"
        model_file = model_log.open("w")
        model = subprocess.Popen(plan["model_server"], stdin=subprocess.DEVNULL, stdout=model_file,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        report["model_server_starts"] += 1
        report["model_server_pid"] = model.pid
        persist(acceptance, report)
        wait_ready(model, model_log, plan["ready_line"], plan["timeouts"]["model_ready"])
        command(plan["g1_2_query"], output / "G1.2-query.log", plan["timeouts"]["G1.2"])
        if model.poll() is not None:
            raise StageError("checkpoint server exited after G1.2")

    def three():
        if model is None or model.poll() is not None:
            raise StageError("the G1.2 checkpoint service is no longer running")
        if plan.get("memory_probe"):
            samples = memory_samples(output / "gpu-memory.csv")
            if not samples or "simulator_peak_mib" not in report:
                raise StageError("missing GPU memory evidence; joint execution was not started")
            raw = subprocess.check_output(plan["memory_probe"], timeout=10, universal_newlines=True)
            used, total = [int(value.strip()) for value in raw.strip().split(",")]
            report["checkpoint_idle_mib"] = used
            report["memory_total_mib"] = total
            needed = report["simulator_peak_mib"] + plan["memory_margin_mib"]
            if total - used < needed:
                raise StageError("joint GPU memory headroom {} MiB is below {} MiB; "
                                 "closed loop was not started".format(total - used, needed), 78)
        command(plan["g1_3"], output / "G1.3.log", plan["timeouts"]["G1.3"])

    try:
        if plan.get("monitor"):
            monitor_file = (output / "gpu-memory.csv").open("w")
            monitor = subprocess.Popen(plan["monitor"], stdin=subprocess.DEVNULL, stdout=monitor_file,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        if not run_stage("G1.1", one):
            for name in ("G1.2", "G1.3"):
                report["stages"][name]["reason"] = "no verified native G1.1 observation"
        else:
            samples = memory_samples(output / "gpu-memory.csv")
            if samples:
                report["simulator_peak_mib"] = max(samples)
            if not run_stage("G1.2", two):
                report["stages"]["G1.3"]["reason"] = "checkpoint recovery or real-observation query failed"
            else:
                report["passed"] = run_stage("G1.3", three)
    except KeyboardInterrupt:
        report["interrupted"] = True
    except Exception as exc:
        report["orchestration_error"] = str(exc)
        with (output / "orchestration-exception.log").open("w") as stream:
            traceback.print_exc(file=stream)
    finally:
        for proc, stream in ((model, model_file), (monitor, monitor_file)):
            terminate(proc)
            if stream:
                stream.close()
        report["ended_at"] = utc()
        samples = memory_samples(output / "gpu-memory.csv")
        if samples:
            report["observed_gpu_peak_mib"] = max(samples)
        persist(acceptance, report)
    return 0 if report["passed"] else 1


def memory_samples(path):
    if not path.is_file():
        return []
    samples = []
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split(",")
        if len(fields) == 4 and fields[2].strip().isdigit():
            samples.append(int(fields[2].strip()))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt("signal {}".format(signum))

    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(orchestrate(json.loads(args.plan.read_text()), args.output))


if __name__ == "__main__":
    main()
