"""Bounded native trial batch; no resource submission, process-group owner is host."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def run(entries, output, timeout=1200):
    rows = []
    stopped = None
    for entry in entries:
        if stopped:
            rows.append(
                dict(episode_id=entry["episode_id"], status="not_run", reason=stopped)
            )
            continue
        started = time.monotonic()
        try:
            process = subprocess.run(entry["argv"], timeout=timeout, check=False)
            result_path = Path(entry["output"]) / "result.json"
            if process.returncode or not result_path.is_file():
                stopped = (
                    f"infrastructure: child exit {process.returncode} or missing result"
                )
                rows.append(
                    dict(
                        episode_id=entry["episode_id"],
                        status="infrastructure_failure",
                        reason=stopped,
                    )
                )
                continue
            result = json.loads(result_path.read_text())
            failure = result.get("failure") or {}
            # Candidate rejection is local to a trial. Actual unsafe geometry,
            # synchronization/observations, timeouts and service failures stop
            # the campaign. Never resubmit or replace an initial state.
            unsafe_actual = any(
                text in failure.get("detail", "")
                for text in (
                    "measured native geometry blocked",
                    "measured fingers exceed",
                    "arm joint outside limits",
                )
            )
            infrastructure = result["outcome"] in (
                "timeout",
                "interrupted",
                "policy_failed",
                "environment_error",
                "adapter_error",
            )
            if unsafe_actual or infrastructure:
                stopped = (
                    "measured_safety_stop" if unsafe_actual else "infrastructure_stop"
                )
            rows.append(
                dict(
                    episode_id=entry["episode_id"],
                    status="completed",
                    outcome=result["outcome"],
                    success=result["success"],
                    executed=result["executed"],
                    wall_s=time.monotonic() - started,
                )
            )
        except subprocess.TimeoutExpired:
            # Descendants remain in the host-owned stage process group. Abort
            # the batch so its owner kills/reaps that group before statistics.
            stopped = "hard_trial_timeout"
            rows.append(
                dict(
                    episode_id=entry["episode_id"],
                    status="infrastructure_failure",
                    reason=stopped,
                )
            )
            output.write_text(
                json.dumps(dict(rows=rows, stopped=stopped), indent=2) + "\n"
            )
            raise
        output.write_text(json.dumps(dict(rows=rows, stopped=stopped), indent=2) + "\n")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("entries", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    rows = run(json.loads(a.entries.read_text()), a.output)
    raise SystemExit(0 if all(r["status"] == "completed" for r in rows) else 1)
