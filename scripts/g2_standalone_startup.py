"""Bounded Slurm allocation reads and explicit launch-stage logging."""

from datetime import datetime, timezone
import json
import subprocess
import time


def event(stage, state="entered", **details):
    print(
        json.dumps(
            dict(
                stage=stage,
                state=state,
                utc=datetime.now(timezone.utc).isoformat(),
                **details,
            ),
            default=str,
        ),
        flush=True,
    )


def read_parallel_allocation(job, output, *, query=subprocess.run, pause=time.sleep):
    """Retry transport/configuration only. An identity mismatch fails immediately."""
    attempts = []
    for timeout in (15, 20, 25):
        try:
            reply = query(
                ["scontrol", "show", "job", "-o", job],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            fields = dict(
                word.split("=", 1) for word in reply.stdout.split() if "=" in word
            )
            if fields.get("JobId") != job:
                raise PermissionError("scontrol returned a different JobId")
            (output / "slurm-allocation.txt").write_text(reply.stdout)
            if fields.get("JobState") == "CONFIGURING":
                raise ValueError("Slurm is still configuring this allocation")
            if fields.get("JobState") != "RUNNING":
                raise PermissionError(
                    "unexpected Slurm state: " + str(fields.get("JobState"))
                )
            start, end = (
                datetime.fromisoformat(fields[k]).timestamp()
                for k in ("StartTime", "EndTime")
            )
            event("slurm_query", "passed", attempt=len(attempts) + 1, timeout_s=timeout)
            return fields, start, end
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            KeyError,
            ValueError,
        ) as error:
            attempts.append(f"{type(error).__name__}: {error}")
            event(
                "slurm_query",
                "transient_failure",
                attempt=len(attempts),
                timeout_s=timeout,
                error=attempts[-1],
                stderr=getattr(error, "stderr", None),
            )
            if timeout != 25:
                pause(2)
    raise TimeoutError(
        "unable to read running allocation after three attempts: " + "; ".join(attempts)
    )
