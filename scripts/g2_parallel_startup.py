"""Shared launch evidence and prerequisite for the parallel G2 batch."""

from datetime import datetime, timezone
import json
import subprocess
import time

from g2_parallel_spec import digest

PROBE_WALL_S = 600
PROBE_STAGES = (
    "allocation_validated",
    "output_ready",
    "runtime_identity_verified",
    "graphics_ready",
    "simulation_imports_ready",
    "model_imports_ready",
)


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


def probe_path(root, job):
    if not job.isascii() or not job.isdigit() or not int(job):
        raise ValueError("expected an actual probe Job ID")
    return root / "results/pace-g2-startup-probes" / job / "result.json"


def validate_probe(root, job, batch_path, query):
    """Check the actual completed Slurm probe and bind its receipt to all seven packages."""
    path = probe_path(root, job)
    data = json.loads(path.read_text())
    batch = json.loads(batch_path.read_text())
    if (
        data.get("outcome") != "STARTUP_PROBE_PASSED"
        or data.get("job_id") != job
        or data.get("node") != "gnode39"
        or data.get("batch_sha256") != digest(batch_path)
        or data.get("candidate_manifests")
        != {r["id"]: r["manifest_sha256"] for r in batch["rows"]}
        or data.get("completed_startup_stages") != list(PROBE_STAGES)
        or data.get("cleanup")
        != {
            "probe_graphics_group_reaped": True,
            "probe_sim_group_reaped": True,
            "probe_model_group_reaped": True,
        }
    ):
        raise PermissionError(
            "successful compute-node probe for this exact batch is required"
        )
    rows = [
        line.split("|")
        for line in query(
            "sacct",
            "-j",
            job,
            "-nP",
            "-o",
            "JobIDRaw,State,ExitCode,NodeList,AllocTRES",
        ).splitlines()
    ]
    top = [r for r in rows if r[0] == job]
    if (
        len(top) != 1
        or top[0][1:4] != ["COMPLETED", "0:0", "gnode39"]
        or not any(
            f.startswith("gres/gpu") and f.endswith("=1") for f in top[0][4].split(",")
        )
        or not any(
            r[:4] == [job + ".batch", "COMPLETED", "0:0", "gnode39"] for r in rows
        )
    ):
        raise PermissionError(
            "Slurm has not confirmed the probe allocation and batch completion"
        )
    return path, digest(path)
