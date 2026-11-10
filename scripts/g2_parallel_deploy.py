"""Launch one independently approved parallel experiment inside its Slurm allocation."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import os
from pathlib import Path
from shutil import disk_usage
import socket
import subprocess
import time

from g2e_host_supervisor import interrupt_signals
from g2_final_campaign import gpu_samples, reap_process_group, verified_scene_cleanup
from g2_final_deploy import (
    allocation_matches,
    authorization,
    commands,
    read,
    run_deployment_stage,
    verify,
)
from g2_job_admission import read_allocation
from g2_parallel_spec import digest, load
from pace_g1_orchestrate import persist, wait_ready

CONFIG = "parallel-config.json"
MIN_DISK_GIB = 8
CACHE_DIRS = (
    "sim-xdg",
    "sim-cuda",
    "sim-torch",
    "sim-tmp",
    "sim-py",
    "model-xdg",
    "model-hf",
    "model-cuda",
)


def verify_package(release):
    plan = verify(release)
    spec = load(release / CONFIG)
    r = spec["resources"]
    if (
        (plan["checkpoint"]["revision"], plan["checkpoint"]["norm_stats_sha256"])
        != (spec["checkpoint_revision"], spec["norm_stats_sha256"])
        or read(release / "parallel-admission.json")
        != {
            "submission_allowed": False,
            "automatic_retry": False,
            "spec": spec,
        }
        or r["graphics_s"] + r["model_start_s"] + r["trial_hard_s"] + r["cleanup_s"]
        >= r["wall_s"]
    ):
        raise ValueError("parallel release identity or allocation envelope differs")
    return plan, spec


def bound_authorization(release, path, job, spec, *, starting=True):
    if (
        not job.isdigit()
        or int(job) <= 0
        or os.environ.get("SLURM_JOB_NAME") != "cpf-pace-g2"
    ):
        raise PermissionError("parallel worker requires its own Slurm allocation")
    auth = authorization(release, path, job, starting=starting)
    expected = dict(
        job_id=job,
        experiment=spec["id"],
        episode_id=spec["episode_id"],
        configuration_sha256=digest(release / CONFIG),
        resources=spec["resources"],
        allocations_authorized=1,
        automatic_retry=False,
    )
    if any(auth.get(k) != v for k, v in expected.items()):
        raise PermissionError("approval differs from the independently bound job")
    return auth


def reap(proc, name, proof):
    try:
        reap_process_group(proc)
        proof[name] = True
    except Exception as error:
        proof[name] = False
        proof[name + "_error"] = str(error)


def run_trial(command, model, output, cwd, deadline, cleanup):
    with (output / "trial.log").open("xb") as log, ExitStack() as stack:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stack.callback(reap, child, "trial_group_reaped", cleanup)
        while child.poll() is None:
            if model.poll() is not None:
                raise RuntimeError("independent model service exited during trial")
            if time.monotonic() >= deadline:
                return "host_trial_hard_timeout", None
            time.sleep(0.25)
        return None, child.returncode


def run(root, release, approval, plan, spec):
    job = os.environ.get("SLURM_JOB_ID", "")
    auth = bound_authorization(release, approval, job, spec)
    r = spec["resources"]
    output = root / spec["output_dir"] / job
    output.mkdir(parents=True, exist_ok=False)
    cleanup = {}
    status = dict(
        job_id=job,
        config=spec,
        outcome="INFRASTRUCTURE_FAILURE",
        cleanup=cleanup,
        started_at=datetime.now(timezone.utc).isoformat(),
        candidate_sha256=auth["candidate_sha256"],
        archive_sha256=auth["archive_sha256"],
    )
    try:
        fields, start, end = read_allocation(job, output)
        minimum = sum(
            r[k] for k in ("graphics_s", "model_start_s", "trial_hard_s", "cleanup_s")
        )
        if (
            not allocation_matches(fields)
            or fields.get("Comment") != auth["comment"]
            or end - start != r["wall_s"]
            or not start <= time.time() <= end - minimum
        ):
            raise PermissionError(
                "actual allocation differs from the approved envelope"
            )
        if disk_usage(output).free < MIN_DISK_GIB * 1024**3:
            raise OSError("insufficient result headroom")
        cache = output / "cache"
        for key in CACHE_DIRS:
            (cache / key).mkdir(parents=True)
        port = spec["port_base"] + int(job) % 2000
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        uri = f"ws://127.0.0.1:{port}"
        with approval.with_suffix(".used").open("x") as f:
            f.write(job + "\n")
        persist(
            output / "allocation-receipt.json",
            dict(job_id=job, start=start, end=end, port=port),
        )
        with interrupt_signals():
            run_deployment_stage(
                [
                    "env",
                    f"G2_STANDALONE_CACHE={cache}",
                    "bash",
                    str(release / "scripts/g2_graphics.sh"),
                    "--g2",
                    str(release),
                    str(output / "graphics"),
                ],
                output / "graphics.log",
                r["graphics_s"] - 5,
                cwd=root,
            )
            graphics = read(output / "graphics/sim-launch.json")
            status["gpu_uuid"] = graphics["gpu_uuid"]
            rt = commands(
                root, release, output, job, plan, graphics, r["wall_s"], approval
            )
            model_cmd = rt["model"]
            model_cmd[model_cmd.index(str(release / "scripts/g2e_model_entry.py"))] = (
                str(release / "scripts/g2_author_native_h50_service.py")
            )
            model_cmd[model_cmd.index("--port") + 1] = str(port)
            model_cmd.insert(1, f"G2_STANDALONE_CACHE={cache}")
            template = next(iter(rt["commands"].values()))
            sim = template[: template.index(str(release / "scripts/g2_final_trial.py"))]
            pace = template[template.index("--pace") + 1]
            trial = output / "trial"
            startup = output / "model/startup.json"
            with ExitStack() as stack:

                def launch(command, name, log_path):
                    log = stack.enter_context(log_path.open("xb"))
                    proc = subprocess.Popen(
                        command,
                        cwd=root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    stack.callback(reap, proc, name, cleanup)
                    return proc

                launch(rt["monitor"], "monitor_group_reaped", output / "gpu.csv")
                model_log = output / "model.log"
                model = launch(model_cmd, "model_group_reaped", model_log)
                model_start = time.monotonic()
                wait_ready(
                    model,
                    model_log,
                    f"checkpoint policy on {uri}",
                    r["model_start_s"],
                    rt["model_phases_file"],
                    rt["model_phase_timeouts"],
                )
                status["model_ready_s"] = time.monotonic() - model_start
                if end - time.time() < r["trial_hard_s"] + r["cleanup_s"]:
                    status.update(
                        outcome="RESOURCE_CENSORED", reason="allocation_headroom"
                    )
                else:
                    started = time.monotonic()
                    command = sim + [
                        str(release / "scripts/g2_parallel_worker.py"),
                        "--release",
                        str(release),
                        "--output",
                        str(trial),
                        "--pace",
                        pace,
                        "--uri",
                        uri,
                        "--started",
                        str(started),
                        "--model-startup",
                        str(startup),
                        "--authorization",
                        str(approval),
                    ]
                    reason, code = run_trial(
                        command,
                        model,
                        output,
                        root,
                        started + r["trial_hard_s"] - 10,
                        cleanup,
                    )
                    receipt = (
                        read(trial / "receipt.json")
                        if (trial / "receipt.json").exists()
                        else {}
                    )
                    status.update(worker_exit_code=code, worker_receipt=receipt)
                    if reason:
                        status.update(
                            outcome="RESOURCE_CENSORED",
                            reason=reason,
                            physical_cleanup_unverified=not receipt,
                        )
                    elif code != 0 or model.poll() is not None or not receipt:
                        status.update(reason="trial_or_service_failed")
                    elif (
                        spec["mode"] == "protected"
                        and not verified_scene_cleanup(receipt)
                    ) or (
                        spec["mode"] == "author_native"
                        and not receipt.get("scene_closed")
                    ):
                        status.update(reason="scene_cleanup_unverified")
                    else:
                        status["outcome"] = receipt["outcome"]
            if list(trial.glob("request-*.npz")) and end - time.time() >= 300:
                try:
                    run_deployment_stage(
                        sim
                        + [str(release / "scripts/g2_standalone_video.py"), str(trial)],
                        output / "query-video.log",
                        100,
                        cwd=root,
                    )
                    status["query_video"] = (trial / "query-cameras.mp4").is_file()
                except Exception as error:
                    status["video_error"] = str(error)
            status["gpu_peak_mib"] = max(
                (used for used, _ in gpu_samples(output / "gpu.csv")), default=None
            )
    except (Exception, KeyboardInterrupt) as error:
        status.update(error=type(error).__name__, detail=str(error))
    finally:
        if any(v is False for v in cleanup.values()) or status.get("error"):
            status.update(
                previous_outcome=status["outcome"], outcome="INFRASTRUCTURE_FAILURE"
            )
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        persist(output / "result.json", status)
    return int(status["outcome"] == "INFRASTRUCTURE_FAILURE")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("release", type=Path)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--authorization", type=Path, required=True)
    a = p.parse_args()
    release = a.release.resolve()
    plan, spec = verify_package(release)
    return run(a.root.resolve(), release, a.authorization.resolve(), plan, spec)


if __name__ == "__main__":
    raise SystemExit(main())
