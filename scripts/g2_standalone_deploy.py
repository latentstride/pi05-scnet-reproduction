"""One approved Slurm allocation for the frozen author H50 or ARX seed1 trial."""

import argparse
import json
import os
from pathlib import Path
from shutil import disk_usage
import socket
import time

from g2_author_h50_campaign import CAMPAIGN_RESERVE_S, bind_runtime, REVIEW_FILE
from g2_author_h50_limits import EPISODE
from g2_final_deploy import (
    allocation_matches,
    authorization,
    commands,
    digest,
    read,
    run_deployment_stage,
    verify,
)
from g2_final_campaign import supervise
from g2_job_admission import read_allocation
from g2e_host_supervisor import interrupt_signals
from pace_g1_orchestrate import persist

BASE_MANIFEST = "9270db1e97eafedfb6f94711c8b8c9f33efd76d04ab800af99bf11a42556f19a"
BASE_ARCHIVE = "c8eeff121eefb1bff98e8103cc3314ef6d95669e362889f3d68ac35507389e99"
ARX = "unified-seed1-arx-full-w4"
SPEC_FILE = "standalone-review.json"
OVERHEAD = dict(graphics_s=300, statistics_s=120, cleanup_s=120)
MODES = {
    "author": dict(
        episode=EPISODE,
        wall_s=4800,
        startup_s=1080,
        hard_s=2700,
        port_base=35000,
        min_disk_gib=12,
        fee_cap="4.80",
    ),
    "arx": dict(
        episode=ARX,
        wall_s=5400,
        startup_s=1680,
        hard_s=2600,
        port_base=39000,
        min_disk_gib=8,
        fee_cap="5.40",
    ),
}


def spec(mode):
    return dict(
        mode=mode,
        **MODES[mode],
        submission_allowed=False,
        base_manifest_sha256=BASE_MANIFEST,
        base_archive_sha256=BASE_ARCHIVE,
        overhead=OVERHEAD,
        host_reserve_s=CAMPAIGN_RESERVE_S,
        allocations=1,
        model_services=1,
        automatic_retry=False,
    )


def verify_standalone(release):
    plan = verify(release)
    selected = read(release / SPEC_FILE)
    if selected != spec(selected["mode"]):
        raise ValueError("standalone review spec differs")
    archive_manifest = read(release / "standalone-base-members.json")
    if archive_manifest["manifest_sha256"] != BASE_MANIFEST:
        raise ValueError("standalone parent identity differs")
    for name, checksum in archive_manifest["members"].items():
        if digest(release / name) != checksum:
            raise ValueError(f"frozen runtime member differs: {name}")
    return plan, selected


def isolated_port(selected, job):
    return selected["port_base"] + int(job) % 4000


def bind(plan, runtime, release, output, selected, job):
    mode, episode = selected["mode"], selected["episode"]
    if mode == "author":
        plan, runtime = bind_runtime(
            plan, runtime, release, output, review=read(release / REVIEW_FILE)
        )
    else:
        runtime["probes"] = {
            key: val for key, val in runtime["probes"].items() if val[0] == episode
        }
        if len(runtime["probes"]) != 1:
            raise ValueError("ARX startup admission not bound")
    (row,) = [r for r in plan["trials"] if r["episode_id"] == episode]
    if row["hard_timeout_s"] != selected["hard_s"]:
        raise ValueError("standalone trial deadline differs")
    overhead = dict(OVERHEAD, shared_model_and_dynamic_entry_s=selected["startup_s"])
    bounded = row["hard_timeout_s"] + sum(overhead.values())
    if bounded + CAMPAIGN_RESERVE_S > selected["wall_s"]:
        raise ValueError("standalone allocation cannot cover its full budget")
    execution = dict(
        plan,
        trials=[row],
        budget=dict(
            plan["budget"],
            trials_s=row["hard_timeout_s"],
            overhead=overhead,
            proposed_allocation_s=selected["wall_s"],
            bounded_execution_s=bounded,
            proposed_fee_cap_yuan=float(selected["fee_cap"]),
        ),
    )
    old_port = 50000 + int(job) % 14000
    port = isolated_port(selected, job)
    old_uri, uri = (f"ws://127.0.0.1:{p}" for p in (old_port, port))

    def bind_uri(cmd):
        args = cmd.copy()
        i = args.index("--uri") + 1
        if args[i] != old_uri:
            raise ValueError("unexpected inherited policy URI")
        args[i] = uri
        return args

    runtime["commands"] = {episode: bind_uri(runtime["commands"][episode])}
    runtime["probes"] = (
        {}
        if mode == "author"
        else {k: (name, bind_uri(cmd)) for k, (name, cmd) in runtime["probes"].items()}
    )
    runtime.pop("validate_startup", None)
    model = runtime["model"].copy()
    model[model.index("--port") + 1] = str(port)
    model.insert(1, f"G2_STANDALONE_CACHE={output / 'cache'}")
    runtime.update(
        model=model,
        ready_marker=f"checkpoint policy on {uri}",
        startup_cap_s=selected["startup_s"],
    )
    return execution, runtime


def output_for(root, selected, job):
    return root / "results" / f"pace-g2-standalone-{selected['mode']}-{job}"


def run(root, release, approval, plan, selected):
    job = os.environ.get("SLURM_JOB_ID", "")
    if (
        not job.isdigit()
        or job == "127408633"
        or os.environ.get("SLURM_JOB_NAME") != "cpf-pace-g2"
    ):
        raise PermissionError("standalone entry requires its own allocated job")
    auth = authorization(release, approval, job)
    if (
        auth.get("job_id") != job
        or auth.get("standalone_mode") != selected["mode"]
        or auth.get("allocation_s") != selected["wall_s"]
        or auth.get("allocations_authorized") != 1
    ):
        raise PermissionError(
            "authorization does not bind the standalone mode and walltime"
        )
    output = output_for(root, selected, job)
    output.mkdir(parents=True, exist_ok=False)
    persist(output / "authorization-receipt.json", auth)
    try:
        fields, start, end = read_allocation(job, output)
        minimum = (
            selected["hard_s"]
            + selected["startup_s"]
            + sum(OVERHEAD.values())
            + CAMPAIGN_RESERVE_S
        )
        if (
            not allocation_matches(fields)
            or end - start != selected["wall_s"]
            or not start <= time.time() <= end - minimum
        ):
            raise PermissionError(
                "standalone allocation differs from its reserved budget"
            )
        if disk_usage(output).free < selected["min_disk_gib"] * 1024**3:
            raise OSError("insufficient result space for standalone evidence")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", isolated_port(selected, job)))
        for name in (
            "sim-xdg",
            "sim-cuda",
            "sim-torch",
            "sim-tmp",
            "sim-py",
            "model-xdg",
            "model-hf",
            "model-cuda",
        ):
            (output / "cache" / name).mkdir(parents=True, exist_ok=False)
        with approval.with_suffix(".used").open("x") as f:
            f.write(job + "\n")
        persist(
            output / "allocation-receipt.json", dict(job_id=job, start=start, end=end)
        )
        with interrupt_signals():
            run_deployment_stage(
                [
                    "env",
                    f"G2_STANDALONE_CACHE={output / 'cache'}",
                    "bash",
                    str(release / "scripts/g2_graphics.sh"),
                    "--g2",
                    str(release),
                    str(output / "graphics"),
                ],
                output / "graphics.log",
                295,
                cwd=root,
            )
            timeout = (
                end
                - time.time()
                - OVERHEAD["statistics_s"]
                - OVERHEAD["cleanup_s"]
                - CAMPAIGN_RESERVE_S
            )
            runtime = commands(
                root,
                release,
                output,
                job,
                plan,
                read(output / "graphics/sim-launch.json"),
                timeout,
                approval,
            )
            execution, runtime = bind(plan, runtime, release, output, selected, job)
            persist(
                output / "effective-launch.json",
                dict(
                    model=runtime["model"],
                    trials=runtime["commands"],
                    probes=list(runtime["probes"].values()),
                    selected=selected,
                ),
            )
            result = supervise(
                execution, runtime, output / "supervisor", allocated=True
            )
            row = result["rows"][selected["episode"]]
            # The frozen ARX path uses infrastructure_failure for a timeout.
            # Annotate resource censoring without changing its stop decision.
            receipt = row.get("result") or {}
            detail = receipt.get("result") or {}
            reason = row.get("reason", result["stop_reason"])
            outcome = (
                receipt.get("outcome") or row.get("outcome") or detail.get("outcome")
            )
            if (
                reason == "trial hard timeout"
                or outcome == "timeout"
                or (detail.get("failure") or {}).get("code") == "timeout"
            ):
                outcome = "RESOURCE_CENSORED"
            persist(
                output / "standalone-result.json",
                dict(
                    job_id=job,
                    selected=selected,
                    outcome=outcome,
                    row=row,
                    stop_reason=result["stop_reason"],
                ),
            )
            if (
                selected["mode"] == "author"
                and receipt
                and list((output / "trials" / EPISODE).glob("request-*.npz"))
            ):
                child = runtime["commands"][EPISODE]
                py = child[
                    : child.index(str(release / "scripts/g2_author_h50_trial.py"))
                ]
                run_deployment_stage(
                    py
                    + [
                        str(release / "scripts/g2_standalone_video.py"),
                        str(output / "trials" / EPISODE),
                    ],
                    output / "query-video.log",
                    100,
                    cwd=root,
                )
        return 1 if result["stop_reason"] or row["status"] != "completed" else 0
    except (Exception, KeyboardInterrupt, SystemExit) as e:
        persist(
            output / "deployment-failure.json",
            dict(error=type(e).__name__, detail=str(e)),
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("release", type=Path)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--root", type=Path)
    p.add_argument("--authorization", type=Path)
    a = p.parse_args()
    release = a.release.resolve()
    plan, selected = verify_standalone(release)
    if a.verify:
        print(
            json.dumps(
                dict(selected=selected, manifest_sha256=digest(release / "SHA256SUMS"))
            )
        )
        return 0
    if not (a.root and a.authorization):
        p.error("allocated execution requires --root and --authorization")
    return run(a.root.resolve(), release, a.authorization.resolve(), plan, selected)


if __name__ == "__main__":
    raise SystemExit(main())
