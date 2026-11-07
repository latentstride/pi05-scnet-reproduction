"""Verify the candidate offline or start it inside a separately approved allocation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import time

from g2_final_campaign import reap_process_group, supervise
from g2_job_admission import read_allocation
from g2_single_launch import OPENPI, PACE
from g2e_host_supervisor import interrupt_signals
from pace_g1_orchestrate import persist


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify(release):
    manifest = release / "SHA256SUMS"
    declared = {}
    for line in manifest.read_text().splitlines():
        checksum, name = line.split("  ", 1)
        path = release / name
        if (
            path.resolve().parent != path.parent
            or release not in path.parents
            or ".." in Path(name).parts
            or path.is_symlink()
            or name in declared
            or digest(path) != checksum
        ):
            raise ValueError(f"invalid candidate file: {name}")
        declared[name] = checksum
    actual = {
        str(p.relative_to(release))
        for p in release.rglob("*")
        if p.is_file() and p != manifest
    }
    if set(declared) != actual:
        raise ValueError("candidate manifest coverage differs")
    plan = read(release / "unified-plan.json")
    admission = read(release / "release-admission.json")
    if (
        admission["submission_allowed"] is not False
        or plan["submission_allowed"] is not False
    ):
        raise ValueError("the immutable candidate must have submission disabled")
    unified = plan.get("schema_version") == 3
    if unified:
        from g2_unified_plan import validate

        validate(plan)
    ids = {row["episode_id"] for row in plan["trials"]}
    counts = Counter(
        (r["seed"], r["embodiment"], "expert" in r["method"]) for r in plan["trials"]
    )
    expected = Counter(
        {
            (s, e, x): n
            for s in (0, 1)
            for e, x, n in (
                ("aloha-agilex", False, 3),
                ("ARX-X5", False, 3),
                ("ARX-X5", True, 1),
            )
        }
    )
    if (
        len(ids) != len(plan["trials"])
        or (not unified and (len(ids) != 14 or counts != expected))
        or ids != set(plan["trial_bindings"])
    ):
        raise ValueError("candidate trial roster differs")
    budget = plan["budget"]
    if not unified and (
        plan["allocations_planned"],
        plan["model_server_starts"],
        budget["gpu_count"],
        budget["bounded_execution_s"],
        budget["proposed_allocation_s"],
        budget["proposed_fee_cap_yuan"],
    ) != (1, 1, 1, 29700, 34200, 34.20):
        raise ValueError("candidate resource envelope differs")
    if budget["rate_status"] == "catalog_verified_at_observation":
        quote = read(release / "catalog-rate.json")
        if (
            budget["rate_evidence_sha256"] != declared["catalog-rate.json"]
            or quote["listed_yuan_per_gpu_hour"] != "3.60"
            or quote["hard_cap_yuan"] != "34.20"
        ):
            raise ValueError("candidate quote evidence differs")
    if (
        sum(row["hard_timeout_s"] for row in plan["trials"])
        + sum(budget["overhead"].values())
        != budget["bounded_execution_s"]
    ):
        raise ValueError("trial and overhead sum differs")
    model = read(release / "pace_download_manifest.json")
    if plan["checkpoint"] != dict(
        revision=model["revision"],
        norm_stats_sha256=model["norm_stats_sha256"],
        manifest_sha256=declared["pace_download_manifest.json"],
    ):
        raise ValueError("frozen checkpoint identity differs")
    native_pairs = {}
    for row in plan["trials"]:
        episode = row["episode_id"]
        binding = plan["trial_bindings"][episode]
        name = f"configs/{episode}.json"
        cfg = read(release / name)
        fixture = sha256(
            json.dumps(
                cfg["reset_fixture"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if (
            declared[name] != binding["config_sha256"]
            or fixture != row["reset_fixture_sha256"]
            or fixture != binding["expected_initial"]["reset_fixture_sha256"]
            or model["revision"] != row["checkpoint_revision"]
            or (
                cfg["episode_id"],
                cfg["condition"],
                cfg["embodiment"],
                cfg["scene_seed"],
                cfg["window"],
                cfg["policy_rng_seed"],
                cfg["prompt"],
                cfg["timeout_s"],
            )
            != (
                episode,
                row["condition"],
                row["embodiment"],
                row["seed"],
                row["window"],
                row["policy_rng_seed"],
                row["prompt"],
                row["hard_timeout_s"] - 50,
            )
        ):
            raise ValueError(f"trial configuration differs from roster: {episode}")
        if unified and row["embodiment"] == "aloha-agilex":
            from g2e_protocol import CONTACT_MODE_NATIVE, PROTOCOL, validate_window

            validate_window(
                cfg["window"],
                cfg["execution_protocol"],
                cfg["embodiment"],
                allow_native_h50=True,
            )
            if (
                cfg["execution_protocol"],
                cfg["contact_mode"],
                cfg.get("native_release_semantics"),
            ) != (PROTOCOL, CONTACT_MODE_NATIVE, "physical_aperture_v1"):
                raise ValueError("native safety mode differs from the frozen protocol")
            shared = {
                k: v
                for k, v in cfg.items()
                if k not in ("episode_id", "condition", "window")
            }
            previous = native_pairs.setdefault(
                row["seed"], (shared, binding["expected_initial"])
            )
            if previous != (shared, binding["expected_initial"]):
                raise ValueError("native pair differs outside the execution window")
        if "expert" in row["method"]:
            if binding["trace"] not in declared or binding["source"] not in declared:
                raise ValueError("expert evidence is unbound")
            if read(release / binding["source"])["seed"] != row["seed"]:
                raise ValueError("expert evidence belongs to a different seed")
    return plan


def authorization(release, path, job, *, starting=True):
    auth = read(path)
    expires = datetime.fromisoformat(auth["quote_valid_until"])
    rate = Decimal(str(auth["yuan_per_gpu_hour"]))
    if (
        auth.get("approved") is not True
        or auth["candidate_sha256"] != digest(release / "SHA256SUMS")
        or auth.get("job_id", job) != job
        or auth["node"] != "gnode39"
        or auth["quote_source"] != "SCNet"
        or expires.tzinfo is None
        or (starting and expires <= datetime.now(timezone.utc))
        or not re.fullmatch(r"[a-f0-9]{64}", auth["quote_sha256"])
        or not rate.is_finite()
        or not Decimal("0") < rate <= Decimal("3.60")
    ):
        raise PermissionError("allocation approval or current SCNet quote is invalid")
    if digest(path.with_suffix(".quote")) != auth["quote_sha256"]:
        raise PermissionError("SCNet quote evidence differs from approval")
    return auth


def commands(root, release, output, job, plan, graphics, timeout, auth_path):
    unified = plan.get("schema_version") == 3
    gpu = graphics["gpu_uuid"]
    if not re.fullmatch(r"GPU-[a-fA-F0-9-]{36}", gpu):
        raise ValueError("invalid graphics GPU identity")
    pace = root / "source" / f"PACE-{PACE}"
    openpi = root / "source" / f"Openpi-PACE-{OPENPI}"
    uri = f"ws://127.0.0.1:{50000 + int(job) % 14000}"
    model_dir = output / "model"
    spec = read(release / "launch-spec.json")
    model_receipt = root / spec["model_cpu_receipt"].split("<PROJECT_ROOT>/", 1)[1]
    if digest(model_receipt) != spec["model_cpu_receipt_sha256"]:
        raise ValueError("model CPU receipt differs from frozen release")
    pythonpath = ":".join(
        map(
            str,
            (
                release / "scripts",
                root / "tools/pace-smoke-deps",
                openpi / "packages/openpi-client/src",
            ),
        )
    )
    sim = graphics["sim_argv"] + [
        "env",
        f"SLURM_JOB_ID={job}",
        "SLURM_JOB_NAME=cpf-pace-g2",
        f"PYTHONPATH={pythonpath}",
        f"G2_FINAL_AUTHORIZATION={auth_path}",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONFAULTHANDLER=1",
        "MPLBACKEND=Agg",
        "python",
    ]
    model = [
        "env",
        f"PI05_ROOT={root}",
        "PYTHONDONTWRITEBYTECODE=1",
        "bash",
        str(release / "scripts/pace_g1_model_env.sh"),
        gpu,
        str(release / "scripts/g2e_model_entry.py"),
        "--port",
        uri.rsplit(":", 1)[1],
        "--require-gpu",
        "--openpi-root",
        str(openpi),
        "--download-dir",
        str(root / "cache/pace-handover"),
        "--diagnostics-dir",
        str(model_dir),
        "--verified-model",
        str(model_receipt),
    ]

    def destination(episode, probe):
        return output / ("startup" if probe else "trials") / episode

    def trial(episode, probe=False):
        cmd = sim + [
            str(release / "scripts/g2_final_trial.py"),
            "--episode",
            episode,
            "--release",
            str(release),
            "--pace",
            str(pace),
            "--uri",
            uri,
            "--output",
            str(destination(episode, probe)),
            "--model-startup",
            str(model_dir / "startup.json"),
        ]
        if probe:
            return cmd + ["--probe"]
        row = next(r for r in plan["trials"] if r["episode_id"] == episode)
        if unified and row["embodiment"] == "aloha-agilex":
            primary = f"unified-seed{row['seed']}-native-w31"
            cmd += [
                "--expected-first-exchange",
                str(destination(primary, True) / "receipt.json"),
            ]
        return cmd

    probes = {}
    for row in plan["trials"]:
        episode = row["episode_id"]
        key = row["embodiment"], row["seed"]
        if unified and row["embodiment"] == "aloha-agilex":
            if row["window"] == 4:
                continue
            if row["window"] == 50:
                key = (*key, 50)
        probes.setdefault(key, (episode, trial(episode, True)))
    runtime = dict(
        mode="allocated",
        timeout_s=timeout,
        model=model,
        probes=probes,
        admission={},
        model_timeout_s=960,
        model_phases_file=model_dir / "phases.jsonl",
        model_phase_timeouts=dict(
            container_startup=180,
            checkpoint_and_tokenizer_sha256=180,
            openpi_import=480,
            jax_device=120,
            orbax_restore_and_policy_create=600,
        ),
        ready_marker=f"checkpoint policy on {uri}",
        commands={
            row["episode_id"]: trial(row["episode_id"]) for row in plan["trials"]
        },
        read_result=lambda episode, probe=False: read(
            destination(episode, probe) / "receipt.json"
        ),
        monitor=[
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=timestamp,uuid,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "--loop=1",
        ],
    )

    if unified:
        from functools import partial
        from g2_unified_plan import validate_native_startup

        runtime["startup_cap_s"] = plan["budget"]["overhead"][
            "shared_model_and_dynamic_entry_s"
        ]
        runtime["validate_startup"] = partial(validate_native_startup, probes=probes)
    return runtime


def allocation_matches(fields):
    tres = fields.get("AllocTRES") or fields.get("TRES", "")
    return (
        fields.get("JobState") == "RUNNING"
        and fields.get("JobName") == "cpf-pace-g2"
        and fields.get("NodeList") == "gnode39"
        and fields.get("Partition") == "ksagnormal01"
        and fields.get("Requeue") == "0"
        and fields.get("NumNodes") == "1"
        and fields.get("NumCPUs") == "4"
        and fields.get("MinMemoryNode") in ("32G", "32768M")
        and bool(re.search(r"(?:^|,)gres/gpu(?::[^=,]+)?=1(?:,|$)", tres))
    )


def run_deployment_stage(command, log, timeout, *, cwd=None):
    with interrupt_signals(), log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            start_new_session=True,
        )
        try:
            if process.wait(timeout=timeout):
                raise RuntimeError(f"{log.stem} process failed")
        finally:
            reap_process_group(process)


def run(root, release, output, auth_path, plan):
    job = os.environ.get("SLURM_JOB_ID", "")
    if (
        not job.isdigit()
        or os.environ.get("SLURM_JOB_NAME") != "cpf-pace-g2"
        or job in plan.get("protected_jobs", [plan["protected_existing_job"]])
    ):
        raise PermissionError("a separately authorized G2 allocation is required")
    auth = authorization(release, auth_path, job)
    unified = plan.get("schema_version") == 3
    if unified and auth.get("job_id") != job:
        raise PermissionError("unified approval must name the allocation")
    output.mkdir(parents=True, exist_ok=False)
    persist(output / "authorization-receipt.json", auth)
    fields, start, end = read_allocation(job, output)
    allocation_s = plan["budget"]["proposed_allocation_s"]
    bound_s = plan["budget"]["bounded_execution_s"]
    min_duration = allocation_s if unified else bound_s
    if (
        not allocation_matches(fields)
        or not start <= time.time() < end
        or not min_duration <= end - start <= allocation_s
        or end - time.time() < bound_s
    ):
        raise PermissionError("running allocation differs from the approved envelope")
    persist(output / "allocation-receipt.json", dict(job_id=job, start=start, end=end))
    spec = read(release / "launch-spec.json")
    receipt = root / spec["model_cpu_receipt"].split("<PROJECT_ROOT>/", 1)[1]
    if digest(receipt) != spec["model_cpu_receipt_sha256"]:
        raise ValueError("missing or mismatched frozen model CPU receipt")
    with auth_path.with_suffix(".used").open("x") as consumed:
        consumed.write(job + "\n")
    try:
        graphics_command = [
            "bash",
            str(release / "scripts/g2_graphics.sh"),
            "--g2",
            str(release),
            str(output / "graphics"),
        ]
        run_deployment_stage(graphics_command, output / "graphics.log", 295, cwd=root)
        graphics = read(output / "graphics/sim-launch.json")
        budget = plan["budget"]["overhead"]
        timeout = end - time.time() - budget["cleanup_s"] - budget["statistics_s"]
        runtime = commands(
            root, release, output, job, plan, graphics, timeout, auth_path
        )
        persist(
            output / "effective-launch.json",
            dict(
                model=runtime["model"],
                probes=list(runtime["probes"].values()),
                trials=runtime["commands"],
            ),
        )
        summary = supervise(plan, runtime, output / "supervisor", allocated=True)
        run_deployment_stage(
            [
                "python3",
                str(release / "scripts/g2_final_statistics.py"),
                str(release / "unified-plan.json"),
                str(output),
            ],
            output / "statistics.log",
            budget["statistics_s"],
        )
        return 1 if summary["stop_reason"] else 0
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        persist(
            output / "deployment-failure.json",
            dict(detail=str(exc), formal_score_eligible=False),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--verify", action="store_true")
    for name in ("root", "output", "authorization"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    release = args.release.resolve()
    plan = verify(release)
    if args.verify:
        print(
            json.dumps(
                dict(
                    trials=len(plan["trials"]),
                    checksums="verified",
                    submission_allowed=False,
                    manifest_sha256=digest(release / "SHA256SUMS"),
                )
            )
        )
        return 0
    if not (args.root and args.output and args.authorization):
        parser.error(
            "--root, --output and --authorization are required for allocated execution"
        )
    return run(
        args.root.resolve(),
        release,
        args.output.resolve(),
        args.authorization.resolve(),
        plan,
    )


if __name__ == "__main__":
    raise SystemExit(main())
