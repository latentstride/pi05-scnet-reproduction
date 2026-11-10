"""Verify, hold, bind and release the seven independent jobs exactly once."""

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from shutil import disk_usage
import subprocess

from g2_standalone_matrix import (
    CONFIG,
    MIN_DISK_GIB,
    bound_authorization,
    verify_package,
)
from g2_parallel_spec import RESOURCES, digest, extract, specs
from g2_parallel_identity import runtime_preflight
from g2_standalone_submit import fields, query, read, require, save, seconds

GOODS = "821418588551249920"
PROJECT_BASE = "a5a67cfb27a787da5562f82e03c3e8d67c257ec3"
MAX_RATE = Decimal("3.60")
MIN_FREE_GIB = 7 * MIN_DISK_GIB + 4
JOB_NAME = "cpf-pace-g2"


def budget(stage, batch_path):
    q = read(stage / "platform-quote.json")
    permit = read(stage / "batch-approval.json")
    observed = datetime.fromisoformat(q["observed_at"])
    rate = Decimal(q["displayed_yuan_per_gpu_hour"])
    cap = Decimal(permit["approved_total_cap_yuan"])
    ceiling = rate * Decimal(7 * RESOURCES["wall_s"]) / 3600
    proof = stage / "platform-quote-evidence"
    require(
        q["source"] == "SCNet"
        and q["goods_id"] == GOODS
        and q["region"] == "华东一区【昆山】"
        and q["account_funds_cover_batch"] is True
        and observed.tzinfo is not None
        and timedelta(0)
        <= datetime.now(timezone.utc) - observed
        <= timedelta(minutes=30)
        and rate.is_finite()
        and cap.is_finite()
        and 0 < rate <= MAX_RATE
        and ceiling <= cap <= Decimal("50.40")
        and Decimal(str(q["account_available_yuan"])) >= ceiling
        and permit.get("submission_scheme") == "standalone-seven-v1"
        and permit.get("approved") is True
        and permit.get("allocations_authorized") == 7
        and permit.get("automatic_retry") is False
        and permit["batch_sha256"] == digest(batch_path)
        and permit["quote_sha256"] == digest(stage / "platform-quote.json")
        and q["evidence_sha256"] == digest(proof),
        "fresh official quote, aggregate approval or available funds missing",
    )
    return q, permit, ceiling, observed


def packages(stage):
    batch_path = stage / "batch.json"
    batch = read(batch_path)
    require(
        batch.get("submission_allowed") is False
        and batch.get("baseline") == PROJECT_BASE
        and [r["id"] for r in batch["rows"]] == [s["id"] for s in specs()],
        "batch roster differs",
    )
    results = []
    for row, spec in zip(batch["rows"], specs()):
        key = row["id"]
        name = f"g2-parallel-{key}"
        require(row["name"] == name, "archive basename differs")
        archive_path = stage / f"{name}.tar.gz"
        release = stage / name
        require(digest(archive_path) == row["archive_sha256"], f"{key} archive differs")
        if not release.exists():
            extract(archive_path, stage, name)
        _, selected = verify_package(release)
        loaded = selected["config"]
        require(
            loaded == spec
            and digest(release / CONFIG) == row["config_sha256"]
            and digest(release / "SHA256SUMS") == row["manifest_sha256"]
            and len((release / "SHA256SUMS").read_text().splitlines())
            == row["members"],
            f"{key} extracted identity differs",
        )
        rcpt = release / "parallel-cpu-receipt.json"
        cpu = read(rcpt) if rcpt.exists() else {}
        cpu_ok = (
            cpu.get("admission") == "PASS"
            and cpu.get("id") == key
            and cpu.get("config_sha256") == row["config_sha256"]
            and digest(rcpt) == row["cpu_sha256"]
        )
        auth_dir = stage / "authorizations" / key
        auth_dir.mkdir(parents=True, exist_ok=True)
        results.append(
            dict(
                spec=spec,
                row=row,
                release=release,
                cpu_ok=cpu_ok,
                approval=auth_dir / "approval.json",
            )
        )
        subprocess.run(
            ["bash", "-n", str(release / "scripts/pace_g2_parallel.slurm")],
            check=True,
            timeout=10,
        )
    return results


def platform(root, account, qos):
    r = RESOURCES
    part = fields(query("scontrol", "show", "partition", r["partition"], "-o"))
    node = fields(query("scontrol", "show", "node", r["node"], "-o"))
    require(
        part["State"] == "UP"
        and seconds(part["MaxTime"]) >= r["wall_s"]
        and (
            part["AllowAccounts"] == "ALL"
            or account in part["AllowAccounts"].split(",")
        )
        and (part["AllowQos"] == "ALL" or qos in part["AllowQos"].split(","))
        and account not in part.get("DenyAccounts", "").split(",")
        and qos not in part.get("DenyQos", "").split(",")
        and r["partition"] in node["Partitions"].split(",")
        and re.search(r"gpu:" + re.escape(r["gpu"]) + r":[1-9]", node["Gres"])
        and int(node["CPUTot"]) >= r["cpus"]
        and int(node["RealMemory"]) >= r["memory_mib"]
        and not any(
            bad in node["State"]
            for bad in ("DOWN", "DRAIN", "FAIL", "MAINT", "NO_RESPOND")
        ),
        "partition or GPU node not eligible",
    )
    user = os.environ["USER"]
    assoc = [
        x
        for x in (
            row.split("|")
            for row in query(
                "sacctmgr",
                "-nP",
                "show",
                "assoc",
                "where",
                "user=" + user,
                "format=Account,Partition,MaxWall,MaxJobs,MaxSubmit,GrpJobs,GrpSubmit",
            ).splitlines()
        )
        if x[0] == account and x[1] in ("", r["partition"])
    ]
    require(
        assoc and all(not x[2] or seconds(x[2]) >= r["wall_s"] for x in assoc),
        "no eligible Slurm association",
    )
    qrows = []
    for name in {qos, part["QoS"]} - {"", "N/A", "(null)"}:
        x = query(
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            "where",
            "name=" + name,
            "format=Name,MaxWall,MaxJobsPU,MaxSubmitPU,GrpJobs,GrpSubmit",
        ).split("|")
        require(
            len(x) == 6 and x[0] == name and (not x[1] or seconds(x[1]) >= r["wall_s"]),
            "QoS limits missing",
        )
        qrows.append(x[2:])
    submit = [int(x) for x in [a[4] for a in assoc] + [q[1] for q in qrows] if x]
    run = [int(x) for x in [a[3] for a in assoc] + [q[0] for q in qrows] if x]
    active = query("squeue", "-h", "-u", user, "-o", "%i").splitlines()
    require(
        submit
        and run
        and len(active) + 7 <= min(submit)
        and not any(a[5] or a[6] for a in assoc)
        and not any(q[2] or q[3] for q in qrows),
        "seven submission slots or account-wide limits unverified",
    )
    free = disk_usage(root / "results").free
    quota = query("quota", "-s")
    require(
        free >= MIN_FREE_GIB * 1024**3 and (not quota or "none" in quota.lower()),
        "aggregate disk reserve or account disk quota unverified",
    )
    return dict(
        active=len(active),
        max_submitted=min(submit),
        max_running=min(run),
        available_bytes=free,
        node_state=node["State"],
        gres=node["Gres"],
    )


def command(root, entry, account, qos, comment, *, test):
    release, approval = entry["release"], entry["approval"]
    require(
        not any(c in str(root) + str(release) + str(approval) for c in ",\n\r"),
        "staging paths cannot contain Slurm export delimiters",
    )
    return [
        "sbatch",
        "--test-only" if test else "--hold",
        "--parsable",
        "--comment=" + comment,
        "--account=" + account,
        "--qos=" + qos,
        "--chdir=" + str(root),
        f"--export=ALL,G2_PROJECT_ROOT={root},G2_FINAL_RELEASE={release},G2_FINAL_APPROVAL={approval}",
        "--output=" + str(approval.parent / "slurm-%j.out"),
        str(release / "scripts/pace_g2_parallel.slurm"),
    ]


class Scheduler:
    def __init__(self, root):
        self.root = root

    def call(self, *args):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("SBATCH_")
            and k not in ("SLURM_CLUSTERS", "SLURM_DEPENDENCY")
        }
        result = subprocess.run(
            args,
            env=env,
            cwd=self.root,
            check=True,
            text=True,
            capture_output=True,
            timeout=40,
        )
        return (
            result.stdout + result.stderr if "--test-only" in args else result.stdout
        ).strip()

    def record(self, job):
        return fields(self.call("scontrol", "show", "job", "-o", job))

    def lookup(self, comment):
        rows = self.call(
            "squeue", "-h", "-u", os.environ["USER"], "-o", "%i|%k"
        ).splitlines()
        return [
            job
            for job, sep, tag in (row.partition("|") for row in rows)
            if sep and tag == comment and re.fullmatch(r"[1-9]\d*", job)
        ]


def verify_held(f, job, account, qos, comment, root, release):
    r = RESOURCES
    expected = dict(
        JobId=job,
        JobName=JOB_NAME,
        JobState="PENDING",
        Reason="JobHeldUser",
        Partition=r["partition"],
        ReqNodeList=r["node"],
        NumCPUs="4",
        NumTasks="1",
        Account=account,
        QOS=qos,
        Requeue="0",
        Dependency="(null)",
        Comment=comment,
        WorkDir=str(root),
        Command=str(release / "scripts/pace_g2_parallel.slurm"),
    )
    require(
        all(f.get(k) == v for k, v in expected.items())
        and seconds(f["TimeLimit"]) == seconds(f["TimeMin"]) == r["wall_s"]
        and f["NumNodes"] in ("1", "1-1")
        and f["MinMemoryNode"] in ("32G", "32768M")
        and f["TresPerNode"] in ("gres:gpu:1", "gres/gpu:1"),
        "held Slurm record differs from its approved scope",
    )


def bind(entry, job, comment, batch_sha, quote, quote_path, observed):
    row, spec, approval = entry["row"], entry["spec"], entry["approval"]
    with approval.with_suffix(".quote").open("xb") as f:
        f.write(quote_path.read_bytes())
    save(
        approval,
        dict(
            approved=True,
            job_id=job,
            experiment=spec["id"],
            episode_id=spec["episode_id"],
            candidate_sha256=row["manifest_sha256"],
            archive_sha256=row["archive_sha256"],
            configuration_sha256=row["config_sha256"],
            batch_sha256=batch_sha,
            standalone_mode="matrix",
            allocation_s=spec["resources"]["wall_s"],
            node=RESOURCES["node"],
            resources=RESOURCES,
            quote_source="SCNet",
            quote_sha256=digest(quote_path),
            quote_valid_until=(observed + timedelta(days=7)).isoformat(),
            yuan_per_gpu_hour=quote["displayed_yuan_per_gpu_hour"],
            comment=comment,
            automatic_retry=False,
            allocations_authorized=1,
        ),
        exclusive=True,
    )
    bound_authorization(entry["release"], approval, job, spec)


def submit_all(
    scheduler, entries, root, account, qos, batch_sha, receipt_path, binding
):
    require(not receipt_path.exists(), "submission receipt already exists")
    rows = [
        dict(
            experiment=e["spec"]["id"],
            job_id=None,
            state="not_submitted",
            comment=f"cpf-g2-s7-{batch_sha[:20]}-{e['spec']['id']}",
            archive_sha256=e["row"]["archive_sha256"],
            config_sha256=e["row"]["config_sha256"],
        )
        for e in entries
    ]
    receipt = dict(batch_sha256=batch_sha, retries=0, rows=rows)

    def store():
        save(receipt_path, receipt)

    store()
    try:
        for row in rows:
            require(not scheduler.lookup(row["comment"]), "duplicate scheduler comment")
        for entry, row in zip(entries, rows):
            row["state"] = "submission_attempted"
            store()  # durable before submitting; missing reply is not permission to retry
            raw = scheduler.call(
                *command(root, entry, account, qos, row["comment"], test=False)
            )
            match = re.fullmatch(r"([1-9]\d*)(?:;[^\s]+)?", raw)
            require(match is not None, "scheduler did not return an unambiguous job ID")
            row["job_id"] = match[1]
            store()
            f = scheduler.record(row["job_id"])
            verify_held(
                f, row["job_id"], account, qos, row["comment"], root, entry["release"]
            )
            row.update(state="held", submit_time=f.get("SubmitTime"))
            store()
        require(len({row["job_id"] for row in rows}) == len(rows), "duplicate job ID")
        for entry, row in zip(entries, rows):
            binding(entry, row["job_id"], row["comment"])
            row["state"] = "bound"
            store()
        # Slurm has no transaction for release; every job is already separately approved.
        for row in rows:
            scheduler.call("scontrol", "release", row["job_id"])
            row["state"] = "released"
            store()
        for entry, row in zip(entries, rows):
            f = scheduler.record(row["job_id"])
            row.update(
                queue_state=f.get("JobState"),
                queue_reason=f.get("Reason"),
                result_dir=entry["spec"]["output_dir"] + "/" + row["job_id"],
            )
            store()
        return receipt
    except (Exception, KeyboardInterrupt) as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        for row in rows:
            if row["state"] in ("not_submitted", "released"):
                continue
            try:
                matches = (
                    [row["job_id"]]
                    if row["job_id"]
                    else scheduler.lookup(row["comment"])
                )
                require(
                    len(matches) == 1,
                    "ambiguous scheduler visibility; manual reconciliation",
                )
                job = matches[0]
                f = scheduler.record(job)
                require(
                    f.get("Comment") == row["comment"] and f.get("JobId") == job,
                    "refusing to touch an unrelated job",
                )
                row["job_id"] = job
                if f.get("JobState") == "PENDING" and f.get("Reason") == "JobHeldUser":
                    scheduler.call("scancel", job)
                    row["state"] = "rollback_cancel_requested"
                else:
                    row.update(state="left_untouched", queue_state=f.get("JobState"))
            except Exception as cleanup_error:
                row["reconciliation_required"] = str(cleanup_error)
            store()
        raise


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--qos", required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--scheduler-test", action="store_true")
    mode.add_argument("--execute", action="store_true")
    a = p.parse_args()
    require(
        all(re.fullmatch(r"[\w.-]+", value) for value in (a.account, a.qos)),
        "invalid account or QoS",
    )
    root, stage = a.root.resolve(), a.stage.resolve()
    batch_path = stage / "batch.json"
    batch_sha = digest(batch_path)
    entries = packages(stage)
    scheduler = Scheduler(root)
    tests = []
    if a.execute:
        quote, _, total, observed = budget(stage, batch_path)
        require(
            all(e["cpu_ok"] for e in entries),
            "at least one core CPU receipt is missing",
        )
        live = platform(root, a.account, a.qos)
        runtime_preflight(root, entries[0]["release"])
        save(stage / "platform-preflight.json", dict(live, max_cost_yuan=str(total)))
    for entry in entries:
        key = entry["spec"]["id"]
        cmd = command(
            root,
            entry,
            a.account,
            a.qos,
            f"cpf-g2-s7-{batch_sha[:20]}-{key}",
            test=a.scheduler_test or a.execute,
        )
        row = dict(experiment=key, cpu_ok=entry["cpu_ok"], argv=cmd)
        if a.scheduler_test or a.execute:
            try:
                row["result"] = scheduler.call(*cmd)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                row["error"] = str(error)
                row["stderr"] = str(error.stderr)
        tests.append(row)
    print(json.dumps(tests, indent=2), flush=True)
    if a.scheduler_test or a.execute:
        save(stage / "scheduler-test.json", tests)
        require(not any("error" in row for row in tests), "scheduler dry-run failed")
    if not a.execute:
        return
    intent_dir = root / "results/pace-g2-parallel/.batch-intents"
    intent_dir.mkdir(parents=True, exist_ok=True)
    require(not any(e["approval"].exists() for e in entries), "stale per-job approval")
    # Stable across staging locations and repackaging of the same experiment roster.
    roster_sha = sha256(
        "\n".join(e["row"]["config_sha256"] for e in entries).encode()
    ).hexdigest()
    legacy_intent = intent_dir / f"{roster_sha}.json"
    if legacy_intent.exists():
        historical_batch = (
            "9ae1fb32d441b23d80311c71c9594ec946829f1a8d8ab8dca9adb193999fcc7d"
        )
        require(
            read(legacy_intent)["batch"] == historical_batch
            and batch_sha != historical_batch,
            "an earlier intent for this roster cannot be superseded",
        )
        old_jobs = {
            "127556961",
            "127556962",
            "127556963",
            "127556966",
            "127556967",
            "127556975",
            "127556977",
        }
        old_rows = [
            line.split("|")
            for line in query(
                "sacct",
                "-X",
                "-nP",
                "-j",
                ",".join(sorted(old_jobs)),
                "-o",
                "JobIDRaw,State",
            ).splitlines()
        ]
        require(
            {row[0] for row in old_rows if row[0] in old_jobs and row[1] == "FAILED"}
            == old_jobs,
            "historical roster is not fully confirmed failed",
        )
    require(
        not (intent_dir / f"{roster_sha}-startup-v2.json").exists(),
        "an earlier replacement batch already has a durable intent",
    )
    save(
        intent_dir / f"{roster_sha}-standalone-v1.json",
        dict(batch=batch_sha, roster_sha256=roster_sha, stage=str(stage)),
        exclusive=True,
    )
    receipt = submit_all(
        scheduler,
        entries,
        root,
        a.account,
        a.qos,
        batch_sha,
        stage / "submission.json",
        lambda e, job, tag: bind(
            e,
            job,
            tag,
            batch_sha,
            quote,
            stage / "platform-quote.json",
            observed,
        ),
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
