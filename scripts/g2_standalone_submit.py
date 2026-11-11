"""Verify and submit one held standalone job from its private SCNet staging directory."""

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
import sys
import time

A = "127408633"
ARX_REFERENCE = "127510363"
ARX_AUTHORIZATION_SHA = "06de3ce0f1e8a924ecf2b2d8b4f50c49c6d8ff5a2ea9e9ac325c722725047c3b"
NODE = "gnode39"
PARTITION = "ksagnormal01"
PARENT = "93b6c0691e236d0b871745852f358a4cea85416b"
MODES = {
    "author": ("author-native-h50-seed100000", 4800, "4.80"),
    "arx": ("unified-seed1-arx-full-w4", 5400, "5.40"),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data, *, exclusive=False):
    if exclusive:
        with path.open("x") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(path)


def query(*command):
    return subprocess.check_output(command, text=True, timeout=25).strip()


def fields(raw):
    return dict(word.split("=", 1) for word in raw.split() if "=" in word)


def seconds(value):
    if value in ("UNLIMITED", "INFINITE"):
        return float("inf")
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    h, m, s = map(int, clock.split(":"))
    return int(days) * 86400 + 3600 * h + 60 * m + s


def job_record(job):
    for attempt in range(8):
        try:
            return query("scontrol", "show", "job", "-o", job)
        except subprocess.CalledProcessError:
            if attempt == 7:
                raise
            time.sleep(3)


def archived_arx_reference(root):
    receipt = root / "results" / f"pace-g2-standalone-arx-{ARX_REFERENCE}" / "authorization-receipt.json"
    require(digest(receipt) == ARX_AUTHORIZATION_SHA, "historical ARX authorization receipt differs")
    rows = query("sacct", "-nPX", "-j", ARX_REFERENCE, "-o",
                 "JobIDRaw,User,Account,QOS,Partition,NodeList,Timelimit,State,ExitCode,Submit").splitlines()
    require(len(rows) == 1, "historical ARX allocation is ambiguous or missing")
    job, user, account, qos, partition, node, limit, state, exit_code, submitted = rows[0].split("|")
    require((job, user, partition, node, seconds(limit), state, exit_code)
            == (ARX_REFERENCE, os.environ["USER"], PARTITION, NODE, MODES["arx"][1],
                "COMPLETED", "0:0") and account and qos and submitted,
            "historical ARX accounting record differs")
    return dict(Account=account, QOS=qos, JobState=state, SubmitTime=submitted)


def unified_reference(root, mode):
    parent = fields(job_record(A))
    require(
        parent["Comment"] == "cpf-g2-93b6c06-c8eeff12-20261109-single"
        and parent["TimeLimit"] == "07:00:00"
        and parent["ReqNodeList"] == NODE
        and parent["Partition"] == PARTITION,
        "A scheduler binding differs",
    )
    original = Path(parent["Command"]).parents[2] / "submission-public.json"
    a_receipt = read(original)
    require(
        a_receipt["job_id"] == A
        and a_receipt["commit"] == PARENT
        and a_receipt["archive_sha256"]
        == "c8eeff121eefb1bff98e8103cc3314ef6d95669e362889f3d68ac35507389e99"
        and len(a_receipt["roster"]) == 11
        and a_receipt["roster"][-1] == MODES["author"][0],
        "A receipt is not the frozen 11-trial candidate",
    )
    if mode == "author" and parent["JobState"] != "PENDING":
        outcome = (
            root / "results" / f"pace-g2-unified-{A}" / "trials" / MODES["author"][0] / "receipt.json"
        )
        require(not outcome.exists() or read(outcome).get("success") is False,
                "A already wrote a successful or unverified author result; review before duplicating")
    return parent


def platform(root, duration, comment, mode, *, projection=False):
    require(not projection or mode == "arx", "projection requires standalone ARX mode")
    parent = archived_arx_reference(root) if projection else unified_reference(root, mode)
    reference_job = ARX_REFERENCE if projection else A
    part = fields(query("scontrol", "show", "partition", PARTITION, "-o"))
    node = fields(query("scontrol", "show", "node", NODE, "-o"))
    require(
        part["State"] == "UP"
        and seconds(part["MaxTime"]) >= duration
        and (part["AllowAccounts"] == "ALL" or parent["Account"] in part["AllowAccounts"].split(","))
        and PARTITION in node["Partitions"].split(",")
        and re.search(r"gpu:RTX4090-PCIE-24GB-LS:[1-9]", node["Gres"])
        and int(node["CPUTot"]) >= 4
        and int(node["RealMemory"]) - int(node.get("MemSpecLimit", 0)) >= 32768
        and not any(bad in node["State"] for bad in ("DOWN", "DRAIN", "FAIL", "MAINT", "NO_RESPOND")),
        "partition or GPU node is ineligible",
    )
    user = os.environ["USER"]
    association = [
        row for row in (line.split("|") for line in query(
            "sacctmgr", "-nP", "show", "assoc", "where", "user=" + user,
            "format=Account,Partition,MaxWall,MaxJobs,MaxSubmit,GrpJobs,GrpSubmit",
        ).splitlines())
        if row[0] == parent["Account"] and row[1] in ("", PARTITION)
    ]
    require(association and all(not row[2] or seconds(row[2]) >= duration for row in association), "Slurm user association missing or time limit exceeded")
    qos = {}
    for name in (parent["QOS"], part["QoS"]):
        row = query("sacctmgr", "-nP", "show", "qos", "where", "name=" + name,
                    "format=Name,MaxWall,MaxJobsPU,MaxSubmitPU,GrpJobs,GrpSubmit").split("|")
        require(len(row) == 6 and row[0] == name and (not row[1] or seconds(row[1]) >= duration), "QoS identity or time limit differs")
        qos[name] = row[2:]
    partition_limits = qos[part["QoS"]]
    require(partition_limits[:2] == ["12", "20"], "live partition QoS concurrency limits changed")
    max_running = min(int(value) for value in [row[3] for row in association] + [row[0] for row in qos.values()] if value)
    max_submitted = min(int(value) for value in [row[4] for row in association] + [row[1] for row in qos.values()] if value)
    require(not any(row[5] or row[6] for row in association) and not any(row[2] or row[3] for row in qos.values()),
            "account-wide QoS or association limits require additional verification")
    jobs = [line.split("|") for line in query(
        "squeue", "-h", "-u", user, "-o", "%i|%T|%k"
    ).splitlines()]
    require(len(jobs) < max_submitted and not any(row[2] == comment for row in jobs), "submission quota or duplicate job")
    free = disk_usage(root / "results").free
    quota = query("quota", "-s")
    require(free >= 32 * 1024**3 and (not quota or "none" in quota.lower()), "insufficient aggregate output headroom or unverified disk quota")
    return parent, dict(
        original_job=reference_job, original_state=parent["JobState"], original_submit=parent["SubmitTime"],
        prior_active_jobs=len(jobs), max_active_submissions_per_user=max_submitted,
        max_running_per_user=max_running, available_result_bytes=free,
        disk_reserve_gib=32, partition_max_time=part["MaxTime"], node_state=node["State"],
        node_configured_gpus=node["Gres"],
    )


def verify_package(root, stage, mode, archive_sha, manifest_sha, config_sha):
    archive = stage / f"g2-standalone-{mode}.tar.gz"
    release = stage / f"g2-standalone-{mode}"
    require(digest(archive) == archive_sha, "uploaded archive digest differs")
    if not release.exists():
        subprocess.run(["tar", "-mxzf", str(archive), "-C", str(stage)], check=True, timeout=30)
    require(digest(release / "SHA256SUMS") == manifest_sha and
            digest(release / "standalone-review.json") == config_sha, "manifest or standalone configuration digest differs")
    sys.path.insert(0, str(release / "scripts"))
    from g2_standalone_deploy import bind, verify_standalone
    from g2_final_deploy import commands
    from g2_single_launch import OPENPI, PACE

    plan, selected = verify_standalone(release)
    require(selected["mode"] == mode and selected["episode"] == MODES[mode][0]
            and selected["wall_s"] == MODES[mode][1] and selected["fee_cap"] == MODES[mode][2],
            "standalone job parameters differ")
    for repo, commit in (("PACE", PACE), ("Openpi-PACE", OPENPI)):
        source = root / "source" / f"{repo}-{commit}"
        require(source.is_dir(), f"frozen {repo} source missing")
        if (source / ".git").exists():
            require(query("git", "--git-dir=" + str(source / ".git"), "rev-parse", "HEAD") == commit, f"{repo} revision differs")
    for relpath in (
        "containers/openpi-ubuntu", "research/robotwin/robotwin-cu122/opt/robotwin/logs/BUILD_COMPLETE",
        "tools/pace-model-venv/bin/python",
    ):
        require((root / relpath).exists(), f"deployed runtime missing: {relpath}")
    from pace_download import load_manifest, local_path

    model = load_manifest()
    download = root / "cache/pace-handover"
    receipt = read(root / "results/pace-preflight-02ff7dd/model/startup.json")
    require(receipt["status"] == "cpu_preflight_passed"
            and receipt["checkpoint_revision"] == model["revision"], "frozen checkpoint receipt differs")
    snapshot = {}
    for item in model["files"]:
        path = local_path(download, item["path"])
        require(path.is_file() and path.stat().st_size == item["bytes"], "checkpoint file missing or wrong size")
        snapshot[item["path"]] = [path.stat().st_size, path.stat().st_mtime_ns, item["sha256"]]
    require(snapshot == receipt["checkpoint_files"]
            and digest(download / model["norm_stats"]) == model["norm_stats_sha256"]
            and digest(root / "cache/openpi-pace/big_vision/paligemma_tokenizer.model")
            == snapshot["tokenizer.model"][2], "checkpoint snapshot, norm stats or tokenizer differs")
    runtime = commands(root, release, stage / "offline-check", "1", plan,
                       dict(gpu_uuid="GPU-00000000-0000-0000-0000-000000000001", sim_argv=["offline-only"]),
                       selected["wall_s"], stage / "approval.json")
    execution, bound = bind(plan, runtime, release, stage / "offline-check", selected, "1")
    require(list(bound["commands"]) == [selected["episode"]] and len(execution["trials"]) == 1
            and len(bound["probes"]) == (mode == "arx"), "standalone dispatch differs")
    slurm = release / "scripts/pace_g2_standalone.slurm"
    subprocess.run(["bash", "-n", str(slurm)], check=True, timeout=10)
    return release, selected, len((release / "SHA256SUMS").read_text().splitlines())


def submit(root, stage, mode, archive_sha, manifest_sha, config_sha, commit):
    require(re.fullmatch(r"[a-f0-9]{40}", commit), "missing verified release commit")
    intent = stage / "submission-intent.json"
    approval = stage / "approval.json"
    require(not intent.exists() and not approval.exists(), "a submission was already attempted or prepared")
    release, selected, member_count = verify_package(root, stage, mode, archive_sha, manifest_sha, config_sha)
    from g2_final_deploy import authorization

    episode, duration, cap = MODES[mode]
    comment = f"cpf-g2-standalone-{mode}-{archive_sha[:12]}-single"
    config = read(release / "configs" / f"{episode}.json") if mode == "arx" else {}
    parent, live = platform(root, duration, comment, mode,
                            projection=config.get("task_tcp_projection") == "bounded_task_tcp_projection_v1")
    quote_path = stage / "approval.quote"
    q = read(quote_path)
    now = datetime.now(timezone.utc)
    observed = datetime.fromisoformat(q["observed_at"])
    price = Decimal(q["displayed_yuan_per_gpu_hour"])
    cost = (price * duration / 3600).quantize(Decimal("0.01"))
    require(q["source"] == "SCNet" and q["goods_id"] == "821418588551249920"
            and q["region"] == "华东一区【昆山】" and q["account_funds_cover_all_three"] is True
            and timedelta(0) <= now - observed <= timedelta(minutes=30)
            and Decimal(0) < price <= Decimal("3.60") and price * duration <= Decimal(cap) * 3600,
            "fresh official quote, account funds or cost cap verification failed")
    resources = dict(gpu_count=1, gpu_type="RTX4090", cpu_count=4, memory_gib=32, wall_s=duration,
                     node=NODE, partition=PARTITION)
    auth = dict(approved=True, pending_job_binding=True, approved_at=now.isoformat(),
                commit=commit, archive_sha256=archive_sha, candidate_sha256=manifest_sha,
                configuration_sha256=config_sha, quote_source="SCNet", quote_sha256=digest(quote_path),
                quote_valid_until=(observed + timedelta(days=7)).isoformat(),
                yuan_per_gpu_hour=str(price), max_cost_yuan=str(cost), resources=resources,
                node=NODE, allocations_authorized=1, automatic_retry=False,
                standalone_mode=mode, allocation_s=duration)
    save(approval, auth, exclusive=True)
    authorization(release, approval, "0")
    result_path = stage / "submission-public.json"
    public = dict(mode=mode, episode=episode, resources=resources, live=live, commit=commit,
                  archive_sha256=archive_sha, manifest_sha256=manifest_sha, configuration_sha256=config_sha,
                  price_yuan_per_gpu_hour=str(price), max_cost_yuan=str(cost),
                  quote_observed_at=q["observed_at"], quote_valid_until=auth["quote_valid_until"],
                  candidate_members=member_count, submission_attempts=0, job_id=None, dependency=None)
    save(result_path, public)
    command = [
        "sbatch", "--parsable", "--hold", "--comment=" + comment, "--account=" + parent["Account"],
        "--chdir=" + str(root), f"--export=ALL,G2_FINAL_RELEASE={release},G2_FINAL_APPROVAL={approval}",
        "--output=" + str(stage / "slurm-%j.out"), str(release / "scripts/pace_g2_standalone.slurm"),
    ]
    save(intent, dict(argv=command, attempt_at=now.isoformat(), no_retry=True), exclusive=True)
    public["submission_attempts"] = 1
    save(result_path, public)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_") and k != "SLURM_CLUSTERS"}
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=45)
    save(stage / "sbatch-receipt.json", dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    require(result.returncode == 0, "Slurm rejected the sole submission")
    match = re.fullmatch(r"(\d+)(?:;[^\s]+)?\s*", result.stdout)
    require(match is not None and match[1] != A, "no unambiguous standalone Job ID")
    job = match[1]
    public.update(job_id=job, job_state="HELD", result_directory=f"results/pace-g2-standalone-{mode}-{job}")
    save(result_path, public)
    print("UNIQUE_SLURM_JOB_ID", mode, job, flush=True)
    raw = job_record(job)
    (stage / "held-job.txt").write_text(raw + "\n")
    f = fields(raw)
    expected = dict(JobId=job, JobName="cpf-pace-g2", JobState="PENDING", Reason="JobHeldUser",
                    Partition=PARTITION, ReqNodeList=NODE, NumCPUs="4", NumTasks="1",
                    Account=parent["Account"], QOS=parent["QOS"], Requeue="0",
                    Dependency="(null)", Comment=comment)
    require(all(f.get(k) == v for k, v in expected.items())
            and seconds(f["TimeLimit"]) == seconds(f["TimeMin"]) == duration
            and f["NumNodes"] in ("1", "1-1") and f["MinMemoryNode"] in ("32G", "32768M")
            and f["TresPerNode"] in ("gres:gpu:1", "gres/gpu:1"), "held Slurm record differs; job stays held")
    auth.pop("pending_job_binding")
    auth["job_id"] = job
    save(approval, auth)
    authorization(release, approval, job)
    public.update(authorization_bound_before_release=True, submit_time=f["SubmitTime"],
                  allocations=1, model_services=1, automatic_retry=False)
    save(result_path, public)
    query("scontrol", "release", job)
    f = fields(job_record(job))
    public.update(job_state=f["JobState"], reason=f["Reason"], released=True)
    save(result_path, public)
    print(json.dumps(public, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", type=Path)
    parser.add_argument("archive_sha")
    parser.add_argument("manifest_sha")
    parser.add_argument("config_sha")
    parser.add_argument("commit")
    args = parser.parse_args()
    submit(args.root.resolve(), args.stage.resolve(), args.mode,
           args.archive_sha, args.manifest_sha, args.config_sha, args.commit)
