"""CPU-only fault injection for six independent Slurm jobs and process groups."""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import subprocess
import sys
import time

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from g2_parallel_deploy import run_trial  # noqa: E402
from g2_parallel_spec import BATCH_SCHEME, RESOURCES, digest, specs  # noqa: E402
from g2_parallel_submit import budget, command, submit_all  # noqa: E402
from g2_standalone_submit import read, save  # noqa: E402


class FakeSlurm:
    def __init__(self, *, fail_submit=None, fail_release=None):
        self.held = {}
        self.active = set()
        self.cancelled = []
        self.bindings = set()
        self.count = 0
        self.commands = {}
        self.fail_submit = fail_submit
        self.fail_release = fail_release

    def call(self, *args):
        if args[0] == "sbatch":
            assert "--hold" in args
            assert not any("dependency" in a or "array" in a for a in args)
            self.count += 1
            job = str(1000 + self.count)
            self.commands[job] = args
            self.held[job] = next(
                a.split("=", 1)[1] for a in args if a.startswith("--comment=")
            )
            if self.count == self.fail_submit:
                raise subprocess.TimeoutExpired(args, 40)
            return job
        if args[:2] == ("scontrol", "release"):
            assert len(self.bindings) == 6
            if args[2] == self.fail_release:
                raise RuntimeError("simulated release failure")
            self.active.add(args[2])
            return ""
        assert args[0] == "scancel" and args[1] not in self.active
        self.cancelled.append(args[1])
        self.held.pop(args[1])
        return ""

    def lookup(self, comment):
        return [job for job, tag in self.held.items() if tag == comment]

    def record(self, job):
        return dict(
            JobId=job,
            JobName="cpf-pace-g2",
            JobState="PENDING",
            Reason="Resources" if job in self.active else "JobHeldUser",
            Partition=RESOURCES["partition"],
            ReqNodeList=RESOURCES["node"],
            NumCPUs="4",
            NumTasks="1",
            NumNodes="1",
            Account="test",
            QOS="qos",
            Requeue="0",
            Dependency="(null)",
            Comment=self.held[job],
            TimeLimit="04:00:00",
            TimeMin="04:00:00",
            MinMemoryNode="32G",
            TresPerNode="gres/gpu:1",
            WorkDir=next(
                a.split("=", 1)[1]
                for a in self.commands[job]
                if a.startswith("--chdir=")
            ),
            Command=self.commands[job][-1],
        )


def entries(tmp_path):
    return [
        dict(
            spec=spec,
            row=dict(archive_sha256=str(i), config_sha256=str(i)),
            release=tmp_path / spec["id"],
            approval=tmp_path / spec["id"] / "approval.json",
        )
        for i, spec in enumerate(specs())
    ]


def batch(tmp_path, scheduler, *, fail_bind=False):
    def bind(entry, job, comment):
        assert scheduler.held[job] == comment and not scheduler.active
        if fail_bind and entry["spec"]["id"] == "G4":
            raise RuntimeError("authorization invalid")
        scheduler.bindings.add(job)

    return submit_all(
        scheduler,
        entries(tmp_path),
        tmp_path,
        "test",
        "qos",
        "0" * 64,
        tmp_path / "receipt.json",
        bind,
    )


def test_isolated_roster_and_hold_before_release(tmp_path):
    frozen = specs()
    assert (
        len({s["episode_id"] for s in frozen})
        == len({s["output_dir"] for s in frozen})
        == 6
    )
    assert all(
        b["port_base"] - a["port_base"] == 2000 for a, b in zip(frozen, frozen[1:])
    )
    for i in range(3):
        shared = (
            "scene_seed",
            "scene_cache_sha256",
            "checkpoint_revision",
            "model_rng_seed",
        )
        assert all(frozen[i][k] == frozen[i + 3][k] for k in shared)
    scheduler = FakeSlurm()
    receipt = batch(tmp_path, scheduler)
    assert len(scheduler.active) == 6
    assert {x["state"] for x in receipt["rows"]} == {"released"}
    assert len({r["result_dir"] for r in receipt["rows"]}) == 6
    assert "--test-only" in command(
        tmp_path, entries(tmp_path)[0], "test", "qos", "x", test=True
    )


def test_lost_submit_response_reconciles_by_exact_comment(tmp_path):
    scheduler = FakeSlurm(fail_submit=4)
    with pytest.raises(subprocess.TimeoutExpired):
        batch(tmp_path, scheduler)
    assert scheduler.count == 4
    assert not scheduler.active and scheduler.cancelled == [
        "1001",
        "1002",
        "1003",
        "1004",
    ]
    assert read(tmp_path / "receipt.json")["rows"][3]["job_id"] == "1004"


def test_auth_failure_releases_none(tmp_path):
    scheduler = FakeSlurm()
    with pytest.raises(RuntimeError, match="authorization invalid"):
        batch(tmp_path, scheduler, fail_bind=True)
    assert not scheduler.active
    assert len(scheduler.cancelled) == 6


def test_release_failure_leaves_already_released_untouched(tmp_path):
    scheduler = FakeSlurm(fail_release="1003")
    with pytest.raises(RuntimeError, match="release failure"):
        batch(tmp_path, scheduler)
    assert scheduler.active == {"1001", "1002"}
    assert scheduler.cancelled == ["1003", "1004", "1005", "1006"]


def test_duplicate_comment_neither_submits_nor_cancels(tmp_path):
    scheduler = FakeSlurm()
    scheduler.held["777"] = f"cpf-g2-s6h4-{'0' * 20}-G4"
    with pytest.raises(RuntimeError, match="duplicate"):
        batch(tmp_path, scheduler)
    assert scheduler.count == 0 and scheduler.cancelled == []


def test_process_failure_and_timeout_do_not_stop_other_trial(tmp_path):
    model = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        for idx, (script, wait, expected) in enumerate(
            (
                ("raise SystemExit(7)", 2, (None, 7)),
                (
                    "import time; time.sleep(20)",
                    0.15,
                    ("host_trial_hard_timeout", None),
                ),
                ("pass", 2, (None, 0)),
            )
        ):
            out = tmp_path / str(idx)
            out.mkdir()
            proof = {}
            assert (
                run_trial(
                    [sys.executable, "-c", script],
                    model,
                    out,
                    out,
                    time.monotonic() + wait,
                    proof,
                )
                == expected
            )
            assert proof == {"trial_group_reaped": True} and model.poll() is None
    finally:
        model.terminate()
        model.wait(timeout=2)


def test_model_death_reaps_its_own_trial(tmp_path):
    model = subprocess.Popen([sys.executable, "-c", "pass"])
    model.wait(timeout=2)
    proof = {}
    with pytest.raises(RuntimeError, match="model service exited"):
        run_trial(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            model,
            tmp_path,
            tmp_path,
            time.monotonic() + 2,
            proof,
        )
    assert proof == {"trial_group_reaped": True}


def test_new_cost_scope_requires_fresh_six_job_approval(tmp_path):
    path = tmp_path / "batch.json"
    save(path, {"submission_allowed": False})
    proof = tmp_path / "platform-quote-evidence"
    proof.write_text("simulated quote")
    quote = tmp_path / "platform-quote.json"
    save(
        quote,
        dict(
            source="SCNet",
            goods_id="821418588551249920",
            region="华东一区【昆山】",
            account_funds_cover_batch=True,
            account_available_yuan="86.40",
            observed_at=datetime.now(timezone.utc).isoformat(),
            displayed_yuan_per_gpu_hour="3.60",
            evidence_sha256=digest(proof),
        ),
    )
    permit = tmp_path / "batch-approval.json"
    approval = dict(
        approved=True,
        automatic_retry=False,
        approved_total_cap_yuan="86.40",
        allocations_authorized=6,
        submission_scheme=BATCH_SCHEME,
        batch_sha256=digest(path),
        quote_sha256=digest(quote),
    )
    save(permit, approval)
    assert budget(tmp_path, path)[2] == Decimal("86.40")
    save(
        permit,
        dict(
            approval,
            allocations_authorized=7,
            submission_scheme="standalone-seven-v1",
            approved_total_cap_yuan="50.40",
        ),
    )
    with pytest.raises(RuntimeError, match="aggregate approval"):
        budget(tmp_path, path)
    save(permit, approval)
    save(quote, dict(read(quote), account_available_yuan="86.39"))
    save(permit, dict(approval, quote_sha256=digest(quote)))
    with pytest.raises(RuntimeError, match="available funds"):
        budget(tmp_path, path)
