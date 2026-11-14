import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from g2_parallel_spec import digest  # noqa: E402
from g2_parallel_startup import PROBE_STAGES, validate_probe  # noqa: E402
from g2_standalone_startup import read_parallel_allocation  # noqa: E402
from g2_parallel_status import monitor  # noqa: E402


def test_allocation_recovers_from_slow_controller(tmp_path):
    attempts = []

    def query(argv, *, timeout, **_):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(
            argv,
            0,
            "JobId=17 JobState=RUNNING "
            "StartTime=2026-11-11T00:00:00 EndTime=2026-11-11T02:00:00",
        )

    record, start, end = read_parallel_allocation(
        "17", tmp_path, query=query, pause=lambda _: None
    )
    assert record["JobId"] == "17" and end - start == 7200
    assert attempts == [15, 20]


def test_allocation_fails_closed(tmp_path):
    attempts = []

    def no_reply(argv, *, timeout, **_):
        attempts.append(timeout)
        raise subprocess.TimeoutExpired(argv, timeout)

    with pytest.raises(TimeoutError, match="three attempts"):
        read_parallel_allocation("17", tmp_path, query=no_reply, pause=lambda _: None)
    assert attempts == [15, 20, 25]
    with pytest.raises(PermissionError, match="different JobId"):
        read_parallel_allocation(
            "17",
            tmp_path,
            query=lambda *a, **k: subprocess.CompletedProcess(a, 0, "JobId=18"),
        )


def test_probe_requires_finished_real_allocation_and_exact_batch(tmp_path):
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps(dict(rows=[dict(id="G1", manifest_sha256="abc")])))
    path = tmp_path / "results/pace-g2-startup-probes/17/result.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            dict(
                job_id="17",
                node="gnode39",
                outcome="STARTUP_PROBE_PASSED",
                batch_sha256=digest(batch),
                candidate_manifests={"G1": "abc"},
                completed_startup_stages=PROBE_STAGES,
                cleanup={
                    "probe_graphics_group_reaped": True,
                    "probe_model_group_reaped": True,
                    "probe_sim_group_reaped": True,
                },
            )
        )
    )
    records = "17|COMPLETED|0:0|gnode39|cpu=4,gres/gpu=1\n17.batch|COMPLETED|0:0|gnode39|cpu=4"
    assert validate_probe(tmp_path, "17", batch, lambda *_: records) == (
        path,
        digest(path),
    )
    with pytest.raises(PermissionError, match="Slurm"):
        validate_probe(
            tmp_path, "17", batch, lambda *_: records.replace("COMPLETED", "FAILED")
        )
    batch.write_text("{}")
    with pytest.raises(PermissionError, match="exact batch"):
        validate_probe(tmp_path, "17", batch, lambda *_: records)


def test_shell_logs_before_python(tmp_path):
    # Empty environment intentionally omits SLURM_SUBMIT_DIR and all authorization.
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/pace_g2_parallel.slurm")],
        env={"PATH": "/bin:/usr/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "stage=slurm_shell" in result.stdout
    assert "outcome=INFRASTRUCTURE_FAILURE exit=1" in result.stderr


def test_bootstrap_reports_import_failure(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("import g2_missing_dependency_for_test\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/g2_parallel_bootstrap.py"), str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "python_bootstrap" in result.stdout
    assert (
        "ModuleNotFoundError" in result.stderr
        and "INFRASTRUCTURE_FAILURE" in result.stderr
    )


def test_monitor_stops_at_first_terminal_report():
    reports = []
    assert (
        monitor(
            ["17", "18"],
            query=lambda _: {"17": {"state": "FAILED"}, "18": {"state": "COMPLETED"}},
            emit=reports.append,
            pause=lambda _: pytest.fail("must not poll again"),
            queries=5,
        )
        == 0
    )
    assert len(reports) == 1


def test_monitor_reports_consecutive_connection_failures():
    reports = []

    def fail(_):
        raise ConnectionError("scheduler offline")

    assert (
        monitor(
            ["17"], query=fail, emit=reports.append, pause=lambda _: None, queries=5
        )
        == 2
    )
    assert len(reports) == 2 and "scheduler offline" in reports[-1]


def test_shell_uses_execution_directory_not_submit_directory(tmp_path):
    project = tmp_path / "project"
    runner = tmp_path / "staging"
    scripts = runner / "scripts"
    scripts.mkdir(parents=True)
    project.mkdir()
    approval = runner / "approval.json"
    approval.write_text("{}")
    (scripts / "g2_standalone_deploy.py").touch()
    (scripts / "g2_parallel_bootstrap.py").write_text(
        "import os,sys; assert sys.argv[sys.argv.index('--root')+1] == os.getcwd(); print('ROOT_OK')"
    )
    env = dict(
        os.environ,
        SLURM_JOB_ID="17",
        SLURM_SUBMIT_DIR=str(runner),
        G2_PROJECT_ROOT=str(project),
        G2_FINAL_RELEASE=str(runner),
        G2_FINAL_APPROVAL=str(approval),
    )
    cmd = ["bash", str(ROOT / "scripts/pace_g2_parallel.slurm")]
    success = subprocess.run(cmd, cwd=project, env=env, capture_output=True, text=True)
    assert success.returncode == 0 and "ROOT_OK" in success.stdout
    failed = subprocess.run(cmd, cwd=runner, env=env, capture_output=True, text=True)
    assert failed.returncode == 1 and "working directory differs" in failed.stderr


def test_original_graphics_exit_reproduced_and_fixed(tmp_path):
    import tarfile

    project = tmp_path / "code_cpf/pi05"
    runner = project / "staging"
    runner.mkdir(parents=True)
    marker = (
        project / "research/robotwin/robotwin-cu122/opt/robotwin/logs/BUILD_COMPLETE"
    )
    marker.parent.mkdir(parents=True)
    marker.touch()
    # Prevent execution of any device command. The fixed script must reach its
    # first visibility phase, while the archived script exits at the cwd check.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "timeout"
    stub.write_text("#!/bin/sh\nexit 1\n")
    stub.chmod(0o755)
    package = ROOT / "research/g2/parallel-batch/packages/g2-parallel-G1.tar.gz"
    with tarfile.open(package) as source:
        historical = source.extractfile("g2-parallel-G1/scripts/g2_graphics.sh").read()
    graphics = tmp_path / "graphics.sh"
    env = dict(
        os.environ,
        HOME=str(tmp_path),
        SLURM_SUBMIT_DIR=str(runner),
        SLURM_JOB_ID="17",
        G2_PROJECT_ROOT=str(project),
        G2_STANDALONE_CACHE=str(tmp_path / "cache"),
        PATH=str(bin_dir) + ":/bin:/usr/bin",
    )
    for contents, expected in (
        (historical, False),
        (
            historical.replace(
                b'cd "${SLURM_SUBMIT_DIR:?Submit from personal project root}"',
                b'cd "${G2_PROJECT_ROOT:?Explicit parallel project root required}"',
            ),
            True,
        ),
    ):
        graphics.write_bytes(contents)
        out = tmp_path / str(expected)
        cmd = ["bash", "-x", str(graphics), "--g2", str(project), str(out)]
        result = subprocess.run(
            cmd, cwd=project, env=env, capture_output=True, text=True
        )
        assert result.returncode == 1
        assert ("stage nvidia_visibility" in result.stderr) is expected
        assert (out / "gpu_visibility.txt").exists() is expected
        assert (out / "FAILED").exists() is expected
