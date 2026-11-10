"""Exercise the seven standalone package entry points and existing supervisor."""

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from g2_final_campaign import supervise  # noqa: E402
from g2_final_deploy import commands  # noqa: E402
from g2_parallel_package import package  # noqa: E402
from g2_parallel_spec import digest, extract  # noqa: E402
from g2_standalone_deploy import bind, output_for  # noqa: E402
from g2_standalone_matrix import verify_package  # noqa: E402


@pytest.fixture(scope="module")
def releases(tmp_path_factory):
    path = tmp_path_factory.mktemp("matrix").resolve()
    parent = ROOT / "research/g2/standalone-review/g2-standalone-author.tar.gz"
    evidence = ROOT / "research/g2/parallel-batch"
    result = package(ROOT, parent, evidence, path)
    for row in result["rows"]:
        extract(path / (row["name"] + ".tar.gz"), path, row["name"])
    return path, result


def runtime_for(root, release, job):
    plan, selected = verify_package(release)
    out = output_for(root, selected, job)
    out.mkdir(parents=True)
    spec = json.loads((release / "launch-spec.json").read_text())
    with patch("g2_final_deploy.digest", return_value=spec["model_cpu_receipt_sha256"]):
        runtime = commands(
            root,
            release,
            out,
            job,
            plan,
            dict(
                gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
                sim_argv=["cpu-stub"],
            ),
            selected["wall_s"],
            root / "approval.json",
        )
    return (*bind(plan, runtime, release, out, selected, job), out)


def test_all_packages_use_historical_standalone_chain(releases, tmp_path):
    path, manifest = releases
    old_path = tmp_path.resolve()
    extract(
        ROOT / "research/g2/standalone-review/g2-standalone-author.tar.gz",
        old_path,
        "g2-standalone-author",
    )
    old = old_path / "g2-standalone-author" / "scripts"
    ports, directories = set(), set()
    for i, row in enumerate(manifest["rows"]):
        release = path / row["name"]
        scripts = release / "scripts"
        shell = (scripts / "pace_g2_parallel.slurm").read_text()
        assert (
            "g2_standalone_deploy.py" in shell and "g2_parallel_deploy.py" not in shell
        )
        assert not (scripts / "g2_parallel_deploy.py").exists()
        assert "G2_STARTUP_PROBE_BATCH" not in shell and "--dependency" not in shell
        graphics = (scripts / "g2_graphics.sh").read_text()
        assert graphics == (old / "g2_graphics.sh").read_text().replace(
            'cd "${SLURM_SUBMIT_DIR:?Submit from personal project root}"',
            'cd "${G2_PROJECT_ROOT:?Explicit parallel project root required}"',
        )
        for name in (
            "g2_author_native_h50_service.py",
            "pace_g1_model_env.sh",
            "robotwin_gpu_smoke.py",
            "robotwin_graphics_driver.py",
        ):
            assert digest(scripts / name) == digest(old / name)
        result = subprocess.check_output(
            [
                sys.executable,
                "-S",
                "-B",
                str(scripts / "g2_standalone_deploy.py"),
                str(release),
                "--verify",
            ],
            text=True,
        )
        assert (
            json.loads(result.splitlines()[-1])["manifest_sha256"]
            == row["manifest_sha256"]
        )
        plan, runtime, out = runtime_for(tmp_path, release, str(i + 1))
        assert (
            len(plan["trials"]) == len(runtime["commands"]) == 1
            and not runtime["probes"]
        )
        assert runtime["standalone_trial"]["budget"].hard_s == 4500
        assert runtime["standalone_trial"]["budget"].soft_s == 4200
        assert plan["budget"]["proposed_allocation_s"] == 7200
        ports.add(runtime["model"][runtime["model"].index("--port") + 1])
        directories.add(out)
    assert len(ports) == len(directories) == 7


def test_real_supervisor_runs_one_matrix_trial(releases, tmp_path):
    path, manifest = releases
    release = path / manifest["rows"][0]["name"]
    plan, runtime, out = runtime_for(tmp_path, release, "17")
    (episode,) = runtime["commands"]
    runtime.update(
        model=[
            sys.executable,
            "-u",
            "-c",
            "import time; print('READY'); time.sleep(50)",
        ],
        ready_marker="READY",
        model_timeout_s=5,
        model_phases_file=None,
        monitor=[
            sys.executable,
            "-u",
            "-c",
            "import time; print('t,g,1,24000'); time.sleep(50)",
        ],
    )
    # Keep the shared supervisor and its real subprocess handling. The child
    # verifies that it received the supervisor's deadline, then writes a receipt.
    fixture = """import json,sys
from pathlib import Path
from g2_author_h50_limits import AuthorBudget,TrialClock
target, episode, *_ = sys.argv[1:]
start = float(sys.argv[sys.argv.index('--trial-start')+1])
hard = float(sys.argv[sys.argv.index('--hard-deadline')+1])
clock = TrialClock(AuthorBudget(soft_s=4200,hard_s=4500),start,hard)
Path(target).write_text(json.dumps(dict(mode='author_native',method='fixed_h50',scene_closed=True,
episode_id=episode,trial_clock=clock.receipt(),outcome='SUCCESS',completed_actions=1,success=True)))
"""
    target = out / "trials" / episode / "receipt.json"
    target.parent.mkdir()
    runtime["commands"][episode] = [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0,"
        + repr(str(ROOT / "scripts"))
        + ");\n"
        + fixture,
        str(target),
        episode,
    ]
    result = supervise(plan, runtime, out / "supervisor", allocated=True)
    assert result["model_server_starts"] == 1
    assert result["stop_reason"] is None
    assert result["rows"][episode]["status"] == "completed"
    assert result["rows"][episode]["process_group_reaped"] is True
