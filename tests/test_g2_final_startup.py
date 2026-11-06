"""Exercise early-exit and stalled startup using CPU subprocesses only."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
import time
import unittest

from g2_final_campaign import process_group_gone, supervise
from g2_remaining_plan import build
from g2_final_prepare_review import archive_candidate

PROJECT = Path(__file__).resolve().parents[1]


class StartupTests(unittest.TestCase):
    def test_clean_model_container_receives_frozen_source_cache_setting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "containers/openpi-ubuntu").mkdir(parents=True)
            python = root / "tools/pace-model-venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nexit 0\n")
            python.chmod(0o755)
            container = root / "container"
            container.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            container.chmod(0o755)
            env = dict(os.environ, PI05_ROOT=str(root), APPTAINER_BIN=str(container))
            result = subprocess.run(
                [
                    "bash",
                    str(PROJECT / "scripts/pace_g1_model_env.sh"),
                    "cpu",
                    "--help",
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            self.assertIn("PYTHONDONTWRITEBYTECODE=1", result.stdout.splitlines())
            self.assertIn("JAX_PLATFORMS=cpu", result.stdout.splitlines())

    def test_archive_contains_only_manifest_members_across_platforms(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "release"
            output.mkdir()
            (output / "source.py").write_text("pass\n")
            (output / "._source.py").write_bytes(b"macOS metadata")
            (output / "SHA256SUMS").write_text(
                "unused-in-this-archive-test  source.py\n"
            )
            destination = root / "release.tar.gz"
            archive_candidate(output, destination)
            with tarfile.open(destination) as archive:
                self.assertEqual(
                    archive.getnames(), ["release/SHA256SUMS", "release/source.py"]
                )
                self.assertTrue(all(row.isfile() for row in archive.getmembers()))

    def test_early_exit_silent_hang_and_phase_hang_preserve_failure_receipts(self):
        for mode in ("exit", "silent", "phase"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                pid = root / "model.pid"
                phases = root / "phases.jsonl"
                code = (
                    "import os,time,pathlib; "
                    f"pathlib.Path({str(pid)!r}).write_text(str(os.getpid())); "
                )
                if mode == "exit":
                    code += "print('injected missing dependency',flush=True); raise SystemExit(5)"
                elif mode == "phase":
                    event = (
                        json.dumps(dict(event="start", stage="openpi_import")) + "\n"
                    )
                    code += f"pathlib.Path({str(phases)!r}).write_text({event!r}); time.sleep(20)"
                else:
                    code += "time.sleep(20)"
                runtime = dict(
                    mode="cpu_rehearsal",
                    timeout_s=5,
                    model_timeout_s=0.75,
                    model=[sys.executable, "-u", "-c", code],
                    ready_marker="READY",
                    commands={},
                    admission={},
                    read_result=lambda _: {},
                    model_phases_file=phases,
                    model_phase_timeouts={"openpi_import": 0.25},
                )
                started = time.monotonic()
                summary = supervise(
                    build(PROJECT), runtime, root / "run", rehearsal=True
                )
                self.assertLess(time.monotonic() - started, 4)
                self.assertEqual(summary["model_server_starts"], 1)
                self.assertTrue(summary["stop_reason"])
                self.assertTrue(
                    all(row["status"] == "not_run" for row in summary["rows"].values())
                )
                self.assertEqual(
                    json.loads((root / "run/campaign.json").read_text()), summary
                )
                self.assertTrue(process_group_gone(int(pid.read_text())))
                if mode == "exit":
                    self.assertIn("exited before ready", summary["stop_reason"])
                    self.assertIn(
                        "missing dependency", (root / "run/model.log").read_text()
                    )
                elif mode == "phase":
                    self.assertIn("openpi_import timed out", summary["stop_reason"])
                else:
                    self.assertIn("did not reach readiness", summary["stop_reason"])

    def test_batch_entry_binds_paths_without_running_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / "python3"
            python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            python.chmod(0o755)
            env = dict(
                os.environ,
                PATH=str(root) + os.pathsep + os.environ["PATH"],
                SLURM_SUBMIT_DIR=str(root),
                SLURM_JOB_ID="9001",
                G2_FINAL_RELEASE=str(root / "release"),
                G2_FINAL_APPROVAL=str(root / "approval.json"),
            )
            result = subprocess.run(
                ["bash", str(PROJECT / "scripts/pace_g2_final.slurm")],
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            self.assertEqual(
                result.stdout.splitlines(),
                [
                    str(root / "release/scripts/g2_final_deploy.py"),
                    str(root / "release"),
                    "--root",
                    str(root),
                    "--output",
                    str(root / "results/pace-g2-final-9001"),
                    "--authorization",
                    str(root / "approval.json"),
                ],
            )
            env.pop("G2_FINAL_RELEASE")
            result = subprocess.run(
                ["bash", str(PROJECT / "scripts/pace_g2_final.slurm")],
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("candidate path required", result.stderr)
