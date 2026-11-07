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
from types import SimpleNamespace
from unittest.mock import patch

from g2_final_campaign import process_group_gone, supervise
from g2_remaining_plan import build
from g2_final_prepare_review import archive_candidate

PROJECT = Path(__file__).resolve().parents[1]


class StartupTests(unittest.TestCase):
    def rehearse_admission(self, later_scene_s):
        plan = build(PROJECT)
        clock = SimpleNamespace(now=0.0)
        clock.monotonic = lambda: clock.now

        def advance(seconds):
            clock.now += seconds

        clock.sleep = advance
        probes = {}
        for row in plan["trials"]:
            episode = row["episode_id"]
            probes.setdefault(
                (row["embodiment"], row["seed"]), (episode, ["probe", episode])
            )
        durations = dict(
            zip(
                (episode for episode, _ in probes.values()),
                (338, later_scene_s, 320, 320),
            )
        )
        launched = []

        def spawn(command, *, stdout, **kwargs):
            stage = command[0]
            launched.append(command)
            if stage == "monitor":
                stdout.write(b"timestamp, GPU-mock, 14314, 24564\n")
            finish = float("inf")
            if stage == "probe":
                finish = clock.now + durations[command[1]]
            elif stage == "trial":
                finish = clock.now + 1
            return SimpleNamespace(
                pid=len(launched),
                returncode=0,
                poll=lambda: 0 if clock.now >= finish else None,
            )

        runtime = dict(
            mode="allocated",
            timeout_s=33600,
            model_timeout_s=960,
            ready_marker="READY",
            model=["model"],
            monitor=["monitor"],
            probes=probes,
            admission={},
            commands={
                row["episode_id"]: ["trial", row["episode_id"]]
                for row in plan["trials"]
            },
            read_result=lambda episode, probe=False: dict(
                episode_id=episode,
                category="admitted" if probe else "diagnostic_complete",
                fresh_reset_verified=True,
            ),
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("g2_final_campaign.time", clock),
            patch("g2_final_campaign.subprocess.Popen", side_effect=spawn),
            patch("g2_final_campaign.stop_group"),
            patch(
                "g2_final_campaign.wait_ready", side_effect=lambda *args: advance(320)
            ),
        ):
            summary = supervise(plan, runtime, Path(temp) / "run", allocated=True)
        return summary, launched, clock.now

    def test_slow_second_scene_admits_and_runs_all_trials_in_order(self):
        result, launched, _ = self.rehearse_admission(275)
        self.assertIsNone(result["stop_reason"])
        self.assertEqual(result["model_server_starts"], 1)
        self.assertEqual(
            [command[1] for command in launched if command[0] == "trial"],
            [row["episode_id"] for row in build(PROJECT)["trials"]],
        )
        second = result["startup"]["remaining-seed0-native-w31"]
        self.assertGreater(second["wall_s"], 145)
        self.assertGreater(second["allowance_s"], second["wall_s"])

    def test_stalled_second_scene_remains_bounded_and_keeps_failure_evidence(self):
        result, launched, elapsed = self.rehearse_admission(20000)
        self.assertIn(
            "remaining-seed0-native-w31: trial hard timeout", result["stop_reason"]
        )
        self.assertTrue(
            all(row["status"] == "not_run" for row in result["rows"].values())
        )
        self.assertFalse(any(command[0] == "trial" for command in launched))
        self.assertLess(elapsed, 6400)
        second = result["startup"]["remaining-seed0-native-w31"]
        self.assertEqual(second["status"], "failed")
        self.assertAlmostEqual(second["allowance_s"], second["wall_s"], delta=0.03)

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
