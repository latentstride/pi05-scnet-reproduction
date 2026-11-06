import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from g2_fixtures import ROOT
from g2_orchestrator import STAGES, orchestrate, validate_plan


def child(code):
    return [sys.executable, "-u", "-c", code]


def plan(root):
    pid = root / "model.pid"
    server = child(f"import os,time;from pathlib import Path;Path({str(pid)!r}).write_text(str(os.getpid())); print('ready');time.sleep(60)")
    def consume():
        return child(f"import os;from pathlib import Path;os.kill(int(Path({str(pid)!r}).read_text()),0)")
    return dict(data_kind="synthetic", total_timeout_s=8, model_ready_timeout_s=2,
                model_server=server, ready_line="ready",
                stages={name: dict(argv=child("print('CPU independent fixture')") if name in ("dynamic", "statistics") else consume(),
                                   requires=[] if name in ("dynamic", "aloha", "statistics") else ["dynamic"],
                                   timeout_s=3, needs_model=name not in ("dynamic", "statistics")) for name in STAGES})


class OrchestratorTests(unittest.TestCase):
    def run_case(self, scenario):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = plan(root)
            if scenario == "dynamic":
                config["stages"]["dynamic"]["argv"] = child("raise RuntimeError('injected dynamics error')")
            if scenario == "model":
                config["model_server"] = child("raise RuntimeError('injected checkpoint error')")
            if scenario in ("timeout", "deadline"):
                config["stages"]["arx"]["argv"] = child("import time;time.sleep(60)")
                if scenario == "timeout":
                    config["stages"]["arx"]["timeout_s"] = .15
                else:
                    config["total_timeout_s"] = .6
            rc = orchestrate(config, root / "out")
            report = json.loads((root / "out/g2_acceptance.json").read_text())
            if (root / "model.pid").exists():
                with self.assertRaises(ProcessLookupError):
                    os.kill(int((root / "model.pid").read_text()), 0)
            return rc, report

    def test_one_model_and_all_stages(self):
        rc, r = self.run_case("ok")
        self.assertEqual(rc, 0)
        self.assertTrue(r["complete"])
        self.assertEqual(r["model_server_starts"], 1)
        self.assertTrue(all(s["status"] == "passed" for s in r["stages"].values()))

    def test_dynamic_failure_allows_aloha_and_statistics_but_skips_arx(self):
        rc, r = self.run_case("dynamic")
        self.assertEqual(rc, 1)
        self.assertEqual(r["stages"]["aloha"]["status"], "passed")
        self.assertEqual(r["stages"]["statistics"]["status"], "passed")
        for name in ("arx", "ablation", "expert"):
            self.assertEqual(r["stages"][name]["status"], "skipped")
            self.assertIn("dynamic", r["stages"][name]["reason"])

    def test_model_failure_never_restarts(self):
        rc, r = self.run_case("model")
        self.assertEqual(rc, 1)
        self.assertEqual(r["model_server_starts"], 1)
        self.assertIn("model_start_error", r)
        self.assertEqual(r["stages"]["statistics"]["status"], "passed")

    def test_stage_and_shared_deadlines(self):
        for scenario in ("timeout", "deadline"):
            with self.subTest(scenario=scenario):
                start = time.monotonic()
                rc, r = self.run_case(scenario)
                self.assertEqual(rc, 1)
                self.assertEqual(r["stages"]["arx"]["exit_code"], 124)
                self.assertLess(time.monotonic() - start, 3)

    def test_no_resource_allocation_or_formal_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            config = plan(Path(temp))
            config["model_server"] = ["sbatch", "bad"]
            with self.assertRaisesRegex(ValueError, "allocate"):
                validate_plan(config)
            config = plan(Path(temp))
            config["data_kind"] = "formal"
            with self.assertRaisesRegex(ValueError, "formal"):
                validate_plan(config)

    def test_sigterm_writes_final_report_and_reaps_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = plan(root)
            config["stages"]["arx"]["argv"] = child("import time;time.sleep(60)")
            config["stages"]["arx"]["timeout_s"] = 30
            path = root / "plan.json"
            path.write_text(json.dumps(config))
            proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/g2_orchestrator.py"), str(path), str(root / "out")])
            try:
                end = time.monotonic() + 5
                while time.monotonic() < end:
                    report = root / "out/g2_acceptance.json"
                    if report.exists() and json.loads(report.read_text())["stages"]["arx"]["status"] == "running":
                        break
                    time.sleep(.03)
                else:
                    self.fail("ARX fixture did not start")
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=4)
                r = json.loads(report.read_text())
                self.assertTrue(r["interrupted"])
                self.assertEqual(r["stages"]["arx"]["exit_code"], 130)
                with self.assertRaises(ProcessLookupError):
                    os.kill(r["model_server_pid"], 0)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
