"""Exercise the exact stdlib host supervisor used by the proposed release."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from g2e_host_supervisor import orchestrate
from test_g2_orchestrator import child


class G2EHostTests(unittest.TestCase):
    def run_case(self, scenario):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    dict(
                        formal_score_eligible=False,
                        all_trials_complete=scenario == "ok",
                    )
                )
            )
            cfg = dict(
                data_kind="synthetic",
                total_timeout_s=5,
                model_ready_timeout_s=1,
                model_server=child("import time;print('ready');time.sleep(60)"),
                ready_line="ready",
                evaluation_summary_file=str(summary),
                stages={
                    name: dict(
                        argv=child("print('fixture')"),
                        requires=[] if name == "dynamic" else ["dynamic"],
                        needs_model=name != "statistics",
                        timeout_s=1,
                    )
                    for name in ("dynamic", "aloha", "statistics")
                },
            )
            if scenario == "timeout":
                cfg["stages"]["aloha"].update(
                    argv=child("import time;time.sleep(60)"), timeout_s=0.1
                )
            if scenario == "model":
                cfg["model_server"] = child("raise RuntimeError('model failed')")
            started = time.monotonic()
            rc = orchestrate(cfg, root / "out")
            report = json.loads((root / "out/g2_acceptance.json").read_text())
            if report.get("model_server_pid"):
                with self.assertRaises(ProcessLookupError):
                    os.kill(report["model_server_pid"], 0)
            self.assertLess(time.monotonic() - started, 3)
            return rc, report

    def test_native_only_one_shared_model_and_complete_roster(self):
        rc, report = self.run_case("ok")
        self.assertEqual(rc, 0)
        self.assertTrue(report["execution_complete"])
        self.assertEqual(report["model_server_starts"], 1)
        self.assertEqual(set(report["stages"]), {"dynamic", "aloha", "statistics"})

    def test_timeout_reaps_group_but_keeps_statistics(self):
        rc, report = self.run_case("timeout")
        self.assertEqual(rc, 1)
        self.assertEqual(report["stages"]["aloha"]["exit_code"], 124)
        self.assertEqual(report["stages"]["statistics"]["status"], "passed")
        self.assertFalse(report["execution_complete"])

    def test_model_start_failure_is_not_retried(self):
        rc, report = self.run_case("model")
        self.assertEqual(rc, 1)
        self.assertEqual(report["model_server_starts"], 1)
        self.assertEqual(report["stages"]["aloha"]["status"], "skipped")
        self.assertFalse(report["execution_complete"])
