"""Exercise G1 supervision with local child processes; no Slurm or GPU."""
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pace_g1_orchestrate import orchestrate, wait_ready
from pace_g1_plan import build


class G1IntegrationTest(unittest.TestCase):
    def test_readiness_handles_invalid_utf8_and_split_markers(self):
        with TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_bytes(b"native log: \xff\nrea")
            def finish():
                time.sleep(0.1)
                with log.open("ab") as stream:
                    stream.write(b"dy\n")
            writer = threading.Thread(target=finish)
            writer.start()
            try:
                wait_ready(SimpleNamespace(poll=lambda: None), log, "ready", 2)
            finally:
                writer.join()

    def test_fixed_launch_plan_uses_one_model_service_and_distinct_runtimes(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            release = out / "release"
            release.mkdir()
            (release / "RELEASE_COMMIT").write_text("fixture")
            with patch.dict(os.environ, PI05_G1_DEADLINE_EPOCH="3000"):
                plan = build(out, release, out / "preflight", out, "42", "GPU-fixture", ["sim-container"])
            self.assertEqual(plan["g1_1"][0], "sim-container")
            self.assertEqual(plan["g1_2_query"][0], "sim-container")
            self.assertEqual(plan["g1_3"][0], "sim-container")
            self.assertIn(str(release / "pace_g1_model_env.sh"), plan["model_server"])
            uri = plan["ready_line"].split()[-1]
            self.assertIn(uri, plan["g1_2_query"])
            self.assertIn(uri, plan["g1_3"])
            self.assertIn(str(out / "g1_1" / "rollout"), plan["g1_2_query"])
            self.assertIn(str(out / "g1_1" / "rollout"), plan["g1_3"])
            self.assertEqual(plan["deadline_epoch"], 3000)
            self.assertEqual(plan["timeouts"]["model_ready"], 30 * 60)
            self.assertEqual(plan["model_phase_timeouts"]["container_startup"], 180)
            self.assertEqual(plan["model_phase_timeouts"]["orbax_restore_and_policy_create"], 600)

    def run_fixture(self, scenario):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            observation = output / "observation"
            pid = output / "model-pid"

            def child(source):
                return [sys.executable, "-u", "-c", source]

            def stage1():
                if scenario == "scene":
                    return child("raise RuntimeError('injected setup_demo failure')")
                return child("from pathlib import Path; Path({!r}).write_text('fake observation')"
                             .format(str(observation)))

            def server():
                if scenario == "model":
                    return child("raise RuntimeError('injected checkpoint error')")
                return child("import os,time; from pathlib import Path; "
                             "Path({!r}).write_text(str(os.getpid())); print('ready'); time.sleep(60)"
                             .format(str(pid)))

            def consumer(extra):
                return child("import os,time; from pathlib import Path; "
                             "assert Path({!r}).read_text()=='fake observation'; "
                             "os.kill(int(Path({!r}).read_text()),0); {}"
                             .format(str(observation), str(pid), extra))

            query = "print('real-observation query fixture')"
            if scenario == "bad_action":
                query = "raise ValueError('expected finite (50, 14)')"
            if scenario == "timeout":
                query = "time.sleep(30)"
            if scenario == "deadline":
                query = "time.sleep(30)"
            if scenario == "restore_timeout":
                def server():
                    return child("import json,time; from pathlib import Path; "
                                 "Path({!r}).write_text(json.dumps({{'stage': "
                                 "'orbax_restore_and_policy_create', 'event': 'start'}})+'\\n'); "
                                 "time.sleep(30)".format(str(output / "model-phases.jsonl")))
            plan = {"job_id": "offline", "release_commit": "fixture", "ready_line": "ready",
                    "g1_1": stage1(), "model_server": server(), "g1_2_query": consumer(query),
                    "g1_3": consumer("print('same model process is still alive')"),
                    "timeouts": {"G1.1": 2, "model_ready": 2, "G1.2": 0.1 if scenario == "timeout" else 2,
                                 "G1.3": 2}}
            if scenario == "deadline":
                plan["deadline_epoch"] = time.time() + 0.8
            if scenario == "restore_timeout":
                plan.update(model_phases_file=str(output / "model-phases.jsonl"),
                            model_phase_timeouts={"container_startup": 1,
                                                  "orbax_restore_and_policy_create": 0.1})
            if scenario == "memory":
                plan.update(monitor=child("import time; print('now,GPU-fixture,100,24000',flush=True); "
                                          "time.sleep(60)"),
                            memory_probe=child("print('23900,24000')"), memory_margin_mib=1024)
            rc = orchestrate(plan, output)
            result = json.loads((output / "g1_acceptance.json").read_text())
            if pid.is_file():
                model_pid = int(pid.read_text())
                self.assertEqual(result["model_server_pid"], model_pid)
                with self.assertRaises(ProcessLookupError):
                    os.kill(model_pid, 0)
            logs = {path.name: path.read_text() for path in output.glob("*.log")}
            return rc, result, logs

    def test_one_model_child_spans_three_ordered_stages(self):
        rc, report, logs = self.run_fixture("ok")
        self.assertEqual(rc, 0)
        self.assertEqual(report["model_server_starts"], 1)
        for stage in report["stages"].values():
            self.assertEqual(stage["status"], "passed")
            self.assertEqual(stage["exit_code"], 0)
        first, second, third = report["stages"].values()
        self.assertLessEqual(first["ended_at"], second["started_at"])
        self.assertLessEqual(second["ended_at"], third["started_at"])
        self.assertIn("same model process", logs["G1.3.log"])

    def test_failed_prerequisites_skip_dependents_and_keep_logs(self):
        for scenario, stage, log, detail in (
            ("scene", "G1.1", "G1.1.log", "setup_demo failure"),
            ("model", "G1.2", "model-server.log", "checkpoint error"),
            ("bad_action", "G1.2", "G1.2-query.log", "finite (50, 14)"),
            ("timeout", "G1.2", "G1.2-exception.log", "timed out"),
        ):
            with self.subTest(scenario=scenario):
                rc, report, logs = self.run_fixture(scenario)
                self.assertEqual(rc, 1)
                self.assertEqual(report["stages"][stage]["status"], "failed")
                self.assertEqual(report["stages"]["G1.3"]["status"], "skipped")
                self.assertIn(detail, logs[log])
                if scenario == "scene":
                    self.assertEqual(report["model_server_starts"], 0)
                if scenario == "timeout":
                    self.assertEqual(report["stages"][stage]["exit_code"], 124)

    def test_insufficient_headroom_prevents_checkpoint_control(self):
        rc, report, logs = self.run_fixture("memory")
        self.assertEqual(rc, 1)
        self.assertEqual(report["stages"]["G1.2"]["status"], "passed")
        self.assertEqual(report["stages"]["G1.3"]["exit_code"], 78)
        self.assertNotIn("G1.3.log", logs)
        self.assertIn("memory headroom", logs["G1.3-exception.log"])

    def test_allocation_deadline_caps_query_and_preserves_evidence(self):
        rc, report, logs = self.run_fixture("deadline")
        self.assertEqual(rc, 1)
        self.assertEqual(report["stages"]["G1.2"]["exit_code"], 124)
        self.assertEqual(report["stages"]["G1.3"]["status"], "skipped")
        self.assertLess(report["effective_timeouts_s"]["G1.2"], 0.8)
        self.assertIn("timed out", logs["G1.2-exception.log"])

    def test_restore_timeout_names_the_phase_and_skips_control(self):
        rc, report, logs = self.run_fixture("restore_timeout")
        self.assertEqual(rc, 1)
        self.assertEqual(report["stages"]["G1.2"]["exit_code"], 124)
        self.assertEqual(report["stages"]["G1.3"]["status"], "skipped")
        self.assertIn("orbax_restore_and_policy_create timed out", logs["G1.2-exception.log"])


if __name__ == "__main__":
    unittest.main()
