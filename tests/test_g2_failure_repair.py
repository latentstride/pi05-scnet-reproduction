from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from g2_locked_score import collect
from g2_locked_task import save_terminal_record
from g2_single_statistics import assessment

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research/g2/failure-audit-127054052-r1"


class FailureRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = [
            json.loads(p.read_text())
            for p in sorted((EVIDENCE / "selected").glob("*/formal-record.json"))
        ]
        if len(cls.raw) != 8:
            raise RuntimeError(f"eight actual G2 records required at {EVIDENCE}")
        cls.supervisor = json.loads(
            (EVIDENCE / "selected/supervisor/g2_acceptance.json").read_text()
        )

    def test_original_formal_labels_contradict_actual_diagnostic_supervisor(self):
        with self.assertRaisesRegex(ValueError, "contradicts"):
            assessment(self.raw, self.supervisor)

    def test_all_stages_complete_does_not_count_as_any_task_success(self):
        rows = [dict(row, data_kind="diagnostic") for row in self.raw]
        report = assessment(rows, self.supervisor)
        self.assertTrue(self.supervisor["complete"])
        self.assertTrue(report["trial_records_complete"])
        self.assertEqual(report["trial_count"], 8)
        self.assertEqual(report["task_success_count"], 0)
        self.assertFalse(report["all_tasks_successful"])
        self.assertFalse(report["formal_score_eligible"])
        self.assertEqual(
            report["outcomes"],
            dict(task_unsuccessful=4, planning_failed=3, adapter_error=1),
        )

    def test_diagnostic_configuration_does_not_promote_terminal_record(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            row = save_terminal_record(
                output, {"formal_release_ready": False}, self.raw[0]
            )
            self.assertEqual(row["data_kind"], "diagnostic")
            self.assertTrue((output / "diagnostic-record.json").is_file())
            self.assertFalse((output / "formal-record.json").exists())
            self.assertEqual(self.raw[0]["data_kind"], "formal")

    def test_diagnostic_collection_preserves_locked_roster_and_missing_detection(self):
        lock = EVIDENCE / "release-context/run_lock.json"
        digest = sha256(lock.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for row in self.raw:
                out = root / row["episode_id"]
                out.mkdir()
                (out / "diagnostic-record.json").write_text(
                    json.dumps(dict(row, data_kind="diagnostic"))
                )
            rows = collect(
                lock, digest, root, root / "rows.json", data_kind="diagnostic"
            )
            self.assertEqual(len(rows), 8)
            (root / self.raw[0]["episode_id"] / "diagnostic-record.json").unlink()
            with self.assertRaisesRegex(ValueError, "no terminal diagnostic record"):
                collect(
                    lock, digest, root, root / "missing.json", data_kind="diagnostic"
                )
            self.assertFalse((root / "missing.json").exists())

    def test_static_cpu_replay_reproduces_all_four_real_failures(self):
        path = ROOT / "research/g2/offline-repair-20261106-r1/static-replay.json"
        report = json.loads(path.read_text())
        self.assertEqual(len(report), 4)
        for case in report.values():
            self.assertFalse(case["reconstructed"]["accepted"])
            self.assertEqual(case["reconstructed"]["status"], "collision")
            self.assertEqual(case["scene_physics_steps"], 0)
            self.assertFalse(case["gpu_used"])
            self.assertTrue(
                all(not x["accepted"] for x in case["recorded_candidate_checks"])
            )
            self.assertTrue(
                all(x["result"]["accepted"] for x in case["expert_task_space_controls"])
            )
        self.assertEqual(report["seed1-B-arx-w1"]["world"], "source_collision")

    def test_repeated_future_closure_is_outside_recorded_executed_prefix(self):
        actions = json.loads((EVIDENCE / "action-audit.json").read_text())
        for name in ("seed0-A-native-w1", "seed1-A-native-w1", "seed1-A-native-w4"):
            rows = actions[name]
            self.assertTrue(
                all(min(p[6] for p in r["executed_prefix"]) > 0.98 for r in rows)
            )
            self.assertGreater(
                sum(r["first_close_index"][0] is not None for r in rows),
                len(rows) * 0.9,
            )
            self.assertTrue(all(r["finite"] for r in rows))

    def test_task_object_capture_uses_actual_actor_pose_in_task_frame(self):
        from types import SimpleNamespace as NS
        import numpy as np
        from g2_capture import capture_task_objects

        actor = NS(get_pose=lambda: NS(p=[1.0, 2.0, 3.0], q=[1.0, 0.0, 0.0, 0.0]))
        frame = np.eye(4)
        frame[:3, 3] = [0.5, 1.0, 1.5]
        objects = capture_task_objects(NS(box=actor), frame)
        np.testing.assert_allclose(objects["block"][:3, 3], [0.5, 1.0, 1.5])
        self.assertNotIn("target", objects)
        self.assertIsNone(capture_task_objects(NS(), frame))

    def test_inventory_reports_missing_events_and_retains_input_bytes(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from g2_failure_inventory import audit

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            trial = raw / "trials/case"
            trial.mkdir(parents=True)
            marker = raw / "marker"
            marker.write_bytes(b"original evidence")
            before = marker.read_bytes()
            with redirect_stdout(StringIO()):
                audit(raw, root / "analysis")
            report = json.loads((root / "analysis/trial-audit.json").read_text())
            self.assertIn("events.jsonl", report["case"]["missing"])
            self.assertIn("result.json", report["case"]["missing"])
            self.assertEqual(marker.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "outside"):
                audit(raw, raw / "forbidden-analysis")

    def test_legacy_statistics_refuses_writing_into_original_evidence(self):
        from g2_single_statistics import summarize

        trials = EVIDENCE / "selected"
        with self.assertRaisesRegex(ValueError, "outside the original run"):
            summarize(
                EVIDENCE / "release-context/run_lock.json",
                "unused",
                trials,
                trials / "forbidden-new-summary",
                EVIDENCE / "expert-replay.json",
                trials / "supervisor/g2_acceptance.json",
                True,
            )
        self.assertFalse((trials / "forbidden-new-summary").exists())
