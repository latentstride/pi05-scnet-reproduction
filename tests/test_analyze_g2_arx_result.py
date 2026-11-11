"""Exercise archived projection events and resource censoring without a model."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analyze_g2_arx_result import analyze, analyze_events, jsonl


class ArchiveAnalysisTests(unittest.TestCase):
    def test_actual_cpu_projection_events_link_only_fresh_requests(self):
        data = json.loads(Path("research/g2/arx-projection-online/cpu-exchanges.json").read_bytes())
        for query, expected_approvals in (("35", 2), ("36", 1)):
            with self.subTest(archive=query):
                result = analyze_events(enumerate(data[query]["selected_events"], 1))
                first, second = result["queries"]
                projection, = result["projection_plans"]
                self.assertEqual(first["stamp"], projection["stamp"])
                self.assertEqual(projection["executed_actions"], 4)
                self.assertEqual(projection["discarded_rows"], 46)
                self.assertEqual(projection["fresh_request"], second["query"])
                self.assertTrue(second["projection_reobservation"]["clock_advanced"])
                self.assertTrue(second["projection_reobservation"]["arms_logged"])
                self.assertIn("block", second["projection_reobservation"]["objects_logged"])
                self.assertEqual(result["counts"]["approved_plan"], expected_approvals)

    def test_hard_censor_preserves_partial_events_and_missing_contact(self):
        with TemporaryDirectory() as temporary:
            run = Path(temporary)
            episode = "unified-seed1-arx-full-w4"
            trial = run / "trials" / episode / "episode"
            trial.mkdir(parents=True)
            request = {"kind": "policy_request", "stamp": {"episode": episode, "tick": 0, "sim_s": 0.0}}
            (trial / "events.jsonl").write_text(json.dumps(request) + '\n{"kind":')
            (trial / "result.json").write_text(json.dumps({"outcome": "running", "success": False, "queries": 0}))
            (run / "standalone-result.json").write_text(json.dumps({
                "selected": {"episode": episode}, "outcome": "RESOURCE_CENSORED",
                "row": {"status": "infrastructure_failure", "reason": "trial hard timeout"}}))
            result = analyze(run)
            self.assertEqual(result["termination"]["outcome"], "RESOURCE_CENSORED")
            self.assertIsNone(result["termination"]["task_success"])
            self.assertEqual(result["counters"]["logged_model_requests"], 1)
            self.assertEqual(result["counters"]["result_queries"], 0)
            self.assertEqual(result["activity"]["partial_final_line"], 2)
            self.assertIsNone(result["collision_audits"]["target"])

    def test_nonterminal_json_corruption_is_not_silently_ignored(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.jsonl"
            path.write_text('{}\n{"kind":\n{}\n')
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                list(jsonl(path))


if __name__ == "__main__":
    unittest.main()
