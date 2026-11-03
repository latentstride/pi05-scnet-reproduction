"""Synthetic-only regression tests for the stage 6–7 evaluation contract."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize_experiments.py"
SHA = "a" * 64
POSE = [0, 0, 0, 1, 0, 0, 0]
LOCK_FIELDS = ("episode_id", "task", "seed", "comparison_id", "pair_id", "condition", "environment")


def episode(episode_id, side="source", outcome="success"):
    embodiment = "aloha-agilex" if side != "target" else "arx-x5"
    expert = side == "expert"
    return dict(
        schema_version=1, data_kind="synthetic", run_lock_sha256=SHA,
        episode_id=episode_id, task="handover_block", seed=12,
        comparison_id=None if expert else "main", pair_id=None if expert else "p1",
        condition=dict(id=side, embodiment=embodiment,
                       action_source="expert_replay" if expert else f"{side}_policy",
                       checkpoint=None if expert else dict(id="mock", revision="mock",
                                                           manifest_sha256=SHA, norm_stats_sha256=SHA),
                       expert_representation="expert_qpos" if expert else None, execution="qpos",
                       window_actions=4, config_sha256=SHA, observation_mapping=side, action_mapping=side),
        environment=dict(runtime_id="mock-runtime", robotwin_revision="mock-rw", pace_revision="mock-pace",
                         assets_revision="mock-assets", scene_config_sha256=SHA, embodiment_config_sha256=SHA),
        initial=dict(robot_base_world=dict(left=POSE.copy(), right=POSE.copy()), task_frame_world=POSE.copy(),
                     actual_arm_qpos_rad=dict(left=[0] * 6, right=[0] * 6), actual_gripper=dict(left=0, right=0),
                     objects_task={"block": POSE.copy()},
                     reset_fixture_sha256=SHA, workspace_check_id="mock-check", common_workspace_pass=True),
        match_status="not_applicable" if expert else "matched", outcome=outcome,
        failure=None if outcome == "success" else dict(stage="ik", code="mock", detail="synthetic only",
                                                       step=1, artifact="mock/error.json"),
        metrics=dict(
            trajectory=dict(reference_id="mock-alignment", n=4, position_rmse_m=0.002, position_p95_m=0.003,
                            orientation_mean_deg=0.5, orientation_p95_deg=0.8),
            planning=dict(attempts=4, failures=1), limits=dict(checks=8, violations=1),
            gripper=dict(samples=4, state_mismatches=1, final_left_actual=0.1, final_right_actual=0.9),
            inference=None if expert else dict(calls=2, wall_s=0.5),
            control=dict(calls=5, wall_s=0.25),
            actions=dict(windows=2, requested=8, executed=5, sim_elapsed_s=0.4)),
        episode_artifact=f"mock/{episode_id}",
    )


class SummarizerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.output = self.directory / "report"

    def json_file(self, filename, payload):
        path = self.directory / filename
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def invoke(self, *inputs, kind="synthetic", lock=None):
        args = [sys.executable, str(SCRIPT), *map(str, inputs), "--data-kind", kind, "--output", str(self.output)]
        if lock is not None:
            args.extend(["--run-lock", str(lock)])
        return subprocess.run(args, capture_output=True, text=True, check=False)

    def table(self, name):
        with (self.output / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def reject(self, records, message, **options):
        result = self.invoke(self.json_file("episodes.json", records), **options)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(message, result.stderr)
        self.assertFalse(self.output.exists())

    def test_schema_is_valid(self):
        Draft202012Validator.check_schema(json.loads((ROOT / "evaluation_schema.json").read_text()))

    def test_mixed_file_formats_sources_and_metrics(self):
        source = episode("s")
        target = episode("t", "target", "ik_unreachable")
        expert = episode("e", "expert")
        target["seed"] = 99
        target["initial"]["objects_task"]["block"][3] = -1
        target["initial"]["robot_base_world"]["left"][0] = 0.2
        csv_path = self.directory / "episodes.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=target)
            writer.writeheader()
            writer.writerow({key: json.dumps(value) for key, value in target.items()})
        result = self.invoke(self.json_file("episodes.json", [source, expert]), csv_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        groups = {row["action_source"]: row for row in self.table("condition_summary")}
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups["expert_replay"]["inference_wall_s_per_calls"], "")
        stats = groups["target_policy"]
        self.assertEqual(stats["outcome_ik_unreachable"], "1")
        for metric, value in {
            "success_rate": 0, "planning_failures_per_attempts": 0.25, "limits_violations_per_checks": 0.125,
            "inference_wall_s_per_calls": 0.25, "control_wall_s_per_calls": 0.05,
            "actions_executed_per_windows": 2.5, "gripper_state_mismatches_per_samples": 0.25,
            "mean_episode_position_rmse_m": 0.002, "mean_episode_orientation_p95_deg": 0.8,
        }.items():
            self.assertAlmostEqual(float(stats[metric]), value)
        self.assertEqual(self.table("paired_summary")[0]["mean_success_delta_b_minus_a"], "-1.0")
        self.assertEqual(self.table("failure_index")[0]["episode_id"], "t")
        archive = json.loads((self.output / "episodes.json").read_text())
        self.assertEqual(archive, [source, expert, target])

    def test_seed_does_not_establish_pair(self):
        a, b = episode("s"), episode("t", "target")
        b["initial"]["objects_task"]["block"][0] = 0.005
        self.reject([a, b], "differ beyond tolerance")

    def test_same_robot_ablation_requires_matching_gripper(self):
        a, b = episode("s"), episode("s2")
        b["condition"]["id"] = "source-window-1"
        b["condition"]["window_actions"] = 1
        b["metrics"]["actions"] = None
        b["initial"]["actual_gripper"]["left"] = 0.2
        self.reject([a, b], "differ beyond tolerance")

    def test_fixed_checkpoint_and_drift(self):
        a, b = episode("s"), episode("t", "target")
        b["condition"]["checkpoint"]["revision"] = "changed"
        self.reject([a, b], "matching policy weights")
        b = copy.deepcopy(a)
        b.update(episode_id="s2", pair_id="p2")
        b["condition"]["window_actions"] = 2
        b["metrics"]["actions"] = None
        self.reject([a, b], "condition or environment drift")

    def test_reject_invalid_mixed_or_duplicate_records(self):
        a = episode("s")
        self.reject([a, a], "duplicate episode")
        bad = episode("e", "expert")
        bad["condition"]["checkpoint"] = a["condition"]["checkpoint"]
        self.reject([bad], "checkpoint")
        self.reject([a], "expected data_kind=diagnostic", kind="diagnostic")
        a["metrics"]["limits"]["violations"] = 9
        self.reject([a], "limits.violations exceeds checks")

    def test_incomplete_and_unverified_pairs(self):
        a = episode("s")
        b = episode("t", "target", "environment_error")
        b.update(initial=None, match_status="unverified")
        b["failure"]["stage"] = "reset"
        c = episode("s2")
        c["pair_id"] = "p2"
        result = self.invoke(self.json_file("episodes.json", [a, b, c]))
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = self.table("paired_summary")[0]
        self.assertEqual((summary["n_missing"], summary["n_not_admitted"], summary["n_matched"]), ("1", "1", "0"))
        self.assertEqual(summary["mean_success_delta_b_minus_a"], "")
        self.assertEqual(self.table("failure_index")[0]["outcome"], "environment_error")

    def test_formal_mode_enforces_locked_roster(self):
        a, b = episode("s"), episode("t", "target", "ik_failed")
        roster = [{key: row[key] for key in LOCK_FIELDS} for row in (a, b)]
        lock = self.json_file("run_lock.json", dict(schema_version=1, episodes=roster))
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        for row in (a, b):
            row.update(data_kind="formal", run_lock_sha256=digest)
        self.reject([a], "missing 1 locked episodes", kind="formal", lock=lock)
        a["seed"] += 1
        self.reject([a, b], "does not match the locked roster", kind="formal", lock=lock)
        a["seed"] -= 1
        result = self.invoke(self.json_file("episodes.json", [a, b]), kind="formal", lock=lock)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
