import copy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from g2_fixtures import ROOT, fixture
from g2_episode import run_episode, save_json
from g2_export import episode_record
from test_g2_integration import MockRobot, FixedPolicy, spec

SHA = sha256(b"synthetic_evaluation_fixture").hexdigest()


def identity(config, embodiment="arx-x5", contrast=None):
    return dict(run_lock_sha256=SHA, episode_id=config["episode_id"], task="handover_block", seed=0,
                comparison_id=contrast, pair_id="synthetic_pair0" if contrast else None,
                match_status="matched" if contrast else "not_applicable",
                trajectory_reference_id="synthetic_tracking",
                condition=dict(id=config["condition"], embodiment=embodiment,
                               action_source="source_policy" if embodiment == "aloha-agilex" else "target_policy",
                               checkpoint=dict(id="synthetic_policy", revision="synthetic", manifest_sha256=SHA,
                                               norm_stats_sha256=SHA),
                               expert_representation=None, execution="qpos", window_actions=config["window"],
                               config_sha256=SHA, observation_mapping=embodiment, action_mapping=embodiment),
                environment=dict(runtime_id="synthetic_runtime", robotwin_revision="synthetic",
                                 pace_revision="synthetic", assets_revision="synthetic",
                                 scene_config_sha256=SHA, embodiment_config_sha256=SHA))


class ExportTests(unittest.TestCase):
    def test_real_episode_evidence_through_existing_summarizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for condition, embodiment in (("Aloha-w1", "aloha-agilex"), ("ARX-w1", "arx-x5")):
                adapter, qa, qr, bases = fixture()
                config = spec(1, condition)
                config["episode_id"] = condition
                path = root / condition
                run_episode(config, MockRobot(adapter, qa, qr, bases), FixedPolicy(qa), path,
                            adapter=adapter if embodiment == "arx-x5" else None)
                row = episode_record(path, identity(config, embodiment, "synthetic_transfer"))
                rows.append(row)
                self.assertEqual(row["metrics"]["actions"]["executed"], 4)
                self.assertEqual(row["metrics"]["control"]["calls"], 4)
            save_json(root / "records.json", rows)
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/summarize_experiments.py"),
                                   str(root / "records.json"), "--data-kind", "synthetic",
                                   "--output", str(root / "report")], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            text = (root / "report/paired_summary.csv").read_text()
            self.assertIn("synthetic_transfer", text)
            self.assertIn(",0.0", text)
            self.assertTrue((root / "report/condition_summary.csv").exists())

    def test_timeout_and_infrastructure_failure_are_preserved(self):
        for fault in ("reset", "model", "planning", "timeout", "task", "interrupt"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                adapter, qa, qr, bases = fixture()
                config = spec(1, "ARX-w1")
                if fault == "planning":
                    adapter.target_collision = lambda q, g: False
                if fault == "timeout":
                    config["timeout_s"] = .06
                path = Path(temp) / "episode"
                result = run_episode(config, MockRobot(adapter, qa, qr, bases, fault),
                                     FixedPolicy(qa, fault), path, adapter=adapter)
                row = episode_record(path, identity(config))
                self.assertIsNotNone(row["failure"])
                self.assertEqual(row["failure"]["code"], result["failure"]["code"])
                if fault == "timeout":
                    self.assertEqual(row["failure"]["code"], "timeout")
                    self.assertIn(row["outcome"], ("control_error", "adapter_error"))
                if fault == "reset":
                    self.assertIsNone(row["initial"])
                    self.assertEqual(row["outcome"], "environment_error")

    def test_unknown_metrics_remain_null_and_metadata_cannot_relabel(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter, qa, qr, bases = fixture()
            config = spec(1, "ARX-w1")
            path = Path(temp) / "episode"
            run_episode(config, MockRobot(adapter, qa, qr, bases), FixedPolicy(qa), path, adapter=adapter)
            meta = identity(config)
            row = episode_record(path, meta)
            self.assertIsNone(row["metrics"]["gripper"])
            self.assertIsNone(row["metrics"]["limits"])
            meta["condition"]["window_actions"] = 4
            with self.assertRaisesRegex(ValueError, "window"):
                episode_record(path, meta)
            meta = identity(config)
            meta["condition"]["id"] = "fake"
            with self.assertRaisesRegex(ValueError, "condition"):
                episode_record(path, meta)

    def test_no_promoting_synthetic_to_formal(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter, qa, qr, bases = fixture()
            config = spec(1, "ARX-w1")
            path = Path(temp) / "episode"
            run_episode(config, MockRobot(adapter, qa, qr, bases), FixedPolicy(qa), path, adapter=adapter)
            result = json.loads((path / "result.json").read_text())
            result["data_kind"] = "formal"
            save_json(path / "result.json", result)
            with self.assertRaisesRegex(ValueError, "formal export"):
                episode_record(path, identity(config))
