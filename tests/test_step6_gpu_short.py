"""Exercise checkpoint admission and the short campaign scoring boundary on CPU."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.step6_arx.analyze import summarize
from scripts.step6_arx.checkpoint import CheckpointUnavailable, inspect_checkpoint
from scripts.step6_arx.metrics import Stages


PLAN = dict(protocol="test", dataset_sha256="dataset", checkpoint_revision="pace")


class CheckpointTest(unittest.TestCase):
    def test_partial_training_cannot_start_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with self.assertRaisesRegex(CheckpointUnavailable, "missing Step 5 selection"):
                inspect_checkpoint(run, PLAN)
            (run / "identity.json").write_text(json.dumps(dict(
                dataset_sha256="dataset", pace_checkpoint="pace", settings={"train_steps": 400})))
            (run / "selected.json").write_text(json.dumps(dict(step=100, last_step=200)))
            with self.assertRaisesRegex(CheckpointUnavailable, "incomplete"):
                inspect_checkpoint(run, PLAN)
            (run / "selected.json").write_text(json.dumps(dict(step=100, last_step=400)))
            (run / "gpu_gates.json").write_text(json.dumps(dict(status="passed", frozen_unchanged=True)))
            for step in (100, 400):
                checkpoint = run / "checkpoints" / f"step-{step:04d}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "_METADATA").touch()
                (checkpoint / "manifest.ocdbt").touch()
                (checkpoint / "_CHECKPOINT_METADATA").write_text(
                    json.dumps(dict(commit_timestamp_nsecs=1 if step == 100 else 0)))
            with self.assertRaisesRegex(CheckpointUnavailable, "incomplete Orbax"):
                inspect_checkpoint(run, PLAN)
            (checkpoint / "_CHECKPOINT_METADATA").write_text(json.dumps(dict(commit_timestamp_nsecs=2)))
            selection, _, selected_path = inspect_checkpoint(run, PLAN)
            self.assertEqual(selection["step"], 100)
            self.assertEqual(selected_path.name, "step-0100")


class ShortStatsTest(unittest.TestCase):
    def test_accepted_short_pair_cannot_enter_formal_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "campaign.json").write_text(json.dumps(dict(
                plan=PLAN, seeds=[0], fixture=False, acceptance_only=True)))
            for variant in ("frozen", "lora"):
                trial = root / f"seed-00-{variant}"
                trial.mkdir()
                (trial / "trajectory.jsonl").touch()
                (trial / "receipt.json").write_text(json.dumps(dict(
                    schema="test", scene_seed=0, variant=variant, source="step5_model",
                    pair_match=True, outcome="ACTION_LIMIT", task_success=False,
                    rgb_status="OBSERVED", video="head.mp4", physics_ticks=20, failure=None,
                    analysis=Stages().summary())))
            result = summarize(root)
            self.assertEqual(result["pairs"][0]["excluded"], "GPU_SHORT_ACCEPTANCE")
            self.assertEqual(result["formal_groups"]["frozen"]["n"], 0)
            self.assertEqual(result["formal_groups"]["lora"]["n"], 0)
