"""Check native replay evidence and failure isolation without launching PhysX."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import subprocess
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from g2_native_expert_batch import REVISION, batch, read, write
from g2_verify_native_expert_100 import dense_contacts, final_predicate, verify_episode


class TestNativeExpertEvidence(unittest.TestCase):
    def test_original_success_limits_and_right_request(self):
        tray = [0, 0, 0, 1, 0, 0, 0]
        block = [0, 0, .105, 1, 0, 0, 0]
        self.assertTrue(final_predicate(block, tray, .81)[0])
        self.assertFalse(final_predicate(block, tray, .8)[0])
        self.assertFalse(final_predicate([.03, *block[1:]], tray, .81)[0])
        self.assertFalse(final_predicate([0, 0, .116, *block[3:]], tray, .81)[0])

    def test_contact_phases_require_shared_before_right_takeover(self):
        trace = {
            "contacts": np.array([[12], [3], [3], [15], [12], [16]]),
            "block_pose": np.array([[0, 0, z] for z in [0, .005, .025, .025, .025, 0]]),
        }
        self.assertEqual(dense_contacts(trace), dict(left_bilateral_lift=2, both_bilateral=3,
                                                   right_bilateral_after_shared=4, target_contact=5))

    def test_timeout_is_isolated_and_completed_receipts_are_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robot = "arx-x5"
            inputs = root / "inputs"
            results = root / "results"
            args = SimpleNamespace(input=inputs, output=results, robots=[robot], episodes=[0, 1, 2],
                                   assets=root, audit=root, source=root, timeout=1)
            for ep in range(3):
                source = inputs / robot / f"episode{ep:02}"
                source.mkdir(parents=True)
                write(source / "input.json", dict(robot=robot, episode=ep, dataset_revision=REVISION))
            receipt = results / robot / "episode00" / "receipt.json"
            receipt.parent.mkdir(parents=True)
            write(receipt, {"original": True})
            previous = results / robot / "episode01" / "attempt"
            previous.mkdir(parents=True)
            (previous / "worker.log").write_text("interrupted original")
            completed = []

            def run(command, stdout, **kwargs):
                attempt = Path(command[5])
                if attempt.parent.name == "episode01":
                    stdout.write(b"PHYSICS_PROGRESS 200\n")
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                write(attempt / "receipt.json", {"status": "SUCCESS", "independent": True})
                completed.append(attempt)
                return subprocess.CompletedProcess(command, 0)

            with patch("g2_native_expert_batch.subprocess.run", side_effect=run) as fake:
                batch(args)
                batch(args)
                self.assertEqual(fake.call_count, 2)
            self.assertEqual(read(receipt), {"original": True})
            failed = results / robot / "episode01"
            failure, _, _ = verify_episode(failed, robot, 1)
            self.assertEqual(failure["status"], "NOT_REPRODUCIBLE")
            self.assertIsNone(failure["physical_steps"])
            self.assertEqual(failure["lower_bound_completed_physical_steps"], 200)
            self.assertEqual(len(list(failed.glob("interrupted-*/worker.log"))), 1)
            self.assertEqual(len(completed), 1)
            self.assertTrue(read(completed[0].parent / "receipt.json")["independent"])


if __name__ == "__main__":
    unittest.main()
