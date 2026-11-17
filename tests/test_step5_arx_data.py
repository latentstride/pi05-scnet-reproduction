import os
from pathlib import Path
import unittest

import numpy as np

from scripts.step5_arx.adapter import MODEL_CAMERAS, NativeAdapter, native_observation, normalization
from scripts.step5_arx.data import ARM, CAMERAS, GRIP, HORIZON, PROMPT, Dataset, absolute, delta, labels, partitions


class Step5DataTest(unittest.TestCase):
    def setUp(self):
        self.q = np.arange(52 * 14, dtype=np.float32).reshape(-1, 14) / 1000
        self.q[:, GRIP] = np.linspace(1, 0, len(self.q))[:, None]
        self.images = {name: np.full((240, 320, 3), i, dtype=np.uint8)
                       for i, name in enumerate(CAMERAS)}
        stats = {"q01": [-0.2] * 14, "q99": [0.9] * 14}
        self.adapter = NativeAdapter({"state": stats, "actions": stats})

    def test_partitions_are_disjoint_exhaustive_and_fixed(self):
        splits = partitions()
        self.assertEqual([len(v) for v in splits.values()], [40, 5, 5])
        self.assertEqual(sorted(sum(splits.values(), [])), list(range(50)))
        self.assertEqual(splits["test"], [6, 29, 33, 38, 48])

    def test_labels_exclude_current_and_stop_at_episode_boundary(self):
        state, action = labels(self.q, 1)
        np.testing.assert_array_equal(state, self.q[1])
        np.testing.assert_array_equal(action, self.q[2:52])
        with self.assertRaises(IndexError):
            labels(self.q, 2)
        with self.assertRaises(IndexError):
            labels(self.q, -1)

    def test_delta_and_normalization_round_trip_including_extremes(self):
        state, target = labels(self.q, 0)
        target[[0, -1], 0] = [3, -3]
        d = delta(state, target)
        np.testing.assert_array_equal(d[:, GRIP], target[:, GRIP])
        self.assertAlmostEqual(float(d[-1, ARM[1]]), float(target[-1, ARM[1]] - state[ARM[1]]))
        np.testing.assert_allclose(absolute(state, d), target, rtol=0, atol=5e-7)
        x = self.adapter.inputs(dict(state=state, actions=target, images=self.images, prompt=PROMPT))
        np.testing.assert_allclose(self.adapter.outputs(state, np.pad(x["actions"], ((0, 0), (0, 18)))),
                                   target, rtol=0, atol=5e-7)

    def test_native_schema_keeps_cameras_and_requires_exact_prompt(self):
        sample = native_observation({"joint_action": {"vector": self.q[0]},
                                     "observation": {name: {"rgb": im} for name, im in self.images.items()}})
        x = self.adapter.inputs(sample)
        self.assertEqual(tuple(x["image"]), MODEL_CAMERAS)
        for name, im in zip(MODEL_CAMERAS, self.images.values(), strict=True):
            np.testing.assert_array_equal(x["image"][name], im)
        self.assertTrue(all(x["image_mask"].values()))
        self.assertNotIn("actions", x)
        with self.assertRaises(ValueError):
            self.adapter.inputs(dict(sample, prompt=""))
        missing = dict(sample, images={name: self.images[name] for name in CAMERAS[:2]})
        with self.assertRaises(ValueError):
            self.adapter.inputs(missing)
        with self.assertRaises(ValueError):
            self.adapter.outputs(self.q[0], np.full((HORIZON, 32), np.nan))
        p = np.zeros((HORIZON, 32), dtype=np.float32)
        p[:, GRIP] = 6
        clipped = self.adapter.control_outputs(self.q[0], p)
        np.testing.assert_array_equal(clipped[:, GRIP], 1.0)
        np.testing.assert_array_equal(clipped[:, ARM], self.adapter.outputs(self.q[0], p)[:, ARM])

    @unittest.skipUnless(os.environ.get("STEP5_ARX_DATASET"), "converted archive not supplied")
    def test_real_archive_all_50_episode_boundaries(self):
        with Dataset(Path(os.environ["STEP5_ARX_DATASET"])) as ds:
            self.assertEqual({key: len(tuple(ds.indices(key))) for key in partitions()},
                             {"train": 5736, "val": 728, "test": 716})
            for split, episodes in partitions().items():
                for ep in episodes:
                    q = ds.groups[split][ep]["drive_target"][:]
                    state = q[:-HORIZON]
                    targets = np.stack([labels(q, t)[1] for t in range(len(state))])
                    delta_norm = normalization(delta(state, targets), ds.norm_stats["actions"])
                    estimate = absolute(state, normalization(delta_norm, ds.norm_stats["actions"], inverse=True))
                    np.testing.assert_allclose(estimate, targets, rtol=0, atol=1e-6)
                    sample = ds.read(split, ep, len(q) - HORIZON - 1)
                    np.testing.assert_array_equal(sample["actions"][-1], q[-1])
                    with self.assertRaises(IndexError):
                        ds.read(split, ep, len(q) - HORIZON)


if __name__ == "__main__":
    unittest.main()
