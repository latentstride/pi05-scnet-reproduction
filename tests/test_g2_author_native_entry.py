"""CPU protocol-contract tests against the pinned upstream rollout source."""

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import unittest

import numpy as np

from g2_author_native_entry import (
    ArchivedFirstChunk,
    CounterfactualBoundary,
    MAX_ACTIONS,
    fixed_h50,
)


class AuthorEnvironment:
    """Expose only the author environment methods used by PACE run_episode."""

    def __init__(self, *, success_at=None):
        self.take_action_cnt = 0
        self.step_lim = MAX_ACTIONS
        self.eval_success = False
        self.success_at = success_at
        self.observations, self.actions = [], []

    def get_obs(self):
        self.observations.append(self.take_action_cnt)
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        return {
            "joint_action": {"vector": np.zeros(14)},
            "observation": {
                key: {"rgb": rgb}
                for key in ("head_camera", "left_camera", "right_camera")
            },
        }

    def get_instruction(self):
        return "unseen instruction"

    def take_action(self, action):
        self.actions.append(action.copy())
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt == self.success_at


class AuthorNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        location = os.environ.get("G2_PINNED_PACE")
        if not location:
            raise unittest.SkipTest(
                "G2_PINNED_PACE is required for the actual upstream rollout"
            )
        source = Path(location) / "pace/src/pace/rollout.py"
        if (
            sha256(source.read_bytes()).hexdigest()
            != "6ea198bcf47d6b6e62bd9206b6c5694466ca6346b30ad9c20885e4009af91325"
        ):
            raise ValueError("unexpected upstream rollout revision")
        spec = spec_from_file_location("pinned_pace_rollout", source)
        cls.upstream = module_from_spec(spec)
        spec.loader.exec_module(cls.upstream)

    def test_exact_rows_and_first_new_observation_at_fifty(self):
        env = AuthorEnvironment()
        chunk = np.linspace(-1.1, 1.1, 700, dtype=np.float64).reshape(50, 14)
        policy = ArchivedFirstChunk(chunk, env.get_instruction())
        with self.assertRaises(CounterfactualBoundary):
            self.upstream.run_episode(env, policy, fixed_h50)
        np.testing.assert_array_equal(env.actions, chunk)
        self.assertEqual(env.observations, [0, 50])
        self.assertEqual(policy.requests, 1)

    def test_upstream_success_ends_block_and_800_limit_is_native(self):
        for success_at, expected in ((43, (True, 43, 1)), (None, (False, 800, 16))):
            env = AuthorEnvironment(success_at=success_at)

            class Policy:
                def infer(self, request):
                    return {"actions": np.zeros((50, 14))}

            result = self.upstream.run_episode(env, Policy(), fixed_h50)
            self.assertEqual(
                tuple(result[k] for k in ("success", "steps", "queries")), expected
            )


if __name__ == "__main__":
    unittest.main()
