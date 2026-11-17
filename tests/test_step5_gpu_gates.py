"""Optional CPU regression of the real ARX input and its diagnostic AOT path."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

DATASET = os.getenv("STEP5_ARX_DATASET")


@unittest.skipUnless(DATASET, "Run in the pinned CPU container with STEP5_ARX_DATASET")
class GateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.step5_arx import train
        cls.t = train
        assert train.platform().platform == "cpu"

    def test_actual_arx_observation_initial_gate(self):
        from openpi.models.model import Observation
        from jaxtyping import TypeCheckError
        t = self.t
        with t.Dataset(DATASET) as data:
            policy = t.model.NativePolicy(data.norm_stats)
            indices = tuple(data.indices("train"))
            obs, target = policy.batch([data.read("train", *indices[len(indices) // 2])])
        key = t.jax.random.key(t.SETTINGS.seed)
        self.assertEqual(list(target.shape), [1, 50, 32])
        print("REAL_ARX_INPUT", json.dumps(t.arx_input_contract(obs, target, key)))

        @t.jax.jit
        def loss(state, key, observation: Observation, actions):
            adapter, _ = state
            return (t.jnp.mean(observation.state) + t.jnp.mean(actions)
                    + adapter["a"] * adapter["b"] + t.jax.random.uniform(key))

        initial = {"a": t.jnp.float32(3), "b": t.jnp.float32(0)}
        with self.assertRaisesRegex(TypeCheckError, "ArgInfo"):
            loss.lower((initial, {}), key, obs, target)
        with TemporaryDirectory() as tmp:
            report = Path(tmp) / "initial_loss.json"
            # The deliberately different vanilla output checks its diagnostic-only role.
            frozen_diag = lambda _, k, o, a: loss((initial, {}), k, o, a) + 0.01
            first, fixed = t.check_initial_loss(loss, frozen_diag, initial, {}, {}, key, obs, target, report)
            self.assertEqual(first, fixed(initial))
            receipt = json.loads(report.read_text())
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["same_graph_abs_diff"], 0)
            self.assertGreater(receipt["vanilla_abs_diff"], 0)
            print("REAL_ARX_AOT_GATE", json.dumps(receipt))
