"""Regression for the zero-residual gate. Run in the pinned OpenPI-PACE environment."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("openpi") and importlib.util.find_spec("jax"),
                     "Requires the Step 5 OpenPI-PACE JAX environment")
class InitialLossTest(unittest.TestCase):
    def setUp(self):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        from scripts.step5_arx import model
        from scripts.step5_arx.train import check_initial_loss

        self.nnx = nnx
        self.model = model
        self.check = check_initial_loss
        self.rng = jax.random.key(42)
        self.observation = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
        self.target = jnp.array([[0.0]], dtype=jnp.float32)
        self.frozen = jnp.array([[1.0], [1.0]], dtype=jnp.float32)
        self.a = jnp.array([[1.0], [1.0]], dtype=jnp.float32)
        self.zero = jnp.zeros((1, 1), dtype=jnp.float32)
        self.initial = nnx.State({"lora_a": nnx.Param(self.a).to_state(),
                                  "lora_b": nnx.Param(self.zero).to_state()})

        @jax.jit
        def loss(state, rng, obs, target):
            adapter, frozen = state
            params = adapter.to_pure_dict()
            out = obs @ frozen + (obs @ params["lora_a"]) @ params["lora_b"]
            return jnp.mean(jnp.square(out - target))

        @jax.jit
        def diagnostic(frozen, rng, obs, target):
            # Model the type of independent executable drift seen in the job.
            return jnp.mean(jnp.square(obs @ frozen - target)) * 0.99465

        self.loss = loss
        self.diagnostic = diagnostic

    def test_strict_same_executable_gate_records_separate_vanilla_difference(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "initial.json"
            value = self.check(self.loss, self.diagnostic, self.initial, self.frozen,
                               self.frozen, self.rng, self.observation, self.target, out)
            result = json.loads(out.read_text())
            self.assertEqual(value, 9.0)
            self.assertEqual(result["same_graph_abs_diff"], 0.0)
            self.assertGreater(result["vanilla_rel_diff"], result["rtol"])
            self.assertEqual(result["rtol"], 1e-3)
            self.assertEqual(result["atol"], 1e-5)

    def test_nonzero_b_is_rejected_by_both_parameter_and_forward_checks(self):
        updated = self.nnx.State({"lora_a": self.nnx.Param(self.a).to_state(),
                                  "lora_b": self.nnx.Param(self.zero + 0.25).to_state()})
        with self.assertRaisesRegex(ValueError, "LoRA B must be zero"):
            self.model.check_adapter_initialization(updated)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "initial.json"
            with self.assertRaises(AssertionError):
                self.check(self.loss, self.diagnostic, updated, self.frozen, self.frozen,
                           self.rng, self.observation, self.target, out)
            self.assertGreater(json.loads(out.read_text())["same_graph_abs_diff"], 0.0)

    def test_zero_a_is_rejected_to_preserve_gradient_at_zero_b(self):
        updated = self.nnx.State({"lora_a": self.nnx.Param(self.a * 0).to_state(),
                                  "lora_b": self.nnx.Param(self.zero).to_state()})
        with self.assertRaisesRegex(ValueError, "LoRA A must be finite and nonzero"):
            self.model.check_adapter_initialization(updated)


if __name__ == "__main__":
    unittest.main()
