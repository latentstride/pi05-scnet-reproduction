"""SCNet CPU integration for the production ARX training and checkpoint entry points."""
import dataclasses
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

DATASET = os.environ.get("STEP5_ARX_DATASET")


@unittest.skipUnless(DATASET, "Set STEP5_ARX_DATASET in the pinned CPU container")
class TrainingLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import flax.nnx as nnx
        from flax.traverse_util import flatten_dict
        import jax
        import jax.numpy as jnp
        import numpy as np
        from openpi.models import gemma, pi0

        from scripts.step5_arx import model, train
        from scripts.step5_arx.data import Dataset

        assert train.platform().platform == "cpu"
        cls.jax, cls.np, cls.jnp = jax, np, jnp
        cls.nnx, cls.model, cls.train, cls.dataset = nnx, model, train, Dataset
        cls.key = jax.random.key(train.SETTINGS.seed)
        with Dataset(DATASET) as data:
            cls.policy = model.NativePolicy(data.norm_stats)
            indices = tuple(data.indices("train"))
            cls.obs, cls.target = cls.policy.batch([data.read("train", *indices[len(indices) // 2])])
        print("ARX_INPUT", json.dumps(train.arx_input_contract(cls.obs, cls.target, cls.key)), flush=True)

        # Real OpenPI Pi0/NNX and LoRA operators with reduced constructor dimensions.
        # Keep pi05, vocabulary, observation, augmentation, actions and optimizer unchanged.
        pace, adapted = (dataclasses.replace(config, dtype="float32") for config in model.model_configs())
        gemma_config = gemma.get_config
        vision_factory = pi0._siglip.Module

        def small_gemma(name):
            config = gemma_config(name)
            return dataclasses.replace(config, width=32, depth=1, mlp_dim=64, num_heads=2,
                                       num_kv_heads=1, head_dim=16,
                                       lora_configs={k: dataclasses.replace(v, rank=2, alpha=2)
                                                     for k, v in config.lora_configs.items()})

        def small_vision(*args, **kwargs):
            return vision_factory(*args, **{**kwargs, "variant": "mu/32"})

        with mock.patch.object(gemma, "get_config", small_gemma), mock.patch.object(pi0._siglip, "Module", small_vision):
            cls.base_graph, cls.base_state = nnx.split(pace.create(cls.key))
            cls.graph, cls.spec, flat, paths, cls.predicate, report = model.audit_graph(adapted)
        cls.weights = jax.tree.map(lambda x: x.astype(jnp.bfloat16), cls.base_state.to_pure_dict())
        cls.frozen = model.bind_frozen_weights(cls.spec, cls.base_state, cls.weights, cls.predicate)
        cls.initial, cls.optimizer, cls.opt = model.initialize_adapters(cls.spec, flat, paths, cls.predicate)
        report["lora_initialization"] = model.check_adapter_initialization(cls.initial)
        cls.report = report
        cls.loss = staticmethod(train.flow_loss(cls.graph, adapted=True))
        cls.vanilla = staticmethod(train.flow_loss(cls.base_graph, adapted=False))
        cls.step = staticmethod(train.optimizer_step(cls.graph, cls.optimizer))
        cls.frozen_paths = flatten_dict(cls.frozen.to_pure_dict())
        print("SMALL_MODEL", json.dumps(dict(compute_dtype=pace.dtype, frozen_dtype="bfloat16",
                                             trainable=report["trainable"]["elements"],
                                             frozen=report["frozen"]["elements"])), flush=True)

    def gate(self, directory, *, initial=None, target=None):
        return self.train.check_initial_loss(
            self.loss, self.vanilla, self.initial if initial is None else initial, self.frozen,
            self.base_state, self.key, self.obs, self.target if target is None else target,
            Path(directory) / "initial_loss.json")

    def assert_tree_equal(self, before, after):
        self.assertEqual(self.jax.tree.structure(before), self.jax.tree.structure(after))
        for lhs, rhs in zip(self.jax.tree.leaves(before), self.jax.tree.leaves(after), strict=True):
            self.np.testing.assert_array_equal(lhs, rhs)

    def test_actual_gate_with_real_observation_and_original_aot_failure(self):
        import jaxtyping

        with self.assertRaisesRegex(jaxtyping.TypeCheckError, "ArgInfo"):
            self.loss.lower((self.initial, self.frozen), self.key, self.obs, self.target).compile()
        with TemporaryDirectory() as temp:
            value, fixed_eval = self.gate(temp)
            receipt = json.loads((Path(temp) / "initial_loss.json").read_text())
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["same_graph_abs_diff"], 0)
            self.assertEqual(receipt["same_graph_rel_diff"], 0)
            self.assertEqual((receipt["rtol"], receipt["atol"]), (1e-3, 1e-5))
            self.assertEqual(fixed_eval(self.initial), value)
            self.np.testing.assert_allclose(value, receipt["vanilla_frozen"], rtol=1e-3, atol=1e-5)
            self.np.testing.assert_allclose(
                self.loss((self.initial, self.frozen), self.key, self.obs, self.target), value, rtol=1e-3, atol=1e-5)
            print("SMALL_REAL_INITIAL_LOSS", json.dumps(receipt), flush=True)

    def test_nonzero_b_zero_a_and_inconsistent_bottom_weights_rejected(self):
        # Alter B using the production NNX partition, leaving all A tensors intact.
        from flax.traverse_util import flatten_dict, unflatten_dict

        bad = self.jax.tree.map(lambda value: value, self.initial)
        bad.replace_by_pure_dict(unflatten_dict({
            path: self.jnp.full_like(x, 4) if path[-1].endswith("_b") else x
            for path, x in flatten_dict(self.initial.to_pure_dict()).items()
        }))
        with self.assertRaisesRegex(ValueError, "LoRA B must be zero"):
            self.model.check_adapter_initialization(bad)
        with TemporaryDirectory() as temp, self.assertRaises(AssertionError):
            self.gate(temp, initial=bad)
        with self.assertRaisesRegex(ValueError, "LoRA A must be finite and nonzero"):
            self.model.check_adapter_initialization(self.jax.tree.map(self.jnp.zeros_like, self.initial))
        missing = {**self.weights}
        del missing["action_in_proj"]
        with self.assertRaisesRegex(ValueError, "Orbax paths differ"):
            self.model.bind_frozen_weights(self.spec, self.base_state, missing, self.predicate)
        wrong = {**self.weights, "action_in_proj": {**self.weights["action_in_proj"],
                                                    "kernel": self.weights["action_in_proj"]["kernel"][:1]}}
        with self.assertRaisesRegex(ValueError, "Incompatible Orbax leaf"):
            self.model.bind_frozen_weights(self.spec, self.base_state, wrong, self.predicate)
        # Simulate a graph binding that silently failed to replace one array.
        inconsistent = self.jax.tree.map(lambda value: value, self.base_state)
        changed = {**self.weights, "action_in_proj": {**self.weights["action_in_proj"],
                                                      "kernel": self.weights["action_in_proj"]["kernel"] + 1}}
        inconsistent.replace_by_pure_dict(changed)
        with mock.patch.object(type(inconsistent), "replace_by_pure_dict", return_value=None):
            with self.assertRaisesRegex(ValueError, "do not share the exact checkpoint array"):
                self.model.bind_frozen_weights(self.spec, inconsistent, self.weights, self.predicate)

    def test_nonfinite_real_loss_rejected_with_receipt(self):
        for invalid in (self.np.nan, self.np.inf):
            with self.subTest(invalid=invalid), TemporaryDirectory() as temp:
                target = self.target.copy()
                target[0, 0, 0] = invalid
                with self.assertRaisesRegex(RuntimeError, "Nonfinite initialization losses"):
                    self.gate(temp, target=target)
                receipt = json.loads((Path(temp) / "initial_loss.json").read_text())
                self.assertEqual(receipt["status"], "failed")
                self.assertFalse(receipt["all_losses_finite"])

    def test_actual_gradient_update_and_orbax_round_trip(self):
        key, obs, target = self.key, self.obs, self.target
        train = self.train
        before = self.jax.tree.map(self.np.array, self.frozen)
        fitted, opt, metrics = train.update(self.step, self.initial, self.frozen, self.opt, key, obs, target)
        self.assertGreater(metrics["grad_norm"], 0)
        self.assertGreater(metrics["adapter_delta_norm"], 0)
        self.assert_tree_equal(before, self.frozen)
        adam = opt[1][0]
        self.assertEqual(int(adam.count), 1)
        for moment in (adam.mu, adam.nu):
            self.assertEqual(self.jax.tree.structure(moment), self.jax.tree.structure(fitted))
            self.assertTrue(all(x.dtype == self.jnp.float32 for x in self.jax.tree.leaves(moment)))
        with TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "step-0001"
            train.save(checkpoint, fitted, opt)
            (Path(temp) / "step-0002").mkdir()
            (Path(temp) / "step-0003.orbax-checkpoint-tmp-0").mkdir()
            self.assertEqual(train.latest_checkpoint(Path(temp)), checkpoint)
            reloaded, reloaded_opt = train.restore(checkpoint, self.initial, self.opt)
            self.assert_tree_equal((fitted, opt), (reloaded, reloaded_opt))
            next_key = self.jax.random.fold_in(key, 2)
            next_original = train.update(self.step, fitted, self.frozen, opt, next_key, obs, target)
            next_resumed = train.update(self.step, reloaded, self.frozen, reloaded_opt, next_key, obs, target)
            self.assert_tree_equal(next_original[:2], next_resumed[:2])
            self.assertEqual(int(next_resumed[1][1][0].count), 2)
            with self.assertRaises(FileExistsError):
                train.save(checkpoint, fitted, opt)
            wrong = Path(temp) / "step-0007"
            checkpoint.rename(wrong)
            with self.assertRaisesRegex(ValueError, "count disagrees"):
                train.restore(wrong, self.initial, self.opt)
            print("SMALL_REAL_STEP_ORBAX", json.dumps({**metrics, "saved_step": 1, "resumed_step": 2,
                                                      "optimizer_leaves": len(self.jax.tree.leaves(opt)),
                                                      "frozen_leaves": len(self.frozen_paths)}), flush=True)

    def test_bad_optimizer_metrics_rejected(self):
        # Deliberate faults in the metrics returned by optimizer_step, isolated from numerical integration.
        for metrics in ((1., 0., 1.), (1., self.np.nan, 1.), (1., 1., 0.), (1., 1., self.np.inf)):
            with self.subTest(metrics=metrics), self.assertRaisesRegex(RuntimeError, "Invalid optimizer update"):
                self.train.update(lambda *args: (self.initial, self.opt, metrics), self.initial, self.frozen,
                                  self.opt, self.key, self.obs, self.target)

    def test_run_fit_gate_order_overfit_and_resume(self):
        train = self.train
        first_step = train.update

        class LimitedValidationData(self.dataset):
            def indices(self, split, stride=1):
                indices = super().indices(split, stride)
                yield from indices if split == "train" else [next(indices)]

        with TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(project_root=root, run=root, resume=False, dataset=Path(DATASET),
                                   stop_file=root / "stop")
            args.stop_file.touch()
            updates = []

            def stop_after_first_formal(*args_updates):
                fitted, opt, metrics = first_step(*args_updates)
                updates.append(int(opt[1][0].count))
                if len(updates) == 22:
                    args.stop_file.write_text("stop\n")
                return fitted, opt, metrics

            # The small random checkpoint is not a model-quality test. Exercise the 1% decision
            # explicitly, while running the real final eval forward before substituting its result.
            original_gate = train.check_initial_loss
            final_results = []
            gate_calls = []

            def control_overfit_result(*gate_args):
                value, evaluate = original_gate(*gate_args)
                gate_calls.append(value)
                if len(gate_calls) == 1:
                    args.stop_file.write_text("stop\n")

                def final_eval(adapters):
                    final_results.append(evaluate(adapters))
                    return value * (0.995 if len(final_results) == 1 else 0.98)
                return value, final_eval

            identity = {"cpu_integration": True}
            real_device = train.platform()
            parts = self.graph, self.initial, self.frozen, self.optimizer, self.opt
            with (mock.patch.object(train, "SETTINGS", dataclasses.replace(
                    train.SETTINGS, train_steps=2, checkpoint_interval=2, validation_interval=2)),
                  mock.patch.object(train, "Dataset", LimitedValidationData),
                  mock.patch.object(train.model, "validate_install", return_value=root),
                  mock.patch.object(train.model, "load_models",
                                    return_value=(parts, (self.base_graph, self.base_state), self.report)),
                  mock.patch.object(train, "update", side_effect=stop_after_first_formal),
                  mock.patch.object(train, "check_initial_loss", side_effect=control_overfit_result),
                  mock.patch("signal.signal")):
                self.assertEqual(real_device.platform, "cpu")
                self.assertEqual(train.run_fit(args, identity), 75)
                self.assertEqual(updates, [])
                self.assertTrue((root / "initial_loss.json").is_file())
                args.resume = True
                args.stop_file.write_text("")
                with self.assertRaisesRegex(RuntimeError, "20-step overfit gate failed"):
                    train.run_fit(args, identity)
                self.assertEqual(updates, [1, *range(1, 21)])
                self.assertFalse((root / "gpu_gates.json").exists())
                self.assertFalse((root / "val-frozen.json").exists())
                self.assertIsNone(train.latest_checkpoint(root / "checkpoints"))
                updates.clear()
                args.resume = True
                self.assertEqual(train.run_fit(args, identity), 75)
                self.assertEqual(updates, [1, *range(1, 21), 1])
                committed = train.latest_checkpoint(root / "checkpoints")
                self.assertEqual(committed.name, "step-0001")
                (root / "checkpoints/step-0002").mkdir()
                args.stop_file.write_text("")
                self.assertEqual(train.run_fit(args, identity), 0)
                self.assertEqual(updates[-2:], [1, 2])
                self.assertEqual(len(updates), 23)
                self.assertEqual(len(final_results), 2)
                self.assertTrue(self.np.isfinite(final_results).all())
            self.assertEqual(len(json.loads((root / "gpu_gates.json").read_text())["overfit"]), 20)
            self.assertEqual(json.loads((root / "selected.json").read_text())["last_step"], 2)
            self.assertEqual(json.loads((root / "val-0002.json").read_text())["windows"], 1)
            print("SMALL_RUN_FIT_CONTROL", json.dumps({"executed_overfit_eval_losses": final_results,
                                                      "tested_improvements": [0.005, 0.02],
                                                      "stop_before_backward": True,
                                                      "passed_updates": updates, "formal_resume": [1, 2]}), flush=True)


if __name__ == "__main__":
    unittest.main()
