"""Real ARX launch contract and deterministic GPU control-flow substitutes on CPU."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

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

    def test_refuses_pending_legacy_and_stale_admission_before_device(self):
        t = self.t
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            receipt = root / "review.json"
            args = SimpleNamespace(project_root=root, run=root / "new-run", cpu_receipt=receipt,
                                   dataset=root / "data.h5", data_audit=root / "audit.json",
                                   resume=False, command="fit")
            scope = t.admission.make_scope(args, "a" * 32, "test_account", root / "slurm-%j.out")
            for packet in ({"status": "cpu_pass"}, {"status": "CPU_STRUCTURAL_ONLY"},
                           {"status": "CPU_NUMERIC_EXCEPTION", "scope": scope,
                            "authorization": {"state": "PENDING"}}):
                receipt.write_text(json.dumps(packet))
                with (mock.patch.object(t.model, "cuda_identity") as cuda,
                      self.assertRaises(PermissionError)):
                    t.admit_gpu(args)
                cuda.assert_not_called()
            receipt.write_text('{"status":"CPU_VERIFIED","identity":{"code_sha256":"old"}}')
            with (mock.patch.object(t, "fingerprint", return_value={"code_sha256": "new"}),
                  self.assertRaisesRegex(RuntimeError, "stale")):
                t.inspect_admission(args)

    def exercise(self, root, *, invalid=None, stop_at=None):
        t = self.t
        stop = root / "stop"
        stop.touch()
        args = SimpleNamespace(project_root=root, run=root, resume=(root / "identity.json").exists(),
                               dataset=Path(DATASET), stop_file=stop)
        trace = []
        initial = t.nnx.State({"weight": t.nnx.Param(t.jnp.asarray([0.], dtype=t.jnp.float32)).to_state()})
        frozen = {"weight": t.jnp.asarray([5.], dtype=t.jnp.bfloat16)}
        initial_opt = ((t.jnp.int32(0),), (t.optax.ScaleByAdamState(
            count=t.jnp.int32(0), mu=t.jnp.float32(0), nu=t.jnp.float32(0)),))
        gpu = SimpleNamespace(memory_stats=lambda: {"peak_bytes_in_use": 13})
        identity = {"test": "gpu-control-flow"}

        class StubDataset:
            norm_stats = {}

            def __init__(self, path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def indices(self, split):
                return [(0, 0)]

            def read(self, *args):
                return None

        def update(state, actual_frozen, opt, key, obs, target):
            i = int(opt[1][0].count) + 1
            prior = float(state.to_pure_dict()["weight"][0])
            stage = json.loads((root / "gpu_progress.json").read_text())["stage"]
            trace.append((stage, i, prior))
            if invalid == "oom" and stage == "gpu_stage_a" and i == 2:
                raise RuntimeError("RESOURCE_EXHAUSTED: simulated CUDA out of memory")
            grad = float("inf") if invalid == "inf" and stage == "gpu_stage_a" and i == 2 else 1.
            if invalid == "frozen" and stage == "gpu_stage_a" and i == 2:
                actual_frozen["weight"] = t.jnp.asarray([6.], dtype=t.jnp.bfloat16)
            if stop_at == i and stage == "gpu_formal_train":
                stop.write_text("stop\n")
            fitted = t.jax.tree.map(lambda x: x + 1., state)
            count = opt[1][0].count + 1
            result_opt = (opt[0], (opt[1][0]._replace(count=count, mu=opt[1][0].mu + 1),))
            return fitted, result_opt, (2., grad, 1.)

        def startup_gate(*args):
            args[-1].write_text('{"status":"passed"}')
            return 2., lambda fitted: 1.99 if invalid == "overfit" else 1.9

        results = dict(episode_mean=dict(flow_loss=2., arm_mae_rad=1., gripper_mae=1.))
        parts = (None, initial, frozen, None, initial_opt)
        output = io.StringIO()
        with (mock.patch.object(t, "platform", return_value=gpu),
              mock.patch.object(t, "Dataset", StubDataset),
              mock.patch.object(t, "flow_loss"),
              mock.patch.object(t, "check_initial_loss", side_effect=startup_gate),
              mock.patch.object(t, "optimizer_step", return_value=update),
              mock.patch.object(t, "evaluate", return_value=results),
              mock.patch.object(t.model, "NativePolicy", return_value=SimpleNamespace(batch=lambda _: (None, None))),
              mock.patch.object(t.model, "sampler"), mock.patch.object(t.model, "validate_install"),
              mock.patch.object(t.model, "load_models", return_value=(parts, (None, {}), {})),
              mock.patch("signal.signal"), redirect_stdout(output)):
            try:
                status = t.run_fit(args, identity)
            except RuntimeError as error:
                t.write_gpu_failure(root, error)
                status = 1
        return status, trace

    def test_two_steps_then_twenty_then_four_hundred_and_orbax_resume(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            status, trace = self.exercise(root, stop_at=26)
            self.assertEqual(status, 75)
            self.assertEqual(trace[:2], [("gpu_stage_a", 1, 0.), ("gpu_stage_a", 2, 1.)])
            self.assertEqual(trace[2:22], [("gpu_overfit", i, float(i - 1)) for i in range(1, 21)])
            self.assertEqual(trace[22:], [("gpu_formal_train", i, float(i - 1)) for i in range(1, 27)])
            self.assertEqual(self.t.latest_checkpoint(root / "checkpoints").name, "step-0026")
            (root / "checkpoints/step-0050").mkdir()
            (root / "stop").write_text("")
            status, trace = self.exercise(root)
            self.assertEqual(status, 0)
            self.assertEqual(trace, [("gpu_formal_train", i, float(i - 1)) for i in range(27, 401)])
            self.assertEqual(self.t.latest_checkpoint(root / "checkpoints").name, "step-0400")
            self.assertTrue(list((root / "checkpoints").glob(".incomplete-step-0050-*")))
            self.assertTrue(all((root / f"val-{i:04d}.json").exists() for i in (100, 200, 300, 400)))
            self.assertEqual(json.loads((root / "selected.json").read_text())["last_step"], 400)
            gates = json.loads((root / "gpu_gates.json").read_text())
            self.assertEqual([r["adamw_count"] for r in gates["stage_a"]], [1, 2])
            self.assertEqual([r["adamw_count"] for r in gates["overfit"]], list(range(1, 21)))
            self.assertEqual(gates["memory"]["peak_bytes_in_use"], 13)

    def test_failures_stop_before_formal(self):
        for invalid, phase, count in (("inf", "gpu_stage_a", 2), ("oom", "gpu_stage_a", 2),
                                      ("frozen", "gpu_frozen_check_after", 2),
                                      ("overfit", "gpu_overfit_eval", 22)):
            with self.subTest(invalid=invalid), TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                status, trace = self.exercise(root, invalid=invalid)
                self.assertEqual((status, len(trace)), (1, count))
                self.assertFalse((root / "gpu_gates.json").exists())
                self.assertIsNone(self.t.latest_checkpoint(root / "checkpoints"))
                failure = json.loads((root / "gpu_failure.json").read_text())
                self.assertEqual(failure["last_progress"]["stage"], phase)
                self.assertEqual(failure["memory"]["peak_bytes_in_use"], 13)
                if count == 2:
                    self.assertTrue((root / "gpu_stage_a_0001.json").exists())
                    self.assertEqual((root / "gpu_stage_a_0002.json").exists(), invalid == "frozen")
