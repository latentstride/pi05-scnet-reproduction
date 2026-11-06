"""G2-E admission, actual execution/abort receipts, RNG and native safety."""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import numpy as np
from g2_adapter import G2Error, Stamp
from g2_contact import ContactTracker
from g2_episode import run_episode
from g2e_native_guard import NativeGuard
from g2e_protocol import PROTOCOL, validate_window
from g2e_model_entry import reset_rng
from g2_run_task import execute
from g2_task_runtime import validate_config
from test_g2_native_backend import native_fixture
from test_g2_integration import FixedPolicy, spec
from g2_fixtures import robot_arms
from robotwin_control_geometry import Status

ROOT = Path(__file__).resolve().parents[1]


class GuardStub:
    protocol = PROTOCOL

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at
        self.previous_drive = None

    def candidate(self, *args):
        self.calls += 1
        if self.calls == self.fail_at:
            raise G2Error(Status.COLLISION, "injected candidate collision")
        args[-1]()

    def measured(self, *args):
        pass

    def tracking(self, *args):
        pass

    def metrics(self):
        return {"guard_calls": self.calls}


def guarded_fixture(root, fail_at=None):
    backend, state, closed, physical = native_fixture(root)
    guard = GuardStub(fail_at)
    backend.guard = guard
    backend.execution_protocol = PROTOCOL
    backend.env.scene.before_step = backend._before_step
    physical.get_contacts = lambda: []
    backend.env.check_success = lambda: False
    return backend, state, closed, physical


class G2ETests(unittest.TestCase):
    def test_only_preselected_windows_and_native_are_admitted(self):
        for k in (1, 4, 26, 31):
            validate_window(k, PROTOCOL)
        for k in (0, 8, 16, 32, 50, True):
            with self.assertRaises(ValueError):
                validate_window(k, PROTOCOL)
        with self.assertRaises(ValueError):
            validate_window(26)
        with self.assertRaises(ValueError):
            validate_window(26, PROTOCOL, "ARX-X5")

    def test_actual_native_long_window_executes_closure_without_state_replacement(self):
        for k in (26, 31):
            with self.subTest(k=k), tempfile.TemporaryDirectory() as tmp:
                backend, state, closed, physical = guarded_fixture(Path(tmp))
                cfg = dict(spec(k, "Aloha"), max_actions=k, execution_protocol=PROTOCOL)

                class CloseLater(FixedPolicy):
                    def infer(self, request):
                        result = super().infer(request)
                        result["actions"][24:, 6] = 0.0
                        return result

                result = run_episode(
                    cfg, backend, CloseLater(np.zeros((2, 6))), Path(tmp) / "episode"
                )
                self.assertEqual(result["executed"], k, result)
                self.assertEqual(result["queries"], 1)
                self.assertEqual(state["gripper"][0], 0.0)
                self.assertEqual(backend.guard.calls, k)
                self.assertEqual(backend.tick, k)
                self.assertIs(backend.env.scene, physical)
                self.assertEqual(closed, [True])

    def test_abort_after_three_completed_actions_keeps_partial_motion_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend, state, closed, physical = guarded_fixture(Path(tmp), 4)
            cfg = dict(spec(31, "Aloha"), max_actions=31, execution_protocol=PROTOCOL)
            row = run_episode(
                cfg, backend, FixedPolicy(np.zeros((2, 6))), Path(tmp) / "episode"
            )
            self.assertEqual(row["executed"], 3, row)
            self.assertEqual(backend.tick, 3)
            self.assertEqual(row["failure"]["code"], "collision")
            self.assertEqual(row["control_calls"], 3)
            self.assertTrue(np.all(state["velocity"] == 0))
            self.assertEqual(closed, [True])
            events = (Path(tmp) / "episode/events.jsonl").read_text()
            self.assertIn("partial_prefix_abort", events)
            self.assertIn("native_guard_rejection", events)

    def test_missing_guard_cannot_silently_execute_long_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend, state, closed, physical = native_fixture(Path(tmp))
            cfg = dict(spec(26, "Aloha"), execution_protocol=PROTOCOL)
            row = run_episode(
                cfg, backend, FixedPolicy(np.zeros((2, 6))), Path(tmp) / "episode"
            )
            self.assertEqual(row["executed"], 0)
            self.assertEqual(row["queries"], 0)
            self.assertEqual(row["failure"]["code"], "invalid_input")

    def test_rng_reset_is_explicit_and_does_not_infer(self):
        policy = NS(_is_pytorch_model=False, _rng="old")
        reset_rng(policy, 20261106, lambda k: ("key", k))
        self.assertEqual(policy._rng, ("key", 20261106))
        for seed in (-1, 2**32, True):
            with self.assertRaises(ValueError):
                reset_rng(policy, seed, lambda x: x)
        with self.assertRaises(ValueError):
            reset_rng(NS(), 1, lambda x: x)

    def test_normal_config_entry_preserves_opt_in_and_rejects_arx(self):
        cfg = json.loads(
            (
                ROOT
                / "research/g2/failure-audit-127054052-r1/release-context/configs/seed0-A-native-w1.json"
            ).read_text()
        )
        cfg.update(
            episode_id="native",
            condition="Aloha",
            window=31,
            max_actions=31,
            execution_protocol=PROTOCOL,
            contact_mode="g2e_native_measured_contact_v1",
            policy_rng_seed=20261106,
        )
        validate_config(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            backend, state, closed, physical = guarded_fixture(Path(tmp))

            @contextmanager
            def factory(specification):
                self.assertEqual(specification["execution_protocol"], PROTOCOL)
                yield backend, None

            row = execute(
                cfg, factory, FixedPolicy(np.zeros((2, 6))), Path(tmp) / "episode"
            )
            self.assertEqual(row["executed"], 31, row)
        invalid = deepcopy(cfg)
        invalid["embodiment"] = "ARX-X5"
        with self.assertRaises(ValueError):
            validate_config(invalid)

    def test_real_guard_separates_negative_drive_intent_and_contact_geometry(self):
        arms = robot_arms("aloha-agilex")
        tracker = ContactTracker(99, {"left": [1, 2], "right": [3, 4]})
        probes = []

        class Collision:
            last_collisions = []

            def refresh(self):
                pass

            def set_start(self, *args):
                pass

            def project_fingers(self, q, g, check, previous):
                check()
                probes.append(g.copy())
                return np.maximum(g, 0.02), [{"rule": "synthetic contact projection"}]

            def __call__(self, q, g):
                return True

        world = Collision()
        world.tracker = tracker
        guard = NativeGuard(world, arms, 0.02, 0.03, 10.0)
        drive = np.full((2, 2), -0.01)
        guard.candidate(
            np.zeros((2, 6)), np.full((2, 2), 0.03), np.zeros((2, 6)), drive, lambda: 1
        )
        np.testing.assert_array_equal(drive, np.full((2, 2), -0.01))
        np.testing.assert_array_equal(probes[0], np.zeros((2, 2)))
        np.testing.assert_array_equal(
            guard.metrics()["predicted_fingers"], np.full((2, 2), 0.02)
        )
        target = np.tile(np.eye(4), (2, 1, 1))
        actual = target.copy()
        actual[0, 0, 3] = 0.031
        with self.assertRaisesRegex(G2Error, "tracking"):
            guard.tracking(target, actual)
        q = np.zeros((2, 6))
        q[0, 0] = 100
        with self.assertRaises(G2Error):
            guard.candidate(np.zeros((2, 6)), np.zeros((2, 2)), q, drive, lambda: 1)

    def test_actual_contact_loss_and_premature_release_stop(self):
        arms = robot_arms("aloha-agilex")
        tracker = ContactTracker(99, {"left": [1, 2], "right": [3, 4]})
        tracker.holders["left"] = np.eye(4)
        with self.assertRaises(G2Error):
            tracker.guard_opening(np.zeros((2, 2)), np.full((2, 2), 0.02), arms)
        with self.assertRaises(G2Error):
            tracker.observe(
                Stamp("x", 1, 0.004), [], np.tile(np.eye(4), (2, 1, 1)), np.eye(4)
            )


if __name__ == "__main__":
    unittest.main()


class G2EReleaseTests(unittest.TestCase):
    def test_release_bindings_all_native_caps_and_dry_host_plan(self):
        from hashlib import sha256
        from g2_locked_task import load_trial
        from g2e_launch import build
        from g2e_host_supervisor import validate_plan

        release = ROOT / "research/g2/g2e-20261106-r1/release"
        lock = release / "run_lock.json"
        raw = lock.read_bytes()
        roster = json.loads(raw)["episodes"]
        self.assertEqual(len(roster), 8)
        for row in roster:
            cfg, expected, _ = load_trial(
                lock,
                sha256(raw).hexdigest(),
                row["episode_id"],
                release / "configs" / f"{row['episode_id']}.json",
            )
            self.assertEqual(cfg["execution_protocol"], PROTOCOL)
            self.assertEqual(cfg["policy_rng_seed"], 20261106 + row["seed"])
            self.assertEqual(cfg["embodiment"], "aloha-agilex")
        spec = json.loads((release / "launch-spec.json").read_text())
        self.assertLessEqual(
            sum(spec["stage_timeout_s"].values()) + 300 + 60,
            spec["approved_allocation_s"],
        )
        self.assertEqual(spec["approved_allocation_s"], 12600)
        with tempfile.TemporaryDirectory() as tmp:
            plan = build(
                Path("/scnet-fixture"),
                release,
                Path(tmp),
                "1",
                dict(gpu_uuid="GPU-fake", sim_argv=["no-execution-placeholder"]),
                9999999999,
            )
            validate_plan(plan)
            self.assertEqual(set(plan["stages"]), {"dynamic", "aloha", "statistics"})
            commands = json.loads((Path(tmp) / "native-trials.json").read_text())
            self.assertEqual(len(commands), 8)
            self.assertTrue(
                all(c["argv"][0] == "no-execution-placeholder" for c in commands)
            )
            self.assertIn(
                str(release / "scripts/g2e_model_entry.py"), plan["model_server"]
            )
            self.assertFalse(
                any(
                    Path(arg).name in ("sbatch", "srun", "salloc")
                    for c in commands
                    for arg in c["argv"]
                )
            )
        admission = json.loads((release / "release-admission.json").read_text())
        self.assertEqual(admission["gpu_release_ready"], admission["status"] == "ready")
        if admission["gpu_release_ready"]:
            self.assertTrue(admission["cpu_release_validated"])
            self.assertIn("user_authorization", admission)
            self.assertEqual(admission["approved_allocation_s"], 12600)
        else:
            self.assertEqual(admission["status"], "awaiting_user_approval")

    def test_partial_statistics_never_counts_missing_as_failure_or_success(self):
        from g2e_statistics import summarize

        release = ROOT / "research/g2/g2e-20261106-r1/release"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trials").mkdir()
            result = summarize(
                release / "run_lock.json", root / "trials", root / "statistics"
            )
            self.assertEqual(result["planned_trials"], 8)
            self.assertEqual(result["completed_trials"], 0)
            self.assertEqual(result["task_success_count"], 0)
            self.assertEqual(len(result["missing_or_not_run"]), 8)
            self.assertIsNone(result["success_fraction_of_completed"])
            self.assertFalse(result["formal_score_eligible"])

    def test_rng_extension_works_over_real_websocket_and_requires_reset(self):
        from threading import Thread
        from unittest.mock import patch
        from websockets.sync.server import serve
        from g2e_model_entry import handler, reset_remote
        from pace_runtime import open_policy

        class Model:
            _is_pytorch_model = False
            _rng = 0

            def infer(self, request):
                self._rng += 1
                return {"actions": np.tile(request["state"], (50, 1))}

        model = Model()
        reset = lambda p, s: reset_rng(p, s, lambda x: x)
        with patch("g2e_model_entry.reset_rng", reset):
            with serve(
                handler(model, "checkpoint"), "127.0.0.1", 0, close_timeout=0.1
            ) as server:
                worker = Thread(target=server.serve_forever, daemon=True)
                worker.start()
                uri = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
                request = dict(
                    state=np.zeros(14, np.float32),
                    prompt="synthetic",
                    images={
                        k: np.zeros((3, 224, 224), np.uint8)
                        for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")
                    },
                )
                try:
                    with open_policy(uri, "checkpoint", 3) as remote:
                        reset_remote(remote, 123, "trial")
                        remote.infer(request)
                        self.assertEqual(model._rng, 124)
                    with open_policy(uri, "checkpoint", 3) as remote:
                        reset_remote(remote, 123, "other-window")
                        remote.infer(request)
                        self.assertEqual(model._rng, 124)
                    with open_policy(uri, "checkpoint", 3) as remote:
                        with self.assertRaisesRegex(RuntimeError, "reset required"):
                            remote.infer(request)
                finally:
                    server.shutdown()
                    worker.join(2)


class G2EPhysicalBudgetTests(unittest.TestCase):
    def test_physical_cap_rejects_before_next_scene_advance(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend, state, closed, physical = guarded_fixture(Path(tmp))
            backend.guard.max_physics_steps = 1
            cfg = dict(spec(31, "Aloha"), max_actions=31, execution_protocol=PROTOCOL)
            row = run_episode(
                cfg, backend, FixedPolicy(np.zeros((2, 6))), Path(tmp) / "episode"
            )
            self.assertEqual(backend.tick, 1)
            self.assertEqual(row["executed"], 1)
            self.assertEqual(row["outcome"], "timeout")

    def test_success_predicate_does_not_replace_physical_sequence_evidence(self):
        from g2e_statistics import physical_sequence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def state(tick, holders):
                return dict(
                    kind="native_guard_state",
                    tick=tick,
                    contact_state=dict(holders=holders),
                )

            events = [
                state(1, ["left"]),
                state(2, ["left", "right"]),
                state(3, ["right"]),
                state(4, []),
            ]
            (root / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
            self.assertFalse(physical_sequence(root)["complete"])
            events.insert(
                3,
                dict(
                    kind="native_contact_sample",
                    contacts=[
                        dict(
                            body_names=["box", "target_box"],
                            points=[dict(separation=0.0, impulse=[0, 0, 0.001])],
                        )
                    ],
                ),
            )
            (root / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
            self.assertTrue(physical_sequence(root)["complete"])
