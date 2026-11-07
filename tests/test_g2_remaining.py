"""CPU acceptance for bounded rerouting, action accounting and physical recovery."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
from unittest.mock import patch
from g2_adapter import G2Error, Result
from g2_remaining_geometry import interpolate_tools, refined_pair
from g2_remaining_router import JointRouter
from g2_remaining_coordinator import RecoveryCoordinator
from g2_motion import QuinticMotion
from g2_fixtures import fixture, snapshot
from test_g2_integration import MockRobot, FixedPolicy
from robotwin_control_geometry import Status


def motion():
    return QuinticMotion(
        0.02,
        np.ones((2, 6)) * 0.5,
        np.ones((2, 6)),
        np.ones((2, 2)) * 0.03,
        np.ones((2, 2)) * 0.1,
        0.02,
        0.001,
        "explicit CPU test bounds",
        "synthetic obstacle",
        max_ticks=2500,
    )


class RoutingTests(unittest.TestCase):
    def test_real_detour_preserves_original_goal_and_policy_action_count(self):
        q = np.zeros((2, 6))
        q[0, 0] = -0.6
        end = q.copy()
        end[0, 0] = 0.6
        bounds = np.tile(np.array([-1.0, 1.0]), (2, 6, 1))
        g = np.zeros((2, 2))

        def clear(q, g):
            return not (abs(q[0, 0]) < 0.17 and abs(q[0, 1]) < 0.25)

        router = JointRouter(motion(), bounds, max_nodes=384)
        path = router.plan(q, g, [(end, g)], clear, lambda: 1)
        self.assertTrue(router.last["rerouted"])
        self.assertEqual(len(path.waypoint_ticks), 1)
        np.testing.assert_allclose(path.q[-1], end, atol=1e-12)
        self.assertTrue(all(clear(value, g) for value in path.q))
        self.assertLessEqual(float(abs(path.velocity).max()), 0.5 + 1e-9)
        self.assertLessEqual(
            float(abs(np.diff(path.velocity, axis=0) / path.period_s).max()), 1.0 + 1e-6
        )
        self.assertEqual(path.waypoint_ticks[-1], len(path.q) - 1)

    def test_colliding_endpoint_is_never_rerouted_into_acceptance(self):
        q = np.zeros((2, 6))
        end = q.copy()
        end[0, 0] = 0.5
        g = np.zeros((2, 2))
        router = JointRouter(motion(), np.tile([-1.0, 1.0], (2, 6, 1)))
        with self.assertRaisesRegex(G2Error, "endpoint collision"):
            router.route(q, end, g, lambda q, g: q[0, 0] < 0.4, lambda: 1)

    def test_holding_cannot_use_free_space_random_search(self):
        q = np.zeros((2, 6))
        end = q.copy()
        end[0, 0] = 0.5
        g = np.zeros((2, 2))

        class Blocked:
            tracker = NS(holders={"right": np.eye(4)}, releasing=set())

            def __call__(self, q, g):
                return q[0, 0] < 0.4

        router = JointRouter(motion(), np.tile([-1.0, 1.0], (2, 6, 1)))
        with self.assertRaisesRegex(G2Error, "holding"):
            router.plan(q, g, [(end, g)], Blocked(), lambda: 1)

    def test_budget_expiry_stops_route_without_physics(self):
        q = np.zeros((2, 6))
        end = q.copy()
        end[0, 0] = 0.5
        router = JointRouter(motion(), np.tile([-1.0, 1.0], (2, 6, 1)))

        def expired():
            raise G2Error(Status.TIMEOUT, "deadline")

        with self.assertRaisesRegex(G2Error, "deadline"):
            router.route(q, end, np.zeros((2, 2)), lambda q, g: True, expired)

    def test_pose_interpolation_is_finite_rigid_and_explicit(self):
        a = np.tile(np.eye(4), (2, 1, 1))
        b = a.copy()
        b[:, 0, 3] = 0.1
        np.testing.assert_allclose(interpolate_tools(a, b, 0.25)[:, 0, 3], 0.025)
        with self.assertRaises(ValueError):
            interpolate_tools(a, b, 1.1)


class CoordinatorTests(unittest.TestCase):
    def setup_pair(self):
        adapter, qa, qr, bases = fixture()
        backend = MockRobot(adapter, qa, qr, bases)
        backend.reset(dict(episode_id="remaining", condition="ARX"))

        # Geometry callable in the synthetic fixture has no refresh method.
        class World:
            def refresh(self):
                pass

            def __call__(self, q, g):
                return True

        adapter.target_collision = World()
        return adapter, backend, qa

    def test_complete_cycle_queries_only_real_mapped_state(self):
        adapter, backend, qa = self.setup_pair()
        runner = RecoveryCoordinator(adapter, backend)
        policy = FixedPolicy(qa)
        events = []
        row = runner.cycle(policy, "synthetic handover", 1, lambda: 5, events.append)
        self.assertEqual(len(policy.inputs), 1)
        self.assertEqual(row["policy_actions"], 1)
        self.assertEqual(row["recovery_actions"], 0)
        self.assertFalse(row["receipt"]["repaired"])

    def test_failed_proxy_never_queries_model_or_fabricates_state(self):
        adapter, backend, qa = self.setup_pair()
        runner = RecoveryCoordinator(adapter, backend, max_recoveries=0)
        adapter.observe = lambda *a: Result(
            Status.COLLISION,
            "proxy rejected",
            dict(collision_world="source_collision"),
            None,
        )
        policy = FixedPolicy(qa)
        with self.assertRaisesRegex(G2Error, "cap"):
            runner.cycle(policy, "synthetic", 1, lambda: 5, lambda e: None)
        self.assertEqual(len(policy.inputs), 0)
        self.assertEqual(backend.actions, 0)

    def test_source_recovery_requires_fresh_actual_observation_before_query(self):
        adapter, backend, qa = self.setup_pair()
        runner = RecoveryCoordinator(adapter, backend)
        real_observe = adapter.observe
        calls = []

        def observe(snap, prompt, budget):
            calls.append(snap.stamp.tick)
            if len(calls) == 1:
                return Result(
                    Status.COLLISION,
                    "source collision",
                    dict(collision_world="source_collision"),
                    None,
                )
            return real_observe(snap, prompt, budget)

        adapter.observe = observe
        with patch(
            "g2_remaining_coordinator.refined_pair",
            return_value=(backend.q.copy(), [], {}),
        ):
            row = runner.cycle(
                FixedPolicy(qa), "synthetic", 1, lambda: 5, lambda e: None
            )
        self.assertEqual(len(calls), 2)
        self.assertGreater(calls[1], calls[0])
        self.assertEqual(row["recovery_actions"], 1)
        self.assertEqual(row["policy_actions"], 1)
        self.assertEqual(backend.actions, 2)

    def test_later_prefix_collision_replans_only_previous_actions(self):
        adapter, backend, qa = self.setup_pair()
        runner = RecoveryCoordinator(adapter, backend)
        snap, _ = backend.capture()
        mapped = adapter.observe(snap, "synthetic", 5)
        self.assertTrue(mapped.ok)
        original = adapter.actions
        windows = []

        def actions(chunk, snapshot, window, budget):
            windows.append(window)
            if window == 4:
                return Result(
                    Status.COLLISION,
                    "row 2 blocked",
                    dict(
                        collision_world="target_collision",
                        requests=[dict(row=0), dict(row=1), dict(row=2)],
                    ),
                    None,
                )
            return original(chunk, snapshot, window, budget)

        adapter.actions = actions
        plan, receipt = runner.action_plan(
            FixedPolicy(qa).infer(mapped.payload)["actions"],
            snap,
            4,
            lambda: 5,
            lambda e: None,
        )
        self.assertEqual(windows, [4, 2])
        self.assertEqual(len(plan.waypoint_ticks), 2)
        self.assertTrue(receipt["receipt"]["requires_new_observation"])


if __name__ == "__main__":
    unittest.main()


class RemainingHarnessTests(unittest.TestCase):
    def test_full_persistence_cleanup_and_unchanged_model_contract(self):
        import tempfile, json
        from g2_remaining_episode import run

        adapter, qa, qr, bases = fixture()
        backend = MockRobot(adapter, qa, qr, bases)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(
                data_kind="synthetic",
                episode_id="persisted",
                condition="ARX",
                window=4,
                max_actions=8,
                timeout_s=10,
                prompt="synthetic",
            )
            row = run(cfg, backend, adapter, FixedPolicy(qa), Path(tmp) / "episode")
            self.assertTrue(row["success"])
            self.assertEqual(row["policy_actions"], 4)
            self.assertEqual(row["queries"], 1)
            from g2_export import episode_record
            from test_g2_export import identity

            exported = episode_record(Path(tmp) / "episode", identity(cfg))
            self.assertEqual(exported["metrics"]["actions"]["executed"], 4)
            self.assertEqual(exported["data_kind"], "synthetic")
            self.assertTrue(backend.closed)
            self.assertTrue((Path(tmp) / "episode/action-00001.npy").is_file())
            self.assertTrue((Path(tmp) / "episode/plan-00001.npz").is_file())
            self.assertEqual(
                json.loads((Path(tmp) / "episode/result.json").read_text())[
                    "data_kind"
                ],
                "synthetic",
            )

    def test_invalid_policy_output_is_retained_and_stops_before_motion(self):
        import tempfile
        from g2_remaining_episode import run

        adapter, qa, qr, bases = fixture()
        backend = MockRobot(adapter, qa, qr, bases)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(
                data_kind="synthetic",
                episode_id="invalid",
                condition="ARX",
                window=4,
                max_actions=8,
                timeout_s=10,
                prompt="synthetic",
            )
            row = run(
                cfg, backend, adapter, FixedPolicy(qa, "shape"), Path(tmp) / "episode"
            )
            self.assertEqual(row["outcome"], "policy_failed")
            self.assertEqual(row["policy_actions"], 0)
            self.assertEqual(backend.held, 1)
            self.assertTrue(backend.closed)
            self.assertTrue((Path(tmp) / "episode/action-00001.npy").is_file())

    def test_single_unified_campaign_is_complete_bounded_and_cannot_submit(self):
        from g2_remaining_plan import build

        root = Path(__file__).resolve().parents[1]
        plan = build(root)
        self.assertEqual(plan["protected_existing_job"], "127081074")
        self.assertEqual(plan["allocations_planned"], 1)
        self.assertEqual(plan["model_server_starts"], 1)
        self.assertEqual(len(plan["trials"]), 14)
        self.assertFalse(plan["submission_allowed"])
        self.assertEqual(plan["budget"]["bounded_execution_s"], 29700)
        self.assertEqual(plan["budget"]["proposed_fee_cap_yuan"], 34.2)
        self.assertTrue(plan["budget"]["budget_expansion_approved"])
        self.assertEqual(
            plan["budget"]["approved_remaining_campaign_budget_yuan"], 34.2
        )
        self.assertFalse(plan["budget"]["submission_authorized"])
        self.assertLessEqual(29700, plan["budget"]["proposed_allocation_s"])
        self.assertEqual(
            len([r for r in plan["trials"] if r["embodiment"] == "aloha-agilex"]), 6
        )
        self.assertEqual(
            len([r for r in plan["trials"] if "expert" in r["condition"]]), 2
        )
        for seed in (0, 1):
            rows = [r for r in plan["trials"] if r["seed"] == seed]
            self.assertEqual(len(set(r["reset_fixture_sha256"] for r in rows)), 1)
            self.assertEqual(len(set(r["prompt"] for r in rows)), 1)
            self.assertEqual(len(set(r["policy_rng_seed"] for r in rows)), 1)


class UnifiedConditionTests(unittest.TestCase):
    def test_all_policy_configs_bind_original_reset_and_validate(self):
        import json
        from g2_remaining_conditions import configure
        from g2_remaining_plan import build

        root = Path(__file__).resolve().parents[1]
        for row in build(root)["trials"]:
            if "expert" in row["method"]:
                continue
            old = f"seed{row['seed']}-" + (
                "A-native-w1" if row["embodiment"] == "aloha-agilex" else "B-arx-w1"
            )
            base = json.loads(
                (
                    root
                    / "research/g2/failure-audit-127054052-r1/release-context/configs"
                    / f"{old}.json"
                ).read_text()
            )
            result = configure(base, row)
            self.assertEqual(result["reset_fixture"], base["reset_fixture"])
            self.assertEqual(result["calibration"], base["calibration"])
            self.assertEqual(
                result["tracking_position_limit_m"], base["tracking_position_limit_m"]
            )
            self.assertEqual(result["prompt"], row["prompt"])
            self.assertEqual(result["window"], row["window"])

    def test_all_three_arx_runner_variants_persist_and_close(self):
        import json, tempfile
        from contextlib import contextmanager
        from g2_remaining_conditions import configure, run_condition
        from g2_remaining_plan import build

        root = Path(__file__).resolve().parents[1]
        plan = build(root)
        for method in (
            "frozen_exact_mapping",
            "bounded_recovery",
            "bounded_recovery_and_router",
        ):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                row = next(r for r in plan["trials"] if r["method"] == method)
                base = json.loads(
                    (
                        root
                        / "research/g2/failure-audit-127054052-r1/release-context/configs"
                        / f"seed{row['seed']}-B-arx-w1.json"
                    ).read_text()
                )
                config = configure(base, row)
                config["max_actions"] = 4
                config["timeout_s"] = 10
                adapter, qa, qr, bases = fixture()
                backend = MockRobot(adapter, qa, qr, bases)

                @contextmanager
                def factory(spec):
                    yield backend, adapter

                result = run_condition(
                    config, row, factory, FixedPolicy(qa), Path(tmp) / "trial"
                )
                self.assertEqual(result["outcome"], "success", result)
                self.assertTrue(backend.closed)
                self.assertEqual(result["executed"], 4)
