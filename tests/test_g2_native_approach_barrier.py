"""Corrected native prefixes must reobserve and query; stale tails are never reused."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from g2_adapter import G2Error
from g2_episode import run_episode
from g2e_protocol import APPROACH_REPAIR, PROTOCOL, NativeApproachBarrier
from test_g2e import guarded_fixture
from test_g2_integration import spec, FixedPolicy


def enable(backend):
    backend.approach_repair = True

    def correction(row, remaining, log):
        value = row.copy()
        if backend.env.take_action_cnt == 0:
            value[0] += 0.001
        return value

    backend._approach_row = correction


def specification():
    return dict(
        spec(4, "Aloha"),
        max_actions=4,
        execution_protocol=PROTOCOL,
        free_space_approach_repair=APPROACH_REPAIR,
    )


class NativeApproachBarrierTests(unittest.TestCase):
    def test_partial_prefix_queries_again_on_new_measured_state(self):
        with tempfile.TemporaryDirectory() as temp:
            backend, state, closed, _ = guarded_fixture(Path(temp))
            enable(backend)
            policy = FixedPolicy(np.zeros((2, 6)))
            measured_requests = []
            infer = policy.infer

            def infer_measured(request):
                measured_requests.append(request["state"].copy())
                return infer(request)

            policy.infer = infer_measured
            result = run_episode(
                specification(), backend, policy, Path(temp) / "episode"
            )
            self.assertEqual(result["outcome"], "task_unsuccessful", result)
            self.assertEqual((result["executed"], result["queries"]), (4, 2))
            self.assertEqual(backend.tick, 4)
            self.assertAlmostEqual(measured_requests[1][0], 0.001)
            self.assertEqual(closed, [True])
            text = (Path(temp) / "episode/events.jsonl").read_text()
            self.assertIn("bounded_approach_requires_new_observation", text)
            self.assertIn('"discarded_actions": 3', text)

    def test_uncaptured_backend_cannot_execute_an_old_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            backend, _, _, _ = guarded_fixture(Path(temp))
            enable(backend)
            backend.reset(specification())
            rows = np.zeros((4, 14))
            rows[:, [6, 13]] = 0.5
            self.assertEqual(backend.execute(rows, lambda r: None, lambda: 1), 1)
            reason = backend.take_replan()
            self.assertIsInstance(reason, NativeApproachBarrier)
            self.assertEqual((reason.tick, reason.consumed_actions), (1, 1))
            with self.assertRaises(G2Error) as caught:
                backend.execute(rows, lambda r: None, lambda: 1)
            self.assertEqual(caught.exception.status.value, "unsynced")
            self.assertEqual(backend.tick, 1)
            backend.hold()
            backend.close(Path(temp))

    def test_stale_post_barrier_observation_stops_before_next_query(self):
        with tempfile.TemporaryDirectory() as temp:
            backend, _, closed, _ = guarded_fixture(Path(temp))
            enable(backend)
            capture = backend.capture

            def stale():
                snap, native = capture()
                if backend.tick:
                    snap = replace(snap, stamp=replace(snap.stamp, tick=0, sim_s=0.0))
                return snap, native

            backend.capture = stale
            policy = FixedPolicy(np.zeros((2, 6)))
            result = run_episode(
                specification(), backend, policy, Path(temp) / "episode"
            )
            self.assertEqual(result["failure"]["code"], "unsynced")
            self.assertEqual((result["executed"], result["queries"]), (1, 1))
            self.assertEqual(backend.tick, 1)
            self.assertEqual(closed, [True])

    def test_arbitrary_or_unapproved_partial_receipt_never_continues(self):
        for missing_profile in (False, True):
            with (
                self.subTest(missing_profile=missing_profile),
                tempfile.TemporaryDirectory() as temp,
            ):
                backend, _, _, _ = guarded_fixture(Path(temp))
                enable(backend)
                cfg = specification()
                if missing_profile:
                    cfg.pop("free_space_approach_repair")
                else:
                    backend.take_replan = lambda: dict(reason="native approach")
                policy = FixedPolicy(np.zeros((2, 6)))
                result = run_episode(cfg, backend, policy, Path(temp) / "episode")
                self.assertIn(result["failure"]["code"], ("unsynced", "backend_error"))
                self.assertEqual(result["queries"], 1)
                self.assertEqual(backend.tick, 1)


if __name__ == "__main__":
    unittest.main()
