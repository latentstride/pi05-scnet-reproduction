"""Release regressions for a stopped contact failure and independent next trial."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from g2_adapter import G2Error, Snapshot, Stamp, read_snapshot
from g2_capture import capture_arms
from g2_contact import ContactWorld
from g2_episode import run_episode
from g2_final_campaign import supervise, verified_contact_stop
from g2_final_trial import RuntimeReset, classify
from g2_native_backend import SceneClock
from g2_remaining_plan import build
from g2e_native_guard import NativeGuard
from robotwin_control_geometry import Status
from test_g2_contact import contacts, tracker
from test_g2_integration import FixedPolicy, spec
from test_g2_native_backend import native_fixture


PROJECT = Path(__file__).resolve().parents[1]


class ContactStopTests(unittest.TestCase):
    def native_stop(
        self,
        *,
        blocked=False,
        close_fails=False,
        lose_bilateral=False,
        tracking_bad=False,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "episode"
            backend, state, closed, physical = native_fixture(root)
            original = backend.initial
            contact = tracker()
            stamp = Stamp("mock", 0, 0.0)
            snapshot = Snapshot(
                stamp, capture_arms(backend.env.robot, stamp), {}, backend.world_task
            )
            _, _, _, tools = read_snapshot(snapshot, backend.arms)
            for tick in range(3):
                contact.observe(
                    Stamp("seed", tick, tick * 0.004),
                    contacts([1, 2]),
                    tools,
                    np.eye(4),
                )
            # The synthetic guarded episode starts with the established grasp at tick zero.
            contact.last_stamp = Stamp("mock", 0, 0.0)
            physical_contacts = contacts([1] if lose_bilateral else [1, 2])
            for pair in physical_contacts:
                for body in pair.bodies:
                    body.entity.name = str(body.entity.per_scene_id)
                for point in pair.points:
                    point.position = np.zeros(3)
                    point.normal = np.array([1.0, 0.0, 0.0])
                    point.separation = 0.0
            physical.get_contacts = lambda: physical_contacts
            backend.env.box = SimpleNamespace(
                get_pose=lambda: SimpleNamespace(
                    p=[0.006 if backend.tick else 0, 0, 0], q=[1, 0, 0, 0]
                )
            )
            world = Mock(
                tracker=contact, last_collisions=["blocked"] if blocked else []
            )
            world.measured_clear.return_value = not blocked
            world.verify_contact_stop.side_effect = (
                lambda *args: ContactWorld.verify_contact_stop(world, *args)
            )
            guard = NativeGuard(world, backend.arms, 0.02, 0.03, 10)
            guard.candidate = Mock()
            if tracking_bad:
                guard.tracking = Mock(
                    side_effect=G2Error(
                        Status.BACKEND_ERROR,
                        "native tracking exceeds existing 30mm/10deg bounds",
                    )
                )
            backend.guard = guard
            backend.execution_protocol = guard.protocol
            backend.env.scene = SceneClock(
                physical, backend._stepped, backend._before_step
            )
            if close_fails:
                backend.env.close_env = Mock(side_effect=RuntimeError("close failed"))
            failures = []
            trial = RuntimeReset(
                backend, original, output, lambda *args: dict(fcl_clear=True), failures
            )
            # Keep the exact fixture readback when resetting; the synthetic slip starts only at tick one.
            result = run_episode(
                dict(spec(4, "Aloha"), execution_protocol=guard.protocol),
                trial,
                FixedPolicy(np.zeros((2, 6))),
                output,
            )
            if trial.contact_stop:
                trial.contact_stop["scene_context_closed"] = True
            self.assertEqual(backend.tick, 1)
            self.assertTrue(np.all(state["velocity"] == 0))
            self.assertEqual(world.measured_clear.call_count, 1, result)
            self.assertTrue(world.measured_clear.call_args.kwargs["check_unheld"])
            self.assertEqual(closed, [] if close_fails else [True])
            return result, trial.contact_stop, failures

    def test_slip_stops_on_first_tick_and_requires_completed_cleanup(self):
        for lose_bilateral in (False, True):
            with self.subTest(lose_bilateral=lose_bilateral):
                result, proof, failures = self.native_stop(
                    lose_bilateral=lose_bilateral
                )
                self.assertEqual(result["outcome"], "control_error")
                self.assertEqual(
                    proof["kind"], "bilateral_loss" if lose_bilateral else "slip"
                )
                self.assertEqual(result["executed"], 0)
                self.assertFalse(failures)
                self.assertEqual(classify(result, proof), "controlled_contact_stop")
                self.assertEqual(classify(result), "actual_safety_stop")

    def test_concurrent_measured_collision_or_failed_close_cannot_continue(self):
        result, proof, failures = self.native_stop(blocked=True)
        self.assertFalse(proof)
        self.assertFalse(failures)
        self.assertEqual(result["failure"]["code"], Status.COLLISION.value)
        self.assertEqual(classify(result, proof), "actual_safety_stop")
        result, proof, failures = self.native_stop(close_fails=True)
        self.assertTrue(failures)
        self.assertFalse(verified_contact_stop(proof))
        self.assertEqual(classify(result, proof), "actual_safety_stop")

    def test_unverified_fresh_reset_cannot_execute(self):
        backend = Mock()
        trial = RuntimeReset(backend, {}, Path("."), Mock(), [])
        with self.assertRaisesRegex(G2Error, "fresh admitted scene"):
            trial.execute()
        backend.execute.assert_not_called()

    def test_service_or_unsynced_error_cannot_use_contact_receipt(self):
        result, proof, _ = self.native_stop()
        self.assertEqual(
            classify(dict(result, outcome="policy_failed"), proof),
            "infrastructure_failure",
        )

    def test_slip_with_simultaneous_tracking_failure_stops_globally(self):
        result, proof, failures = self.native_stop(tracking_bad=True)
        self.assertFalse(proof)
        self.assertFalse(failures)
        self.assertEqual(classify(result, proof), "actual_safety_stop")
        result["failure"]["code"] = Status.UNSYNCED.value
        self.assertEqual(classify(result, proof), "actual_safety_stop")
        self.assertEqual(
            classify(
                dict(
                    outcome="planning_failed",
                    failure=dict(code="RuntimeError", detail="unknown failure"),
                )
            ),
            "infrastructure_failure",
        )

    def test_campaign_continues_aloha_slip_to_arx_and_expert_only_with_proof(self):
        result, proof, _ = self.native_stop()
        plan = build(PROJECT)
        roster = plan["trials"]
        native, arx = roster[0], roster[2]
        expert = next(
            r for r in roster if "expert" in r["method"] and r["seed"] == arx["seed"]
        )
        selected = (native, arx, expert)
        for row in roster:
            row["hard_timeout_s"] = 1
        plan["budget"]["bounded_execution_s"] = len(roster) + sum(
            plan["budget"]["overhead"].values()
        )
        command = [sys.executable, "-c", "pass"]
        ids = [r["episode_id"] for r in selected]
        for scenario in (
            "verified",
            "danger",
            "proof_missing",
            "group_alive",
            "next_reset_failed",
        ):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp:
                observed = []

                def receipt(episode, probe=False):
                    if probe:
                        return dict(episode_id=episode, category="admitted")
                    observed.append(episode)
                    data = dict(
                        episode_id=episode,
                        category="diagnostic_complete",
                        fresh_reset_verified=True,
                    )
                    if episode == ids[0]:
                        data.update(
                            category=classify(result, proof), contact_stop=proof
                        )
                        if scenario == "danger":
                            data.update(category="actual_safety_stop")
                        if scenario == "proof_missing":
                            data.update(contact_stop=dict(proof, hold_complete=False))
                    if episode == ids[1] and scenario == "next_reset_failed":
                        data.update(
                            category="infrastructure_failure",
                            fresh_reset_verified=False,
                        )
                    return data

                runtime = dict(
                    mode="allocated",
                    timeout_s=10,
                    admission={},
                    model_timeout_s=3,
                    ready_marker="READY",
                    model=[
                        sys.executable,
                        "-u",
                        "-c",
                        "import time; print('READY'); time.sleep(10)",
                    ],
                    monitor=[
                        sys.executable,
                        "-u",
                        "-c",
                        "import time; print('time, GPU-mock, 1, 24576'); time.sleep(10)",
                    ],
                    probes={
                        (r["embodiment"], r["seed"]): (r["episode_id"], command)
                        for r in (native, arx)
                    },
                    commands={episode: command for episode in ids},
                    read_result=receipt,
                )
                patch_group = patch(
                    "g2_final_campaign.process_group_gone", return_value=False
                )
                if scenario == "group_alive":
                    with patch_group:
                        summary = supervise(
                            plan, runtime, Path(temp) / "run", allocated=True
                        )
                else:
                    summary = supervise(
                        plan, runtime, Path(temp) / "run", allocated=True
                    )
                expected = (
                    ids
                    if scenario == "verified"
                    else ids[:2]
                    if scenario == "next_reset_failed"
                    else ids[:1]
                )
                self.assertEqual(observed, expected)
                self.assertEqual(summary["stop_reason"] is None, scenario == "verified")
                self.assertEqual(summary["model_server_starts"], 1)
