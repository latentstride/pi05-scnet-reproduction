"""127192740 provenance, actual paused-state checks and original ten-row scheduling."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch, Mock
import json
import tempfile
import unittest
import numpy as np
from g2_adapter import G2Error, Stamp
from g2_candidate_stop import measure_stop, verified_candidate_stop
from g2_final_trial import RuntimeReset, classify
from g2_remaining_geometry import refined_pair, bounded_progress
from g2_candidate_schedule_cpu import simulate, proof_for
from g2_contact import ContactWorld
from g2e_native_guard import NativeGuard
from robotwin_control_geometry import Status
from test_g2_contact import tracker, contacts
from test_g2_native_backend import native_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "research/g2/candidate-stop-127192740/historical-fixture.json"


def historical():
    return json.loads(FIXTURE.read_text())["receipt"]["result"]


def inject_refinement():
    old = historical()["failure"]["diagnostic"]["attempts"]
    errors = [G2Error(Status.COLLISION, r["detail"], **r["diagnostic"]) for r in old]
    from g2_fixtures import robot_arms

    arms = robot_arms("aloha-agilex")
    with patch("g2_remaining_geometry.solve_pair", side_effect=errors):
        try:
            refined_pair(arms, None, None, None, None, None, lambda: 1)
        except G2Error as e:
            return e
    raise AssertionError("refinement must fail")


def fixture(root, fault=None):
    backend, state, closed, physical = native_fixture(root)
    backend.env.robot.left_entity.get_qvel = lambda: np.zeros(16)
    for i, j in enumerate(backend.env.robot.left_entity.get_active_joints()):
        j.get_drive_velocity_target = lambda i=i: np.array([state["velocity"][i]])
    world = Mock(tracker=tracker(), last_collisions=[])
    world.measured_clear.return_value = fault != "geometry"
    world.verify_contact_stop.side_effect = (
        lambda *args: ContactWorld.verify_contact_stop(world, *args)
    )
    guard = NativeGuard(world, backend.arms, 0.02, 0.03, 10)
    backend.guard = guard
    backend.episode = "cpu-candidate"
    backend.tick = 3
    stamp = Stamp(backend.episode, 3, 3 * backend.dt)
    physical.get_contacts = lambda: []
    world.tracker.observe(stamp, [], np.array([np.eye(4)] * 2), np.eye(4))
    if fault == "stale":
        world.tracker.last_stamp = Stamp(backend.episode, 2, 2 * backend.dt)
    if fault == "tracking":
        guard.tracking = Mock(
            side_effect=G2Error(Status.BACKEND_ERROR, "tracking", geometry="actual")
        )
    if fault == "close":
        backend.env.close_env = Mock(side_effect=RuntimeError("failed close"))
    if fault == "hold":
        backend.hold = Mock(side_effect=RuntimeError("failed hold"))
    if fault in ("hold_velocity", "hold_finger", "hold_base"):
        original = backend.hold

        def hold():
            original()
            if fault == "hold_velocity":
                state["velocity"][0] = 1
            else:
                state["target"][6] += 0.001

        backend.hold = hold
    return backend, state, closed, world


class CandidateStopTests(unittest.TestCase):
    def test_real_failure_refinement_preserves_candidate_provenance(self):
        old = historical()
        self.assertEqual(classify(old), "actual_safety_stop")
        error = inject_refinement()
        self.assertEqual(error.diagnostic["geometry"], "candidate")
        self.assertEqual(error.diagnostic["reason"], "exact_pose_refinements_exhausted")
        self.assertEqual(
            error.diagnostic["attempts"], old["failure"]["diagnostic"]["attempts"]
        )
        # Marking a candidate alone is insufficient to authorize continuation.
        result = deepcopy(old)
        result["failure"]["diagnostic"] = error.diagnostic
        self.assertEqual(classify(result), "actual_safety_stop")
        proof = proof_for(result, "cpu-candidate")
        self.assertEqual(
            classify(result, candidate_stop=proof), "controlled_candidate_stop"
        )
        self.assertEqual(classify(old, candidate_stop=proof), "actual_safety_stop")

    def test_bounded_wrapper_preserves_provenance_and_nested_actual_stops(self):
        from g2_fixtures import robot_arms

        arms = robot_arms("aloha-agilex")
        error = inject_refinement()
        with patch("g2_remaining_geometry.refined_pair", side_effect=error):
            with self.assertRaises(G2Error) as caught:
                bounded_progress(
                    arms,
                    np.array([np.eye(4)] * 2),
                    np.array([np.eye(4)] * 2),
                    np.zeros((2, 6)),
                    np.full((2, 2), 0.02),
                    np.full((2, 2), 0.02),
                    None,
                    None,
                    lambda: 1,
                )
        self.assertEqual(caught.exception.diagnostic["geometry"], "candidate")
        actual = G2Error(
            Status.COLLISION, "nested actual", attempts=[dict(geometry="actual")]
        )
        with patch("g2_remaining_geometry.solve_pair", side_effect=actual):
            with self.assertRaises(G2Error) as caught:
                refined_pair(arms, None, None, None, None, None, lambda: 1)
        self.assertIs(caught.exception, actual)

    def test_arx_retains_grip_drive_and_checks_stop_without_advancing(self):
        from types import SimpleNamespace as NS
        from test_g2_robotwin_backend import fake_backend
        from g2_adapter import Snapshot, read_snapshot
        from g2_capture import capture_arms

        for bad_hold in (False, True):
            with self.subTest(bad_hold=bad_hold), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                b, _, states, _, _ = fake_backend(root)
                b.episode, b.tick = "cpu-arx", 3
                for side, state in zip(("left", "right"), states):
                    entity = getattr(b.env.robot, side + "_entity")
                    entity.get_qvel = lambda: np.zeros(8)
                    for i, joint in enumerate(entity.get_active_joints()):
                        joint.get_drive_target = lambda i=i, state=state: np.array(
                            [state["target"][i]]
                        )
                        joint.get_drive_velocity_target = (
                            lambda i=i, state=state: np.array([state["speed"][i]])
                        )
                b.env.box = NS(get_pose=lambda: NS(p=np.zeros(3), q=[1, 0, 0, 0]))
                b.env.scene.get_contacts = lambda: contacts([1, 2])
                world = Mock(tracker=tracker(), last_collisions=[])
                world.measured_clear.return_value = True
                world.verify_contact_stop.side_effect = (
                    lambda *args: ContactWorld.verify_contact_stop(world, *args)
                )
                b.collision = world
                b.contact_drive = dict(mode=None)
                stamp = Stamp(b.episode, b.tick, b.tick * b.dt)
                _, _, _, tools = read_snapshot(
                    Snapshot(stamp, capture_arms(b.env.robot, stamp), {}, b.world_task),
                    b.arms,
                )
                world.tracker.holders = {"left": np.linalg.inv(tools[0])}
                world.tracker.observe(
                    stamp, b.env.scene.get_contacts(), tools, np.eye(4)
                )
                states[0]["target"][6:] = 0.001
                original_hold = b.hold
                if bad_hold:

                    def hold():
                        original_hold()
                        states[0]["target"][6:] = states[0]["actual"][6:]

                    b.hold = hold
                wrapped = RuntimeReset(b, {}, root, Mock(), [])
                result = historical()
                result["failure"]["diagnostic"]["geometry"] = "candidate"
                wrapped.note_failure(result)
                if bad_hold:
                    with self.assertRaises(G2Error):
                        wrapped.hold()
                else:
                    wrapped.hold()
                wrapped.close(root)
                if wrapped.candidate_stop:
                    wrapped.candidate_stop["scene_context_closed"] = True
                self.assertEqual(
                    verified_candidate_stop(wrapped.candidate_stop, b.episode, result),
                    not bad_hold,
                )
                self.assertEqual(b.tick, 3)
                self.assertNotIn("stepped", states[0])
                if not bad_hold:
                    np.testing.assert_array_equal(
                        states[0]["target"][6:], [0.001, 0.001]
                    )

    def test_actual_nested_failure_never_becomes_candidate(self):
        result = historical()
        result["failure"]["diagnostic"]["geometry"] = "candidate"
        result["failure"]["diagnostic"]["nested"] = {"geometry": "actual"}
        self.assertEqual(
            classify(result, candidate_stop=proof_for(result, "case")),
            "actual_safety_stop",
        )
        from g2_fixtures import robot_arms

        original = G2Error(Status.COLLISION, "actual", geometry="actual")
        with patch("g2_remaining_geometry.solve_pair", side_effect=original):
            with self.assertRaises(G2Error) as c:
                refined_pair(
                    robot_arms("aloha-agilex"), None, None, None, None, None, lambda: 1
                )
        self.assertIs(c.exception, original)

    def stop(self, root, fault=None):
        backend, state, closed, world = fixture(root, fault)
        wrapped = RuntimeReset(backend, {}, root, lambda *a: None, [])
        wrapped.status = "matched"
        error = inject_refinement()
        backend.window = 4
        backend.approach_repair = True
        backend._approach_row = Mock(side_effect=error)
        rows = np.zeros((4, 14))
        rows[:, [6, 13]] = 0.5
        with self.assertRaises(G2Error) as caught:
            wrapped.execute(rows, lambda row: None, lambda: 1)
        error = caught.exception
        with self.assertRaises(G2Error):
            wrapped.execute(rows, lambda row: None, lambda: 1)
        self.assertTrue(wrapped.candidate_rejected)
        try:
            wrapped.hold()
        except (G2Error, RuntimeError):
            pass
        try:
            wrapped.close(root)
        except RuntimeError:
            pass
        if fault != "context":
            wrapped.candidate_stop["scene_context_closed"] = True
        result = historical()
        result["failure"]["diagnostic"] = error.diagnostic
        return wrapped, result, backend, state, closed, world

    def test_live_check_hold_readback_and_cleanup_create_bound_proof_without_steps(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            w, result, b, state, closed, world = self.stop(Path(temp))
            self.assertTrue(
                verified_candidate_stop(w.candidate_stop, "cpu-candidate", result)
            )
            self.assertEqual(b.tick, 3)
            self.assertEqual(closed, [True])
            self.assertEqual(world.measured_clear.call_count, 2)
            self.assertTrue(
                all(
                    c.kwargs["check_unheld"]
                    for c in world.measured_clear.call_args_list
                )
            )
            self.assertEqual(
                classify(result, candidate_stop=w.candidate_stop),
                "controlled_candidate_stop",
            )
            self.assertFalse(w.cleanup_failures)

    def test_terminal_hold_close_or_context_failure_never_has_complete_proof(self):
        for fault in [
            "geometry",
            "tracking",
            "stale",
            "hold",
            "hold_velocity",
            "hold_finger",
            "hold_base",
            "close",
            "context",
        ]:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                w, result, b, state, closed, world = self.stop(Path(temp), fault)
                self.assertFalse(
                    verified_candidate_stop(w.candidate_stop, "cpu-candidate", result)
                )
                self.assertNotEqual(
                    classify(result, candidate_stop=w.candidate_stop),
                    "controlled_candidate_stop",
                )
                self.assertEqual(b.tick, 3)
                self.assertEqual(closed, [] if fault == "close" else [True])

    def test_same_tick_slip_is_rechecked_without_resetting_live_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            b, state, _, world = fixture(Path(temp))
            t = world.tracker
            t.holders = {"left": np.eye(4)}
            t.finger_body_ids = {"left": frozenset([1, 2]), "right": frozenset([3, 4])}
            b.actual_scene.get_contacts = lambda: contacts([1, 2])
            anchor = t.holders["left"].copy()
            stamp = t.last_stamp
            with self.assertRaises(G2Error):
                measure_stop(b)
            np.testing.assert_array_equal(anchor, t.holders["left"])
            self.assertEqual(stamp, t.last_stamp)

    def test_original_ten_rows_continue_only_after_proved_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            run = simulate(ROOT, Path(temp) / "run", historical())
            self.assertEqual(run["roster"], run["launched"])
            self.assertEqual(len(run["launched"]), 10)
            self.assertEqual(len(run["startup_admissions"]), 6)
            self.assertIsNone(run["summary"]["stop_reason"])
            self.assertEqual(run["summary"]["model_server_starts"], 1)
            self.assertTrue(
                all(r["status"] == "completed" for r in run["summary"]["rows"].values())
            )

    def test_faults_at_different_positions_stop_global_schedule(self):
        faults = [
            "actual",
            "nested_actual",
            "unknown",
            "terminal_checks_passed",
            "hold_verified",
            "no_physics_advance",
            "close_complete",
            "scene_context_closed",
            "missing_proof",
            "wrong_episode",
            "wrong_failure",
            "cleanup",
            "reset",
            "group_alive",
            "timeout",
            "service_exit",
            "forged_diagnostic",
            "forged_controlled_actual",
            "forged_controlled_unknown",
        ]
        for i, fault in enumerate(faults):
            pos = [0, 3, 6, 9][i % 4]
            with (
                self.subTest(fault=fault, position=pos),
                tempfile.TemporaryDirectory() as temp,
            ):
                run = simulate(ROOT, Path(temp) / "run", historical(), fault, pos)
                self.assertEqual(run["launched"], run["roster"][: pos + 1])
                self.assertIsNotNone(run["summary"]["stop_reason"])
