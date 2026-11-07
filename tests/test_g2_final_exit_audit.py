"""CPU-only regression for terminal results and the locked campaign order."""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from g2_adapter import G2Error
from g2_episode import finish_episode
from g2_final_campaign import process_group_gone, supervise, verified_scene_cleanup
from g2_final_deploy import run_deployment_stage
from g2_final_expert import execute_expert
from g2_final_trial import RuntimeReset, classify, run as run_trial
from g2_locked_task import InitialMismatch
from g2_remaining_coordinator import RecoveryBudgetExhausted, RecoveryCoordinator
from g2_remaining_episode import run as run_recovery
from g2_remaining_geometry import bounded_progress, refined_pair
from g2_remaining_plan import build
from g2_remaining_router import JointRouter
from g2_task_runtime import RuntimeSession
from robotwin_control_geometry import Status

PROJECT = Path(__file__).resolve().parents[1]
HISTORY = PROJECT / "research/g2/final-cpu-review/result-127139405.json"


def terminal(outcome, code, *, phase="control", **diagnostic):
    return dict(
        outcome=outcome,
        success=outcome == "success",
        failure=dict(
            phase=phase, code=code, detail="CPU injection", diagnostic=diagnostic
        ),
    )


def cleanup():
    return dict(
        scene_opened=True,
        hold_complete=True,
        close_complete=True,
        scene_context_closed=True,
    )


def replay_history(output):
    """Use archived counts to exercise the current runner against an injected scene."""
    archived = json.loads(HISTORY.read_text())
    observed = archived["stop_trigger"]
    backend = Mock()
    backend.tick = 19360
    backend.success.return_value = False
    backend.close.return_value = None
    adapter = Mock()
    adapter.target_collision.tracker = SimpleNamespace(
        holders={}, releasing=set(), contact_fingers={}
    )
    adapter.observe.return_value = SimpleNamespace(
        ok=False,
        status=Status.COLLISION,
        detail="proxy blocked",
        diagnostic=dict(collision_world="source_collision"),
    )
    backend.capture.return_value = object(), None
    original = RecoveryCoordinator.__init__

    def at_observed_counts(coordinator, *args, **kwargs):
        original(coordinator, *args, **kwargs)
        coordinator.recoveries = observed["recovery_actions"]
        coordinator.recovery_actions = observed["recovery_actions"]
        coordinator.policy_actions = observed["policy_actions"]

    wrapped = RuntimeReset(backend, {}, output, Mock(), [])
    wrapped.reset = Mock(return_value={})
    session = RuntimeSession(Mock())

    @contextmanager
    def factory(spec):
        yield wrapped, adapter
        # Evidence of this CPU test context only.
        wrapped.cleanup["scene_context_closed"] = True

    session.factory = factory
    with patch.object(RecoveryCoordinator, "__init__", at_observed_counts):
        result = run_recovery(
            dict(
                data_kind="diagnostic",
                episode_id=observed["episode_id"],
                condition="cpu-replay",
                window=4,
                max_actions=300,
                max_recoveries=4,
                timeout_s=10,
                prompt="handover",
            ),
            session,
            adapter,
            Mock(),
            output,
        )
    return observed, result, backend, wrapped.cleanup


class ExitPathTests(unittest.TestCase):
    def test_deployment_stage_timeout_reaps_process_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "stage.log"
            command = [
                sys.executable,
                "-u",
                "-c",
                "import os,time; print(os.getpid()); time.sleep(10)",
            ]
            with self.assertRaises(subprocess.TimeoutExpired):
                run_deployment_stage(command, log, 0.5)
            self.assertTrue(process_group_gone(int(log.read_text().strip())))
            with self.assertRaisesRegex(RuntimeError, "stage process failed"):
                run_deployment_stage(
                    [sys.executable, "-c", "raise SystemExit(8)"], log, 2
                )

    def test_historical_four_receipts_cannot_authorize_continuation(self):
        archived = json.loads(HISTORY.read_text())
        trials = [row for row in archived["trials"] if row["status"] == "completed"]
        self.assertEqual(len(trials), 4)
        self.assertTrue(all(row["task_predicate"] is False for row in trials))
        self.assertEqual(
            [classify(row) for row in trials],
            [
                "actual_safety_stop",
                "actual_safety_stop",
                "diagnostic_complete",
                "infrastructure_failure",
            ],
        )
        self.assertTrue(all(not verified_scene_cleanup(row) for row in trials))

    def test_actual_collision_is_not_retried_as_a_planning_candidate(self):
        actual = G2Error(Status.COLLISION, "measured collision", geometry="actual")
        with patch("g2_remaining_geometry.solve_pair", side_effect=actual) as solve:
            with self.assertRaises(G2Error) as caught:
                refined_pair((), (), (), (), (), Mock(), Mock())
            self.assertIs(caught.exception, actual)
            solve.assert_called_once()
        motion = Mock()
        motion.plan.side_effect = actual
        router = JointRouter(motion, np.tile([-1, 1], (2, 6, 1)))
        with self.assertRaises(G2Error) as caught:
            router.plan(None, None, [], Mock(), Mock())
        self.assertIs(caught.exception, actual)
        motion.plan.assert_called_once()
        world = SimpleNamespace(tracker=None)
        with (
            patch(
                "g2_remaining_geometry.interpolate_tools", side_effect=lambda *args: []
            ),
            patch("g2_remaining_geometry.refined_pair", side_effect=actual) as refine,
        ):
            with self.assertRaises(G2Error) as caught:
                bounded_progress((), (), (), (), (), (), world, motion, Mock())
            self.assertIs(caught.exception, actual)
            refine.assert_called_once()

    def test_archived_recovery_cap_requires_structured_result_and_new_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            old, new, backend, proof = replay_history(Path(temp) / "episode")
        self.assertEqual(old["failure"]["detail"], "trial recovery cap exhausted")
        self.assertEqual((new["policy_actions"], new["recovery_actions"]), (156, 4))
        self.assertEqual(classify(old), "infrastructure_failure")
        self.assertEqual(new["outcome"], RecoveryBudgetExhausted.reason)
        self.assertEqual(new["failure"]["phase"], "observation")
        self.assertEqual(classify(new), "diagnostic_complete")
        self.assertEqual(proof, cleanup())
        backend.execute.assert_not_called()
        backend.hold.assert_called_once()
        backend.close.assert_called_once()
        for field, value in (
            ("reason", None),
            ("recoveries_used", 3),
            ("recoveries_used", True),
            ("recovery_limit", 5),
        ):
            forged = deepcopy(new)
            forged["failure"]["diagnostic"][field] = value
            self.assertEqual(classify(forged), "infrastructure_failure")

    def test_recovery_cap_and_contact_precedence(self):
        adapter = Mock()
        tracker = SimpleNamespace(holders={}, releasing=set(), contact_fingers={})
        adapter.target_collision.tracker = tracker
        with self.assertRaises(ValueError):
            RecoveryCoordinator(adapter, Mock(), max_recoveries=5)
        coordinator = RecoveryCoordinator(adapter, Mock())
        coordinator.recoveries = 3
        coordinator._free()
        coordinator.recoveries = 4
        with self.assertRaises(RecoveryBudgetExhausted):
            coordinator._free()
        for field, value in (
            ("holders", {"right": 1}),
            ("releasing", {"left"}),
            ("contact_fingers", {"left": {0}}),
        ):
            with self.subTest(field=field), patch.object(tracker, field, value):
                with self.assertRaises(G2Error) as exc:
                    coordinator._free()
                self.assertNotIsInstance(exc.exception, RecoveryBudgetExhausted)
                self.assertEqual(
                    classify(
                        terminal(
                            "planning_failed",
                            exc.exception.status.value,
                            **exc.exception.diagnostic,
                        )
                    ),
                    "actual_safety_stop",
                )

    def test_finish_attempts_close_even_when_hold_fails(self):
        backend = Mock()
        backend.hold.side_effect = KeyboardInterrupt("hold interrupted")
        backend.close.side_effect = OSError("close failed")
        failure = {"phase": "control", "code": "collision"}
        record = dict(outcome="control_error", failure=failure, success=False)
        event = Mock()
        finish_episode(backend, Path("."), record, event)
        backend.close.assert_called_once()
        self.assertIs(record["failure"], failure)
        self.assertEqual(
            [item["kind"] for item in record["cleanup_failures"]],
            ["hold_failed", "close_failed"],
        )
        self.assertEqual(classify(record), "infrastructure_failure")

    def test_session_exits_once_even_when_close_fails(self):
        context = Mock()
        backend = Mock()
        backend.close.side_effect = OSError("close failed")
        context.__enter__ = Mock(return_value=(backend, None))
        context.__exit__ = Mock(return_value=False)
        session = RuntimeSession(Mock(return_value=context))
        session.reset({})
        with self.assertRaises(OSError):
            session.close(Path("."))
        session.close(Path("."))
        context.__exit__.assert_called_once()
        backend.close.assert_called_once()
        with self.assertRaises(RuntimeError):
            session.reset({})

    def test_cleanup_must_not_advance_physics(self):
        backend = Mock(tick=5)

        def advance():
            backend.tick += 1

        backend.hold.side_effect = advance
        failures = []
        reset = RuntimeReset(backend, {}, Path("."), Mock(), failures)
        with self.assertRaisesRegex(RuntimeError, "advanced simulator time"):
            reset.hold()
        self.assertFalse(reset.cleanup["hold_complete"])
        self.assertTrue(failures)

    def test_classifier_exit_paths(self):
        for outcome, code, diagnostics, category in (
            ("success", "none", {}, "diagnostic_complete"),
            ("task_unsuccessful", "action_budget_exhausted", {}, "diagnostic_complete"),
            ("planning_failed", "collision", {}, "diagnostic_complete"),
            (
                "control_error",
                "collision",
                {"geometry": "candidate"},
                "diagnostic_complete",
            ),
            (
                "control_error",
                Status.LIMIT.value,
                {"geometry": "candidate"},
                "diagnostic_complete",
            ),
            ("planning_failed", "ik_failed", {}, "diagnostic_complete"),
            ("planning_failed", "unreachable", {}, "diagnostic_complete"),
            (
                "adapter_error",
                "collision",
                {"collision_world": "source_collision"},
                "diagnostic_complete",
            ),
            (
                "control_error",
                "backend_error",
                {"geometry": "candidate", "reason": "premature_release"},
                "diagnostic_complete",
            ),
            (
                "control_error",
                "collision",
                {"geometry": "actual"},
                "actual_safety_stop",
            ),
            (
                "control_error",
                "backend_error",
                {"geometry": "actual"},
                "actual_safety_stop",
            ),
            (
                "joint_limit",
                Status.LIMIT.value,
                {"geometry": "actual"},
                "actual_safety_stop",
            ),
            (
                "planning_failed",
                "unsynced",
                {"geometry": "candidate"},
                "actual_safety_stop",
            ),
            (
                "planning_failed",
                "timeout",
                {"geometry": "candidate"},
                "infrastructure_failure",
            ),
            ("timeout", "TimeoutError", {}, "infrastructure_failure"),
            ("interrupted", "KeyboardInterrupt", {}, "infrastructure_failure"),
            ("policy_failed", "ConnectionError", {}, "infrastructure_failure"),
            ("environment_error", "RuntimeError", {}, "infrastructure_failure"),
            ("export_failed", "OSError", {}, "infrastructure_failure"),
            ("planning_failed", "RuntimeError", {}, "infrastructure_failure"),
            ("control_error", "backend_error", {}, "actual_safety_stop"),
        ):
            with self.subTest(outcome=outcome, code=code, diagnostic=diagnostics):
                self.assertEqual(
                    classify(terminal(outcome, code, **diagnostics)), category
                )

    def test_expert_preserves_actual_provenance_and_requires_measured_handoff(self):
        backend, adapter = Mock(), Mock()
        backend.capture.return_value = object(), None
        adapter.begin_expert.return_value = SimpleNamespace(ok=True)
        adapter.actions.return_value = SimpleNamespace(
            ok=False,
            status=Status.COLLISION,
            detail="measured",
            diagnostic={"geometry": "actual"},
        )
        with self.assertRaises(G2Error) as caught:
            execute_expert(backend, adapter, [1], [[0] * 14], None, Mock(), lambda: 10)
        self.assertEqual(caught.exception.diagnostic["geometry"], "actual")
        backend.execute.assert_not_called()
        adapter.target_collision.tracker.holders = {}
        with self.assertRaises(G2Error) as caught:
            execute_expert(backend, adapter, [], [], None, Mock(), lambda: 10)
        self.assertEqual(caught.exception.status, Status.PLAN_FAILED)
        backend.execute.assert_not_called()


class TrialReceiptTests(unittest.TestCase):
    def test_expert_and_probe_receipts_require_real_context_exit(self):
        for kind, injected, expected in (
            ("expert", "task_failure", "diagnostic_complete"),
            ("expert", "invalid_trace", "infrastructure_failure"),
            ("expert", "hold", "infrastructure_failure"),
            ("expert", "close", "infrastructure_failure"),
            ("expert", "context", "infrastructure_failure"),
            ("probe", "initial_mismatch", "not_admitted"),
            ("probe", "actual_collision", "actual_safety_stop"),
            ("probe", "context", "infrastructure_failure"),
        ):
            with (
                self.subTest(kind=kind, injected=injected),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                (root / "configs").mkdir()
                (root / "configs/example.json").write_text('{"timeout_s":10}')
                (root / "source-audit.json").write_text("{}")
                (root / "approval.used").write_text("9000")
                initial = {"objects_task": {"target": []}}
                (root / "unified-plan.json").write_text(
                    json.dumps(
                        dict(
                            protected_existing_job="1",
                            checkpoint={},
                            trials=[
                                dict(
                                    episode_id="example",
                                    method="expert",
                                    embodiment="ARX-X5",
                                )
                            ],
                            trial_bindings={
                                "example": dict(
                                    expected_initial=initial,
                                    trace="trace",
                                    source="source",
                                )
                            },
                        )
                    )
                )
                backend = Mock(tick=0)
                if injected in ("hold", "close"):
                    getattr(backend, injected).side_effect = OSError(injected)
                if injected == "task_failure":
                    expert_error = G2Error(Status.PLAN_FAILED, "expert holder missing")
                else:
                    expert_error = None
                trace_error = (
                    ValueError("invalid trace") if injected == "invalid_trace" else None
                )

                def reset(gate, spec):
                    if injected == "initial_mismatch":
                        gate.status = "unmatched"
                        raise InitialMismatch("different reset")
                    if injected == "actual_collision":
                        raise G2Error(Status.COLLISION, "blocked", geometry="actual")
                    gate.status = "matched"
                    if kind == "probe":
                        raise RuntimeError("synthetic probe failure")

                @contextmanager
                def scene(*args):
                    yield backend, Mock()
                    if injected == "context":
                        raise OSError("scene context exit failed")

                args = SimpleNamespace(
                    release=root,
                    pace=root,
                    output=root / "out",
                    episode="example",
                    model_startup=root / "model.json",
                    uri="unused",
                    probe=kind == "probe",
                )
                with (
                    patch.dict(
                        "os.environ",
                        {
                            "SLURM_JOB_ID": "9000",
                            "SLURM_JOB_NAME": "cpf-pace-g2",
                            "G2_FINAL_AUTHORIZATION": str(root / "approval"),
                        },
                    ),
                    patch("g2_final_trial.authorization"),
                    patch("g2_final_trial.verify_runtime"),
                    patch("g2_final_trial.verify_model_receipt"),
                    patch("g2_final_trial.inspect_assets"),
                    patch("g2_final_trial.real_scene", side_effect=scene),
                    patch.object(RuntimeReset, "reset", reset),
                    patch(
                        "g2_final_trial.expert_prefix",
                        return_value=([], []),
                        side_effect=trace_error,
                    ),
                    patch(
                        "g2_final_trial.execute_expert",
                        return_value=False,
                        side_effect=expert_error,
                    ),
                ):
                    run_trial(args)
                receipt = json.loads((args.output / "receipt.json").read_text())
                self.assertEqual(receipt["category"], expected)
                backend.close.assert_called_once()
                if kind == "expert":
                    backend.hold.assert_called_once()
                if injected == "context":
                    self.assertFalse(receipt["cleanup"]["scene_context_closed"])
                if injected in ("hold", "close"):
                    self.assertTrue(receipt["cleanup_failures"])


class CampaignExitTests(unittest.TestCase):
    def simulate(self, root, fault=None, position=3):
        plan = build(PROJECT)
        rows = plan["trials"]
        ids = [row["episode_id"] for row in rows]
        probes = {}
        for row in rows:
            episode = row["episode_id"]
            probes.setdefault(
                (row["embodiment"], row["seed"]), (episode, ["probe", episode])
            )
        _, recovery, _, proof = replay_history(root / "cpu-replay")
        results = [
            terminal("control_error", "collision", geometry="candidate"),
            terminal("task_unsuccessful", "action_budget_exhausted"),
            terminal("planning_failed", "ik_failed", phase="planning"),
            recovery,
            terminal("success", "none"),
        ] + [terminal("task_unsuccessful", "episode_budget_or_expert_exhausted")] * 9
        processes = []
        clock = Mock()
        now = 0

        def sleep(seconds):
            nonlocal now
            now += seconds

        clock.monotonic.side_effect = lambda: now
        clock.sleep.side_effect = sleep

        def spawn(command, *, stdout, **kwargs):
            kind, _, episode = command[0].partition(":")
            if len(command) > 1:
                episode = command[1]
            pos = ids.index(episode) if kind == "trial" else -1
            started = now
            if kind == "monitor":
                stdout.write(b"time, CPU-double, 1, 24576\n")
            proc = SimpleNamespace(
                pid=len(processes) + 1, returncode=0, kind=kind, pos=pos
            )
            processes.append(proc)

            def poll():
                if kind in ("model", "monitor"):
                    active = processes[-1]
                    if (
                        fault == kind + "_exit"
                        and active.pos == position
                        and now > started
                    ):
                        return 1
                    return None
                if kind == "trial" and pos == position:
                    if fault == "hard_timeout":
                        return None
                    if fault == "exit_nonzero":
                        proc.returncode = 17
                return proc.returncode if now > started else None

            proc.poll = poll
            return proc

        def group_gone(pid):
            proc = processes[pid - 1]
            return not (
                fault == "live_group" and proc.kind == "trial" and proc.pos == position
            )

        def receipt(episode, probe=False):
            if probe:
                return dict(
                    episode_id=episode,
                    category="admitted",
                    fresh_reset_verified=True,
                    physical_admitted=True,
                    cleanup=cleanup(),
                )
            pos = ids.index(episode)
            result = deepcopy(results[pos])
            fields = dict(
                episode_id=episode,
                fresh_reset_verified=True,
                cleanup=dict(proof) if pos == 3 else cleanup(),
            )
            if result["success"]:
                fields["cleanup"]["hold_complete"] = False
            if pos == position:
                failures = {
                    "actual_collision": terminal(
                        "control_error", "collision", geometry="actual"
                    ),
                    "tracking": terminal(
                        "control_error", "backend_error", geometry="actual"
                    ),
                    "unsynced": terminal("control_error", "unsynced"),
                    "soft_timeout": terminal("timeout", "timeout"),
                    "unknown": terminal("planning_failed", "RuntimeError"),
                    "model_io": terminal("policy_failed", "ConnectionError"),
                    "unproven_slip": dict(
                        outcome="control_error",
                        success=False,
                        failure=dict(
                            code="backend_error",
                            detail="object slipped relative to confirmed holder",
                        ),
                    ),
                }
                result = failures.get(fault, result)
                if fault in ("hold_complete", "close_complete", "scene_context_closed"):
                    fields["cleanup"][fault] = False
                if fault == "cleanup_exception":
                    fields["cleanup_failures"] = ["failed close"]
                if fault == "fresh_reset":
                    fields["fresh_reset_verified"] = False
                if fault == "wrong_receipt":
                    fields["episode_id"] = "wrong"
                if fault == "missing_receipt":
                    raise FileNotFoundError("no receipt")
                if fault == "malformed_receipt":
                    return None
                if fault == "stop_flag":
                    fields["stop_campaign"] = True
                if fault == "cap_mismatch":
                    result["failure"]["diagnostic"].update(
                        recovery_limit=3, recoveries_used=3
                    )
            return dict(
                fields,
                category=classify(result),
                success=result["success"],
                result=result,
            )

        runtime = dict(
            mode="allocated",
            timeout_s=33600,
            model_timeout_s=960,
            ready_marker="READY",
            model=["model"],
            monitor=["monitor"],
            probes=probes,
            admission={},
            commands={episode: ["trial", episode] for episode in ids},
            read_result=receipt,
        )
        with (
            patch("g2_final_campaign.time", clock),
            patch("g2_final_campaign.subprocess.Popen", side_effect=spawn),
            patch("g2_final_campaign.wait_ready"),
            patch("g2_final_campaign.stop_group") as stop,
            patch("g2_final_campaign.process_group_gone", side_effect=group_gone),
        ):
            summary = supervise(plan, runtime, root / "campaign", allocated=True)
        launched = [ids[proc.pos] for proc in processes if proc.kind == "trial"]
        self.assertEqual(len(summary["startup"]), 4)
        self.assertEqual(summary["model_server_starts"], 1)
        self.assertEqual(
            {proc.pid for proc in processes},
            {call.args[0].pid for call in stop.call_args_list},
        )
        self.assertEqual(
            summary, json.loads((root / "campaign/campaign.json").read_text())
        )
        return ids, launched, summary

    def test_all_14_continue_after_fourth_recovery_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            ids, launched, summary = self.simulate(Path(temp))
        self.assertEqual(launched, ids)
        self.assertIsNone(summary["stop_reason"])
        self.assertEqual(
            summary["rows"][ids[3]]["result"]["result"]["outcome"],
            RecoveryBudgetExhausted.reason,
        )
        self.assertTrue(
            all(row["status"] == "completed" for row in summary["rows"].values())
        )
        for name in ("native-w26", "native-w4", "expert", "recovery", "baseline"):
            self.assertTrue(any(name in episode for episode in ids[4:]), name)

    def test_global_failures_never_launch_a_later_trial(self):
        for fault, position in (
            ("actual_collision", 0),
            ("tracking", 13),
            ("unsynced", 3),
            ("soft_timeout", 5),
            ("hard_timeout", 3),
            ("unknown", 13),
            ("model_io", 0),
            ("model_exit", 3),
            ("monitor_exit", 8),
            ("unproven_slip", 0),
            ("hold_complete", 3),
            ("close_complete", 3),
            ("scene_context_closed", 3),
            ("fresh_reset", 4),
            ("cleanup_exception", 3),
            ("live_group", 3),
            ("exit_nonzero", 10),
            ("wrong_receipt", 8),
            ("missing_receipt", 6),
            ("malformed_receipt", 11),
            ("stop_flag", 0),
            ("cap_mismatch", 3),
        ):
            with (
                self.subTest(fault=fault, position=position),
                tempfile.TemporaryDirectory() as temp,
            ):
                ids, launched, summary = self.simulate(Path(temp), fault, position)
                self.assertEqual(launched, ids[: position + 1])
                self.assertTrue(summary["stop_reason"])
                self.assertTrue(
                    all(
                        summary["rows"][episode]["status"] == "not_run"
                        for episode in ids[position + 1 :]
                    )
                )
