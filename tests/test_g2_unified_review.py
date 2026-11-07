"""CPU acceptance for the separate unified roster and native H50 opt-in."""

from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from g2_episode import run_episode
from g2_final_campaign import supervise
from g2_final_deploy import commands, verify
from g2_final_prepare_review import prepare
from g2_final_trial import classify
from g2_initial import digest_json
from g2_unified_exchange import CheckedFirstExchange, exchange_receipt
from g2_unified_plan import build, validate, validate_native_startup
from g2e_protocol import PROTOCOL, validate_window
from test_g2_integration import FixedPolicy, spec
from test_g2e import guarded_fixture

PROJECT = Path(__file__).resolve().parents[1]


class LateClosure(FixedPolicy):
    def infer(self, request):
        result = super().infer(request)
        result["actions"][39, 13] = 0
        return result


class UnifiedReviewTests(unittest.TestCase):
    def test_roster_builds_guarded_candidate_without_changing_legacy(self):
        plan = build(PROJECT)
        validate(plan)
        self.assertEqual(
            (len(plan["trials"]), plan["budget"]["bounded_execution_s"]), (10, 21000)
        )
        invalid = deepcopy(plan)
        invalid["trials"].reverse()
        with self.assertRaises(ValueError):
            validate(invalid)
        with self.assertRaises(ValueError):
            validate_window(50, PROTOCOL)
        with self.assertRaises(ValueError):
            validate_window(50, PROTOCOL, "ARX-X5", allow_native_h50=True)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / "release"
            prepare(PROJECT, path, unified=True)
            verified = verify(path)
            self.assertEqual(
                [r["episode_id"] for r in verified["trials"]],
                [r["episode_id"] for r in plan["trials"]],
            )
            # The verified model receipt lives on the cluster. Exercise argument
            # construction only, including the exact pinned receipt path.
            with patch(
                "g2_final_deploy.digest",
                return_value=json.loads((path / "launch-spec.json").read_text())[
                    "model_cpu_receipt_sha256"
                ],
            ):
                runtime = commands(
                    path,
                    path,
                    path / "results",
                    "9000",
                    verified,
                    {
                        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                        "sim_argv": ["simulator"],
                    },
                    25000,
                    path / "approval",
                )
            self.assertEqual(len(runtime["probes"]), 6)
            self.assertEqual(len(runtime["commands"]), 10)
            for row in verified["trials"]:
                command = runtime["commands"][row["episode_id"]]
                if row["embodiment"] == "aloha-agilex":
                    self.assertEqual(command[-2], "--expected-first-exchange")
                    self.assertIn(
                        f"seed{row['seed']}-native-w31/receipt.json", command[-1]
                    )
                else:
                    self.assertNotIn("--expected-first-exchange", command)

    def test_h50_executes_late_close_and_guards_each_step(self):
        for window, blocked_at, completed in (
            (31, None, 31),
            (50, None, 50),
            (50, 40, 39),
        ):
            with (
                self.subTest(window=window, blocked_at=blocked_at),
                tempfile.TemporaryDirectory() as temp,
            ):
                path = Path(temp)
                backend, state, closed, _ = guarded_fixture(path, blocked_at)
                options = dict(
                    spec(window, "Aloha"),
                    max_actions=window,
                    execution_protocol=PROTOCOL,
                )
                result = run_episode(
                    options,
                    backend,
                    LateClosure(np.zeros((2, 6))),
                    path / "episode",
                    allow_native_h50=True,
                )
                events = [
                    json.loads(line)
                    for line in (path / "episode/events.jsonl").read_text().splitlines()
                ]
                requests = [
                    row for row in events if row.get("kind") == "native_action_request"
                ]
                closed_commands = [
                    row["action_index_in_episode"]
                    for row in requests
                    if row["normalized_qpos_request"][13] < 0.5
                ]
                self.assertEqual(closed_commands, [39] if window == 50 else [])
                self.assertEqual(result["executed"], completed)
                self.assertEqual(result["queries"], 1)
                self.assertEqual(backend.guard.calls, blocked_at or completed)
                self.assertEqual(backend.tick, completed)
                self.assertEqual(closed, [True])

    def test_h50_late_envelope_and_physics_cap_are_fail_closed(self):
        class LateLimit(LateClosure):
            def infer(self, request):
                response = super().infer(request)
                response["actions"][39, 13] = 1.1
                return response

        for policy, expected, calls in (
            (LateLimit, "joint_limit", 0),
            (LateClosure, "timeout", 35),
        ):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                backend, _, closed, _ = guarded_fixture(path)
                backend.guard.max_physics_steps = 35
                options = dict(
                    spec(50, "Aloha"), max_actions=50, execution_protocol=PROTOCOL
                )
                result = run_episode(
                    options,
                    backend,
                    policy(np.zeros((2, 6))),
                    path / "episode",
                    allow_native_h50=True,
                )
                self.assertEqual(result["outcome"], expected)
                self.assertEqual(backend.tick, calls)
                self.assertEqual(closed, [True])

    def test_first_exchange_uses_content_and_blocks_motion_on_difference(self):
        request = dict(
            state=np.zeros(14, dtype=np.float32),
            prompt="task",
            images={
                name: np.zeros((3, 4, 4), np.uint8)
                for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
            },
        )
        reply = dict(actions=np.zeros((50, 14), np.float32), service_inference_s=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initial = root / "initial.json"
            initial.write_text('{"seed":0}')
            reference = dict(
                category="admitted",
                initial_sha256=digest_json({"seed": 0}),
                first_exchange=exchange_receipt(request, reply),
            )
            policy = Mock()
            policy.infer_raw.return_value = dict(reply, service_inference_s=9)
            checked = CheckedFirstExchange(
                policy, reference, root / "receipt.json", initial
            )
            checked.infer_raw(request)
            self.assertIsNone(checked.reference)
            for fault in ("image", "chunk", "initial"):
                with self.subTest(fault=fault):
                    changed = deepcopy(request)
                    answer = deepcopy(reply)
                    if fault == "image":
                        changed["images"]["cam_left_wrist"][0, 0, 0] = 1
                    if fault == "chunk":
                        answer["actions"][49, 13] = 1
                    initial.write_text(
                        '{"seed":1}' if fault == "initial" else '{"seed":0}'
                    )
                    policy.infer_raw.return_value = answer
                    checked = CheckedFirstExchange(
                        policy, reference, root / "receipt.json", initial
                    )
                    with self.assertRaisesRegex(ValueError, "paired first"):
                        checked.infer_raw(changed)


class UnifiedCampaignTests(unittest.TestCase):
    def rehearsal(self, root, fault):
        plan = build(PROJECT)
        ids = [row["episode_id"] for row in plan["trials"]]
        probes = {}
        for row in plan["trials"]:
            key = row["embodiment"], row["seed"]
            if key[0] == "aloha-agilex" and row["window"] == 4:
                continue
            if row["window"] == 50:
                key = (*key, 50)
            probes.setdefault(key, (row["episode_id"], ["probe", row["episode_id"]]))
        clock = SimpleNamespace(now=0.0)
        clock.monotonic = lambda: clock.now
        clock.sleep = lambda seconds: setattr(clock, "now", clock.now + seconds)
        launches = []

        def spawn(command, *, stdout, **kwargs):
            stage = command[0]
            now = clock.now
            launches.append(command)
            if stage == "monitor":
                stdout.write(b"time, CPU-fixture, 14500, 24576\n")
            if stage in ("model", "monitor"):

                def poll():
                    return (
                        1
                        if fault == "service_exit"
                        and len([c for c in launches if c[0] == "trial"]) >= 2
                        else None
                    )
            elif stage == "trial" and command[1] == ids[1] and fault == "hard_timeout":

                def poll():
                    return None
            else:

                def poll():
                    return 0 if clock.now > now else None

            return SimpleNamespace(pid=len(launches), poll=poll, returncode=0)

        def receipt(episode, probe=False):
            cleanup = dict(
                scene_opened=True,
                hold_complete=True,
                close_complete=True,
                scene_context_closed=True,
            )
            report = dict(
                episode_id=episode,
                fresh_reset_verified=True,
                cleanup=cleanup,
                success=False,
            )
            if probe:
                return dict(
                    report,
                    category="admitted",
                    physical_admitted=True,
                    initial_sha256="i"
                    if not (fault == "pair_mismatch" and episode == ids[0])
                    else "different",
                    first_exchange=dict(request_sha256="r", chunk_sha256="c"),
                )
            if fault == "budget_remaining" and episode == ids[0]:
                clock.now += 25000
            if episode == ids[1]:
                if fault in ("actual_safety_stop", "infrastructure_failure"):
                    return dict(report, category=fault)
                if fault == "cleanup":
                    cleanup["close_complete"] = False
                if fault == "reset":
                    report["fresh_reset_verified"] = False
            if episode == ids[0]:
                return dict(
                    report,
                    category="controlled_contact_stop",
                    contact_stop=dict(
                        kind="slip", tick=9, terminal_checks_passed=True, **cleanup
                    ),
                )
            if episode == ids[5]:
                result = dict(
                    outcome="recovery_budget_exhausted",
                    success=False,
                    failure=dict(
                        code="plan_failed",
                        phase="observation",
                        diagnostic=dict(
                            reason="recovery_budget_exhausted",
                            recoveries_used=4,
                            recovery_limit=4,
                        ),
                    ),
                )
                return dict(report, category=classify(result), result=result)
            return dict(
                report,
                category="diagnostic_complete",
                result=dict(
                    outcome="task_unsuccessful",
                    failure={"code": "action_budget_exhausted"},
                ),
            )

        runtime = dict(
            mode="allocated",
            timeout_s=25200,
            model_timeout_s=960,
            startup_cap_s=3600,
            model=["model"],
            monitor=["monitor"],
            ready_marker="READY",
            probes=probes,
            admission={},
            validate_startup=partial(validate_native_startup, probes=probes),
            commands={ep: ["trial", ep] for ep in ids},
            read_result=receipt,
        )
        with (
            patch("g2_final_campaign.time", clock),
            patch("g2_final_campaign.subprocess.Popen", side_effect=spawn),
            patch("g2_final_campaign.wait_ready"),
            patch("g2_final_campaign.stop_group"),
            patch(
                "g2_final_campaign.process_group_gone",
                side_effect=lambda pid: not (
                    fault == "group" and launches[pid - 1] == ["trial", ids[1]]
                ),
            ),
        ):
            result = supervise(plan, runtime, root / "run", allocated=True)
        return ids, [cmd[1] for cmd in launches if cmd[0] == "trial"], result

    def test_complete_roster_and_isolated_native_pair_rejection(self):
        for fault in (None, "pair_mismatch"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                ids, launched, result = self.rehearsal(Path(temp), fault)
                self.assertEqual(len(result["startup"]), 6)
                self.assertEqual(result["model_server_starts"], 1)
                self.assertEqual(
                    launched,
                    ids
                    if fault is None
                    else [ep for ep in ids if "seed0-native" not in ep],
                )
                self.assertIsNone(result["stop_reason"])

    def test_unsafe_or_unproven_completion_stops_immediately(self):
        for fault in (
            "actual_safety_stop",
            "infrastructure_failure",
            "service_exit",
            "hard_timeout",
            "cleanup",
            "reset",
            "group",
            "budget_remaining",
        ):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                ids, launched, result = self.rehearsal(Path(temp), fault)
                count = 1 if fault == "budget_remaining" else 2
                self.assertEqual(launched, ids[:count])
                self.assertTrue(result["stop_reason"])
                self.assertTrue(
                    all(result["rows"][ep]["status"] == "not_run" for ep in ids[count:])
                )
