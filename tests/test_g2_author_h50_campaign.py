"""CPU integration through the real unified supervisor and author entry."""

import argparse
from contextlib import contextmanager, nullcontext
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from g2_author_h50_campaign import EPISODE, bind_runtime, build_overlay, review_record
from g2_author_h50_limits import BUDGET, ResourceCensored, TrialClock
from g2_final_campaign import supervise
from g2_final_deploy import commands, digest, verify
from g2_final_statistics import summarize

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    ROOT / "research/g2/candidate-stop-127192740/g2-candidate-stop-candidate.tar.gz"
)
ROLLOUT_SHA = "6ea198bcf47d6b6e62bd9206b6c5694466ca6346b30ad9c20885e4009af91325"


def author_worker(root, mode, clock):
    """Real author wrapper/rollout, with an injected simulator, model and clock."""
    import numpy as np

    from g2_adapter import G2Error, Stamp
    from g2_author_native_h50 import AuthorBackend, run_diagnostic
    from g2_native_backend import NativeBackend
    from robotwin_control_geometry import Status
    from test_g2_candidate_stop import fixture

    source = Path(os.environ["PACE_H50_ROLLOUT"])
    if digest(source) != ROLLOUT_SHA:
        raise ValueError("fixed upstream rollout differs")
    spec = spec_from_file_location("author_rollout", source)
    rollout = module_from_spec(spec)
    spec.loader.exec_module(rollout)
    now = [time.monotonic]

    @contextmanager
    def scene(_pace, _release, _reference, episode, log, trial_clock, **_):
        stub, _, closed, _ = fixture(root)
        env = stub.env
        env.box.actor = env.box
        env.scene = stub.actual_scene
        env.step_lim = 800
        env.eval_success = False
        env.check_success = lambda: env.take_action_cnt == {
            "success": 4,
            "wire": 51,
        }.get(mode, -1)
        env.get_obs = lambda: dict(
            joint_action={"vector": np.array([0.0] * 6 + [0.5] + [0.0] * 6 + [0.5])},
            observation={
                k: {"rgb": np.zeros((240, 320, 3), np.uint8)}
                for k in ("head_camera", "left_camera", "right_camera")
            },
        )
        guard = stub.guard
        guard.collision.tracker.last_stamp = None
        guard.max_physics_steps = trial_clock.budget.physics_steps
        backend = AuthorBackend(
            env, stub.arms, stub.initial, guard, episode, log, trial_clock.remaining
        )
        backend.grip_max_clip = 0.01
        backend.sample_initial(2500)
        original_step = backend.actual_scene.step
        original_measured = guard.measured

        def step():
            original_step()
            env.eval_success = env.check_success()
            if mode in ("soft", "danger_at_soft"):
                now[0] = lambda: trial_clock.soft

        def measured(*args):
            original_measured(*args)
            if mode == "danger_at_soft":
                raise G2Error(
                    Status.COLLISION, "injected measured danger", geometry="actual"
                )

        def candidate(*args):
            args[-1]()
            if mode in ("tick0", "missing", "close", "hold") or (
                mode == "mid" and backend.completed == 51
            ):
                if mode == "missing":
                    guard.collision.tracker.last_stamp = None
                raise G2Error(
                    Status.COLLISION, "injected candidate", geometry="candidate"
                )
            if mode == "unknown":
                raise RuntimeError("unknown injected failure")

        guard.candidate = candidate
        guard.measured = measured
        backend.actual_scene.step = step
        if mode == "physics":
            # Fast-forward the surrogate counter, not the real scene. The
            # separate PhysX cutoff acceptance uses an actual one-step budget.
            backend.tick = trial_clock.budget.physics_steps - 1
            guard.collision.tracker.last_stamp = Stamp(
                episode, backend.tick, backend.tick * backend.dt
            )
        if mode == "full800":
            now[0] = lambda: trial_clock.started + 1201
        if mode == "hold":
            backend.hold = lambda: (_ for _ in ()).throw(
                RuntimeError("injected hold failure")
            )
        try:
            yield backend, rollout.run_episode
        finally:
            env.close_env()
            assert closed == [True]
            if mode == "close":
                raise RuntimeError("injected close failure")

    class Policy:
        def infer(self, request):
            if mode == "inflight":
                now[0] = lambda: clock.soft
                raise ResourceCensored("soft_wall_clock", request_in_flight=True)
            return {"actions": np.tile(request["state"], (50, 1))}

    if mode == "wire":
        from g2_author_native_h50_service import shared_connection

        policy_factory = shared_connection(
            (root / "uri").read_text(), EPISODE, clock.remaining
        )
    else:

        def policy_factory():
            return nullcontext(Policy())

    with (
        patch("g2_author_native_h50.author_scene", scene),
        patch("g2_author_h50_limits.monotonic", lambda: now[0]()),
        patch("g2_author_native_h50.monotonic", lambda: now[0]()),
        # The stub collision object cannot project fingers. Retain the actual
        # author budget gate and measured-step/hold/receipt path.
        patch.object(
            NativeBackend,
            "_before_step",
            lambda backend: backend.guard.candidate(
                None, None, None, None, backend.remaining
            ),
        ),
    ):
        run_diagnostic(
            root,
            root,
            root,
            root / "author",
            EPISODE,
            policy_factory,
            clock=clock,
            configure_cpu=True,
        )


class AuthorCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name).resolve()
        with tarfile.open(ARCHIVE) as tar:
            tar.extractall(cls.root, filter="data")
        cls.base = cls.root / "g2-candidate-stop-release"
        cls.plan = verify(cls.base)

    def test_overlay_keeps_all_original_inputs_and_budget_ledgers(self):
        output = self.root / "overlay"
        review = build_overlay(ROOT, self.base, output)
        self.assertEqual(verify(output), self.plan)
        self.assertEqual(
            [row["remainder_before_term_s"] for row in review["ledgers"].values()],
            [3840, 1140, 2340],
        )
        from g2_author_h50_campaign import ADAPTED_SCRIPTS

        excluded = {"SHA256SUMS", *("scripts/" + n for n in ADAPTED_SCRIPTS)}
        for path in self.base.rglob("*"):
            rel = str(path.relative_to(self.base))
            if path.is_file() and rel not in excluded:
                self.assertEqual(digest(path), digest(output / rel), rel)
        # Real command builder, with only the host path/receipt mocked.
        with patch(
            "g2_final_deploy.digest",
            return_value=json.loads((output / "launch-spec.json").read_text())[
                "model_cpu_receipt_sha256"
            ],
        ):
            runtime = commands(
                self.root,
                output,
                self.root,
                "901",
                self.plan,
                dict(
                    gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
                    sim_argv=["sim"],
                ),
                24_000,
                self.root / "approval.json",
            )
        extended, bound = bind_runtime(
            self.plan, runtime, output, self.root, review=review
        )
        self.assertEqual(extended["trials"][:-1], self.plan["trials"])
        self.assertEqual(bound["probes"], runtime["probes"])
        self.assertEqual(
            {k: v for k, v in bound["commands"].items() if k != EPISODE},
            runtime["commands"],
        )
        self.assertIn(
            str(output / "scripts/g2_author_native_h50_service.py"), bound["model"]
        )
        self.assertIn(
            str(output / "scripts/g2_author_h50_trial.py"), bound["commands"][EPISODE]
        )
        with self.assertRaises(ValueError):
            bind_runtime(self.plan, runtime, output, self.root, review={})

    def test_host_can_verify_the_prior_tick_zero_receipt_without_runtime_dependencies(
        self,
    ):
        prior = ROOT / "research/g2/author-native-h50/cpu-physics-e2e.json"
        code = """
import json, sys
from pathlib import Path
from g2_author_h50_receipt import may_continue
receipt = json.loads(Path(sys.argv[1]).read_text())['cases']['tick-zero']['receipt']
assert may_continue(receipt, 'cpu-tick-zero', process_group_reaped=True, shared_service_alive=True)
assert not {'numpy', 'openpi_client', 'sapien'}.intersection(sys.modules)
"""
        subprocess.run(
            [sys.executable, "-S", "-B", "-c", code, str(prior)],
            env=dict(os.environ, PYTHONPATH=str(ROOT / "scripts")),
            check=True,
        )

    def test_clock_rejects_missing_mismatched_future_and_expired_limits(self):
        started = time.monotonic()
        with self.assertRaises(ValueError):
            TrialClock(BUDGET, started, started + 2699)
        with self.assertRaises(ValueError):
            TrialClock(BUDGET, started + 10, started + 2710).validate_entry()
        with self.assertRaises(ResourceCensored):
            TrialClock(BUDGET, started - 2500, started + 200).validate_entry()
        with self.assertRaises(TypeError):
            TrialClock(BUDGET, started)

    @unittest.skipUnless(
        os.environ.get("PACE_H50_ROLLOUT"),
        "set PACE_H50_ROLLOUT to pinned upstream rollout.py",
    )
    def test_supervisor_uses_one_real_websocket_for_legacy_and_author_children(self):
        summary, root = self.run_supervisor("wire")
        self.assertIsNone(summary["stop_reason"])
        self.assertEqual((root / "legacy-rng").read_text(), "legacy connection used")
        row = summary["rows"][EPISODE]
        self.assertEqual(row["result"]["query_action_counts"], [0, 50])
        records = [
            json.loads(line)
            for line in (root / "author/events.jsonl").read_text().splitlines()
        ]
        reset = next(
            r["receipt"] for r in records if r["kind"] == "author_model_rng_reset"
        )
        chunks = [r["rng"] for r in records if r.get("kind") == "author_chunk"]
        self.assertEqual(reset["key"]["words"], [0, 0])
        self.assertEqual(chunks[0]["after"], chunks[1]["before"])
        self.assertEqual([chunk["query"] for chunk in chunks], [0, 1])

    def run_supervisor(self, mode):
        root = self.root / mode
        root.mkdir()
        command = [sys.executable, "-c", "pass"]
        long_model = [
            sys.executable,
            "-u",
            "-c",
            "import time;print('READY');time.sleep(60)",
        ]
        runtime = dict(
            mode="allocated",
            timeout_s=24_000,
            model=long_model,
            ready_marker="READY",
            model_timeout_s=5,
            startup_cap_s=3600,
            admission={},
            monitor=[
                sys.executable,
                "-u",
                "-c",
                "import time;print('t,id,12,24000');time.sleep(60)",
            ],
            commands={row["episode_id"]: command for row in self.plan["trials"]},
        )
        ready = dict(
            category="admitted",
            fresh_reset_verified=True,
            physical_admitted=True,
            cleanup=dict(
                scene_opened=True, close_complete=True, scene_context_closed=True
            ),
        )
        legacy = dict(
            category="diagnostic_complete",
            fresh_reset_verified=True,
            success=False,
            cleanup=dict(ready["cleanup"], hold_complete=True),
        )
        legacy_keys = {(row["embodiment"], row["seed"]) for row in self.plan["trials"]}
        runtime["probes"] = {
            key: (f"probe-{index}", command)
            for index, key in enumerate(sorted(legacy_keys))
        }
        runtime["read_result"] = lambda episode, probe=False: dict(
            ready if probe else legacy, episode_id=episode
        )
        # Bind actual paths before substituting the process environments.
        release = self.base
        runtime["model"] = [str(release / "scripts/g2e_model_entry.py")]
        runtime["commands"]["unified-seed0-native-w50"] = [
            "python",
            str(release / "scripts/g2_final_trial.py"),
            "--release",
            str(release),
            "--pace",
            "fixture",
            "--uri",
            "fixture",
            "--model-startup",
            "fixture",
        ]
        plan, bound = bind_runtime(
            self.plan, runtime, release, root, review=review_record(self.plan)
        )
        bound["model"] = long_model
        bound["commands"]["unified-seed0-native-w50"] = command
        bound["commands"][EPISODE] = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            mode,
            "--root",
            str(root),
        ]
        read_legacy = bound["read_result"]
        bound["read_result"] = lambda episode, probe=False: (
            json.loads((root / "author/receipt.json").read_bytes())
            if episode == EPISODE
            else read_legacy(episode, probe=probe)
        )
        if mode == "wire":
            worker = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--root",
                str(root),
                "--worker",
            ]
            bound["model"] = worker + ["serve"]
            bound["commands"]["unified-seed0-native-w50"] = worker + ["legacy"]
        if mode == "service_dead":
            # The fake shared service exits after the author starts. Leave it
            # alive for all earlier probes and original trial processes.
            marker = root / "entered"
            bound["model"] = [
                sys.executable,
                "-u",
                "-c",
                "import pathlib,sys,time;print('READY');p=pathlib.Path(sys.argv[1]);"
                "exec('while not p.exists(): time.sleep(.01)')",
                str(marker),
            ]
        context = nullcontext()
        if mode == "hard":
            marker = root / "entered"
            context = patch(
                "g2_final_campaign.time",
                SimpleNamespace(
                    monotonic=lambda: time.monotonic()
                    + (2690.1 if marker.exists() else 0),
                    sleep=time.sleep,
                ),
            )
        with context:
            summary = supervise(plan, bound, root / "supervisor", allocated=True)
        self.assertEqual(summary["model_server_starts"], 1)
        self.assertTrue(
            all(
                summary["rows"][r["episode_id"]]["status"] == "completed"
                for r in self.plan["trials"]
            )
        )
        return summary, root

    @unittest.skipUnless(
        os.environ.get("PACE_H50_ROLLOUT"),
        "set PACE_H50_ROLLOUT to pinned upstream rollout.py",
    )
    def test_real_supervisor_and_author_rollout_positive_and_fail_closed(self):
        expected = {
            "full800": ("diagnostic_complete", "TASK_INCOMPLETE", 800),
            "success": ("diagnostic_complete", "SUCCESS", 4),
            "physics": ("resource_censored", "RESOURCE_CENSORED", 1),
            "soft": ("resource_censored", "RESOURCE_CENSORED", 0),
            "tick0": ("controlled_candidate_stop", "SAFETY_STOP", 0),
            "mid": ("controlled_candidate_stop", "SAFETY_STOP", 51),
            "missing": ("infrastructure_failure", "SAFETY_STOP", 0),
            "hold": ("infrastructure_failure", "SAFETY_STOP", 0),
            "close": ("infrastructure_failure", "SAFETY_STOP", 0),
            "danger_at_soft": ("actual_safety_stop", "SAFETY_STOP", 0),
            "inflight": ("infrastructure_failure", "RESOURCE_CENSORED", 0),
            "unknown": ("infrastructure_failure", None, 0),
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                summary, root = self.run_supervisor(mode)
                row = summary["rows"][EPISODE]
                receipt = row["result"]
                self.assertEqual(
                    (
                        receipt["category"],
                        receipt.get("outcome"),
                        receipt["completed_actions"],
                    ),
                    values,
                )
                good = mode in ("full800", "success", "physics", "soft", "tick0", "mid")
                self.assertEqual(row["status"] == "completed", good)
                self.assertEqual(summary["stop_reason"] is None, good)
                self.assertTrue(row["process_group_reaped"])
                report = summarize(self.plan, summary, root)["author_native_h50"]
                if mode == "full800":
                    self.assertEqual(
                        receipt["query_action_counts"], list(range(0, 800, 50))
                    )
                    self.assertIs(report["task_predicate"], False)
                elif mode == "success":
                    self.assertIs(report["task_predicate"], True)
                else:
                    self.assertIsNone(report["task_predicate"])
                if mode in ("physics", "soft"):
                    self.assertEqual(
                        receipt["resource_stop"]["tick"], receipt["physics_steps"]
                    )
        for mode, reason in (
            ("hard", "trial hard timeout"),
            ("service_dead", "shared model exited"),
        ):
            with self.subTest(mode=mode):
                summary, root = self.run_supervisor(mode)
                row = summary["rows"][EPISODE]
                self.assertEqual(summary["stop_reason"], reason)
                self.assertEqual(row["status"], "infrastructure_failure")
                self.assertTrue(row["process_group_reaped"])
                report = summarize(self.plan, summary, root)["author_native_h50"]
                self.assertIsNone(report["task_predicate"])
                if mode == "hard":
                    self.assertEqual(report["outcome"], "RESOURCE_CENSORED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trial-start", type=float)
    parser.add_argument("--hard-deadline", type=float)
    args = parser.parse_args()
    if args.worker == "serve":
        from websockets.sync.server import serve
        from g2_author_native_h50_service import handler
        from g2e_model_entry import reset_rng
        from test_g2_author_native_h50 import Policy, TestRng

        with patch(
            "g2e_model_entry.reset_rng",
            side_effect=lambda p, seed, factory=TestRng.key: reset_rng(
                p, seed, factory
            ),
        ):
            with serve(
                handler(Policy(), "checkpoint", jax_random=TestRng), "127.0.0.1", 0
            ) as server:
                (args.root / "uri").write_text(
                    f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
                )
                print("READY", flush=True)
                server.serve_forever()
    elif args.worker == "legacy":
        import numpy as np
        from g2e_model_entry import reset_remote
        from pace_runtime import CAMERAS, open_policy

        request = dict(
            state=np.zeros(14),
            prompt="legacy",
            images={k: np.zeros((3, 8, 8), np.uint8) for k in CAMERAS},
        )
        with open_policy((args.root / "uri").read_text(), "checkpoint", 2) as policy:
            reset_remote(policy, 123, "legacy")
            if "author_rng" in policy.infer(request):
                raise AssertionError("legacy response unexpectedly changed")
        (args.root / "legacy-rng").write_text("legacy connection used")
    elif args.worker in ("hard", "service_dead"):
        (args.root / "entered").touch()
        time.sleep(60)
    else:
        author_worker(
            args.root,
            args.worker,
            TrialClock(BUDGET, args.trial_start, args.hard_deadline),
        )
