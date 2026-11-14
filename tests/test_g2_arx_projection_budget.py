"""Exercise standalone ARX budget boundaries with CPU fixtures and subprocesses."""

from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import g2_standalone_deploy as standalone
from g2_adapter import Result, Snapshot, Stamp
from g2_final_campaign import supervise
from g2_final_deploy import commands, digest, verify
from g2_final_trial import classify
from g2_motion import Trajectory
from g2_remaining_episode import run
from g2_standalone_package import package_projection
from g2_task_runtime import validate_config
from g2_unified_plan import ARX_PROJECTION_ACTION_CAP, ARX_PROJECTION_EPISODE, ARX_PROJECTION_METHOD
from pace_runtime import CAMERAS
from robotwin_control_geometry import Status


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "research/g2/standalone-review/g2-standalone-arx.tar.gz"
EPISODE = ARX_PROJECTION_EPISODE
CONFIG = f"configs/{EPISODE}.json"


class Backend:
    def __init__(self):
        self.executed = 0
        self.held = 0
        self.closed = 0

    def reset(self, spec):
        return {}

    def success(self):
        return False

    def capture(self):
        stamp = Stamp(EPISODE, self.executed, self.executed * .004)
        rgb = np.zeros((3, 2, 2), dtype=np.uint8)
        return Snapshot(stamp, (), {name: (stamp, rgb) for name in CAMERAS}, np.eye(4)), None

    def execute(self, plan, log, remaining):
        remaining()
        count = len(plan.waypoint_ticks)
        self.executed += count
        return count

    def hold(self):
        self.held += 1

    def close(self, output):
        self.closed += 1


class ProjectionBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name).resolve()
        cls.archive = cls.root / "projection.tar.gz"
        cls.receipt = package_projection(ROOT, PARENT, cls.archive)
        with tarfile.open(cls.archive) as source:
            source.extractall(cls.root, filter="data")
        cls.release = cls.root / "g2-standalone-arx"
        cls.plan, cls.selected = standalone.verify_standalone(cls.release)
        cls.config = standalone.read(cls.release / CONFIG)

    def test_release_binds_only_projection_action_cap(self):
        self.assertEqual(self.config["max_actions"], 800)
        validate_config(self.config)
        with self.assertRaisesRegex(ValueError, "action cap"):
            validate_config(dict(self.config, max_actions=801))
        self.assertEqual(self.config["timeout_s"], 2550)
        self.assertEqual(self.selected["wall_s"], 5400)
        self.assertEqual(self.selected["hard_s"], 2600)
        self.assertEqual(self.selected["fee_cap"], "5.40")
        with tarfile.open(PARENT) as parent:
            def original(path):
                return parent.extractfile(f"g2-standalone-arx/{path}").read()

            source_plan = json.loads(original("unified-plan.json"))
            for row, before in zip(self.plan["trials"], source_plan["trials"], strict=True):
                if row["episode_id"] == EPISODE:
                    self.assertEqual(row, dict(before, max_policy_actions=800,
                                               task_tcp_projection=ARX_PROJECTION_METHOD))
                    self.assertEqual(row["recovery_cap"], 4)
                else:
                    self.assertEqual(row, before)
                    path = f"configs/{row['episode_id']}.json"
                    self.assertEqual((self.release / path).read_bytes(), original(path))
            self.assertEqual(self.config, dict(json.loads(original(CONFIG)),
                                               max_actions=800, task_tcp_projection=ARX_PROJECTION_METHOD))
            for path in ("scripts/pace_g2_standalone.slurm", "scripts/g2_standalone_deploy.py",
                         "scripts/g2_final_trial.py", "scripts/g2_final_campaign.py",
                         "scripts/g2_adapter.py", "scripts/g2_robotwin_backend.py", "scripts/g2e_model_entry.py",
                         "pace_download_manifest.json"):
                self.assertEqual((self.release / path).read_bytes(), original(path))
        for path in ("scripts/g2_remaining_coordinator.py", "scripts/g2_task_tcp_projection.py"):
            self.assertEqual((self.release / path).read_bytes(), (ROOT / path).read_bytes())
        subprocess.run([sys.executable, "-S", "-B", str(self.release / "scripts/g2_standalone_deploy.py"),
                        str(self.release), "--verify"], check=True, capture_output=True)
        rebuilt = self.root / "rebuilt.tar.gz"
        package_projection(ROOT, PARENT, rebuilt)
        self.assertEqual(digest(rebuilt), self.receipt["archive_sha256"])

    def test_release_rejects_mismatched_cap_even_with_updated_hashes(self):
        import shutil

        release = self.root / "mismatched"
        shutil.copytree(self.release, release)
        config = release / CONFIG
        config.write_text(json.dumps(dict(self.config, max_actions=300)))
        plan = deepcopy(self.plan)
        plan["trial_bindings"][EPISODE]["config_sha256"] = digest(config)
        (release / "unified-plan.json").write_text(json.dumps(plan))
        manifest = release / "SHA256SUMS"
        names = [line.split("  ", 1)[1] for line in manifest.read_text().splitlines()]
        manifest.write_text("".join(f"{digest(release / name)}  {name}\n" for name in names))
        with self.assertRaisesRegex(ValueError, "configuration differs from standalone roster"):
            verify(release)

    def trial(self, name, window=4, expire=False):
        backend = Backend()
        clock = SimpleNamespace(value=0.)
        spec = dict(episode_id=EPISODE, condition="arx-full-w4", data_kind="synthetic",
                    prompt=self.config["prompt"], window=window, max_actions=ARX_PROJECTION_ACTION_CAP,
                    timeout_s=self.config["timeout_s"], max_recoveries=4, approach_repair=True,
                    task_tcp_projection=ARX_PROJECTION_METHOD)

        def observe(snapshot, prompt, seconds):
            return Result(Status.OK, "", {}, dict(state=np.zeros(14), prompt=prompt,
                                                 images={k: image for k, (_, image) in snapshot.images.items()}))

        def plan(chunk, snapshot, count, seconds):
            shape = (count + 1, 2, 6)
            return Result(Status.OK, "", {}, Trajectory(.02, np.zeros(shape), np.zeros(shape),
                          np.zeros((count + 1, 2, 2)), tuple(range(1, count + 1))))

        def infer(request):
            if expire:
                clock.value = self.config["timeout_s"]
            return {"actions": np.zeros((50, 14))}

        adapter = SimpleNamespace(observe=observe, actions=plan)
        with patch("g2_remaining_episode.monotonic", lambda: clock.value), \
             patch("g2_bounded_approach.checked_prefix", lambda _, trajectory, *args: (trajectory, None)):
            result = run(spec, backend, adapter, SimpleNamespace(infer=infer), self.root / name)
        self.assertEqual((backend.held, backend.closed), (1, 1))
        self.assertEqual(backend.executed, result["executed"])
        return result

    def test_action_cap_completes_at_800_and_query_cap_stays_at_300(self):
        action_limited = self.trial("action-cap")
        query_limited = self.trial("query-cap", window=1)
        for result, expected in ((action_limited, (800, 200)), (query_limited, (300, 300))):
            self.assertEqual((result["executed"], result["queries"]), expected)
            self.assertEqual(result["outcome"], "task_unsuccessful")
            self.assertEqual(result["failure"]["code"], "action_budget_exhausted")
            self.assertEqual(classify(result), "diagnostic_complete")
        self.assertEqual(self.host("action-cap", result=action_limited)["outcome"], "task_unsuccessful")

    def host(self, name, *, result=None, hard_expire=False):
        root = self.root / f"host-{name}"
        root.mkdir()
        job = "845001"
        start = int(time.time()) - 1
        approval = root / "approval.json"
        receipt = dict(episode_id=EPISODE, category="admitted", physical_admitted=True,
                       fresh_reset_verified=True,
                       cleanup=dict(scene_opened=True, hold_complete=True, close_complete=True,
                                    scene_context_closed=True))
        command = [sys.executable, "-c"]
        noop = command + ["pass"]
        simulated = {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001", "sim_argv": ["cpu-test"]}
        model_hash = standalone.read(self.release / "launch-spec.json")["model_cpu_receipt_sha256"]
        with patch("g2_final_deploy.digest", return_value=model_hash):
            runtime = commands(root, self.release, root / "child", job, self.plan, simulated, 4900, approval)
        # Bind still reads the original URIs and deadlines. Substitute only its
        # worker processes, which run through the real host supervisor.
        def cpu_supervisor(bound_plan, bound_runtime, output, *, allocated):
            bound_runtime.update(
                model=command + ["import time; print('READY', flush=True); time.sleep(30)"],
                ready_marker="READY", model_timeout_s=5, model_phases_file=None,
                monitor=command + ["import time; print('x,g,1,24000', flush=True); time.sleep(30)"],
                read_result=lambda ep, probe=False: receipt if probe else dict(
                    receipt, category=classify(result), result=result, success=False),
            )
            key, = bound_runtime["probes"]
            bound_runtime["probes"] = {key: (EPISODE, noop)}
            bound_runtime["commands"] = {EPISODE: noop}
            if hard_expire:
                bound_runtime["commands"][EPISODE] = command + ["import time; time.sleep(5)"]
                bound_plan = deepcopy(bound_plan)
                bound_plan["trials"][0]["hard_timeout_s"] = .15
                budget = bound_plan["budget"]
                budget["bounded_execution_s"] = sum(budget["overhead"].values()) + .15
            return supervise(bound_plan, bound_runtime, output, allocated=allocated)

        auth = dict(job_id=job, standalone_mode="arx", allocation_s=5400, allocations_authorized=1)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"SLURM_JOB_ID": job, "SLURM_JOB_NAME": "cpf-pace-g2"}))
            stack.enter_context(patch.object(standalone, "authorization", return_value=auth))
            stack.enter_context(patch.object(standalone, "read_allocation", return_value=({}, start, start + 5400)))
            stack.enter_context(patch.object(standalone, "allocation_matches", return_value=True))
            stack.enter_context(patch.object(standalone, "disk_usage", return_value=SimpleNamespace(free=2**40)))
            stack.enter_context(patch.object(standalone, "run_deployment_stage"))
            stack.enter_context(patch.object(standalone, "read", return_value=simulated))
            stack.enter_context(patch.object(standalone, "commands", return_value=runtime))
            stack.enter_context(patch.object(standalone, "supervise", side_effect=cpu_supervisor))
            code = standalone.run(root, self.release, approval, self.plan, self.selected)
        output = standalone.output_for(root, self.selected, job)
        reported = standalone.read(output / "standalone-result.json")
        self.assertEqual(code, 0 if reported["outcome"] == "task_unsuccessful" else 1)
        self.assertNotIn("cleanup_failures", standalone.read(output / "supervisor/campaign.json"))
        return reported

    def test_soft_deadline_stops_and_is_resource_censored(self):
        result = self.trial("soft", expire=True)
        self.assertEqual((result["executed"], result["queries"]), (0, 1))
        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(result["failure"]["code"], "timeout")
        self.assertEqual(classify(result), "infrastructure_failure")
        self.assertEqual(self.host("soft", result=result)["outcome"], "RESOURCE_CENSORED")

    def test_hard_deadline_reaps_worker_and_is_resource_censored(self):
        result = self.host("hard", hard_expire=True)
        self.assertEqual(result["outcome"], "RESOURCE_CENSORED")
        self.assertEqual(result["row"]["reason"], "trial hard timeout")


if __name__ == "__main__":
    unittest.main()
