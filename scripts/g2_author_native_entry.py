"""One fixed PACE scene through the upstream evaluator and original environment."""

import argparse
from contextlib import contextmanager, nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys

import numpy as np


PACE = "fd2e624cc19325cc9b20a620c82d51c0d8e25cc1"
CACHE = "12ac49e3913308901113cc8ddd9f71f0987ce8c9ebd83f03c308c759bdbeb939"
MANIFEST = "835d60676236e69276032f7bc1ea4837eb6fcda6d4f3dbab2043965063a3d97b"
TASK = "handover_block"
EPISODE = "author-native-h50-seed100000"
MODE = "author_native"
SEED = 100000
WINDOW = 50
MAX_ACTIONS = 800


class CounterfactualBoundary(Exception):
    """No archived second prediction exists."""


@contextmanager
def author_scene(pace, release, *, configure_cpu=None, scene=None):
    from pace_runtime import require_checkout

    require_checkout(pace, PACE)
    previous_cwd, previous_path = Path.cwd(), sys.path[:]
    try:
        os.chdir(pace)
        sys.path[:0] = [
            str(pace),
            str(pace / "pace/src"),
            str(pace / "description/utils"),
        ]
        with (
            configure_cpu(pace, release / "asset-audit.json")
            if configure_cpu
            else nullcontext()
        ):
            from pace.robotwin import (
                load_cached_scenes,
                load_environment,
                restore_cached_scene_state,
                sample_instruction,
            )
            from pace.rollout import run_episode

            from g2_parallel_spec import cache_row, legacy_scene

            scene = legacy_scene() if scene is None else scene
            seed, index = scene["scene_seed"], scene["scene_cache_index"]
            cached, _ = cache_row(scene, load_cached_scenes)
            env, settings = load_environment(TASK)
            if settings["embodiment"] != ["aloha-agilex"]:
                raise ValueError("expected native Aloha")
            if configure_cpu:
                configure_cpu.settings(settings)
            try:
                random.seed(seed)
                env.setup_demo(now_ep_num=index, seed=seed, is_test=True, **settings)
                restore_cached_scene_state(TASK, env, cached["episode_info"])
                env.set_instruction(
                    sample_instruction(
                        TASK, {"info": cached["episode_info"]}, 100, "unseen"
                    )
                )
                if (
                    env.step_lim != MAX_ACTIONS
                    or env.take_action_cnt
                    or env.eval_success
                ):
                    raise ValueError("unexpected upstream reset or action budget")
                yield env, run_episode
            finally:
                if hasattr(env, "scene"):
                    env.close_env(
                        clear_cache=((index + 2) % settings["clear_cache_freq"] == 0)
                    )
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


def fixed_h50(actions):
    if actions.shape != (WINDOW, 14):
        raise ValueError("expected the paper's 50 x 14 policy output")
    return WINDOW


def choose_horizon(pace_root, spec):
    """Delegate adaptive selection to the exact author implementation."""
    from dataclasses import asdict
    from pace import PaceConfig, select_horizon

    if spec["method"] == "fixed_h50":
        return fixed_h50, {"selector": "fixed", "horizon": 50}
    config = PaceConfig()
    file = pace_root / "pace/configs/calibration/handover_block.json"
    raw = file.read_bytes()
    if sha256(raw).hexdigest() != spec["calibration_sha256"]:
        raise ValueError("author calibration differs")
    calibration = json.loads(raw)
    if calibration["task"] != TASK or calibration["selector"] != asdict(config):
        raise ValueError("calibration contract differs")
    threshold = calibration["threshold"]

    def choose(actions):
        fixed_h50(actions)
        return select_horizon(actions, threshold, max_horizon=50, config=config)

    return choose, {
        "selector": "pace",
        "threshold": threshold,
        "config": asdict(config),
        "calibration_sha256": spec["calibration_sha256"],
    }


class ArchivedFirstChunk:
    """CPU only. Refuse any model request beyond the one recorded response."""

    def __init__(self, actions, prompt):
        self.actions, self.prompt, self.requests = actions, prompt, 0

    def infer(self, request):
        if request["prompt"] != self.prompt:
            raise ValueError("replay instruction differs")
        if self.requests:
            raise CounterfactualBoundary
        self.requests += 1
        return {"actions": self.actions.copy()}


class RecordedPolicy:
    """Persist observed requests and full responses without modifying either."""

    def __init__(self, policy, output, log):
        self.policy, self.output, self.log, self.queries = policy, output, log, 0

    def infer(self, request):
        index = self.queries
        with (self.output / f"request-{index:02d}.npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                state=request["state"],
                prompt=request["prompt"],
                **request["images"],
            )
        self.log(dict(kind="request", query=index))
        from time import monotonic

        started = monotonic()
        response = self.policy.infer(request)
        actions = np.asarray(response["actions"])
        with (self.output / f"chunk-{index:02d}.npy").open("xb") as stream:
            np.save(stream, actions, allow_pickle=False)
        self.log(
            dict(
                kind="response",
                query=index,
                rng=response.get("author_rng"),
                chunk_sha256=sha256(
                    (self.output / f"chunk-{index:02d}.npy").read_bytes()
                ).hexdigest(),
                round_trip_s=monotonic() - started,
                service_inference_s=response.get("service_inference_s"),
            )
        )
        self.queries += 1
        return response


def authorized_budget(release, approval):
    from dataclasses import replace

    from g2_author_h50_limits import BUDGET
    from g2_final_deploy import authorization

    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise PermissionError("requires an existing dedicated simulation allocation")
    auth = authorization(release, approval, job)
    if auth.get("job_id") != job or auth.get("standalone_mode") != MODE:
        raise PermissionError("requires dedicated author_native authorization")
    budget = replace(
        BUDGET, soft_s=auth["author_trial_soft_s"], hard_s=auth["author_trial_hard_s"]
    )
    return job, budget, auth


def run_allocated(pace, release, output, uri, startup, approval):
    """Run only as a bounded, separately authorized process."""
    from time import monotonic

    from g2_author_h50_limits import ResourceCensored, TrialClock
    from g2_author_native_h50_service import shared_connection
    from g2_author_physx_observer import ObserveScene
    from g2_locked_task import verify_model_receipt
    from g2e_host_supervisor import interrupt_signals

    job, budget, _ = authorized_budget(release, approval)
    manifest = release / "pace_download_manifest.json"
    if sha256(manifest.read_bytes()).hexdigest() != MANIFEST:
        raise ValueError("frozen checkpoint manifest differs")
    plan = json.loads((release / "unified-plan.json").read_text())
    verify_model_receipt(plan["checkpoint"], manifest, startup)
    output.mkdir()
    started = monotonic()
    clock = TrialClock(budget, started, started + budget.hard_s)
    status = dict(
        mode=MODE, job_id=job, scene_seed=SEED, outcome="INFRASTRUCTURE_FAILURE"
    )
    with (output / "events.jsonl").open("x", buffering=1) as events:

        def log(row):
            events.write(json.dumps(row, allow_nan=False) + "\n")

        try:
            with interrupt_signals(), author_scene(pace, release) as (env, rollout):
                status.update(prompt=env.get_instruction())
                log(
                    dict(
                        kind="reset",
                        prompt=status["prompt"],
                        native_step_limit=env.step_lim,
                    )
                )
                with (output / "physics.jsonl").open("x", buffering=1) as physical:
                    observer = ObserveScene(env.scene, env, physical)
                    env.scene = observer
                    try:
                        with shared_connection(
                            uri,
                            EPISODE,
                            clock.remaining,
                            request_s=clock.budget.request_s,
                        ) as model:
                            log(dict(kind="rng_reset", receipt=model.reset_receipt))
                            policy = RecordedPolicy(model, output, log)
                            result = rollout(
                                env,
                                policy,
                                fixed_h50,
                                lambda step, actions, horizon: log(
                                    dict(kind="block", step=step, horizon=horizon)
                                ),
                            )
                    finally:
                        status.update(
                            physical_steps=observer.tick, steps=env.take_action_cnt
                        )
                        env.scene = observer.original
                if not result["success"] and result["steps"] != MAX_ACTIONS:
                    raise ValueError(
                        "upstream rollout stopped before the native action limit"
                    )
                status.update(
                    result,
                    outcome="SUCCESS" if result["success"] else "TASK_INCOMPLETE",
                )
            status["scene_closed"] = True
        except (Exception, KeyboardInterrupt) as error:
            status.update(
                outcome="RESOURCE_CENSORED"
                if isinstance(error, ResourceCensored)
                else "INFRASTRUCTURE_FAILURE",
                error=type(error).__name__,
                detail=str(error),
            )
            raise
        finally:
            status["elapsed_s"] = monotonic() - started
            log(dict(kind="terminal", **status))
            (output / "result.json").write_text(json.dumps(status, indent=2) + "\n")


def supervise_worker(release, output, approval):
    """Use the existing bounded process group launcher and retain host evidence."""
    from subprocess import TimeoutExpired
    from time import monotonic, time

    from g2_final_deploy import allocation_matches, run_deployment_stage, verify
    from g2_job_admission import read_allocation

    job, budget, auth = authorized_budget(release, approval)
    verify(release)
    if output.exists():
        raise FileExistsError(output)
    host = output.with_name(output.name + ".host")
    host.mkdir()
    status_path = host / "result.json"
    log_path = host / "worker.log"
    fields, start, end = read_allocation(job, host)
    if (
        not allocation_matches(fields)
        or end - start != auth["allocation_s"]
        or not start <= time() <= end - budget.hard_s - auth["host_cleanup_reserve_s"]
        or type(auth["host_cleanup_reserve_s"]) is not int
        or auth["host_cleanup_reserve_s"] < 120
    ):
        raise PermissionError("allocation does not cover the native worker and cleanup")
    started = monotonic()
    result = dict(
        mode=MODE, job_id=job, hard_s=budget.hard_s, outcome="INFRASTRUCTURE_FAILURE"
    )
    with status_path.open("x") as stream:
        try:
            run_deployment_stage(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    *sys.argv[1:],
                ],
                log_path,
                budget.hard_s - budget.reap_s,
            )
            result.update(outcome="WORKER_EXITED", process_group_reaped=True)
        except TimeoutExpired:
            result.update(
                outcome="RESOURCE_CENSORED",
                resource_reason="host_hard_wall",
                process_group_reaped=True,
            )
            raise
        finally:
            result["elapsed_s"] = monotonic() - started
            json.dump(result, stream, indent=2)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pace", "release", "output", "model-startup", "authorization"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = parser.parse_args()
    release, output, approval = (
        a.release.resolve(),
        a.output.resolve(),
        a.authorization.resolve(),
    )
    if a.worker:
        run_allocated(
            a.pace.resolve(),
            release,
            output,
            a.uri,
            a.model_startup.resolve(),
            approval,
        )
    else:
        supervise_worker(release, output, approval)


if __name__ == "__main__":
    main()
