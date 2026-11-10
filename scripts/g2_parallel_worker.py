"""One job-bound trial using the pinned author rollout or the protected H50 path."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np

from g2_author_h50_limits import BUDGET, ResourceCensored, TrialClock
from g2_author_native_entry import RecordedPolicy, author_scene, choose_horizon
from g2_author_native_h50 import run_diagnostic
from g2_author_native_h50_service import request_hash, shared_connection
from g2_author_physx_observer import ObserveScene
from g2e_host_supervisor import interrupt_signals
from g2_final_deploy import read
from g2_locked_task import verify_model_receipt
from g2_parallel_deploy import bound_authorization
from g2_parallel_spec import digest, load
from pace_g1_orchestrate import persist
from pace_runtime import validate_request


class BoundedObserver(ObserveScene):
    """Add a resource stop and passive readbacks to the unchanged native step."""

    def __init__(self, env, stream, clock):
        self.clock = clock
        super().__init__(env.scene, env, stream, extended=True)

    def step(self, *args, **kwargs):
        self.clock.remaining()
        if self.tick >= self.clock.budget.physics_steps:
            raise ResourceCensored("physics_steps")
        return super().step(*args, **kwargs)


class RealCameraPolicy(RecordedPolicy):
    def infer(self, request):
        validate_request(request)
        if any(image.shape != (3, 240, 320) for image in request["images"].values()):
            raise ValueError("author camera shape differs")
        if any(np.ptp(image) == 0 for image in request["images"].values()):
            raise ValueError("an author camera returned a constant frame")
        self.log(
            dict(
                kind="request_identity",
                query=self.queries,
                request_sha256=request_hash(request),
            )
        )
        return super().infer(request)


def native(pace, release, spec, output, policy_factory, clock):
    output.mkdir()
    status = dict(
        mode=spec["mode"],
        method=spec["method"],
        scene_seed=spec["scene_seed"],
        episode_id=spec["episode_id"],
        success=False,
        scene_closed=False,
        outcome="INFRASTRUCTURE_FAILURE",
        trial_clock=clock.receipt(),
    )
    with (output / "events.jsonl").open("x", buffering=1) as events:

        def log(row):
            events.write(json.dumps(row, allow_nan=False) + "\n")

        try:
            clock.validate_entry()
            with author_scene(pace, release, scene=spec) as (env, rollout):
                choose, basis = choose_horizon(pace, spec)
                status.update(prompt=env.get_instruction(), selector=basis)
                log(dict(kind="reset", prompt=status["prompt"], selector=basis))
                with (output / "physics.jsonl").open("x", buffering=1) as stream:
                    observer = BoundedObserver(env, stream, clock)
                    env.scene = observer
                    take_action = env.take_action
                    completed = 0

                    def record_action(action):
                        nonlocal completed
                        tick = observer.tick
                        take_action(action)
                        completed += 1
                        log(
                            dict(
                                kind="action_complete",
                                action=completed,
                                tick=observer.tick,
                                physics_steps=observer.tick - tick,
                                gripper_request=action[[6, 13]].tolist(),
                            )
                        )

                    env.take_action = record_action
                    try:
                        with policy_factory() as model:
                            log(dict(kind="rng_reset", receipt=model.reset_receipt))
                            policy = RealCameraPolicy(model, output, log)
                            result = rollout(
                                env,
                                policy,
                                choose,
                                lambda step, actions, horizon: log(
                                    dict(
                                        kind="block",
                                        step=step,
                                        horizon=horizon,
                                        selected_grippers=actions[
                                            :horizon, [6, 13]
                                        ].tolist(),
                                    )
                                ),
                            )
                        if (
                            not result["success"]
                            and result["steps"] != spec["action_cap"]
                        ):
                            raise ValueError(
                                "native rollout returned before its action cap"
                            )
                        status.update(
                            result,
                            outcome=(
                                "SUCCESS" if result["success"] else "TASK_INCOMPLETE"
                            ),
                        )
                    except ResourceCensored as error:
                        status.update(
                            outcome="RESOURCE_CENSORED",
                            resource_reason=error.reason,
                            request_in_flight=error.request_in_flight,
                        )
                    finally:
                        env.take_action = take_action
                        env.scene = observer.original
                        status.update(
                            completed_actions=completed,
                            native_action_count=env.take_action_cnt,
                            physics_steps=observer.tick,
                        )
            status["scene_closed"] = True
        except (Exception, KeyboardInterrupt) as error:
            status.update(error=type(error).__name__, detail=str(error))
            if isinstance(error, ResourceCensored):
                status.update(outcome="RESOURCE_CENSORED", resource_reason=error.reason)
            else:
                status.update(outcome="INFRASTRUCTURE_FAILURE", success=False)
        finally:
            status["elapsed_s"] = monotonic() - clock.started
            log(dict(kind="terminal", **status))
            persist(output / "receipt.json", status)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("release", "pace", "output", "model-startup", "authorization"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--uri", required=True)
    p.add_argument("--started", type=float, required=True)
    a = p.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "")
    spec = load(a.release / "parallel-config.json")
    bound_authorization(a.release, a.authorization, job, spec, starting=False)
    if a.authorization.with_suffix(".used").read_text().strip() != job:
        raise PermissionError("worker authorization was not activated")
    plan = read(a.release / "unified-plan.json")
    verify_model_receipt(
        plan["checkpoint"], a.release / "pace_download_manifest.json", a.model_startup
    )
    r = spec["resources"]
    budget = replace(
        BUDGET,
        soft_s=r["trial_soft_s"],
        hard_s=r["trial_hard_s"],
        request_s=r["request_s"],
    )
    clock = TrialClock(budget, a.started, a.started + budget.hard_s)
    factory = shared_connection(
        a.uri, spec["episode_id"], clock.remaining, request_s=budget.request_s
    )
    with interrupt_signals():
        if spec["mode"] == "author_native":
            status = native(a.pace, a.release, spec, a.output, factory, clock)
            return int(status["outcome"] == "INFRASTRUCTURE_FAILURE")
        ref = a.release / "parallel-reset-reference.json"
        status = run_diagnostic(
            a.pace,
            a.release,
            ref,
            a.output,
            spec["episode_id"],
            factory,
            clock=clock,
            scene=spec,
            reference_sha256=digest(ref),
        )
        return int(status["category"] == "infrastructure_failure")


if __name__ == "__main__":
    raise SystemExit(main())
