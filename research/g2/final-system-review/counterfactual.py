"""Replay the exact job 127510335 chunk in isolated CPU processes."""

import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np

from g2_author_h50_limits import BUDGET, TrialClock
from g2_author_native_entry import (
    ArchivedFirstChunk,
    CounterfactualBoundary,
    EPISODE,
    author_scene,
    fixed_h50,
)
from g2_author_native_h50 import run_diagnostic
from g2_author_physx_observer import ObserveScene
from g2_author_native_h50_cpu import Headless


CHUNK_SHA = "e1ac8bd5119a1cb24e9761d301aa971ec6e11706eae715be468c5cc4babe14e8"
PROMPT = "Grab the red block using the left arm."


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("author_native", "protected"))
    for key in ("pace", "release", "chunk", "output"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU replay requires hidden CUDA devices")
    pace, release, output = (
        p.resolve() for p in (args.pace, args.release, args.output)
    )
    if sha256(args.chunk.read_bytes()).hexdigest() != CHUNK_SHA:
        raise ValueError("wrong archived GPU chunk")
    actions = np.load(args.chunk, allow_pickle=False)
    fixed_h50(actions)
    output.mkdir()
    policy = ArchivedFirstChunk(actions, PROMPT)
    headless = Headless(output)
    started = monotonic()
    observer = None
    result = dict(
        mode=args.mode, scene_seed=100000, chunk_sha256=CHUNK_SHA, model_inference=False
    )
    with (output / "physics.jsonl").open("x", buffering=1) as stream:

        def attach(env):
            nonlocal observer
            observer = ObserveScene(env.scene, env, stream)
            env.scene = observer

        try:
            if args.mode == "author_native":
                with author_scene(pace, release, configure_cpu=headless) as (
                    env,
                    rollout,
                ):
                    attach(env)
                    try:
                        result.update(
                            rollout(env, policy, fixed_h50), outcome="NATIVE_TERMINAL"
                        )
                    except CounterfactualBoundary:
                        result.update(
                            outcome="ARCHIVED_CHUNK_CONSUMED",
                            completed_actions=env.take_action_cnt,
                            native_success=bool(env.eval_success),
                        )
                    finally:
                        env.scene = observer.original
                result["scene_closed"] = True
            else:
                result["receipt"] = run_diagnostic(
                    pace,
                    release,
                    release / "author-h50-reset-reference.json",
                    output / "protected-trial",
                    EPISODE,
                    lambda: nullcontext(policy),
                    clock=TrialClock(BUDGET, started, started + BUDGET.hard_s),
                    configure_cpu=headless,
                    inject=lambda backend: attach(backend.env),
                )
                result["outcome"] = result["receipt"].get("outcome")
        except (Exception, KeyboardInterrupt) as error:
            result.update(
                outcome="REPLAY_ERROR", error=type(error).__name__, detail=str(error)
            )
            raise
        finally:
            result.update(
                physics_steps=observer.tick if observer else None,
                archived_model_responses=policy.requests,
                elapsed_s=monotonic() - started,
            )
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
