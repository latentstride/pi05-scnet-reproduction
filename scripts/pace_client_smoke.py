#!/usr/bin/env python3
"""Run a bounded single-scene handover_block smoke and record diagnostics."""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack, contextmanager
import json
import math
import os
from pathlib import Path
import signal
import sys
from time import monotonic

import numpy as np

from pace_runtime import CAMERAS, PACE_REVISION, TASK, open_policy, require_checkout, validate_response

PROMPT = "Hand the block to the other arm and place it on the target."
CAMERA_SOURCES = ("head_camera", "left_camera", "right_camera")


def encode(observation, prompt):
    cameras = observation["observation"]
    return {
        "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        "images": {
            target: cameras[source]["rgb"].transpose(2, 0, 1)
            for target, source in zip(CAMERAS, CAMERA_SOURCES, strict=True)
        },
        "prompt": prompt,
    }


class SyntheticEnv:
    """Exercise the client loop and artifacts with deterministic CPU observations."""

    take_action_cnt = 0
    eval_success = False
    step_lim = 20

    def __init__(self):
        self.state = np.zeros(14, dtype=np.float32)

    def get_obs(self):
        image = np.full((64, 64, 3), self.take_action_cnt * 10, dtype=np.uint8)
        return {
            "observation": {name: {"rgb": image} for name in CAMERA_SOURCES},
            "joint_action": {"vector": self.state},
        }

    def take_action(self, action, action_type):
        if action_type != "qpos":
            raise ValueError("expected qpos targets")
        self.state = action.copy()
        self.take_action_cnt += 1


@contextmanager
def environment(pace_root, summary):
    if pace_root is None:
        yield SyntheticEnv(), PROMPT
        return

    require_checkout(pace_root, PACE_REVISION)
    os.chdir(pace_root)  # RoboTwin resolves assets relative to the workspace.
    sys.path[:0] = [str(pace_root), str(pace_root / "pace/src")]
    from pace.robotwin import load_cached_scenes, load_environment, restore_cached_scene_state

    scenes, cache = load_cached_scenes(TASK, seed=0, count=1)
    scene = scenes[0]
    summary.update(scene_seed=scene["scene_seed"], scene_cache=cache, pace_revision=PACE_REVISION)
    env, settings = load_environment(TASK)
    if settings["embodiment"] != ["aloha-agilex"] or not settings["dual_arm_embodied"]:
        raise ValueError("expected native dual-arm Aloha")
    settings["collect_data"] = False
    try:
        env.setup_demo(now_ep_num=0, seed=scene["scene_seed"], is_test=True, **settings)
        restore_cached_scene_state(TASK, env, scene["episode_info"])
        env.set_instruction(PROMPT)
        yield env, env.get_instruction()
    finally:
        env.close_env(clear_cache=True)


class Video:
    def __init__(self, path):
        self.path = path
        self.writer = None

    def frame(self, observation):
        image = np.ascontiguousarray(observation["observation"]["head_camera"]["rgb"])
        if self.writer is None:
            from imageio_ffmpeg import write_frames

            height, width = image.shape[:2]
            self.writer = write_frames(str(self.path), (width, height), fps=10,
                                       codec="libx264", macro_block_size=2,
                                       ffmpeg_log_level="error")
            self.writer.send(None)
        self.writer.send(image)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            if not self.path.is_file() or self.path.stat().st_size == 0:
                raise RuntimeError("MP4 encoder produced no video")


def run(env, policy, prompt, args, summary, step_log, query_log, video):
    limit = min(args.max_steps, env.step_lim)
    obs = env.get_obs()
    video.frame(obs)
    executed = 0
    while executed < limit and not env.eval_success and summary["queries_attempted"] < args.max_queries:
        summary["queries_attempted"] += 1
        query = summary["queries_attempted"]
        start = monotonic()
        try:
            chunk = validate_response(policy.infer(encode(obs, prompt)))
        except Exception:
            query_log.writerow([query, executed, monotonic() - start, "", 0, "error"])
            raise

        inference_s = monotonic() - start
        name = f"chunk_{query:03d}.npy"
        count = min(args.prefix, limit - executed)
        used = 0
        status = "error"
        try:
            np.save(args.output / name, chunk, allow_pickle=False)
            # Reuse this chunk prefix locally. The remaining suffix expires at replan.
            for index, action in enumerate(chunk[:count]):
                before = env.take_action_cnt
                env.take_action(action.copy(), action_type="qpos")
                if env.take_action_cnt != before + 1:
                    raise RuntimeError("RoboTwin did not advance by one control action")
                executed += 1
                used += 1
                summary["control_steps"] = executed
                step_log.writerow([executed, query, index, *action.tolist()])
                obs = env.get_obs()
                video.frame(obs)
                if env.eval_success:
                    break
            status = "ok"
        finally:
            query_log.writerow([query, executed - used, inference_s, name, used, status])
    return ("task_predicate" if env.eval_success else
            "control_step_limit" if executed >= limit else "inference_call_limit")


def execute(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "kind": "interface_smoke",
        "score_eligible": False,
        "task": TASK,
        "backend": args.backend,
        "environment": "robotwin" if args.pace_root else "synthetic",
        "prompt": PROMPT,
        "max_steps": args.max_steps,
        "max_queries": args.max_queries,
        "prefix": args.prefix,
        "control_steps": 0,
        "queries_attempted": 0,
    }
    try:
        with ExitStack() as stack:
            step_file = stack.enter_context((args.output / "steps.csv").open("w", newline="", buffering=1))
            query_file = stack.enter_context((args.output / "queries.csv").open("w", newline="", buffering=1))
            step_log, query_log = csv.writer(step_file), csv.writer(query_file)
            step_log.writerow(["step", "query", "chunk_index", *[f"qpos_{i}" for i in range(14)]])
            query_log.writerow(["query", "step_before", "inference_s", "chunk_file", "used", "status"])
            policy = stack.enter_context(open_policy(args.uri, args.backend, args.timeout))
            env, prompt = stack.enter_context(environment(args.pace_root, summary))
            video = Video(args.output / "head.mp4")
            stack.callback(video.close)
            reason = run(env, policy, prompt, args, summary, step_log, query_log, video)
        summary.update(status="complete", stop_reason=reason)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
        print(summary["error"], file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        temporary = args.output / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(args.output / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="ws://127.0.0.1:8000")
    parser.add_argument("--backend", choices=("fake", "checkpoint"), default="fake")
    parser.add_argument("--pace-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=8, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--max-queries", type=int, default=2, choices=range(1, 6), metavar="1..5")
    parser.add_argument("--prefix", type=int, default=4, choices=range(1, 9), metavar="1..8")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if args.backend == "checkpoint" and args.pace_root is None:
        parser.error("checkpoint smoke requires the pinned native --pace-root")
    if args.pace_root:
        args.pace_root = args.pace_root.resolve()

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(execute(args))


if __name__ == "__main__":
    main()
