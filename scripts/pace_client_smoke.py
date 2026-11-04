#!/usr/bin/env python3
"""Run a bounded single-scene handover_block smoke and record diagnostics."""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
from time import monotonic

import numpy as np

from pace_diagnostics import Diagnostics
from pace_runtime import (
    CAMERAS, PACE_REVISION, TASK, finite_array, open_policy, require_checkout,
    validate_request, validate_response,
)

PROMPT = "Hand the block to the other arm and place it on the target."
CAMERA_SOURCES = ("head_camera", "left_camera", "right_camera")
GRIPPERS = [6, 13]
ARM_INDICES = [*range(6), *range(7, 13)]
JOINT_LIMIT_TOLERANCE_RAD = 0.02


def encode(observation, prompt):
    cameras = observation["observation"]
    payload = {
        "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        "images": {
            target: np.ascontiguousarray(cameras[source]["rgb"].transpose(2, 0, 1))
            for target, source in zip(CAMERAS, CAMERA_SOURCES, strict=True)
        },
        "prompt": prompt,
    }
    validate_request(payload)
    return payload


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
def environment(pace_root, summary, diagnostics, preflight_only=False):
    if pace_root is None:
        yield SyntheticEnv(), PROMPT
        return

    with diagnostics.phase("checkout"):
        require_checkout(pace_root, PACE_REVISION)
    cwd, path = Path.cwd(), sys.path[:]
    try:
        os.chdir(pace_root)  # RoboTwin resolves assets relative to the workspace.
        sys.path[:0] = [str(pace_root), str(pace_root / "pace/src"), str(pace_root / "description/utils")]
        with native_environment(summary, diagnostics, preflight_only) as result:
            yield result
    finally:
        os.chdir(cwd)
        sys.path[:] = path


@contextmanager
def native_environment(summary, diagnostics, preflight_only):
    with diagnostics.phase("pace_module_import"):
        from pace.robotwin import (
            load_cached_scenes, load_environment, restore_cached_scene_state, sample_instruction,
        )

    with diagnostics.phase("expert_cache"):
        scenes, cache = load_cached_scenes(TASK, seed=0, count=1)
    scene = scenes[0]
    summary.update(scene_seed=scene["scene_seed"], scene_cache=cache, cache_split_seed=0,
                   cache_index=0, pace_revision=PACE_REVISION)
    with diagnostics.phase("load_environment"):
        env, settings = load_native_task(load_environment, diagnostics, preflight_only)
        if settings["embodiment"] != ["aloha-agilex"] or not settings["dual_arm_embodied"]:
            raise ValueError("expected native dual-arm Aloha")
        settings.update(collect_data=False, save_data=False, render_freq=0)
    if preflight_only:
        with diagnostics.phase("native_asset_files"):
            import xml.etree.ElementTree as ET

            robot = Path(settings["left_robot_file"])
            config = settings["left_embodiment_config"]
            urdf = robot / config["urdf_path"]
            srdf = robot / config["srdf_path"]
            meshes = {urdf.parent / node.attrib["filename"]
                      for node in ET.parse(urdf).getroot().iter("mesh")}
            required = meshes | {srdf, robot / "curobo_left.yml", robot / "curobo_right.yml"}
            if not meshes or any(not p.is_file() or p.stat().st_size == 0 for p in required):
                raise ValueError("incomplete Aloha URDF, SRDF, meshes or planner configuration")
            summary["aloha_mesh_files"] = len(meshes)
    with diagnostics.phase("planner_imports"):
        from envs.robot import planner

        if not callable(planner.MplibPlanner):
            raise TypeError("expected callable MPlib planner")
        if not summary.get("gpu_only_checks") and not callable(getattr(planner, "CuroboPlanner", None)):
            raise ImportError("PACE did not register CuroboPlanner")
    with diagnostics.phase("task_instruction"):
        # The upstream instruction generator uses both Python and NumPy RNGs.
        py_rng, np_rng = random.getstate(), np.random.get_state()
        try:
            random.seed(0)
            np.random.seed(0)
            instruction = sample_instruction(TASK, {"info": scene["episode_info"]}, 1, "unseen")
        finally:
            random.setstate(py_rng)
            np.random.set_state(np_rng)
        if not instruction.strip():
            raise ValueError("empty PACE task instruction")
    if preflight_only:
        with diagnostics.phase("ffmpeg_probe"):
            from imageio_ffmpeg import get_ffmpeg_exe
            import subprocess

            subprocess.run([get_ffmpeg_exe(), "-version"], check=True, timeout=10,
                           stdout=subprocess.DEVNULL)
        yield env, instruction
        return
    try:
        with diagnostics.phase("setup_demo"):
            env.setup_demo(now_ep_num=0, seed=scene["scene_seed"], is_test=True, **settings)
        with diagnostics.phase("restore_scene"):
            restore_cached_scene_state(TASK, env, scene["episode_info"])
            summary["initial_poses"] = scene_poses(env)
            env.set_instruction(instruction)
        yield env, env.get_instruction()
    finally:
        with diagnostics.phase("close_environment"):
            env.close_env(clear_cache=True)


def load_native_task(loader, diagnostics, preflight_only):
    """On a CPU login node, finish import-only checks after the known CUDA-only import."""
    import traceback

    try:
        return loader(TASK)
    except ImportError as error:
        planner = sys.modules.get("envs.robot.planner")
        if (not preflight_only or error.name != "envs.robot.planner"
                or "cannot import name 'CuroboPlanner'" not in str(error)
                or planner is None or hasattr(planner, "CuroboPlanner")):
            raise
        try:
            # Reproduce the exception swallowed by the upstream planner module.
            from curobo.wrap.reacher.motion_gen import MotionGen
        except RuntimeError as exc:
            frames = traceback.extract_tb(exc.__traceback__)
            if not ("Found no NVIDIA driver on your system" in str(exc)
                    and any(f.filename.endswith("/curobo/wrap/reacher/motion_gen.py") for f in frames)
                    and any(f.filename.endswith("/torch/cuda/__init__.py") and f.name == "_lazy_init"
                            for f in frames)):
                raise
            traceback.print_exc(file=diagnostics.stacks)
            diagnostics.stacks.flush()
            diagnostics.summary["gpu_only_checks"] = ["curobo_motion_gen_import"]
            diagnostics.record("load_environment", "deferred_to_gpu", detail=str(exc))
        else:
            raise error

        def gpu_only(*args, **kwargs):
            raise RuntimeError("CPU preflight cannot construct a CUDA planner")

        # Only the CPU probe executes this branch, and it never calls setup_demo().
        # Satisfy the import of the symbol so unrelated simulator imports can finish.
        planner.CuroboPlanner = gpu_only
        try:
            return loader(TASK)
        finally:
            del planner.CuroboPlanner


def scene_poses(env):
    poses = {
        "left_robot_base": env.robot.left_entity.get_root_pose(),
        "right_robot_base": env.robot.right_entity.get_root_pose(),
        "block": env.box.get_pose(),
        "target": env.target_box.get_pose(),
        "table": env.table.get_pose(),
    }
    result = {}
    for name, pose in poses.items():
        p, q = np.asarray(pose.p), np.asarray(pose.q)
        finite_array(p, (3,))
        finite_array(q, (4,))
        if not np.isclose(np.linalg.norm(q), 1, atol=1e-4):
            raise ValueError(f"invalid quaternion for {name}")
        result[name] = {"position": p.tolist(), "quaternion_wxyz": q.tolist()}
    return result


class NativeState:
    def __init__(self, robot):
        self.robot = robot
        joints = [robot.left_arm_joints, robot.right_arm_joints]
        if any(len(arm) != 6 for arm in joints):
            raise ValueError("expected six controlled joints per Aloha arm")
        self.names = [[joint.get_name() for joint in arm] for arm in joints]
        if len(set(self.names[0] + self.names[1])) != 12:
            raise ValueError("expected distinct left and right Aloha joints")
        self.limits = np.array([joint.get_limits()[0] for arm in joints for joint in arm])
        if self.limits.shape != (12, 2) or not np.all(self.limits[:, 0] < self.limits[:, 1]):
            raise ValueError("invalid Aloha arm limits")

    def check_limits(self, state, label):
        values = state[ARM_INDICES]
        if (np.any(values < self.limits[:, 0] - JOINT_LIMIT_TOLERANCE_RAD)
                or np.any(values > self.limits[:, 1] + JOINT_LIMIT_TOLERANCE_RAD)):
            raise ValueError(f"{label} exceeds Aloha arm limits")

    def measured(self, observation, state):
        joints = observation["joint_action"]
        ordered = np.asarray([*joints["left_arm"], joints["left_gripper"],
                              *joints["right_arm"], joints["right_gripper"]], dtype=np.float32)
        targets = np.asarray(self.robot.get_left_arm_jointState()
                             + self.robot.get_right_arm_jointState(), dtype=np.float32)
        if not (np.array_equal(state, ordered) and np.array_equal(state, targets)):
            raise ValueError("observation does not match left arm/gripper then right arm/gripper")
        actual = np.asarray(self.robot.get_left_arm_real_jointState()
                            + self.robot.get_right_arm_real_jointState(), dtype=np.float32)
        finite_array(actual, (14,))
        self.check_limits(state, "observed drive targets")
        self.check_limits(actual, "measured qpos")
        if np.any((state[GRIPPERS] < 0) | (state[GRIPPERS] > 1)):
            raise ValueError("observed gripper outside [0, 1]")
        return actual.tolist()


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


def run(env, policy, prompt, args, summary, step_log, query_log, obs_log, video, diagnostics):
    native = NativeState(env.robot) if args.pace_root else None
    initial_counter = env.take_action_cnt
    limit = min(args.max_steps, env.step_lim - initial_counter)
    summary.update(initial_take_action_cnt=initial_counter, take_action_cnt=initial_counter,
                   effective_step_limit=limit, observations=0, queries_received=0)
    if native:
        summary.update(arm_joint_names=native.names, arm_joint_limits_rad=native.limits.tolist(),
                       joint_limit_tolerance_rad=JOINT_LIMIT_TOLERANCE_RAD)
    camera_hashes = {name: set() for name in CAMERAS}

    def observe():
        with diagnostics.phase(f"get_obs_{summary['control_steps']:03d}"):
            obs = env.get_obs()
        payload = encode(obs, prompt)
        state, images = payload["state"], payload["images"]
        reference = getattr(args, "reference_run", None)
        if reference and summary["observations"] == 0:
            with np.load(reference / "initial_observation.npz", allow_pickle=False) as data:
                if not np.allclose(state, data["state"], rtol=0, atol=1e-3):
                    raise ValueError("initial state differs from the G1.1 reference")
        actual = native.measured(obs, state) if native else state.tolist()
        cameras = {}
        for name, image in images.items():
            low, high = int(image.min()), int(image.max())
            if native and low == high:
                raise ValueError(f"{name} is a uniform RGB frame (value={low})")
            digest = hashlib.sha256(image).hexdigest()
            camera_hashes[name].add(digest)
            cameras[name] = {"shape": list(image.shape), "dtype": str(image.dtype),
                             "min": low, "max": high, "sha256": digest}
        if native and len({record["sha256"] for record in cameras.values()}) != len(CAMERAS):
            raise ValueError("multiple cameras returned identical RGB frames")
        obs_log.write(json.dumps({"step": summary["control_steps"], "state": state.tolist(),
                                  "measured_qpos": actual, "cameras": cameras}) + "\n")
        if summary["observations"] == 0:
            np.savez_compressed(args.output / "initial_observation.npz", state=state, **images)
        summary["observations"] += 1
        summary["distinct_frames"] = {name: len(hashes) for name, hashes in camera_hashes.items()}
        with diagnostics.phase(f"video_frame_{summary['control_steps']:03d}"):
            video.frame(obs)
        return payload

    payload = observe()
    executed = 0
    while executed < limit and not env.eval_success and summary["queries_attempted"] < args.max_queries:
        summary["queries_attempted"] += 1
        query = summary["queries_attempted"]
        start = monotonic()
        try:
            np.savez_compressed(args.output / f"query_{query:03d}_observation.npz",
                                state=payload["state"], **payload["images"])
            with diagnostics.phase(f"policy_query_{query:03d}"):
                chunk = validate_response(policy.infer(payload))
        except Exception:
            query_log.writerow([query, executed, monotonic() - start, "", 0, "error"])
            raise

        inference_s = monotonic() - start
        summary["queries_received"] += 1
        name = f"chunk_{query:03d}.npy"
        count = min(args.prefix, limit - executed)
        used = 0
        status = "error"
        try:
            np.save(args.output / name, chunk, allow_pickle=False)
            if args.backend == "fake":
                expected = payload["state"].copy()
                expected[GRIPPERS] = expected[GRIPPERS].clip(0, 1)
                if not np.all(chunk == expected):
                    raise ValueError("Fake Policy output does not hold the queried state")
            # Reuse this chunk prefix locally. The remaining suffix expires at replan.
            for index, action in enumerate(chunk[:count]):
                before = env.take_action_cnt
                if native:
                    native.check_limits(action, "requested qpos")
                with diagnostics.phase(f"take_action_{executed + 1:03d}"):
                    try:
                        env.take_action(action.copy(), action_type="qpos")
                    finally:
                        summary["take_action_cnt"] = env.take_action_cnt
                    if env.take_action_cnt != before + 1:
                        raise RuntimeError("RoboTwin did not advance by one control action")
                executed += 1
                used += 1
                summary["control_steps"] = executed
                step_log.writerow([executed, query, index, *action.tolist()])
                payload = observe()
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
    preflight = getattr(args, "preflight_only", False)
    summary = {
        "kind": "cpu_preflight" if preflight else "interface_smoke",
        "score_eligible": False,
        "task": TASK,
        "backend": args.backend,
        "environment": "robotwin" if args.pace_root else "synthetic",
        "max_steps": args.max_steps,
        "max_queries": args.max_queries,
        "prefix": args.prefix,
        "control_steps": 0,
        "queries_attempted": 0,
    }
    try:
        with ExitStack() as stack:
            events = stack.enter_context((args.output / "phases.jsonl").open("w", buffering=1))
            stacks = stack.enter_context((args.output / "stacks.log").open("w", buffering=1))
            diagnostics = Diagnostics(events, stacks, summary, getattr(args, "stall_seconds", 30))
            if preflight:
                _, summary["prompt"] = stack.enter_context(
                    environment(args.pace_root, summary, diagnostics, preflight_only=True)
                )
                summary.update(status="complete", stop_reason="cpu_preflight_complete")
                return 0
            step_file = stack.enter_context((args.output / "steps.csv").open("w", newline="", buffering=1))
            query_file = stack.enter_context((args.output / "queries.csv").open("w", newline="", buffering=1))
            obs_file = stack.enter_context((args.output / "observations.jsonl").open("w", buffering=1))
            step_log, query_log = csv.writer(step_file), csv.writer(query_file)
            step_log.writerow(["step", "query", "chunk_index", *[f"qpos_{i}" for i in range(14)]])
            query_log.writerow(["query", "step_before", "inference_s", "chunk_file", "used", "status"])
            with diagnostics.phase("websocket_connect"):
                policy = stack.enter_context(open_policy(args.uri, args.backend, args.timeout))
            env, prompt = stack.enter_context(environment(args.pace_root, summary, diagnostics))
            summary["prompt"] = prompt
            if getattr(args, "reference_run", None):
                reference = json.loads((args.reference_run / "summary.json").read_text())
                for key in ("scene_seed", "pace_revision", "prompt"):
                    if reference[key] != summary[key]:
                        raise ValueError(f"G1.1 reference mismatch: {key}")
                for name, pose in summary["initial_poses"].items():
                    for field, values in pose.items():
                        if not np.allclose(values, reference["initial_poses"][name][field],
                                           rtol=0, atol=1e-4):
                            raise ValueError(f"G1.1 reference pose mismatch: {name}/{field}")
            video = Video(args.output / "head.mp4")
            def close_video():
                with diagnostics.phase("close_video"):
                    video.close()

            stack.callback(close_video)
            reason = run(env, policy, prompt, args, summary, step_log, query_log, obs_file, video,
                         diagnostics)
        summary.update(status="complete", stop_reason=reason)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if "failed_phase" not in summary:
            import traceback

            summary["failed_phase"] = "contract_or_artifact"
            with (args.output / "stacks.log").open("a") as stack_log:
                traceback.print_exc(file=stack_log)
        summary.update(status="aborted", stop_reason="interrupted" if isinstance(exc, KeyboardInterrupt)
                       else "error", error=f"{type(exc).__name__}: {exc}")
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
    parser.add_argument("--stall-seconds", type=float, default=30,
                        help="write all Python stacks every N seconds within a stalled phase")
    parser.add_argument("--preflight-only", action="store_true",
                        help="check the pinned native loader, planners, cache, prompt and FFmpeg on CPU")
    parser.add_argument("--reference-run", type=Path,
                        help="require the same native initial scene as a verified G1.1 run")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if not math.isfinite(args.stall_seconds) or args.stall_seconds <= 0:
        parser.error("--stall-seconds must be positive and finite")
    if args.preflight_only and (args.pace_root is None or args.backend != "fake"):
        parser.error("--preflight-only requires --pace-root with the fake backend")
    if args.backend == "checkpoint" and args.pace_root is None:
        parser.error("checkpoint smoke requires the pinned native --pace-root")
    if args.pace_root:
        args.pace_root = args.pace_root.resolve()
    if args.reference_run:
        if args.backend != "checkpoint" or not args.pace_root:
            parser.error("--reference-run requires a native checkpoint rollout")
        args.reference_run = args.reference_run.resolve()

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(execute(args))


if __name__ == "__main__":
    main()
