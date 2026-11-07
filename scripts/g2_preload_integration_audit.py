"""CPU action-semantics probe: pinned controller, synthetic planner/contacts only."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import MethodType, SimpleNamespace as NS

import numpy as np

from g2_adapter import G2Error, SIDES, Stamp
from g2_native_preload import NativePreloadHold
from g2_seed1_dynamics_replay_cpu import methods
from g2e_native_guard import NativeReleaseTracker
from g2_contact import ContactLimits
from g2_episode import json_value

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

SOURCE = {
    "envs/_base_task.py": "448f7152b65cb9102217cf5463aa821d72810ca56f63d5a797ec7bd43e23e101",
    "envs/robot/robot.py": "3dcd80acc8cab489a4c5edb507cc460dab1724be0226b8d5c4c1b218dee605cb",
}


class Joint:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, key):
        return getattr(self.wrapped, key)

    @property
    def drive_target(self):
        return self.wrapped.get_drive_target()

    def set_drive_target(self, value):
        self.wrapped.set_drive_target(float(np.asarray(value).item()))

    def set_drive_velocity_target(self, value):
        self.wrapped.set_drive_velocity_target(float(np.asarray(value).item()))


def probe(source, window, enabled, premature=False, controller_type=NativePreloadHold):
    from test_g2_native_backend import native_fixture
    from test_g2_contact import contacts

    with tempfile.TemporaryDirectory() as temp:
        backend, state, closed, physical = native_fixture(Path(temp))
        robot, env = backend.env.robot, backend.env
        entity = robot.left_entity
        joints = [Joint(j) for j in entity.get_active_joints()]
        entity.get_active_joints = lambda: joints
        for i, j in enumerate(joints):
            j.get_drive_velocity_target = lambda i=i: np.array([state["velocity"][i]])
        names = [
            "_entity_qf",
            "set_arm_joints",
            "set_gripper",
            "get_normal_real_gripper_val",
            "get_left_arm_jointState",
            "get_right_arm_jointState",
            "get_left_gripper_val",
            "get_right_gripper_val",
            "get_left_arm_real_jointState",
            "get_right_arm_real_jointState",
        ]
        for name, fn in methods(source / "envs/robot/robot.py", "Robot", names).items():
            setattr(robot, name, MethodType(fn, robot))

        def topp(path, dt, verbose):
            q = np.linspace(path[0], path[-1], 4)
            v = np.repeat(((path[-1] - path[0]) / (3 * dt))[None], 4, axis=0)
            return np.arange(4) * dt, q, v, v * 0, 3 * dt

        for side, arm, start in zip(SIDES, backend.arms, (0, 8)):
            setattr(robot, side + "_arm_joints", joints[start : start + 6])
            setattr(
                robot,
                side + "_gripper",
                [(j, 1.0, 0.0) for j in joints[start + 6 : start + 8]],
            )
            setattr(robot, side + "_gripper_scale", [-0.01, 0.045])
            setattr(robot, side + "_gripper_val", 0.6)
            setattr(robot, side + "_mplib_planner", NS(TOPP=topp))
        state["target"][[6, 7, 14, 15]] = 0.023
        state["actual"] = state["target"].copy()

        def step():
            state["actual"] = state["target"].copy()
            state["actual"][[6, 7, 14, 15]] = np.clip(
                state["actual"][[6, 7, 14, 15]], 0.0, 0.045
            )

        physical.step, physical.get_contacts = step, lambda: []
        env._update_render = lambda: None
        env.render_freq, env.eval_video_path, env.eval_success = 0, None, False
        env.check_success = lambda: False
        env.take_action = MethodType(
            methods(source / "envs/_base_task.py", "Base_Task", ["take_action"])[
                "take_action"
            ],
            env,
        )

        def obs():
            left, right = (
                robot.get_left_arm_jointState(),
                robot.get_right_arm_jointState(),
            )
            return dict(
                observation={
                    c: {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}
                    for c in ("head_camera", "left_camera", "right_camera")
                },
                joint_action=dict(
                    vector=np.array(left + right, dtype=np.float32),
                    left_arm=left[:6],
                    left_gripper=left[6],
                    right_arm=right[:6],
                    right_gripper=right[6],
                ),
            )

        env.get_obs = obs
        tracker = NativeReleaseTracker(
            99, dict(left=[1, 2], right=[3, 4]), ContactLimits()
        )
        phase = ["left"]
        released = set()
        guard = NS(
            protocol="G2-E",
            previous_drive=None,
            max_physics_steps=25000,
            collision=NS(tracker=tracker),
        )

        def candidate(q, fingers, targets, drive, remaining):
            before = drive if guard.previous_drive is None else guard.previous_drive
            tracker.authorize_opening(before, drive, backend.arms)
            guard.previous_drive = drive.copy()

        def measured(stamp, q, fingers, tools, block, cs):
            released.update(tracker.releasing)
            ids = [1, 2] if "left" not in released else []
            if phase[0] == "both":
                ids += [3, 4]
            if "right" in released:
                ids = []
            tracker.observe(
                stamp, contacts(ids), np.array([np.eye(4), np.eye(4)]), np.eye(4)
            )

        guard.candidate, guard.measured = candidate, measured
        guard.metrics, guard.tracking = lambda: {}, lambda *args: None
        backend.guard = guard
        backend.preload = controller_type(tracker, backend.arms) if enabled else None
        env.scene.before_step = backend._before_step
        backend.episode, backend.window = "integration", window
        logs = []
        chunks = []
        for block in range(2):
            chunk = np.zeros((window, 14))
            chunk[:, 0] = (np.arange(window) + block * window) * 0.00001
            chunk[:, 7] = -chunk[:, 0]
            chunk[:, 13] = 0.6 if block == 0 else 0.0
            if block == 0:
                chunk[0, 6] = 0.6
                chunk[-1, 13] = 0.0
            else:
                chunk[2:, 6] = 1.0
            chunks.append(chunk)
        with redirect_stdout(io.StringIO()):
            assert backend.execute(chunks[0], logs.append, lambda: 1) == window
            _, observed = backend.capture()
            assert observed[6] == observed[13] == 0.0
            drive_after_close = backend._drive()[1].copy()
            actual_after_close = state["actual"][[6, 7, 14, 15]].tolist()
            if premature:
                target = chunks[0][-1:].copy()
                target[0, 6] = 1.0
                before = backend.tick
                try:
                    backend.execute(target, logs.append, lambda: 1)
                except G2Error as error:
                    assert "donor opening" in str(error)
                    assert error.diagnostic["completed_actions"] == 0
                    stopped = dict(
                        reason="premature donor release rejected",
                        advanced_ticks=backend.tick - before,
                        completed_actions=0,
                    )
                else:
                    raise AssertionError("donor release was not blocked")
            else:
                phase[0] = "both"
                assert backend.execute(chunks[1], logs.append, lambda: 1) == window
                assert tracker.releasing == {"left"} or set(tracker.holders) == {
                    "right"
                }
                assert "left" not in tracker.holders
                assert "right" in tracker.holders
                assert not backend.preload or "left" not in backend.preload.references
                release = chunks[1][-1:].copy()
                release[0, 13] = 1.0
                assert backend.execute(release, logs.append, lambda: 1) == 1
                if tracker.holders:
                    last = next(
                        r
                        for r in reversed(logs)
                        if r.get("kind") == "native_preload_control"
                    )
                    raise AssertionError(
                        json.dumps(
                            dict(
                                reason="right release blocked despite consumed opening action",
                                window=window,
                                holders=sorted(tracker.holders),
                                logical_right_goal=robot.right_gripper_val,
                                applied_right_drive_m=backend._drive()[1][1],
                                last_control=last,
                            ),
                            default=json_value,
                        )
                    )
                assert not backend.preload or not backend.preload.references
                chunks.append(release)
                requested = [
                    r["normalized_qpos_request"]
                    for r in logs
                    if r.get("kind") == "native_action_request"
                ]
                assert np.array_equal(requested, np.concatenate(chunks))
                assert env.take_action_cnt == 2 * window + 1
                assert backend.tick == 4 * (2 * window + 1)
                stopped = None
        arm_drives = [r["target_arm"] for r in logs if "target_arm" in r]
        request_bytes = np.asarray(
            [
                r["normalized_qpos_request"]
                for r in logs
                if r.get("kind") == "native_action_request"
            ]
        ).tobytes()
        output = dict(
            window=window,
            preload=enabled,
            premature=premature,
            raw_requests_sha256=sha256(request_bytes).hexdigest(),
            arm_drive_sha256=sha256(np.array(arm_drives).tobytes()).hexdigest(),
            fully_consumed_actions=2 * window + 1 if not premature else window,
            controller_action_counter=env.take_action_cnt,
            physics_stub_steps=backend.tick,
            model_state_after_close=observed.tolist(),
            applied_drive_after_close_m=drive_after_close.tolist(),
            measured_stub_fingers_after_close_m=actual_after_close,
            right_close_reached_controller=True,
            clean_release=not premature,
            stop=stopped,
            synthetic_planner_and_contacts=True,
            real_physics=False,
            rendering=False,
            model_inference=False,
        )
        backend.hold()
        backend.close(Path(temp))
        assert closed == [True] and np.all(state["velocity"] == 0.0)
        return output


def replay_recorded_selection(folder):
    from g2_fixtures import robot_arms

    records = json.loads((folder / "trace.json").read_text())
    commands = json.loads((folder / "preload-controls.json").read_text())
    tracker = NativeReleaseTracker(99, dict(left=[1, 2], right=[3, 4]), ContactLimits())
    arms = robot_arms("aloha-agilex")
    controller = NativePreloadHold(tracker, arms)
    previous = np.array(commands[0]["requested_finger_drive_m"])
    for command in commands:
        tick = command["tick"]
        if tick:
            contact = records[tick - 1]["contact_state"]
            tracker.holders = {s: np.array(v) for s, v in contact["tool_block"].items()}
            tracker.releasing = set(contact["releasing"])
            tracker.last_stamp = Stamp("recorded", tick, tick * 0.004)
            controller.observe(tick, previous)
        applied, velocity, _ = controller.select(
            tick,
            previous,
            command["requested_finger_drive_m"],
            command["requested_finger_velocity_m_s"],
        )
        assert np.array_equal(applied, command["applied_finger_drive_m"])
        assert np.array_equal(velocity, command["applied_finger_velocity_m_s"])
        tracker.authorize_opening(previous, applied, arms)
        previous = applied
    return dict(
        passed=True,
        physical_rerun=False,
        recorded_steps=len(commands),
        applied_position_and_velocity_identical=True,
        scope="functional controller replay on recorded measurements, not new physical handover",
    )


def run(source, output, legacy_commit=None, recorded_run=None):
    for name, expected in SOURCE.items():
        assert sha256((source / name).read_bytes()).hexdigest() == expected
    results = []
    legacy_failures = []
    for window in (31, 50):
        a, b = probe(source, window, False), probe(source, window, True)
        assert a["raw_requests_sha256"] == b["raw_requests_sha256"]
        assert a["arm_drive_sha256"] == b["arm_drive_sha256"]
        assert a["physics_stub_steps"] == b["physics_stub_steps"]
        results += [a, b, probe(source, window, True, True)]
    if legacy_commit is not None:
        if legacy_commit != "f95b8bb98764d27cf8563cd65c0fedccb3a9279b":
            raise ValueError("legacy comparison is bound to the reviewed candidate")
        code = subprocess.check_output(
            ["git", "show", legacy_commit + ":scripts/g2_native_preload.py"], cwd=ROOT
        )
        namespace = {}
        exec(compile(code, "verified-f95b8bb-preload", "exec"), namespace)
        for window in (31, 50):
            try:
                probe(
                    source, window, True, controller_type=namespace["NativePreloadHold"]
                )
            except AssertionError as error:
                evidence = json.loads(str(error))
                assert evidence["logical_right_goal"] == 1.0 and evidence[
                    "holders"
                ] == ["right"]
                evidence["source_sha256"] = sha256(code).hexdigest()
                legacy_failures.append(evidence)
            else:
                raise AssertionError("legacy right-release failure was not reproduced")
    output.write_text(
        json.dumps(
            dict(
                passed=True,
                source_hashes=SOURCE,
                probes=results,
                legacy_commit=legacy_commit,
                legacy_failures=legacy_failures,
                recorded_selection_replay=replay_recorded_selection(recorded_run)
                if recorded_run
                else None,
                scope="pinned controller with mock TOPP and contacts; not dynamic handover or model success",
            ),
            indent=2,
            default=json_value,
        )
        + "\n"
    )
    print("Pinned controller integration: six CPU probes passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compare-legacy", default=None)
    p.add_argument("--recorded-run", type=Path)
    args = p.parse_args()
    run(args.source_dir, args.output, args.compare_legacy, args.recorded_run)
