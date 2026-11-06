"""Complete synthetic episodes, including the real G1 WebSocket transport."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
from threading import Thread
from time import sleep
import unittest

import numpy as np
from websockets.sync.server import serve

from g2_fixtures import fixture, snapshot
from g2_episode import ExpertWaypoints, run_episode
from g2_adapter import G2Error, Status
from g2_motion import Trajectory
from pace_runtime import open_policy, msgpack_numpy, validate_request


class MockRobot:
    """Synthetic measured state; no simulation, collision assets or physical claim."""
    def __init__(self, adapter, qa, qr, bases, fault=None):
        self.adapter, self.qa, self.q, self.bases = adapter, qa.copy(), qr.copy(), bases
        self.native_limits = [arm.chain.limits for arm in adapter.aloha]
        self.tick, self.actions, self.held, self.closed = 0, 0, 0, False
        self.fault = fault
        self.frames = []

    def reset(self, spec):
        self.episode = spec["episode_id"]
        self.is_native = "aloha" in spec["condition"].lower()
        if self.fault == "reset":
            raise RuntimeError("injected reset failure")
        from g2_robotwin_backend import pose7
        from hashlib import sha256
        actual_q = self.qa if self.is_native else self.q
        actual_bases = self.adapter.source_frames if self.is_native else self.bases
        return dict(robot_base_world={side: pose7(base) for side, base in zip(("left", "right"), actual_bases)},
                    task_frame_world=[0, 0, 0, 1, 0, 0, 0],
                    actual_arm_qpos_rad={side: q.tolist() for side, q in zip(("left", "right"), actual_q)},
                    actual_gripper=dict(left=.5, right=.5),
                    objects_task={"synthetic_block": [0, 0, 0, 1, 0, 0, 0]},
                    reset_fixture_sha256=sha256(self.q.tobytes()).hexdigest(),
                    workspace_check_id="synthetic_aligned_fixture_only", common_workspace_pass=True)

    def success(self):
        return self.actions >= 4 and self.fault != "task"

    def capture(self):
        from types import SimpleNamespace
        snap = (snapshot(SimpleNamespace(arx=self.adapter.aloha), self.qa, self.adapter.source_frames,
                         self.tick, self.episode) if self.is_native else
                snapshot(self.adapter, self.q, self.bases, self.tick, self.episode))
        self.frames.append(np.ascontiguousarray(snap.images["cam_high"][1].transpose(1, 2, 0)))
        return snap, np.column_stack((self.qa, [.5, .5])).ravel()

    def execute(self, plan, log, remaining):
        if self.fault == "interrupt":
            raise KeyboardInterrupt("injected interrupt")
        if self.fault == "control":
            raise RuntimeError("injected control failure")
        if self.fault == "timeout":
            sleep(.08)
            remaining()
        if isinstance(plan, Trajectory):
            for tick in range(1, len(plan.q)):
                self.q = plan.q[tick].copy()
                self.tick += 1
                log(dict(tick=self.tick, actual_arm=self.q, target_arm=plan.q[tick],
                         actual_fingers=plan.fingers[tick], period_s=plan.period_s, control_call_count=1,
                         position_error_m=[0., 0.], orientation_error_deg=[0., 0.],
                         error_scope="synthetic"))
            completed = len(plan.waypoint_ticks)
        else:
            completed = 0
            for row in plan:
                self.qa = row.reshape(2, 7)[:, :6]
                self.tick += 1
                log(dict(tick=self.tick, actual_aloha=row, target_aloha=row, period_s=.1, control_call_count=1))
                completed += 1
                if self.actions + completed >= 4 and self.fault != "task":
                    break
        self.actions += completed
        return completed

    def hold(self):
        self.held += 1

    def close(self, output):
        self.closed = True
        from imageio_ffmpeg import write_frames
        path = output / "head.mp4"
        writer = write_frames(str(path), (16, 16), fps=10, codec="libx264", ffmpeg_log_level="error")
        writer.send(None)
        for frame in self.frames or [np.zeros((16, 16, 3), dtype=np.uint8)]:
            writer.send(frame)
        writer.close()
        return str(path)


class FixedPolicy:
    def __init__(self, qa, fault=None):
        self.qa, self.fault, self.inputs = qa, fault, []

    def infer(self, request):
        validate_request(request)
        self.inputs.append(request["images"]["cam_high"][0, 0, 0].item())
        if self.fault == "model":
            raise RuntimeError("injected model failure")
        chunk = np.tile(np.column_stack((self.qa, [.5, .5])).ravel(), (50, 1))
        if self.fault == "shape":
            chunk = chunk[:3]
        return {"actions": chunk}


def spec(window, condition="ARX"):
    return dict(data_kind="synthetic", episode_id="mock", condition=condition, prompt="synthetic handover",
                window=window, max_actions=4, timeout_s=10., native_grip_max_clip=.05)


class IntegrationTests(unittest.TestCase):
    def test_reported_zero_actuation_is_preserved_when_gate_rejects(self):
        adapter,qa,qr,bases=fixture();backend=MockRobot(adapter,qa,qr,bases)
        def reject(plan,log,remaining):
            log(dict(kind='planning_gate_rejection',tick=0,control_call_count=0,physics_step_count=0,
                     collisions=[{'object_name1':'robot','object_name2':'probe'}]))
            raise G2Error(Status.COLLISION,'scene changed after planning')
        backend.execute=reject
        with tempfile.TemporaryDirectory() as tmp:
            row=run_episode(spec(1),backend,FixedPolicy(qa),Path(tmp)/'e',adapter=adapter)
            self.assertEqual(row['outcome'],'control_error');self.assertEqual(row['control_calls'],0)
            self.assertEqual(row['executed'],0);self.assertEqual(backend.held,1);self.assertTrue(backend.closed)

    def test_two_robots_two_windows_one_real_websocket_service(self):
        adapter, qa, qr, bases = fixture()
        policy = FixedPolicy(qa)
        connections = []
        def handler(socket):
            connections.append(1)
            socket.send(msgpack_numpy.packb({"backend": "fake", "task": "handover_block", "protocol": 1}))
            for message in socket:
                socket.send(msgpack_numpy.packb(policy.infer(msgpack_numpy.unpackb(message))))
        with tempfile.TemporaryDirectory() as directory:
            with serve(handler, "127.0.0.1", 0, close_timeout=.1) as server:
                worker = Thread(target=server.serve_forever, daemon=True)
                worker.start()
                try:
                    uri = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
                    with open_policy(uri, "fake", 3) as remote:
                        for condition in ("Aloha", "ARX"):
                            for window in (1, 4):
                                adapter, qa, qr, bases = fixture()
                                backend = MockRobot(adapter, qa, qr, bases)
                                output = Path(directory) / f"{condition}-{window}"
                                offset = len(policy.inputs)
                                row = run_episode(spec(window, condition), backend, remote, output,
                                                  adapter=adapter if condition == "ARX" else None)
                                self.assertTrue(row["success"], row)
                                self.assertEqual(row["queries"], 4 // window)
                                self.assertEqual(row["executed"], 4)
                                self.assertEqual(len(list(output.glob("observation-*.npz"))), row["queries"])
                                self.assertTrue((output / "head.mp4").stat().st_size)
                                if window == 1:
                                    self.assertEqual(policy.inputs[offset:offset + 4], [0, 1, 2, 3])
                                self.assertTrue(backend.closed)
                finally:
                    server.shutdown()
                    worker.join()
        self.assertEqual(len(connections), 1)

    def test_failure_cleanup_evidence_and_termination(self):
        cases = [("model", "policy_failed"), ("shape", "policy_failed"),
                 ("task", "task_unsuccessful"), ("reset", "environment_error"),
                 ("control", "control_error"), ("interrupt", "interrupted"),
                 ("timeout", "timeout"), ("planning", "planning_failed")]
        for fault, outcome in cases:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                adapter, qa, qr, bases = fixture()
                backend = MockRobot(adapter, qa, qr, bases, fault)
                config = spec(1)
                if fault == "timeout":
                    config["timeout_s"] = .06
                if fault == "planning":
                    adapter.target_collision = lambda q, g: False
                row = run_episode(config, backend, FixedPolicy(qa, fault), Path(temp) / "episode", adapter=adapter)
                self.assertEqual(row["outcome"], outcome, row)
                self.assertFalse(row["success"])
                self.assertEqual(backend.held, 1)
                self.assertTrue(backend.closed)
                saved = json.loads((Path(temp) / "episode/result.json").read_text())
                self.assertEqual(saved["outcome"], outcome)
                self.assertIsNotNone(saved["failure"])
                if fault == "shape":
                    self.assertEqual(np.load(Path(temp) / "episode/action-00001.npy").shape, (3, 14))

    def test_cleanup_precedes_final_evidence_write_on_disk_failure(self):
        from unittest.mock import patch
        from g2_episode import save_json as real_save
        adapter, qa, qr, bases = fixture()
        backend = MockRobot(adapter, qa, qr, bases)
        calls = []
        def disk_error(path, value):
            calls.append(path)
            if len(calls) >= 2:
                raise OSError("injected full disk")
            return real_save(path, value)
        with tempfile.TemporaryDirectory() as temp, patch("g2_episode.save_json", disk_error):
            with self.assertRaises(OSError):
                run_episode(spec(1), backend, FixedPolicy(qa), Path(temp) / "episode", adapter=adapter)
        self.assertEqual(backend.held, 1)
        self.assertTrue(backend.closed)

    def test_malformed_remote_action_preserved_before_validation(self):
        adapter, qa, qr, bases = fixture()
        def handler(socket):
            socket.send(msgpack_numpy.packb({"backend": "fake", "task": "handover_block", "protocol": 1}))
            socket.recv()
            socket.send(msgpack_numpy.packb({"actions": np.zeros((3, 14))}))
            try:
                socket.recv()
            except Exception:
                pass
        with tempfile.TemporaryDirectory() as temp:
            with serve(handler, "127.0.0.1", 0, close_timeout=.1) as server:
                worker = Thread(target=server.serve_forever, daemon=True)
                worker.start()
                try:
                    uri = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
                    with open_policy(uri, "fake", 3) as remote:
                        output = Path(temp) / "episode"
                        row = run_episode(spec(1), MockRobot(adapter, qa, qr, bases),
                                          remote, output, adapter=adapter)
                        self.assertEqual(row["outcome"], "policy_failed")
                        self.assertEqual(np.load(output / "action-00001.npy").shape, (3, 14))
                finally:
                    server.shutdown()
                    worker.join()

    def test_expert_is_geometry_retimed_and_exhaustion_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter, qa, qr, bases = fixture()
            rows = np.tile(np.column_stack((qa, [.5, .5])).ravel(), (2, 1))
            row = run_episode(spec(4), MockRobot(adapter, qa, qr, bases), None, Path(temp) / "expert",
                              adapter=adapter, expert=ExpertWaypoints(rows, "synthetic_expert_qpos"))
            self.assertEqual(row["outcome"], "task_unsuccessful")
            self.assertEqual(row["executed"], 2)
            text = (Path(temp) / "expert/events.jsonl").read_text()
            self.assertIn('"original_sample_period_s": "unknown"', text)
            self.assertIn('"source_frames": [0, 1]', text)


if __name__ == "__main__":
    unittest.main()
