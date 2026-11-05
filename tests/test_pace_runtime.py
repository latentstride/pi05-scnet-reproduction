"""Offline Stage 5 contract checks. No JAX, RoboTwin, GPU or checkpoint."""
from argparse import Namespace
from contextlib import contextmanager
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from pathlib import Path
import sys
import tempfile
from threading import Thread
import time
import unittest

import numpy as np
from websockets.sync.server import serve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pace_client_smoke import PROMPT, SyntheticEnv, encode, execute
from pace_diagnostics import Diagnostics
from pace_download import download_one, load_manifest, verify_file
from pace_model_smoke import FakePolicy, handler, require_orbax_checkpoint
from pace_runtime import (
    ACTION_SHAPE, TASK, msgpack_numpy, open_policy, validate_request, validate_response,
)
from verify_pace_smoke import verify


@contextmanager
def service(handle):
    with serve(handle, "127.0.0.1", 0, close_timeout=0.1) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        finally:
            server.shutdown()
            thread.join()


class RuntimeTest(unittest.TestCase):
    def test_orbax_mode_rejects_an_alternate_model_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            require_orbax_checkpoint(root)
            (root / "model.safetensors").write_bytes(b"alternate model")
            with self.assertRaisesRegex(ValueError, "alternate PyTorch"):
                require_orbax_checkpoint(root)

    def setUp(self):
        self.request = encode(SyntheticEnv().get_obs(), PROMPT)

    def test_contract(self):
        validate_request(self.request)
        invalid_states = (np.zeros(13), np.full(14, np.nan), np.zeros(14, dtype=np.int32))
        for state in invalid_states:
            with self.assertRaises(ValueError):
                validate_request(dict(self.request, state=state))
        for images in ({}, dict(self.request["images"], cam_high=np.zeros((8, 8, 3)))):
            with self.assertRaises(ValueError):
                validate_request(dict(self.request, images=images))
        with self.assertRaises(ValueError):
            validate_request(dict(self.request, prompt=" "))
        for actions in (np.zeros((49, 14)), np.full(ACTION_SHAPE, np.inf)):
            with self.assertRaises(ValueError):
                validate_response({"actions": actions})

    def test_roundtrip_and_reject_backend_mismatch(self):
        with service(handler(FakePolicy(), "fake")) as uri:
            with open_policy(uri, "fake", 2) as client:
                actions = client.infer(self.request)["actions"]
                self.assertEqual(actions.shape, ACTION_SHAPE)
                np.testing.assert_array_equal(actions[0], self.request["state"])
            with self.assertRaises(ValueError), open_policy(uri, "checkpoint", 2):
                self.fail("accepted fake as checkpoint")

    def test_server_rejects_bad_frame(self):
        with service(handler(FakePolicy(), "fake")) as uri, open_policy(uri, "fake", 2) as client:
            client.socket.send(b"\xc1")
            self.assertIsInstance(client.socket.recv(timeout=2), str)

    def test_backend_exception(self):
        class Broken:
            def infer(self, request):
                raise RuntimeError("injected")

        with service(handler(Broken(), "fake")) as uri, open_policy(uri, "fake", 2) as client:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                client.infer(self.request)

    def test_client_rejects_bad_response_and_times_out(self):
        def respond(socket):
            socket.send(msgpack_numpy.packb({"backend": "fake", "task": TASK, "protocol": 1}))
            socket.recv()
            socket.send(msgpack_numpy.packb({"actions": np.zeros((50, 13))}))
            socket.recv()
            time.sleep(0.3)

        with service(respond) as uri, open_policy(uri, "fake", 0.1) as client:
            with self.assertRaises(ValueError):
                client.infer(self.request)
            with self.assertRaises(TimeoutError):
                client.infer(self.request)

    def test_bounded_loop_artifacts_and_abort(self):
        class Counting:
            calls = 0
            camera_samples = []

            def infer(self, request):
                self.calls += 1
                if self.calls == 4:
                    raise RuntimeError("injected")
                self.camera_samples.append(int(request["images"]["cam_high"][0, 0, 0]))
                return FakePolicy().infer(request)

        policy = Counting()
        with tempfile.TemporaryDirectory() as temp, service(handler(policy, "fake")) as uri:
            root = Path(temp)
            args = Namespace(uri=uri, backend="fake", timeout=2, pace_root=None,
                             output=root / "complete", max_steps=5, max_queries=2, prefix=3)
            self.assertEqual(execute(args), 0)
            result = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(result["stop_reason"], "control_step_limit")
            self.assertFalse(result["score_eligible"])
            self.assertNotIn("success", result)
            with (args.output / "steps.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["chunk_index"]) for row in rows], [0, 1, 2, 0, 1])
            self.assertEqual(np.load(args.output / "chunk_002.npy").shape, ACTION_SHAPE)
            self.assertIn(b"ftyp", (args.output / "head.mp4").read_bytes()[:32])
            self.assertEqual(policy.camera_samples, [0, 30])
            self.assertEqual(verify(args.output)["frames"], 6)

            args.output = root / "query_limit"
            args.max_queries = 1
            self.assertEqual(execute(args), 0)
            result = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(result["stop_reason"], "inference_call_limit")
            self.assertEqual(result["control_steps"], 3)

            args.output = root / "aborted"
            self.assertEqual(execute(args), 1)
            result = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(result["status"], "aborted")
            self.assertEqual(result["control_steps"], 0)
            self.assertEqual(result["queries_attempted"], 1)
            self.assertFalse(list(args.output.glob("chunk_*.npy")))
            self.assertEqual(result["failed_phase"], "policy_query_001")
            self.assertIn("injected", (args.output / "stacks.log").read_text())

    def test_phase_watchdog_records_stack_before_phase_finishes(self):
        with tempfile.TemporaryFile(mode="w+") as events, tempfile.TemporaryFile(mode="w+") as stacks:
            diagnostics = Diagnostics(events, stacks, {}, stall_seconds=0.02)
            with diagnostics.phase("load_environment"):
                threading.Event().wait(0.12)
            events.seek(0)
            records = [json.loads(line) for line in events]
            self.assertEqual([event["event"] for event in records], ["start", "end"])
            self.assertEqual(records[1]["stage"], "load_environment")
            stacks.seek(0)
            self.assertIn("test_phase_watchdog_records_stack_before_phase_finishes", stacks.read())


class ManifestTest(unittest.TestCase):
    def test_manifest_scope_and_checksum(self):
        manifest = load_manifest()
        self.assertEqual(manifest["checkpoint_bytes"], 12_441_552_737)
        self.assertEqual(manifest["total_bytes"], 12_445_816_760)
        self.assertEqual(manifest["revision"], "17ef545270cf26ed81169893b60f845f80ce6557")

    def test_download_resume_and_checksum(self):
        data = b"offline fixture"
        offsets = []

        class RangeHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                offset = int(self.headers["Range"].removeprefix("bytes=").rstrip("-"))
                offsets.append(offset)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {offset}-{len(data) - 1}/{len(data)}")
                self.send_header("Content-Length", str(len(data) - offset))
                self.end_headers()
                self.wfile.write(data[offset:])

            def log_message(self, *args):
                pass

        item = {"path": "handover_block/sample", "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as temp, ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler) as http:
            root = Path(temp).resolve()
            final = root / item["path"]
            final.parent.mkdir()
            partial = final.with_suffix(".part")
            partial.write_bytes(data[:4])
            thread = Thread(target=http.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{http.server_port}/"
                download_one(root, base, item)
                self.assertEqual(offsets, [4])
                verify_file(final, item)
                self.assertFalse(partial.exists())
                final.write_bytes(b"x" * len(data))
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    download_one(root, base, item)
                self.assertEqual(offsets, [4])
            finally:
                http.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
