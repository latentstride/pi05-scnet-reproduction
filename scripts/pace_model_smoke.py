#!/usr/bin/env python3
"""Serve a deterministic CPU fake or the explicitly selected pinned checkpoint."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
import logging
import os
from pathlib import Path
import subprocess
from time import monotonic

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve
from pace_diagnostics import Diagnostics

from pace_runtime import (
    ACTION_SHAPE, CONFIG, MAX_MESSAGE, OPENPI_REVISION, TASK, decode, msgpack_numpy,
    require_checkout, validate_request, validate_response,
)


class FakePolicy:
    def infer(self, observation):
        # Hold current joint positions. This output is solely for smoke tests.
        actions = np.broadcast_to(observation["state"], ACTION_SHAPE).copy()
        actions[:, [6, 13]] = actions[:, [6, 13]].clip(0, 1)
        return {"actions": actions}


def require_orbax_checkpoint(path):
    if (path / "model.safetensors").exists():
        raise ValueError("unexpected alternate PyTorch checkpoint")


def handler(policy, backend, diagnostics=None):
    def handle(socket):
        try:
            socket.send(msgpack_numpy.packb({"backend": backend, "task": TASK, "protocol": 1}))
            for frame in socket:
                try:
                    request = decode(frame)
                    validate_request(request)
                    start = monotonic()
                    phase = diagnostics.phase("inference") if diagnostics else nullcontext()
                    with phase:
                        response = policy.infer(request)
                        validate_response(response)
                    response["service_inference_s"] = monotonic() - start
                    socket.send(msgpack_numpy.packb(response))
                except ConnectionClosed:
                    return
                except Exception as exc:
                    socket.send(f"{type(exc).__name__}: {exc}")
                    socket.close(code=1011)
                    return
        except ConnectionClosed:
            return
    return handle


def load_model(root, download_dir, diagnostics, require_gpu=False, preflight=False, verified_model=None):
    from pace_download import load_manifest, local_path, verify_all, verify_file

    with diagnostics.phase("checkpoint_and_tokenizer_sha256"):
        require_orbax_checkpoint(download_dir / TASK)
        require_checkout(root, OPENPI_REVISION)
        for name, revision in (
            ("aloha", "d1dc83afd89ded4379851257fe5d85632d31d5ec"),
            ("libero", "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"),
        ):
            path = f"third_party/{name}"
            entry = subprocess.check_output(["git", "-C", str(root), "ls-tree", "HEAD", path], text=True)
            if entry.strip() != f"160000 commit {revision}\t{path}" or not any((root / path).iterdir()):
                raise ValueError(f"missing pinned OpenPI submodule: {name}")
        manifest = load_manifest()
        if len(manifest["files"]) != 28:
            raise ValueError("expected 28 pinned model files")
        def snapshot():
            records = {}
            for item in manifest["files"]:
                path = local_path(download_dir, item["path"])
                stat = path.stat()
                if not path.is_file() or stat.st_size != item["bytes"]:
                    raise ValueError(f"missing or wrong-sized checkpoint file: {path}")
                records[item["path"]] = [stat.st_size, stat.st_mtime_ns, item["sha256"]]
            return records

        files = snapshot()
        if verified_model:
            verified = json.loads(verified_model.read_text())
            if (verified.get("status") != "cpu_preflight_passed"
                    or verified.get("checkpoint_revision") != manifest["revision"]
                    or verified.get("checkpoint_files") != files):
                raise ValueError("checkpoint no longer matches the CPU SHA-256 verification")
        else:
            verify_all(download_dir, manifest)
            if snapshot() != files:
                raise ValueError("checkpoint changed during SHA-256 verification")
        diagnostics.summary.update(checkpoint_revision=manifest["revision"], checkpoint_files=files)
        tokenizer = next(item for item in manifest["files"] if item["path"] == "tokenizer.model")
        norm = next(item for item in manifest["files"] if item["path"] == manifest["norm_stats"])
        verify_file(download_dir / norm["path"], norm)
        cache = Path(os.environ["OPENPI_DATA_HOME"]) / "big_vision/paligemma_tokenizer.model"
        verify_file(cache, tokenizer)

    with diagnostics.phase("openpi_import"):
        import jax
        from openpi.policies import policy_config
        from openpi.training import config

        if not Path(config.__file__).resolve().is_relative_to(root):
            raise ValueError("openpi imports must resolve inside the pinned checkout")
        task_config = config.get_config(CONFIG)
        if task_config.model.action_horizon != ACTION_SHAPE[0]:
            raise ValueError("unexpected action horizon")
    with diagnostics.phase("jax_device"):
        devices = jax.devices("gpu") if require_gpu else jax.devices()
        if require_gpu and len(devices) != 1:
            raise ValueError(f"expected exactly one visible JAX GPU, got {devices}")
        if preflight and (not devices or any(device.platform != "cpu" for device in devices)):
            raise ValueError("model preflight must use CPU devices only")
        diagnostics.summary.update(jax_version=jax.__version__, devices=[str(d) for d in devices])
    if preflight:
        return None
    with diagnostics.phase("orbax_restore_and_policy_create"):
        return policy_config.create_trained_policy(task_config, download_dir / TASK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verified-model", type=Path)
    args = parser.parse_args()
    if bool(args.openpi_root) != bool(args.download_dir):
        parser.error("--openpi-root and --download-dir must be supplied together")

    if (args.require_gpu or args.preflight_only) and not args.openpi_root:
        parser.error("model flags require --openpi-root and --download-dir")
    if args.require_gpu and args.preflight_only:
        parser.error("CPU preflight cannot require GPU")
    if args.verified_model and not args.require_gpu:
        parser.error("--verified-model is only for the GPU stage following a full CPU verification")
    if args.openpi_root and not args.diagnostics_dir:
        parser.error("checkpoint mode requires --diagnostics-dir")
    logging.basicConfig(level=logging.INFO)
    with ExitStack() as stack:
        diag = None
        summary = {}
        if args.diagnostics_dir:
            args.diagnostics_dir.mkdir(parents=True, exist_ok=False)
            events = stack.enter_context((args.diagnostics_dir / "phases.jsonl").open("w"))
            traces = stack.enter_context((args.diagnostics_dir / "stacks.log").open("w"))
            diag = Diagnostics(events, traces, summary, 30)
        try:
            if args.openpi_root:
                policy = load_model(args.openpi_root.resolve(), args.download_dir.resolve(),
                                    diag, args.require_gpu, args.preflight_only, args.verified_model)
                backend = "checkpoint"
            else:
                policy, backend = FakePolicy(), "fake"
        except BaseException as exc:
            summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            summary["status"] = "cpu_preflight_passed" if args.preflight_only else "ready"
        finally:
            if args.diagnostics_dir:
                (args.diagnostics_dir / "startup.json").write_text(json.dumps(summary, indent=2) + "\n")
        if not args.preflight_only:
            serve_policy(policy, backend, args.port, diag)


def serve_policy(policy, backend, port, diagnostics):
    # Keep a stateful policy and its JAX RNG on one inference path.
    from threading import Lock

    lock = Lock()
    handle = handler(policy, backend, diagnostics)

    def single_client(socket):
        if not lock.acquire(blocking=False):
            socket.send("policy service already has an active client")
            socket.close(code=1013)
            return
        try:
            handle(socket)
        finally:
            lock.release()

    with serve(single_client, "127.0.0.1", port, max_size=MAX_MESSAGE,
               compression=None, close_timeout=1) as server:
        print(f"{backend} policy on ws://127.0.0.1:{port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
