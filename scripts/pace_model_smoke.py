#!/usr/bin/env python3
"""Serve a deterministic CPU fake or the explicitly selected pinned checkpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

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


def handler(policy, backend):
    def handle(socket):
        try:
            socket.send(msgpack_numpy.packb({"backend": backend, "task": TASK, "protocol": 1}))
            for frame in socket:
                try:
                    request = decode(frame)
                    validate_request(request)
                    response = policy.infer(request)
                    validate_response(response)
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


def load_model(root, download_dir):
    from pace_download import load_manifest, verify_all, verify_file

    require_checkout(root, OPENPI_REVISION)
    manifest = load_manifest()
    verify_all(download_dir, manifest)
    tokenizer = next(item for item in manifest["files"] if item["path"] == "tokenizer.model")
    cache = Path(os.environ["OPENPI_DATA_HOME"]) / "big_vision/paligemma_tokenizer.model"
    verify_file(cache, tokenizer)

    from openpi.policies import policy_config
    from openpi.training import config

    if not Path(config.__file__).resolve().is_relative_to(root):
        raise ValueError("openpi imports must resolve inside the pinned checkout")
    return policy_config.create_trained_policy(config.get_config(CONFIG), download_dir / TASK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--download-dir", type=Path)
    args = parser.parse_args()
    if bool(args.openpi_root) != bool(args.download_dir):
        parser.error("--openpi-root and --download-dir must be supplied together")

    if args.openpi_root:
        policy = load_model(args.openpi_root.resolve(), args.download_dir.resolve())
        backend = "checkpoint"
    else:
        policy = FakePolicy()
        backend = "fake"

    # Keep a stateful policy and its JAX RNG on one inference path.
    from threading import Lock

    lock = Lock()
    handle = handler(policy, backend)

    def single_client(socket):
        if not lock.acquire(blocking=False):
            socket.send("policy service already has an active client")
            socket.close(code=1013)
            return
        try:
            handle(socket)
        finally:
            lock.release()

    with serve(single_client, "127.0.0.1", args.port, max_size=MAX_MESSAGE,
               compression=None, close_timeout=1) as server:
        print(f"{backend} policy on ws://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
