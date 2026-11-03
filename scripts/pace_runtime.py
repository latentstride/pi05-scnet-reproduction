"""Shared PACE smoke contract. Imports neither the simulator nor JAX."""
from __future__ import annotations

from contextlib import contextmanager
import subprocess

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

TASK = "handover_block"
CONFIG = "pi05_robotwin_handover_block"
PACE_REVISION = "fd2e624cc19325cc9b20a620c82d51c0d8e25cc1"
OPENPI_REVISION = "f4e52e6de4cc487a11585a3a7414d610b2883dae"
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
MAX_MESSAGE = 16 * 1024 * 1024
ACTION_SHAPE = (50, 14)


def require_checkout(root, revision):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("rev-parse", "HEAD") != revision or git("status", "--porcelain", "-uno"):
        raise ValueError(f"expected clean tracked checkout {revision} at {root}")


def finite_array(value, shape):
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype.kind != "f" or not np.isfinite(value).all()):
        raise ValueError(f"expected finite float array {shape}")
    return value


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {"state", "images", "prompt"}:
        raise ValueError("expected state, images and prompt")
    finite_array(request["state"], (14,))
    images = request["images"]
    if not isinstance(images, dict) or set(images) != set(CAMERAS):
        raise ValueError("expected all three PACE cameras")
    for image in images.values():
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[0] != 3
                or not all(0 < side <= 1024 for side in image.shape[1:])):
            raise ValueError("expected uint8 CHW RGB, sides <= 1024")
    prompt = request["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2048:
        raise ValueError("expected a nonempty language prompt")


def validate_response(response):
    if not isinstance(response, dict) or "actions" not in response:
        raise ValueError("response missing actions")
    return finite_array(response["actions"], ACTION_SHAPE)


def decode(frame):
    if isinstance(frame, str):
        raise RuntimeError(f"policy service error: {frame[:500]}")
    return msgpack_numpy.unpackb(frame)


class RemotePolicy:
    def __init__(self, socket, timeout):
        self.socket = socket
        self.timeout = timeout

    def infer(self, request):
        validate_request(request)
        self.socket.send(msgpack_numpy.packb(request))
        response = decode(self.socket.recv(timeout=self.timeout))
        validate_response(response)
        return response


@contextmanager
def open_policy(uri, backend, timeout):
    with connect(uri, open_timeout=timeout, close_timeout=1, max_size=MAX_MESSAGE,
                 compression=None, proxy=None) as socket:
        metadata = decode(socket.recv(timeout=timeout))
        expected = {"backend": backend, "task": TASK, "protocol": 1}
        if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError(f"unexpected policy metadata; expected {expected}")
        yield RemotePolicy(socket, timeout)
