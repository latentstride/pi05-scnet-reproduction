"""The Step 5 native WebSocket contract; no JAX dependency on the simulator side."""

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
from time import time_ns
from urllib.parse import urlsplit

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

from scripts.step5_arx.adapter import native_observation
from scripts.step5_arx.data import CAMERAS, DIM, GRIP, HORIZON, PROMPT

SHAPE = (240, 320, 3)
MAX_MESSAGE = 16 * 1024**2
ACTION_SPACE = "arx_x5_absolute_drive_target"


def request_from(obs, query, command):
    request = native_observation(obs)
    validate_request(request, query)
    if not np.array_equal(request["state"], command):
        raise ValueError("sampled RoboTwin command differs from independent drive readback")
    return dict(request, query_index=query)


def validate_request(request, query):
    if (type(query) is not int or not 0 <= query < 2**32
            or set(request) - {"query_index"} != {"state", "images", "prompt"}
            or request.get("query_index", query) != query or request["prompt"] != PROMPT):
        raise ValueError("incorrect Step 5 request envelope")
    state = request["state"]
    if (not isinstance(state, np.ndarray) or state.shape != (DIM,)
            or state.dtype.kind != "f" or not np.isfinite(state).all()
            or np.any((state[GRIP] < 0) | (state[GRIP] > 1))):
        raise ValueError("expected finite 14D native command state")
    if set(request["images"]) != set(CAMERAS):
        raise ValueError("expected exactly three native cameras")
    for name, image in request["images"].items():
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != SHAPE:
            raise ValueError(f"{name}: expected uint8 RGB HWC {SHAPE}")


def actions_from(response):
    chunk = response.get("actions") if isinstance(response, dict) else None
    if (not isinstance(chunk, np.ndarray) or chunk.shape != (HORIZON, DIM)
            or chunk.dtype.kind != "f" or not np.isfinite(chunk).all()):
        raise ValueError("expected a finite native [50,14] WebSocket response")
    # Range-check only consumed rows via the native safety gate; the H50 tail is discarded.
    return chunk


def metadata_for(variant, plan):
    return dict(variant=variant, action_space=ACTION_SPACE, action_horizon=HORIZON,
                action_dim=DIM, seed=plan["policy_seed"], dataset_sha256=plan["dataset_sha256"])


def validate_metadata(metadata, variant, plan, fixture):
    if not isinstance(metadata, dict) or any(
            metadata.get(k) != v for k, v in metadata_for(variant, plan).items()):
        raise ValueError("WebSocket handshake does not match the trial")
    identity = metadata.get("step6_identity")
    step = metadata.get("selected_step")
    if fixture:
        if identity != {"kind": "protocol_fixture"} or step is not None:
            raise ValueError("expected a CPU test fixture, not a model")
        return
    if not isinstance(identity, dict) or identity.get("kind") != "step5_checkpoint":
        raise ValueError("start the Step 6 identity wrapper around Step 5 serve")
    if identity.get("pace_revision") != plan["checkpoint_revision"]:
        raise ValueError("unexpected frozen PACE checkpoint")
    if set(identity) != {"kind", "pace_revision", "pace_manifest_sha256", "norm_stats_sha256",
                         "training_identity_sha256", "selected_checkpoint_sha256"}:
        raise ValueError("incomplete model provenance")
    for k in ("pace_manifest_sha256", "norm_stats_sha256", "training_identity_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", str(identity[k])):
            raise ValueError(f"invalid identity field: {k}")
    selected = identity["selected_checkpoint_sha256"]
    if (variant == "frozen" and (step is not None or selected is not None)
            or variant == "lora" and (type(step) is not int or step <= 0
                                     or not re.fullmatch("[0-9a-f]{64}", str(selected)))):
        raise ValueError("selected checkpoint does not match variant")


@contextmanager
def socket_for(uri, timeout):
    url = urlsplit(uri)
    if url.scheme != "ws" or url.hostname not in {"127.0.0.1", "::1", "localhost"} or url.username:
        raise ValueError("the Step 5 service must use a loopback WebSocket")
    with connect(uri, open_timeout=timeout, close_timeout=1, max_size=MAX_MESSAGE,
                 compression=None, proxy=None, ping_interval=None) as socket:
        yield socket


def decode(frame):
    if isinstance(frame, str):
        raise RuntimeError(f"policy server returned text: {frame[:2000]}")
    return msgpack_numpy.unpackb(frame)


def record_wire(directory, query, direction, frame):
    is_binary = isinstance(frame, bytes)
    contents = frame if is_binary else frame.encode("utf-8")
    file = directory / f"{query:03d}.{direction}.{'msgpack' if is_binary else 'txt'}"
    file.write_bytes(contents)
    return dict(file=str(file.relative_to(directory.parent)), sha256=sha256(contents).hexdigest(),
                nbytes=len(contents), unix_ns=time_ns())
