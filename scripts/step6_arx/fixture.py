"""Loopback Step 5 wire fixture using real Step 5 normalization; never a model."""

from contextlib import contextmanager
from hashlib import sha256
from threading import Thread

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.server import serve

from scripts.step5_arx.adapter import NativeAdapter, MODEL_DIM, normalization
from scripts.step5_arx.data import DIM, HORIZON, Dataset, delta

from .protocol import MAX_MESSAGE, decode, metadata_for, validate_request


def fixture_adapter(path, plan):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    if digest.hexdigest() != plan["dataset_sha256"]:
        raise ValueError("fixture dataset SHA-256 differs from Step 5")
    with Dataset(path) as dataset:
        return NativeAdapter(dataset.norm_stats)


@contextmanager
def serve_fixture(args, variant, plan, adapter):
    metadata = dict(metadata_for(variant, plan), selected_step=None,
                    step6_identity={"kind": "protocol_fixture"})

    def handler(socket):
        socket.send(msgpack_numpy.packb(metadata))
        for frame in socket:
            request = decode(frame)
            query = request["query_index"]
            validate_request(request, query)
            adapter.inputs(request)
            targets = np.repeat(request["state"][None, :].astype(np.float32), HORIZON, axis=0)
            # A small nonzero left joint move is observable but bounded by the original gate.
            targets[:, 0] += np.minimum(np.arange(1, HORIZON + 1), 4) * np.float32(.00025)
            latent = np.zeros((HORIZON, MODEL_DIM), np.float32)
            latent[:, :DIM] = normalization(delta(request["state"].astype(np.float32), targets),
                                             adapter.stats["actions"])
            action = adapter.control_outputs(request["state"], latent)
            if query == 1 and variant == args.invalid_variant:
                action[0, 0] = 99.  # Valid wire shape/dtype; real ARX joint gate must reject it.
            socket.send(msgpack_numpy.packb({"actions": action}))

    with serve(handler, "127.0.0.1", 0, compression=None, max_size=MAX_MESSAGE) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        finally:
            server.shutdown()
            thread.join()
