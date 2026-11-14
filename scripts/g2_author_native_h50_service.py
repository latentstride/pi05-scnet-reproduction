"""Optional shared-service extension for the independent author H50 diagnostic."""

from contextlib import ExitStack, contextmanager, nullcontext
from hashlib import sha256
from time import monotonic

import numpy as np
from websockets.exceptions import ConnectionClosed

from g2_author_h50_limits import BUDGET, ResourceCensored
import g2e_model_entry as legacy
import pace_model_smoke as base
from pace_runtime import (
    TASK,
    decode,
    msgpack_numpy,
    MAX_MESSAGE,
    connect,
    validate_request,
    validate_response,
)

RESET = "author_h50_reset"
INFER = "author_h50_infer"


def request_hash(request):
    validate_request(request)
    canonical = {
        "state": np.ascontiguousarray(request["state"]),
        "images": {
            name: np.ascontiguousarray(p)
            for name, p in sorted(request["images"].items())
        },
        "prompt": request["prompt"],
    }
    return sha256(msgpack_numpy.packb(canonical)).hexdigest()


def key_receipt(jax_random, key):
    return dict(
        impl=str(jax_random.key_impl(key)),
        words=np.asarray(jax_random.key_data(key)).tolist(),
    )


class LegacySocket:
    """Replay the first frame to the unchanged legacy handler."""

    def __init__(self, socket, frames, first):
        self.socket, self.frames, self.first = socket, frames, first
        self.greeting_sent = False

    def send(self, data):
        if not self.greeting_sent:
            self.greeting_sent = True
            return
        self.socket.send(data)

    def __iter__(self):
        yield self.first
        yield from self.frames

    def close(self, **kwargs):
        self.socket.close(**kwargs)


def handler(policy, backend, diagnostics=None, *, jax_random=None):
    """Accept either the unchanged G2 protocol or a single author RNG stream."""
    unchanged = legacy.handler(policy, backend, diagnostics)
    author_first = True

    def handle(socket):
        nonlocal author_first
        try:
            socket.send(
                msgpack_numpy.packb(
                    dict(
                        backend=backend,
                        task=TASK,
                        protocol=1,
                        g2e_rng_reset=1,
                        author_native_h50=1,
                    )
                )
            )
            frames = iter(socket)
            first = next(frames, None)
            if first is None:
                return
            reset = decode(first)
            if not isinstance(reset, dict) or reset.get("command") != RESET:
                unchanged(LegacySocket(socket, frames, first))
                return
            episode = reset.get("episode_id")
            if (
                set(reset) != {"command", "episode_id"}
                or not isinstance(episode, str)
                or not episode
            ):
                raise ValueError("invalid author H50 reset")
            rng = jax_random
            if rng is None:
                from jax import random as rng
            legacy.reset_rng(policy, 0, rng.key)
            socket.send(
                msgpack_numpy.packb(
                    dict(
                        command=RESET,
                        episode_id=episode,
                        seed=0,
                        key=key_receipt(rng, policy._rng),
                    )
                )
            )
            for query, frame in enumerate(frames):
                envelope = decode(frame)
                if (
                    not isinstance(envelope, dict)
                    or set(envelope)
                    != {"command", "episode_id", "query", "request_sha256", "request"}
                    or envelope["command"] != INFER
                    or envelope["episode_id"] != episode
                    or type(envelope["query"]) is not int
                    or envelope["query"] != query
                ):
                    raise ValueError("unexpected author H50 request sequence")
                request = envelope["request"]
                identity = request_hash(request)
                if identity != envelope["request_sha256"]:
                    raise ValueError("author H50 request hash mismatch")
                before = key_receipt(rng, policy._rng)
                next_key, sampling_key = rng.split(policy._rng)
                start = monotonic()
                phase = (
                    "author_h50_first_inference"
                    if author_first
                    else "author_h50_inference"
                )
                with diagnostics.phase(phase) if diagnostics else nullcontext():
                    response = policy.infer(request)
                    validate_response(response)
                author_first = False
                after = key_receipt(rng, policy._rng)
                if after != key_receipt(rng, next_key):
                    raise ValueError(
                        "author policy did not advance the pinned JAX stream exactly once"
                    )
                response = dict(
                    response,
                    author_rng=dict(
                        episode_id=episode,
                        query=query,
                        request_sha256=identity,
                        before=before,
                        sample=key_receipt(rng, sampling_key),
                        after=after,
                    ),
                    service_inference_s=monotonic() - start,
                )
                socket.send(msgpack_numpy.packb(response))
        except ConnectionClosed:
            return
        except Exception as exc:
            socket.send(f"{type(exc).__name__}: {exc}")
            socket.close(code=1011)

    return handle


class AuthorRemotePolicy:
    def __init__(self, socket, episode, remaining, request_s):
        self.socket, self.episode, self.remaining = socket, episode, remaining
        self.request_s = request_s
        self.query = 0
        ack = self.exchange(dict(command=RESET, episode_id=episode))
        if set(ack) != {"command", "episode_id", "seed", "key"} or (
            ack["command"],
            ack["episode_id"],
            ack["seed"],
        ) != (RESET, episode, 0):
            raise ValueError("author model RNG reset receipt mismatch")
        self.next_key = ack["key"]
        self.reset_receipt = ack

    def exchange(self, request):
        limit = monotonic() + min(self.request_s, self.remaining())
        self.socket.send(msgpack_numpy.packb(request))
        try:
            return decode(
                self.socket.recv(
                    timeout=max(0.0, min(limit - monotonic(), self.remaining()))
                )
            )
        except (TimeoutError, ResourceCensored) as error:
            # Once the request is sent a timed-out client cannot establish that
            # the shared worker finished its inference or released its lease.
            try:
                self.remaining()
            except ResourceCensored as elapsed:
                raise ResourceCensored(
                    elapsed.reason, request_in_flight=True
                ) from error
            raise ResourceCensored(
                "model_request_timeout", request_in_flight=True
            ) from error

    def infer(self, request):
        identity = request_hash(request)
        response = self.exchange(
            dict(
                command=INFER,
                episode_id=self.episode,
                query=self.query,
                request_sha256=identity,
                request=request,
            )
        )
        validate_response(response)
        rng = response["author_rng"]
        if (rng["episode_id"], rng["query"], rng["request_sha256"], rng["before"]) != (
            self.episode,
            self.query,
            identity,
            self.next_key,
        ):
            raise ValueError("author model RNG or request chain mismatch")
        self.next_key = rng["after"]
        self.query += 1
        return response


def shared_connection(uri, episode, remaining, *, request_s=BUDGET.request_s):
    """Connection to an existing service with one bound for handshake and greeting."""

    @contextmanager
    def connection():
        limit = monotonic() + min(request_s, remaining())
        with ExitStack() as stack:
            try:
                socket = stack.enter_context(
                    connect(
                        uri,
                        open_timeout=max(0.0, limit - monotonic()),
                        close_timeout=1,
                        max_size=MAX_MESSAGE,
                        compression=None,
                        proxy=None,
                    )
                )
                meta = decode(
                    socket.recv(timeout=max(0.0, min(limit - monotonic(), remaining())))
                )
            except TimeoutError as error:
                raise ResourceCensored("model_connection_timeout") from error
            expected = dict(
                backend="checkpoint", task=TASK, protocol=1, author_native_h50=1
            )
            if not isinstance(meta, dict) or any(
                meta.get(k) != v for k, v in expected.items()
            ):
                raise ValueError("unexpected author H50 service metadata")
            yield AuthorRemotePolicy(socket, episode, remaining, request_s)

    return connection


if __name__ == "__main__":
    # Keep the same frozen loader and single-client lock as the current service.
    base.handler = handler
    base.main()
