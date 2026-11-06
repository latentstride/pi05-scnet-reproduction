"""G2-E model entry: pinned G1 loader, explicit per-trial common RNG reset.

No model weights, transforms, observation contract or sampling settings change.
This entry is never used by G1 or legacy G2.
"""

from contextlib import nullcontext
from time import monotonic
from websockets.exceptions import ConnectionClosed
import pace_model_smoke as base
from pace_runtime import (
    TASK,
    decode,
    msgpack_numpy,
    validate_request,
    validate_response,
)


def reset_rng(policy, seed, key_factory=None):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("RNG seed must be uint32")
    if getattr(policy, "_is_pytorch_model", True) or not hasattr(policy, "_rng"):
        raise ValueError("expected pinned JAX policy with explicit RNG state")
    if key_factory is None:
        from jax.random import key

        key_factory = key
    policy._rng = key_factory(seed)


def handler(policy, backend, diagnostics=None):
    first = True

    def handle(socket):
        nonlocal first
        try:
            socket.send(
                msgpack_numpy.packb(
                    dict(backend=backend, task=TASK, protocol=1, g2e_rng_reset=1)
                )
            )
            reset = False
            for frame in socket:
                try:
                    request = decode(frame)
                    if (
                        isinstance(request, dict)
                        and request.get("command") == "g2e_reset_rng"
                    ):
                        if (
                            reset
                            or set(request) != {"command", "seed", "episode_id"}
                            or not request["episode_id"]
                        ):
                            raise ValueError(
                                "exactly one RNG reset per trial connection"
                            )
                        reset_rng(policy, request["seed"])
                        reset = True
                        socket.send(
                            msgpack_numpy.packb(
                                dict(
                                    command="g2e_rng_ready",
                                    seed=request["seed"],
                                    episode_id=request["episode_id"],
                                )
                            )
                        )
                        continue
                    if not reset:
                        raise ValueError("trial RNG reset required before inference")
                    validate_request(request)
                    start = monotonic()
                    name = "first_jax_compile_and_inference" if first else "inference"
                    with diagnostics.phase(name) if diagnostics else nullcontext():
                        response = policy.infer(request)
                        validate_response(response)
                    first = False
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


def reset_remote(policy, seed, episode_id):
    request = dict(command="g2e_reset_rng", seed=seed, episode_id=episode_id)
    policy.socket.send(msgpack_numpy.packb(request))
    reply = decode(policy.socket.recv(timeout=policy.timeout))
    if reply != dict(command="g2e_rng_ready", seed=seed, episode_id=episode_id):
        raise ValueError("model RNG reset receipt mismatch")
    return reply


if __name__ == "__main__":
    base.handler = handler
    base.main()
