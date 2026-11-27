"""Loopback OpenPI websocket policy for the paired native ARX experiment."""
import argparse
import json
from pathlib import Path

import jax
from openpi.serving import websocket_policy_server

from . import model
from .config import SETTINGS
from .data import Dataset
from .train import ensure_paths, fingerprint, gpu_device, restore


class Session:
    def __init__(self, policy, predict, state, seed):
        self.policy = policy
        self.predict = predict
        self.state = state
        self.seed = seed

    def infer(self, request):
        # Explicit query indices let paired clients reset and reuse the same noise.
        query = request["query_index"]
        if not isinstance(query, int) or not 0 <= query < 2**32 or "actions" in request:
            raise ValueError("Expected a nonnegative query_index and an observation only")
        key = jax.random.fold_in(jax.random.key(self.seed), query)
        return {"actions": self.policy.infer(self.predict, self.state, key, request)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("frozen", "lora"))
    for name in ("project-root", "dataset", "data-audit", "run"):
        parser.add_argument(f"--{name}", required=name != "run", type=Path)
    parser.add_argument("--cpu-receipt", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=SETTINGS.seed)
    args = parser.parse_args()
    if args.variant == "lora" and args.run is None:
        parser.error("Pass the completed training --run for the lora policy")
    ensure_paths(args)
    identity = fingerprint(args)
    gpu_device()
    parts, baseline, _ = model.load_models(model.validate_install(args.project_root, full_hash=True))
    graph, initial, frozen, _, opt = parts
    adapted = args.variant == "lora"
    selected = None
    if adapted:
        if json.loads((args.run / "identity.json").read_text()) != identity:
            raise RuntimeError("Policy data, configuration or release differs from training")
        selected = json.loads((args.run / "selected.json").read_text())["step"]
        fitted, _ = restore(args.run / "checkpoints" / f"step-{selected:04d}", initial, opt)
        state = (fitted, frozen)
    else:
        graph, state = baseline
    with Dataset(args.dataset) as dataset:
        policy = model.NativePolicy(dataset.norm_stats)
    session = Session(policy, model.sampler(graph, adapted=adapted), state, args.seed)
    websocket_policy_server.WebsocketPolicyServer(
        session, host="127.0.0.1", port=args.port,
        metadata={"variant": args.variant, "dataset_sha256": identity["dataset_sha256"],
                  "action_space": "arx_x5_absolute_drive_target", "action_horizon": 50,
                  "action_dim": 14, "selected_step": selected, "seed": args.seed},
    ).serve_forever()


if __name__ == "__main__":
    main()
