"""Read a completed Step 5 run and serve its frozen or adapted policy on loopback."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

if __package__:
    from .checkpoint import file_sha, inspect_checkpoint, read_json
else:
    from checkpoint import file_sha, inspect_checkpoint, read_json


def identity_digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checkpoint_digest(path):
    entries = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("unexpected checkpoint symlink")
        if item.is_file():
            entries[item.relative_to(path).as_posix()] = file_sha(item)
    if not entries:
        raise ValueError("empty LoRA checkpoint")
    return identity_digest(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("frozen", "lora"))
    for name in ("project-root", "dataset", "data-audit", "run"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-only", action="store_true",
                        help="verify checkpoint metadata and the admitted training release on CPU")
    args = parser.parse_args()
    plan = read_json(Path(__file__).resolve().parents[2] / "docs/step6/pairs.json")
    selected, training_identity, selected_path = inspect_checkpoint(args.run, plan)

    # env.sh supplies the Step 5 release that trained this checkpoint.
    from openpi.serving import websocket_policy_server
    from scripts.step5_arx import model, train
    from scripts.step5_arx.data import Dataset
    from scripts.step5_arx.serve import Session

    train.ensure_paths(args)
    if training_identity != train.fingerprint(args) or not train.complete_checkpoint(selected_path):
        raise RuntimeError("training identity or Orbax finalization check failed")
    model_path = model.validate_install(args.project_root)
    if args.verify_only:
        print(json.dumps(dict(status="CHECKPOINT_METADATA_VERIFIED", step=selected["step"],
                              training_identity_sha256=identity_digest(training_identity))))
        return

    device = model.cuda_identity()
    platform = train.platform()
    if platform.platform != "gpu" or "RTX 4090" not in platform.device_kind.upper():
        raise RuntimeError("expected one allocated RTX 4090")
    print(json.dumps(dict(stage="cuda_device", **device)), flush=True)
    adapted = args.variant == "lora"
    selected_sha = checkpoint_digest(selected_path) if adapted else None
    parts, baseline, _ = model.load_models(model_path)
    if adapted:
        graph, initial, frozen, _, opt = parts
        fitted, _ = train.restore(selected_path, initial, opt)
        if checkpoint_digest(selected_path) != selected_sha:
            raise RuntimeError("checkpoint changed during restore")
        state = (fitted, frozen)
    else:
        graph, state = baseline
    del parts, baseline

    with Dataset(args.dataset) as dataset:
        policy = model.NativePolicy(dataset.norm_stats)
        stats_sha = identity_digest(dataset.norm_stats)
    identity = dict(kind="step5_checkpoint", pace_revision=training_identity["pace_checkpoint"],
                    pace_manifest_sha256=training_identity["manifest_sha256"],
                    norm_stats_sha256=stats_sha, training_identity_sha256=identity_digest(training_identity),
                    selected_checkpoint_sha256=selected_sha)
    session = Session(policy, model.sampler(graph, adapted=adapted), state, args.seed)
    metadata = dict(variant=args.variant, dataset_sha256=training_identity["dataset_sha256"],
                    action_space="arx_x5_absolute_drive_target", action_horizon=50, action_dim=14,
                    selected_step=selected["step"] if adapted else None, seed=args.seed,
                    step6_identity=identity)
    websocket_policy_server.WebsocketPolicyServer(
        session, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
