"""Run the unchanged Step 5 service and attach identity from its admitted release."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from unittest.mock import patch


def checkpoint_digest(path):
    entries = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("unexpected checkpoint symlink")
        if item.is_file():
            with item.open("rb") as stream:
                import hashlib
                entries[item.relative_to(path).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    if not entries:
        raise ValueError("empty LoRA checkpoint")
    return identity_digest(entries)


def identity_digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("frozen", "lora"))
    for name in ("project-root", "dataset", "data-audit", "cpu-receipt", "run"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Put the immutable, admitted Step 5 release first when invoking this wrapper.
    # This import must resolve to the exact package bytes used for training.
    from scripts.step5_arx import serve, train
    from scripts.step5_arx.config import SETTINGS
    from scripts.step5_arx.data import Dataset

    train.ensure_paths(args)
    selected = json.loads((args.run / "selected.json").read_text())
    training_identity = json.loads((args.run / "identity.json").read_text())
    if selected.get("last_step") != SETTINGS.train_steps or training_identity != train.fingerprint(args):
        raise RuntimeError("a completed Step 5 run must match the admitted release")
    step = selected["step"]
    if type(step) is not int or not 0 < step <= SETTINGS.train_steps:
        raise ValueError("invalid selected LoRA step")
    selected_path = args.run / "checkpoints" / f"step-{step:04d}"
    if not train.complete_checkpoint(selected_path):
        raise RuntimeError("selected checkpoint is not a completed checkpoint")
    with Dataset(args.dataset) as dataset:
        stats_sha = identity_digest(dataset.norm_stats)
    selected_sha = checkpoint_digest(selected_path) if args.variant == "lora" else None
    identity = dict(kind="step5_checkpoint", pace_revision=training_identity["pace_checkpoint"],
                    pace_manifest_sha256=training_identity["manifest_sha256"],
                    norm_stats_sha256=stats_sha, training_identity_sha256=identity_digest(training_identity),
                    selected_checkpoint_sha256=selected_sha)

    server = serve.websocket_policy_server.WebsocketPolicyServer

    def with_identity(policy, **kwargs):
        metadata = kwargs["metadata"]
        if metadata["selected_step"] != (step if args.variant == "lora" else None):
            raise RuntimeError("served step differs from the recorded selection")
        if selected_sha is not None and checkpoint_digest(selected_path) != selected_sha:
            raise RuntimeError("checkpoint changed during restore")
        return server(policy, **dict(kwargs, metadata=dict(metadata, step6_identity=identity)))

    with patch.object(serve.websocket_policy_server, "WebsocketPolicyServer", with_identity):
        serve.main()


if __name__ == "__main__":
    main()
