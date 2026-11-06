#!/usr/bin/env python3
"""Run a real native/retargeted diagnostic episode using the existing model service."""
import argparse
import json
from pathlib import Path
import signal

import numpy as np

from g2_episode import ExpertWaypoints, run_episode, save_json
from g2_orchestrator import interrupt_signals
from g2_task_runtime import RuntimeSession, SessionAdapter, real_scene, validate_config, verify_runtime, inspect_assets
from pace_runtime import open_policy


def execute(config, factory, policy, output, expert=None):
    validate_config(config)
    output = Path(output).resolve()
    spec = dict(data_kind=config["data_kind"], episode_id=config["episode_id"],
                condition=config["condition"], prompt=config["prompt"], window=config["window"],
                max_actions=config["max_actions"], timeout_s=config["timeout_s"],
                native_grip_max_clip=config["native_grip_max_clip"])
    session = RuntimeSession(factory)
    adapter = SessionAdapter(session) if config["embodiment"] == "ARX-X5" else None
    return run_episode(spec, session, policy, output, adapter=adapter, expert=expert)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pace-root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--asset-audit", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--uri")
    parser.add_argument("--policy-timeout-s", type=float, default=120)
    parser.add_argument("--expert", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["episode_id"] = args.episode_id
    validate_config(config)
    pace_root = args.pace_root.resolve()
    source_audit = json.loads(args.source_audit.read_text())
    verify_runtime(pace_root, source_audit)
    inspection=inspect_assets(config,args.assets.resolve(),args.asset_audit.resolve())
    if args.inspect_only:
        print(json.dumps(dict(**inspection,runtime_source_verified=True, data_kind=config["data_kind"],
                              renderer_created=False, gpu_requested=False, formal_ready=False)))
        return
    output = args.output.resolve()
    def factory(spec):
        return real_scene(config=config, spec=spec, pace_root=pace_root, assets_root=args.assets.resolve(),
                          asset_audit=args.asset_audit.resolve(), source_audit=source_audit, output=output)
    if bool(args.uri) == bool(args.expert):
        raise ValueError("provide exactly one existing model URI or an expert source")
    expert = None
    if args.expert:
        from hashlib import sha256
        expected = config["expert_sha256"]
        if sha256(args.expert.read_bytes()).hexdigest() != expected:
            raise ValueError("expert file identity mismatch")
        expert = ExpertWaypoints(np.load(args.expert, allow_pickle=False), str(args.expert.resolve()))
    with interrupt_signals():
        if expert is None:
            with open_policy(args.uri, "checkpoint", args.policy_timeout_s) as policy:
                result = execute(config, factory, policy, output)
        else:
            result = execute(config, factory, None, output, expert=expert)
    raise SystemExit(0 if result["outcome"] in ("success", "task_unsuccessful") else 1)


if __name__ == "__main__":
    main()
