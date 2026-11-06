"""Reproduce saved native plans and inspect fixed-chunk prefix alternatives on CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from g2_adapter import G2Error
from g2_assets import load_arms
from g2_episode import native_prefix


def audit(raw, release, assets, output):
    raw = raw.resolve(strict=True)
    output = output.resolve()
    if output == raw or raw in output.parents:
        raise ValueError("output must be outside the original run")
    lock = json.loads((release / "run_lock.json").read_text())
    report = {}
    for directory in sorted((raw / "trials").glob("*A-native*")):
        config_path = release / "configs" / f"{directory.name}.json"
        if (
            sha256(config_path.read_bytes()).hexdigest()
            != lock["bindings"][directory.name]["config_sha256"]
        ):
            raise ValueError("configuration differs from the original run lock")
        config = json.loads(config_path.read_text())
        arms = load_arms(
            assets,
            release / "asset-audit.json",
            "aloha-agilex",
            config["calibration"]["aloha-agilex"],
        )
        limits = [arm.chain.limits for arm in arms]
        sweep = {
            str(w): dict(valid_prefixes=0, prefixes_with_left_closure=0)
            for w in range(1, 51)
        }
        matched = 0
        for path in sorted(directory.glob("action-*.npy")):
            actions = np.load(path, allow_pickle=False)
            original = np.load(
                directory / path.name.replace("action-", "plan-"), allow_pickle=False
            )
            actual = native_prefix(
                actions,
                config["window"],
                limits,
                config["native_grip_max_clip"],
                lambda event: None,
            )
            if not np.array_equal(actual, original):
                raise ValueError(f"native plan mismatch at {path}")
            matched += 1
            for window in range(1, 51):
                try:
                    prefix = native_prefix(
                        actions,
                        window,
                        limits,
                        config["native_grip_max_clip"],
                        lambda event: None,
                    )
                except G2Error:
                    continue
                sweep[str(window)]["valid_prefixes"] += 1
                sweep[str(window)]["prefixes_with_left_closure"] += int(
                    np.any(prefix[:, 6] < 0.5)
                )
        report[directory.name] = dict(
            saved_plans_exactly_reproduced=matched,
            prefix_sweep=sweep,
            scope="fixed saved chunks; raw limits/clipping only, not a dynamic rollout",
        )
    with output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "release", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.raw, args.release, args.assets, args.output)
