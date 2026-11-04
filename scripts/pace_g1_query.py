#!/usr/bin/env python3
"""Query the already loaded checkpoint on G1.1's real initial observation."""
import argparse
import json
import math
from pathlib import Path
from time import monotonic

import numpy as np

from pace_client_smoke import ARM_INDICES, GRIPPERS, JOINT_LIMIT_TOLERANCE_RAD
from pace_runtime import CAMERAS, open_policy, validate_request, validate_response


def run(source, output, uri, timeout):
    summary = json.loads((source / "summary.json").read_text())
    if summary["status"] != "complete" or summary["environment"] != "robotwin":
        raise ValueError("G1.2 requires a completed native G1.1 run")
    with np.load(source / "initial_observation.npz", allow_pickle=False) as data:
        request = {"state": data["state"], "images": {key: data[key] for key in CAMERAS},
                   "prompt": summary["prompt"]}
    validate_request(request)
    start = monotonic()
    with open_policy(uri, "checkpoint", timeout) as policy:
        response = policy.infer(request)
    elapsed = monotonic() - start
    actions = validate_response(response)
    np.save(output / "checkpoint_actions.npy", actions, allow_pickle=False)
    limits = np.asarray(summary["arm_joint_limits_rad"])
    arm = actions[:4, ARM_INDICES]
    if (limits.shape != (12, 2) or np.any(arm < limits[:, 0] - JOINT_LIMIT_TOLERANCE_RAD)
            or np.any(arm > limits[:, 1] + JOINT_LIMIT_TOLERANCE_RAD)):
        raise ValueError("checkpoint prefix exceeds native Aloha joint limits")
    service_seconds = float(response["service_inference_s"])
    if not math.isfinite(service_seconds) or service_seconds <= 0:
        raise ValueError("invalid service timing")
    grippers = actions[:, GRIPPERS]
    result = {"passed": True, "scene_seed": summary["scene_seed"], "shape": list(actions.shape),
              "arm_order": summary["arm_joint_names"], "control": "absolute_native_qpos",
              "gripper_raw_min": float(grippers.min()), "gripper_raw_max": float(grippers.max()),
              "gripper_values_clipped_by_native_controller": int(((grippers < 0) | (grippers > 1)).sum()),
              "roundtrip_s": elapsed, "service_inference_s": service_seconds,
              "policy_timing": response.get("policy_timing", {})}
    (output / "query_verification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--timeout", required=True, type=float)
    args = parser.parse_args()
    run(args.source, args.output, args.uri, args.timeout)
