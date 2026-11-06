#!/usr/bin/env python3
"""Convert G2 evidence to the existing evaluation schema without filling unknown metrics."""
import argparse
from copy import deepcopy
from pathlib import Path
import math

import numpy as np
from jsonschema import Draft202012Validator

from g2_episode import save_json
from summarize_experiments import SCHEMA, decode, validate_episode


def finite_metric(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {label}")
    return value


def episode_record(directory, identity):
    directory = Path(directory)
    result = decode((directory / "result.json").read_text())
    if result["data_kind"] not in ("synthetic", "diagnostic"):
        raise ValueError("formal export requires a frozen roster gate")
    if result["episode_id"] != identity["episode_id"]:
        raise ValueError("episode identity does not match evidence")
    if result["outcome"] == "running":
        raise ValueError("unfinished result requires explicit interruption recovery")
    if result["condition"] != identity["condition"]["id"]:
        raise ValueError("condition identity differs from evidence")
    if result["window"] != identity["condition"]["window_actions"]:
        raise ValueError("window differs from condition metadata")
    events = [decode(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    steps = [event for event in events if event.get("kind") == "executed_step"]
    expert = identity["condition"]["action_source"] == "expert_replay"
    observed_expert = any(event.get("kind") == "expert_source" for event in events)
    if expert != observed_expert and result["initial"] is not None:
        raise ValueError("expert/policy identity differs from episode evidence")
    # Missing clock or Cartesian error samples stay unknown rather than becoming zero.
    elapsed = None
    if steps and all("sim_s" in event for event in steps):
        initial_s = result.get("initial_sim_s")
        clock = [event["sim_s"] for event in steps]
        if initial_s is not None:
            initial_s = finite_metric(initial_s, "initial sim time")
            if any(not math.isfinite(x) for x in clock) or any(b <= a for a, b in zip([initial_s] + clock, clock)):
                raise ValueError("nonmonotonic execution clock")
            elapsed = clock[-1] - initial_s
    elif steps and all("period_s" in event for event in steps):
        elapsed = sum(finite_metric(event["period_s"], "executed period") for event in steps)
    trajectory = None
    errors = [event for event in steps if "position_error_m" in event and "orientation_error_deg" in event]
    if errors:
        p = np.asarray([e["position_error_m"] for e in errors], dtype=float).ravel()
        a = np.asarray([e["orientation_error_deg"] for e in errors], dtype=float).ravel()
        if p.size != a.size or not np.isfinite(p).all() or not np.isfinite(a).all() or np.any(p < 0) or np.any((a < 0) | (a > 180)):
            raise ValueError("invalid measured trajectory error")
        trajectory = dict(reference_id=identity["trajectory_reference_id"], n=int(p.size),
                          position_rmse_m=float(np.sqrt(np.mean(p*p))), position_p95_m=float(np.percentile(p, 95)),
                          orientation_mean_deg=float(a.mean()), orientation_p95_deg=float(np.percentile(a, 95)))
    outcome = result["outcome"]
    failure = result["failure"]
    if outcome == "timeout":
        outcome = {"reset": "environment_error", "observation": "adapter_error", "inference": "policy_failed",
                   "planning": "planning_failed", "control": "control_error"}.get(failure["phase"], "environment_error")
    if outcome == "unreachable":
        outcome = "ik_unreachable"
    if failure:
        stage = "ik" if outcome in ("ik_failed", "ik_unreachable") else failure["phase"]
        failure = dict(stage=stage, code=failure["code"], detail=failure["detail"],
                       step=result["executed"], artifact=str(directory / "events.jsonl"))
    control_calls = result.get("control_calls")
    controls = (dict(calls=control_calls, wall_s=result["control_s"]) if control_calls is not None and (control_calls or not result["control_s"]) else None)
    row = {key: deepcopy(identity[key]) for key in
           ("run_lock_sha256", "episode_id", "task", "seed", "comparison_id", "pair_id", "condition", "environment", "match_status")}
    row.update(schema_version=1, data_kind=result["data_kind"], initial=result["initial"],
               outcome=outcome, failure=failure, episode_artifact=str(directory),
               metrics=dict(trajectory=trajectory,
                            planning=dict(attempts=result["plan_calls"], failures=result["plan_failures"]),
                            limits=None, gripper=None,
                            inference=None if expert else dict(calls=result["queries"], wall_s=result["inference_s"]),
                            control=controls,
                            actions=(dict(windows=result["queries"], requested=result["requested"], executed=result["executed"],
                                          sim_elapsed_s=elapsed) if elapsed is not None else None)))
    Draft202012Validator(decode(SCHEMA.read_text())).validate(row)
    validate_episode(row)
    return row


def export_manifest(manifest, output):
    entries = decode(Path(manifest).read_text())
    rows = [episode_record(Path(entry["directory"]), entry["identity"]) for entry in entries]
    identities = [(row["run_lock_sha256"], row["episode_id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate episode identity")
    save_json(Path(output), rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_manifest(args.manifest, args.output)
