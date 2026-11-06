"""Quantify saved policy prefixes and expert timing without evaluating a model."""

import argparse
import json
from pathlib import Path
import numpy as np


def analyze(evidence, experts, output):
    if (
        output.resolve() == evidence.resolve()
        or evidence.resolve() in output.resolve().parents
    ):
        raise ValueError("analysis output must be outside the input evidence")
    actions = json.loads((evidence / "action-audit.json").read_text())
    audit = json.loads((evidence / "trial-audit.json").read_text())
    metrics = {}
    prefix_counts = {}
    for name, rows in actions.items():
        if "A-native" not in name:
            continue
        seed = int(name.split("-")[0][4:])
        events = json.loads(
            (evidence / "selected" / name / "events-selected.json").read_text()
        )
        inputs = [r for r in events if r.get("kind") == "policy_input"]
        states = np.array([r["state"] for r in inputs])
        actual_prefix = np.concatenate([r["executed_prefix"] for r in rows])
        closure = np.flatnonzero(actual_prefix[:, 6] < 0.5)
        future = [r["first_close_index"][0] for r in rows]
        future_found = [x for x in future if x is not None]
        z = np.load(
            experts / f"expert-dense-episode{seed}-20261106/trace.npz",
            allow_pickle=False,
        )
        expert = z["target"]
        frame_ticks = z["frame_ticks"]
        expert_close = np.flatnonzero(expert[:, 6] < 0.5)
        query_of_error = None
        first = audit[name]["first_tracking_error_over_50mm"]
        if first:
            query_of_error = sum(
                r.get("kind") == "snapshot" and r["source_line"] < first["line"]
                for r in events
            )
        metrics[name] = dict(
            queries=len(rows),
            executed=len(actual_prefix),
            prompt=inputs[0]["prompt"],
            observed_state_grip_min=states[:, [6, 13]].min(0).tolist(),
            executed_target_grip_min=actual_prefix[:, [6, 13]].min(0).tolist(),
            first_executed_left_closure_action=int(closure[0])
            if len(closure)
            else None,
            chunks_with_future_left_closure=len(future_found),
            first_future_left_closure_query=next(
                (i + 1 for i, x in enumerate(future) if x is not None), None
            ),
            median_future_left_closure_horizon=float(np.median(future_found))
            if future_found
            else None,
            last_ten_future_left_closure_horizons=future[-10:],
            first_tracking_error_over_50mm_query=query_of_error,
            first_tracking_error_over_50mm_tick=first["event"]["tick"]
            if first
            else None,
            first_tracking_error_over_50mm_sim_s=first["event"]["sim_s"]
            if first
            else None,
            tracking_position_rmse_m=audit[name]["tracking_position_rmse_m"],
            tracking_position_max_m=audit[name]["tracking_position_max_m"],
            expert_frames=len(frame_ticks),
            expert_physics_steps=len(expert) - 1,
            expert_sim_duration_s=(len(expert) - 1) * 0.004,
            expert_first_left_closure_physics_tick=int(expert_close[0])
            if len(expert_close)
            else None,
            actual_sim_duration_s=json.loads(
                (evidence / "selected" / name / "formal-record.json").read_text()
            )["metrics"]["actions"]["sim_elapsed_s"],
            raw_action_shapes=sorted({tuple(r["shape"]) for r in rows}),
            all_raw_actions_finite=all(r["finite"] for r in rows),
            max_raw_grip_clip_amount=max(r["raw_clip_amount"] for r in rows),
            object_trajectory_recorded=False,
        )
        prefix_counts[name] = {
            str(w): dict(
                chunks_retaining_left_close=sum(
                    x is not None and x < w for x in future
                ),
                chunks=len(rows),
            )
            for w in (1, 4, 8, 16, 32, 50)
        }
    output.mkdir(parents=True, exist_ok=False)
    (output / "aloha-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "prefix-counterfactual.json").write_text(
        json.dumps(
            dict(
                scope="fixed recorded chunks only; alternative prefixes change future observations and require closed-loop validation",
                cases=prefix_counts,
            ),
            indent=2,
        )
        + "\n"
    )
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("evidence", "experts", "output"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    analyze(a.evidence, a.experts, a.output)
