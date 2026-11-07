"""Summarize locked paired diagnostic rows and their saved physical evidence."""

import argparse
import json
from pathlib import Path

from pace_g1_orchestrate import persist


def steps(path):
    if not path.exists():
        return {}
    count = calls = 0
    sim_s = 0.0
    position = angle = None
    pairs = []
    holder_transitions = []
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A killed writer can leave a partial last record.
                if line.endswith("\n"):
                    raise
                continue
            count += row.get("physics_step_count") or 0
            calls += row.get("control_call_count") or 0
            sim_s = max(sim_s, row.get("sim_s") or 0.0)
            if row.get("position_error_m"):
                position = max(position or 0.0, *row["position_error_m"])
            if row.get("orientation_error_deg"):
                angle = max(angle or 0.0, *row["orientation_error_deg"])
            if len(pairs) < 10 and row.get("collisions"):
                pairs.append(row["collisions"])
            contact = row.get("contact_state") or {}
            holders = contact.get("holders")
            if holders is not None and (
                not holder_transitions or holder_transitions[-1]["holders"] != holders
            ):
                holder_transitions.append(dict(tick=row.get("tick"), holders=holders))
    return dict(
        physics_steps=count,
        control_calls=calls,
        last_sim_s=sim_s,
        peak_tcp_position_error_m=position,
        peak_tcp_orientation_error_deg=angle,
        first_collision_records=pairs,
        measured_holder_transitions=holder_transitions,
    )


def summarize(plan, summary, output):
    grouped = {}
    for trial in plan["trials"]:
        episode = trial["episode_id"]
        row = summary["rows"][episode]
        receipt = row.get("result", {})
        result = receipt.get("result", receipt)
        observation = steps(output / "trials" / episode / "episode/events.jsonl")
        wall = result.get("elapsed_s")
        if wall and observation:
            observation["sim_seconds_per_wall_second"] = (
                observation["last_sim_s"] / wall
            )
        grouped[trial["seed"], trial["condition"]] = dict(
            episode_id=episode,
            status=row["status"],
            reason=row.get("reason"),
            category=receipt.get("category"),
            outcome=result.get("outcome"),
            task_predicate=result.get("success"),
            failure=result.get("failure"),
            wall_s=wall,
            planning_s=result.get("planning_s"),
            inference_s=result.get("inference_s"),
            control_s=result.get("control_s"),
            policy_actions=result.get("policy_actions", result.get("executed")),
            recovery_actions=result.get("recovery_actions"),
            evidence=observation,
        )
    return dict(
        data_kind="diagnostic",
        formal_score_eligible=False,
        transfer_claim_eligible=False,
        gpu_peak_mib=summary.get("observed_gpu_peak_mib"),
        stop_reason=summary["stop_reason"],
        trials=list(grouped.values()),
        comparisons=[
            dict(
                seed=seed,
                conditions=[a, b],
                left=grouped[seed, a],
                right=grouped[seed, b],
            )
            for seed in (0, 1)
            for a, b in plan["comparisons"]
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    campaign = json.loads((args.output / "supervisor/campaign.json").read_text())
    persist(
        args.output / "diagnostic-statistics.json",
        summarize(json.loads(args.plan.read_text()), campaign, args.output),
    )
