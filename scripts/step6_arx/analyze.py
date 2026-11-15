"""Recompute trajectory metrics, then compare fully matched and scored F/L pairs."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import median

from .metrics import METRICS, STAGES, Stages

SCORED_OUTCOMES = {"TASK_SUCCESS", "ACTION_LIMIT", "SAFETY_STOP"}
VARIANTS = ("frozen", "lora")


def replay(path):
    stage = Stages()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "physics":
                stage.step(row)
            elif row["kind"] == "stop":
                stage.safety(row)
            elif row["kind"] == "row_attempt":
                stage.record("collision_rejection", False, row)
    return stage.summary()


def group(rows):
    rates = {}
    for name in METRICS:
        counts = Counter(r["analysis"]["metrics"][name]["status"] for r in rows)
        observed, not_observed = counts["observed"], counts["not_observed"]
        available = observed + not_observed
        times = [r["analysis"]["metrics"][name]["first"]["sim_s"] for r in rows
                 if r["analysis"]["metrics"][name]["status"] == "observed"]
        rates[name] = dict(observed=observed, not_observed=not_observed, unavailable=counts["unavailable"],
                           rate=observed / available if available else None,
                           first_sim_s_median=median(times) if times else None)
    lifts = [r["analysis"]["peak_block_lift_m"] for r in rows if r["analysis"]["peak_block_lift_m"] is not None]
    successes = sum(r["task_success"] is True for r in rows)
    return dict(n=len(rows), successes=successes, success_rate=successes / len(rows) if rows else None,
                stage_rates={label: rates[name] for label, name in STAGES.items()},
                metric_rates=rates, median_peak_lift_m=median(lifts) if lifts else None)


def summarize(root):
    campaign = json.loads((root / "campaign.json").read_text())
    plan = campaign["plan"]
    pairs, eligible = [], {v: [] for v in VARIANTS}
    receipts = {v: [] for v in VARIANTS}
    for seed in campaign["seeds"]:
        by_variant = {}
        for variant in VARIANTS:
            directory = root / f"seed-{seed:02d}-{variant}"
            if not (directory / "receipt.json").is_file():
                continue
            r = json.loads((directory / "receipt.json").read_text())
            if (r["schema"] != plan["protocol"] or r["scene_seed"] != seed or r["variant"] != variant
                    or r["source"] != ("protocol_fixture" if campaign["fixture"] else "step5_model")):
                raise ValueError(f"incompatible evidence: {directory}")
            if replay(directory / "trajectory.jsonl") != r["analysis"]:
                raise ValueError(f"receipt metrics differ from trajectory: {directory}")
            receipts[variant].append(r)
            by_variant[variant] = r
        f, l = (by_variant.get(v) for v in VARIANTS)
        reason = None
        if campaign["fixture"]:
            reason = "PROTOCOL_FIXTURE"
        elif campaign.get("acceptance_only"):
            reason = "GPU_SHORT_ACCEPTANCE"
        elif f is None or l is None:
            reason = "MISSING_RECEIPT"
        elif l["pair_match"] is not True:
            reason = "INITIAL_OR_PAIR_MISMATCH"
        elif any(r["outcome"] not in SCORED_OUTCOMES or type(r["task_success"]) is not bool for r in (f, l)):
            reason = "UNSCORED_TERMINATION"
        elif any(r["rgb_status"] != "OBSERVED" or not r["video"] for r in (f, l)):
            reason = "RGB_OR_VIDEO_MISSING"
        else:
            for variant in VARIANTS:
                eligible[variant].append(by_variant[variant])
        row = dict(seed=seed, matched=l["pair_match"] if l else None, excluded=reason,
                   delta_success_L_minus_F=(int(l["task_success"]) - int(f["task_success"])
                                            if reason is None else None))
        for tag, r in zip(("F", "L"), (f, l), strict=True):
            row.update({f"{tag}_outcome": r["outcome"] if r else "MISSING",
                        f"{tag}_success": r["task_success"] if r and reason is None else None,
                        f"{tag}_ticks": r["physics_ticks"] if r else None,
                        f"{tag}_failure": r["failure"]["status"] if r and r["failure"] else None})
            for label, metric in STAGES.items():
                row[f"{tag}_{label}"] = r["analysis"]["metrics"][metric]["status"] if r else "unavailable"
            row[f"{tag}_first_stage_not_observed"] = (
                next((label for label, metric in STAGES.items()
                      if r["analysis"]["metrics"][metric]["status"] != "observed"), None)
                if r and r["outcome"] in SCORED_OUTCOMES and r["task_success"] is False else None)
            for metric in ("left_bilateral", "left_grasp", "left_lift", "shared_hold", "right_takeover", "drop"):
                datum = r["analysis"]["metrics"][metric] if r else {"first": None, "status": "unavailable"}
                row[f"{tag}_{metric}_status"] = datum["status"]
                row[f"{tag}_{metric}_first_s"] = datum["first"]["sim_s"] if datum["first"] else None
        for metric in METRICS:
            a = f["analysis"]["metrics"][metric]["status"] if f else "unavailable"
            b = l["analysis"]["metrics"][metric]["status"] if l else "unavailable"
            row[f"delta_{metric}_L_minus_F"] = (int(b == "observed") - int(a == "observed")
                                                if reason is None and "unavailable" not in (a, b) else None)
        pairs.append(row)
    return dict(schema=plan["protocol"], fixture=campaign["fixture"],
                pairs=pairs, formal_groups={v: group(eligible[v]) for v in VARIANTS},
                all_receipts={v: dict(n=len(receipts[v]),
                                      outcomes=dict(Counter(r["outcome"] for r in receipts[v])),
                                      failure_reasons=dict(Counter(f'{r["failure"]["phase"]}/{r["failure"]["status"]}'
                                                                   for r in receipts[v] if r["failure"])))
                              for v in VARIANTS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.input)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["pairs"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["pairs"])


if __name__ == "__main__":
    main()
