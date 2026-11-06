"""One complete proposed remaining G2 campaign; planning only, never submits."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path


def build(project):
    prior = project / "research/g2/failure-audit-127054052-r1/release-context"
    conditions = [
        ("native-w1", "aloha-agilex", 1, "native_guard", 1200),
        ("native-w4", "aloha-agilex", 4, "native_guard", 1200),
        ("arx-baseline-w4", "ARX-X5", 4, "frozen_exact_mapping", 2600),
        ("arx-recovery-no-route-w4", "ARX-X5", 4, "bounded_recovery", 2600),
        ("arx-full-w4", "ARX-X5", 4, "bounded_recovery_and_router", 2600),
        ("arx-full-w1", "ARX-X5", 1, "bounded_recovery_and_router", 2600),
        (
            "arx-expert-repaired",
            "ARX-X5",
            1,
            "official_expert_plus_placement_suffix",
            2200,
        ),
    ]
    trials = []
    for seed in (0, 1):
        for label, robot, window, method, cap in conditions:
            original = f"seed{seed}-" + (
                "A-native-w1" if robot == "aloha-agilex" else "B-arx-w1"
            )
            config = json.loads((prior / "configs" / f"{original}.json").read_text())
            trials.append(
                dict(
                    episode_id=f"remaining-seed{seed}-{label}",
                    seed=seed,
                    pair_id=f"paired-10g-{seed:02d}",
                    condition=label,
                    embodiment=robot,
                    window=window,
                    method=method,
                    hard_timeout_s=cap,
                    max_policy_actions=300 if "expert" not in label else None,
                    reset_fixture_sha256=sha256(
                        json.dumps(
                            config["reset_fixture"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                    prompt=config["prompt"],
                    policy_rng_seed=20261106 + seed,
                    checkpoint_revision="17ef545270cf26ed81169893b60f845f80ce6557",
                    recovery_cap=4 if "recovery" in method else 0,
                    data_kind="diagnostic",
                    formal_score_eligible=False,
                )
            )
    # Native paired admission first, then counterbalance seed order per condition.
    order = []
    for condition_index, condition in enumerate(conditions):
        seeds = (0, 1) if condition_index % 2 == 0 else (1, 0)
        order.extend(
            next(
                t
                for t in trials
                if t["condition"] == condition[0] and t["seed"] == seed
            )
            for seed in seeds
        )
    trial_s = sum(t["hard_timeout_s"] for t in order)
    overhead = dict(
        graphics_s=300,
        shared_model_and_dynamic_entry_s=1900,
        statistics_s=180,
        cleanup_s=120,
    )
    total = trial_s + sum(overhead.values())
    allocation = 34200
    if len(order) != 14 or total > allocation:
        raise ValueError("incomplete roster or budget overflow")
    return dict(
        schema_version=1,
        status="CPU_PLAN_ONLY_NOT_SUBMISSION_READY",
        submission_allowed=False,
        protected_existing_job="127081074",
        existing_job_must_not_be_modified=True,
        allocations_planned=1,
        model_server_starts=1,
        trials=order,
        stages=[
            "same-allocation startup",
            "native paired admission",
            "cross-embodiment policy",
            "recovery/router/window ablations",
            "official expert control",
            "statistics",
        ],
        native_gate="must include at least one full task predicate success plus measured left-only/shared/right-only/released sequence before any ARX policy trial",
        early_stop=[
            "native gate fails: skip all ARX policy and expert trials, produce complete missing/not-run report",
            "model or initial mismatch or actual unsafe collision: stop campaign; never restart/requeue",
            "candidate rejection ends only that trial; later predeclared condition may continue",
            "recovery requires a fresh measured mapping; at most four per trial, one per failed observation",
            "contact/holding forbids generic RRT and partial-goal recovery",
        ],
        budget=dict(
            trials_s=trial_s,
            overhead=overhead,
            bounded_execution_s=total,
            proposed_allocation_s=allocation,
            unassigned_margin_s=allocation - total,
            gpu_count=1,
            cpu_count=4,
            memory_gib=32,
            node="gnode39",
            reference_price_yuan_per_gpu_hour=3.6,
            proposed_fee_cap_yuan=34.2,
            prior_project_budget_yuan=20.0,
            approved_remaining_campaign_budget_yuan=34.2,
            budget_expansion_approved=True,
            budget_authorization="2026-11-06 user: 可以扩大预算; scoped to proposed CNY34.20 remaining campaign",
            submission_authorized=False,
            fits_prior_total_budget=False,
        ),
        comparisons=[
            ["native-w4", "arx-full-w4"],
            ["native-w1", "arx-full-w1"],
            ["arx-baseline-w4", "arx-recovery-no-route-w4"],
            ["arx-recovery-no-route-w4", "arx-full-w4"],
            ["arx-full-w1", "arx-full-w4"],
        ],
        statistics=dict(
            experimental_units="two locked environmental seeds, never individual correlated frames/actions",
            outcomes=[
                "task predicate success",
                "full physical sequence evidence",
                "safety rejection",
                "infrastructure interruption",
                "not_run",
            ],
            metrics=[
                "original vs executed command pose deviation",
                "source roundtrip residual",
                "joint/TCP tracking",
                "collision pair/depth",
                "planning and rerouting calls/CPU wall",
                "recovery actions vs policy actions",
                "physics steps and real-time ratio",
                "model latency and GPU peak",
            ],
            scoring="original task predicate, safety thresholds and schema unchanged; descriptive paired rows, no significance claim with n=2",
        ),
        existing_g2e_results="external exploratory timing evidence only; do not silently pool separate allocation with matched primary native controls",
        unresolved_release_requirements=[
            "dynamic recovery/placement acceptance beyond archived geometry",
            "final end-to-end real-scene persistence harness and frozen model/runtime lock for new conditions",
            "expanded CNY34.20 budget recorded; resource submission remains explicitly forbidden in this task",
        ],
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("project", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(
        json.dumps(build(a.project.resolve()), ensure_ascii=False, indent=2) + "\n"
    )
