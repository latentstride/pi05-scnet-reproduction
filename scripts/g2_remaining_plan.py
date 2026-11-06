"""One complete proposed remaining G2 campaign; planning only, never submits."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path


def build(project):
    prior = project / "research/g2/failure-audit-127054052-r1/release-context"
    conditions = [
        ("native-w31", "aloha-agilex", 31, "native_guard", 1200),
        ("native-w26", "aloha-agilex", 26, "native_guard", 1200),
        ("native-w4", "aloha-agilex", 4, "native_guard", 1200),
        ("arx-baseline-w4", "ARX-X5", 4, "frozen_exact_mapping", 2600),
        ("arx-recovery-no-route-w4", "ARX-X5", 4, "bounded_recovery", 2600),
        ("arx-full-w4", "ARX-X5", 4, "bounded_recovery_and_router", 2600),
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
    # Probe the only seed with observed closure, then the main ARX diagnostic.
    # Keep matched controls and counterbalance later blocks.
    keys = [
        (1, "native-w31"), (0, "native-w31"),
        (0, "arx-full-w4"), (1, "arx-full-w4"),
        (1, "native-w26"), (0, "native-w26"),
        (0, "native-w4"), (1, "native-w4"),
        (1, "arx-expert-repaired"), (0, "arx-expert-repaired"),
        (0, "arx-recovery-no-route-w4"), (1, "arx-recovery-no-route-w4"),
        (1, "arx-baseline-w4"), (0, "arx-baseline-w4"),
    ]
    indexed = {(row["seed"], row["condition"]): row for row in trials}
    order = [indexed[key] for key in keys]
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
        schema_version=2,
        status="CPU_PLAN_ONLY_NOT_SUBMISSION_READY",
        submission_allowed=False,
        protected_existing_job="127081074",
        existing_job_must_not_be_modified=True,
        allocations_planned=1,
        model_server_starts=1,
        trials=order,
        stages=[
            "same-allocation startup",
            "native short startup and long-window diagnostic",
            "cross-embodiment policy",
            "recovery/router/window ablations",
            "official expert control",
            "statistics",
        ],
        native_gate="independent native safety and input admission; no native task-success prerequisite for ARX diagnostics",
        arx_gate="independent same-allocation real initial/asset/geometry, actual state and proxy input contract admission for each seed",
        transfer_claim_gate="matched K4 native predicate plus measured physical sequence and matching ARX evidence; otherwise diagnostic only",
        early_stop=[
            "native task failure retains ARX diagnostics if independently admitted; unavailable conditions are not_run",
            "model/process/time failure or actual unsafe state: stop campaign and report all remaining trials as not_run",
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
            rate_status="historical_reference_only_live_quote_required",
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
            ["native-w26", "native-w31"],
            ["native-w4", "native-w31"],
            ["arx-baseline-w4", "arx-recovery-no-route-w4"],
            ["arx-recovery-no-route-w4", "arx-full-w4"],
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
