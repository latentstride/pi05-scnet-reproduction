"""Locked roster for the combined native-window and transfer diagnostic."""

ORDER = (
    (0, "native-w50"),
    (0, "native-w31"),
    (1, "native-w31"),
    (1, "native-w50"),
    (0, "native-w4"),
    (0, "arx-full-w4"),
    (1, "arx-full-w4"),
    (1, "native-w4"),
    (1, "arx-expert-repaired"),
    (0, "arx-expert-repaired"),
)
CONDITIONS = {
    "native-w50": ("aloha-agilex", 50, "native_guard", 1200),
    "native-w31": ("aloha-agilex", 31, "native_guard", 1200),
    "native-w4": ("aloha-agilex", 4, "native_guard", 1200),
    "arx-full-w4": ("ARX-X5", 4, "bounded_recovery_and_router", 2600),
    "arx-expert-repaired": ("ARX-X5", 1, "official_expert_plus_placement_suffix", 2200),
}
OVERHEAD = dict(
    graphics_s=300,
    shared_model_and_dynamic_entry_s=3600,
    statistics_s=180,
    cleanup_s=120,
)
TRIALS_S = sum(CONDITIONS[label][-1] for _, label in ORDER)
BOUND_S = TRIALS_S + sum(OVERHEAD.values())
ALLOCATION_S = 25200
FEE = 25.20
PROTECTED_JOBS = ("127054052", "127081074", "127117141", "127139405")
COMPARISONS = [
    ["native-w50", "native-w31"],
    ["native-w4", "arx-full-w4"],
    ["arx-full-w4", "arx-expert-repaired"],
]


def build(project):
    from g2_remaining_plan import build as historical_plan

    prior = historical_plan(project)
    indexed = {(row["seed"], row["condition"]): row for row in prior["trials"]}
    roster = []
    for seed, label in ORDER:
        if label == "native-w50":
            row = dict(indexed[seed, "native-w31"], condition=label, window=50)
        else:
            row = dict(indexed[seed, label])
        roster.append(dict(row, episode_id=f"unified-seed{seed}-{label}"))
    return dict(
        schema_version=3,
        status="CPU_PLAN_ONLY_NOT_SUBMISSION_READY",
        submission_allowed=False,
        protected_existing_job="127081074",
        protected_jobs=list(PROTECTED_JOBS),
        existing_job_must_not_be_modified=True,
        allocations_planned=1,
        model_server_starts=1,
        trials=roster,
        comparisons=COMPARISONS,
        early_stop=[
            "native H50/K31 first observation and prediction must match before motion",
            "isolated native admission failure does not block independently admitted ARX",
            "actual unsafe state, service failure, timeout or unverified cleanup stops the allocation",
            "expected trial failures require verified cleanup before the next independent reset",
        ],
        budget=dict(
            trials_s=TRIALS_S,
            overhead=OVERHEAD,
            bounded_execution_s=BOUND_S,
            proposed_allocation_s=ALLOCATION_S,
            proposed_fee_cap_yuan=FEE,
            unassigned_margin_s=ALLOCATION_S - BOUND_S,
            gpu_count=1,
            cpu_count=4,
            memory_gib=32,
            node="gnode39",
            reference_price_yuan_per_gpu_hour=3.6,
            submission_authorized=False,
            rate_status="historical_reference_only_live_quote_required",
        ),
    )


def validate(plan):
    from g2e_protocol import PRELOAD_PROTOCOL, PRELOAD_ESTABLISHMENT, APPROACH_REPAIR

    if plan.get("free_space_approach_repair") not in (None, APPROACH_REPAIR):
        raise ValueError("unknown bounded approach profile in unified release")

    if plan.get("native_preload_control") not in (None, PRELOAD_PROTOCOL):
        raise ValueError("unknown native preload protocol in unified release")
    if plan.get("native_preload_establishment") is not None and (
        plan.get("native_preload_establishment") != PRELOAD_ESTABLISHMENT
        or plan.get("native_preload_control") != PRELOAD_PROTOCOL
    ):
        raise ValueError("unknown contact preload establishment in unified release")
    budget = plan["budget"]
    if (
        plan.get("schema_version") != 3
        or plan["submission_allowed"] is not False
        or plan["allocations_planned"] != 1
        or plan["model_server_starts"] != 1
        or tuple(plan["protected_jobs"]) != PROTECTED_JOBS
        or (
            budget["gpu_count"],
            budget["cpu_count"],
            budget["memory_gib"],
            budget["node"],
        )
        != (1, 4, 32, "gnode39")
        or budget["overhead"] != OVERHEAD
        or budget["trials_s"] != TRIALS_S
        or budget["bounded_execution_s"] != BOUND_S
        or budget["proposed_allocation_s"] != ALLOCATION_S
        or budget["proposed_fee_cap_yuan"] != FEE
        or budget["submission_authorized"] is not False
        or plan["comparisons"] != COMPARISONS
        or tuple((r["seed"], r["condition"]) for r in plan["trials"]) != ORDER
    ):
        raise ValueError("unified review roster or resource envelope differs")
    for row in plan["trials"]:
        label, seed = row["condition"], row["seed"]
        if (
            tuple(row[k] for k in ("embodiment", "window", "method", "hard_timeout_s"))
            != CONDITIONS[label]
            or row["episode_id"] != f"unified-seed{seed}-{label}"
            or row["recovery_cap"] != (4 if label == "arx-full-w4" else 0)
            or row["max_policy_actions"] != (None if "expert" in label else 300)
            or row["formal_score_eligible"] is not False
        ):
            raise ValueError("unified trial differs from frozen condition")


def validate_native_startup(entries, admission, probes):
    """Quarantine only the native scene when its no-motion receipts disagree."""
    for seed in (0, 1):
        primary = ("aloha-agilex", seed)
        ids = [probes[key][0] for key in (primary, ("aloha-agilex", seed, 50))]
        a, b = (entries[episode] for episode in ids)
        if not (
            a["status"] == b["status"] == "admitted"
            and a.get("first_exchange")
            and a["first_exchange"] == b.get("first_exchange")
            and a.get("initial_sha256")
            and a["initial_sha256"] == b.get("initial_sha256")
        ):
            admission[primary] = {
                "category": "not_admitted",
                "reason": "native pair no-motion reset or first exchange unverified",
            }
