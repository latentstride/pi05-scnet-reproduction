"""Build and bind the opt-in, review-only author H50 overlay for the unified release."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil

from g2_author_h50_limits import BUDGET, EPISODE, MODE
from g2_unified_plan import ALLOCATION_S, validate

SIGNAL_RESERVE_S = 180  # TERM@120 plus Slurm's documented 60-second resolution.
HOST_RESERVE_S = 180  # Child-launch/reap and receipt I/O outside existing stage clocks.
CAMPAIGN_RESERVE_S = SIGNAL_RESERVE_S + HOST_RESERVE_S
REVIEW_FILE = "author-h50-review.json"
REFERENCE_FILE = "author-h50-reset-reference.json"
REFERENCE_SHA = "b1ab4f6201174a872d00468489968313bb445f24be7ab8c9b7ebfe73849a3787"
REPLACED = "unified-seed0-native-w50"
MIN_FREE_BYTES = 12 * 1024**3
TRIAL_MIN_FREE_BYTES = 8 * 1024**3
BASE_MANIFEST_SHA = "96834d54c8293a3c4ac6f2b5467660eecfd5adfc287ceb442aafbc047c05007e"
BASE_PLAN_SHA = "0b210c21d406a7db8a2b06dca9a4c8d5950787bd0a489abba68939d2e94a96cb"
ADDED_SCRIPTS = (
    "g2_author_h50_campaign.py",
    "g2_author_h50_limits.py",
    "g2_author_h50_receipt.py",
    "g2_author_h50_trial.py",
    "g2_author_native_h50.py",
    "g2_author_native_h50_service.py",
)
ADAPTED_SCRIPTS = (
    "g2_candidate_stop.py",
    "g2_final_campaign.py",
    "g2_final_deploy.py",
    "g2_final_statistics.py",
)


def ledgers(plan):
    validate(plan)
    overhead = plan["budget"]["overhead"]
    original = sum(row["hard_timeout_s"] for row in plan["trials"])
    replaced = next(
        row["hard_timeout_s"] for row in plan["trials"] if row["episode_id"] == REPLACED
    )
    return {
        name: dict(
            trial_count=count,
            trials_s=trials,
            overhead=overhead,
            total_s=trials + sum(overhead.values()),
            allocation_s=ALLOCATION_S,
            slurm_signal_reserve_s=SIGNAL_RESERVE_S,
            host_transition_reserve_s=HOST_RESERVE_S,
            remainder_before_term_s=ALLOCATION_S
            - CAMPAIGN_RESERVE_S
            - trials
            - sum(overhead.values()),
            native_seed_pairs_retained=pairs,
        )
        for name, count, trials, pairs in (
            ("original", 10, original, 2),
            ("append", 11, original + BUDGET.hard_s, 2),
            ("replace_seed0_h50", 10, original - replaced + BUDGET.hard_s, 1),
        )
    }


def review_record(plan):
    return dict(
        version=1,
        mode=MODE,
        episode_id=EPISODE,
        placement="append_after_locked_ten",
        submission_allowed=False,
        budget=asdict(BUDGET),
        reference_sha256=REFERENCE_SHA,
        min_free_bytes=MIN_FREE_BYTES,
        trial_min_free_bytes=TRIAL_MIN_FREE_BYTES,
        base_manifest_sha256=BASE_MANIFEST_SHA,
        ledgers=ledgers(plan),
    )


def verify_overlay(release, plan, declared):
    expected = review_record(plan)
    if (
        not set("scripts/" + p for p in (*ADDED_SCRIPTS, *ADAPTED_SCRIPTS)).issubset(
            declared
        )
        or declared.get("unified-plan.json") != BASE_PLAN_SHA
        or declared.get(REFERENCE_FILE) != REFERENCE_SHA
        or json.loads((release / REVIEW_FILE).read_bytes()) != expected
        or expected["ledgers"]["append"]["remainder_before_term_s"] <= 0
    ):
        raise ValueError(
            "author H50 review overlay differs from the locked budget or inputs"
        )
    return expected


def bind_runtime(plan, runtime, release, output, *, review):
    """Preserve all ten legacy commands/probes and append one authorized child."""
    if review != review_record(plan):
        raise ValueError("missing or mismatched author H50 review overlay")
    model = runtime["model"].copy()
    old_model = str(release / "scripts/g2e_model_entry.py")
    model[model.index(old_model)] = str(
        release / "scripts/g2_author_native_h50_service.py"
    )
    original = runtime["commands"][REPLACED]
    trial_script = str(release / "scripts/g2_final_trial.py")
    child = original[: original.index(trial_script)] + [
        str(release / "scripts/g2_author_h50_trial.py")
    ]
    for flag in ("--release", "--pace", "--uri", "--model-startup"):
        child.extend([flag, original[original.index(flag) + 1]])
    dest = output / "trials" / EPISODE
    child += ["--output", str(dest)]
    old_reader = runtime["read_result"]

    def read_result(episode, probe=False):
        if episode == EPISODE and not probe:
            return json.loads((dest / "receipt.json").read_bytes())
        return old_reader(episode, probe=probe)

    rows = [
        *plan["trials"],
        dict(
            episode_id=EPISODE,
            embodiment="aloha-agilex",
            seed=100000,
            method=MODE,
            hard_timeout_s=BUDGET.hard_s,
        ),
    ]
    ledger = review["ledgers"]["append"]
    execution_plan = dict(
        plan,
        trials=rows,
        budget=dict(
            plan["budget"],
            trials_s=ledger["trials_s"],
            bounded_execution_s=ledger["total_s"],
            unassigned_margin_s=ALLOCATION_S - ledger["total_s"],
        ),
    )
    return execution_plan, dict(
        runtime,
        model=model,
        commands={**runtime["commands"], EPISODE: child},
        read_result=read_result,
        author_h50_budget=BUDGET,
        campaign_reserve_s=CAMPAIGN_RESERVE_S,
    )


def build_overlay(project, frozen, output):
    """Package only an already verified immutable ten-row release and the overlay."""
    from g2_final_deploy import verify

    plan = verify(frozen)
    if (frozen / REVIEW_FILE).exists() or sha256(
        (frozen / "SHA256SUMS").read_bytes()
    ).hexdigest() != BASE_MANIFEST_SHA:
        raise ValueError("the base must be the original ten-row release")
    shutil.copytree(frozen, output)
    for name in (*ADDED_SCRIPTS, *ADAPTED_SCRIPTS):
        shutil.copyfile(project / "scripts" / name, output / "scripts" / name)
    shutil.copyfile(
        project / "research/g2/author-h50-preflight/cpu-reset-readback.json",
        output / REFERENCE_FILE,
    )
    (output / REVIEW_FILE).write_text(json.dumps(review_record(plan), indent=2) + "\n")
    manifest = output / "SHA256SUMS"
    manifest.write_text(
        "".join(
            f"{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(output)}\n"
            for p in sorted(output.rglob("*"))
            if p.is_file() and p != manifest
        )
    )
    if verify(output) != plan:
        raise ValueError("author overlay changed the frozen plan")
    return review_record(plan)


if __name__ == "__main__":
    import argparse
    import gzip
    import tarfile

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "frozen", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    record = build_overlay(
        *(getattr(args, k).resolve() for k in ("project", "frozen", "output"))
    )
    if args.archive:
        with (
            args.archive.open("xb") as target,
            gzip.GzipFile(fileobj=target, filename="", mode="wb", mtime=0) as stream,
        ):
            with tarfile.open(fileobj=stream, mode="w") as archive:
                for path in sorted(p for p in args.output.rglob("*") if p.is_file()):
                    member = archive.gettarinfo(
                        path,
                        str(
                            Path("g2-author-h50-review") / path.relative_to(args.output)
                        ),
                    )
                    member.uid = member.gid = member.mtime = 0
                    member.uname = member.gname = ""
                    member.mode = 0o755 if member.mode & 0o111 else 0o644
                    with path.open("rb") as source:
                        archive.addfile(member, source)
    print(json.dumps(record, indent=2))
