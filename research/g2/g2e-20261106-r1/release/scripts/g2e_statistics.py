"""Roster-complete diagnostic reporting, preserving planned but unrun trials."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from g2_contact import ContactLimits
from g2_episode import save_json
from g2_locked_score import collect
from summarize_experiments import paired_tables, load_records, LOCK_FIELDS


def physical_sequence(directory):
    """Audit measured holder transitions without changing the frozen score."""
    expected = [["left"], ["left", "right"], ["right"], []]
    limits = ContactLimits()
    milestones = []
    target_support = False
    path = directory / "events.jsonl"
    if not path.is_file():
        return dict(complete=False, missing_telemetry=True)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "native_contact_sample" and len(milestones) >= 3:
                for contact in row["contacts"]:
                    names = contact["body_names"]
                    if any("box" in n and "target" not in n for n in names) and any(
                        "target_box" in n for n in names
                    ):
                        if any(
                            p["separation"] <= limits.penetration_m
                            and sum(float(x) ** 2 for x in p["impulse"])
                            >= limits.minimum_impulse**2
                            for p in contact.get("points", [])
                        ):
                            target_support = True
            if row.get("kind") == "native_guard_state":
                holders = sorted(row["contact_state"]["holders"])
                if len(milestones) < 4 and holders == expected[len(milestones)]:
                    milestones.append(dict(tick=row["tick"], holders=holders))
    return dict(
        complete=len(milestones) == 4 and target_support,
        milestones=milestones,
        target_support_contact_observed=target_support,
        definition="ordered measured left-only, shared, right-only, released holders plus target support contact; unchanged task score reported separately",
    )


def summarize(lock, trials, output):
    output.mkdir(parents=True, exist_ok=False)
    raw = lock.read_bytes()
    roster = json.loads(raw)["episodes"]
    rows = []
    missing = []
    for entry in roster:
        path = trials / entry["episode_id"] / "diagnostic-record.json"
        if path.is_file():
            rows.append(json.loads(path.read_text()))
        else:
            missing.append(entry["episode_id"])
    # Full roster verification stays mandatory when all eight terminate.
    if not missing:
        rows = collect(
            lock,
            sha256(raw).hexdigest(),
            trials,
            output / "records.json",
            data_kind="diagnostic",
        )
    else:
        save_json(output / "records.json", rows)
    # Validate partial records through the same schema and metric checks, while
    # preserving absent roster entries as absent (never synthetic failures).
    rows = load_records([output / "records.json"], "diagnostic") if rows else []
    planned = {r["episode_id"]: r for r in roster}
    for row in rows:
        if row["run_lock_sha256"] != sha256(raw).hexdigest() or {
            f: row[f] for f in LOCK_FIELDS
        } != planned.get(row["episode_id"]):
            raise ValueError("partial record differs from locked roster")
    contrasts = []
    for baseline in (1, 4):
        for treatment in (26, 31):
            selected = [
                dict(r, comparison_id=f"w{treatment}-vs-w{baseline}")
                for r in rows
                if r["condition"]["window_actions"] in (baseline, treatment)
            ]
            if selected:
                pair_rows, pair_summary = paired_tables(selected)
                contrasts.append(
                    dict(
                        baseline=baseline,
                        treatment=treatment,
                        pairs=pair_rows,
                        summary=pair_summary,
                    )
                )
    save_json(output / "paired-window-contrasts.json", contrasts)
    physical = {
        row["episode_id"]: physical_sequence(trials / row["episode_id"]) for row in rows
    }
    save_json(output / "physical-sequence-audit.json", physical)
    verified = [
        r["episode_id"]
        for r in rows
        if r["outcome"] == "success" and physical[r["episode_id"]]["complete"]
    ]
    counts = Counter(r["outcome"] for r in rows)
    summary = dict(
        data_kind="diagnostic",
        formal_score_eligible=False,
        planned_trials=len(roster),
        completed_trials=len(rows),
        missing_or_not_run=missing,
        outcomes=dict(counts),
        task_success_count=counts["success"],
        all_trials_complete=not missing,
        success_fraction_of_completed=f"{counts['success']}/{len(rows)}"
        if rows
        else None,
        verified_native_full_task_episode_ids=verified,
        arx_authorized=False,
        arx_gate="separate review only after native full-task success with physical sequence evidence",
    )
    save_json(output / "evaluation-summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("lock", "trials", "output"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    summarize(a.lock, a.trials, a.output)
