"""Summarize roster-bound trials with explicit execution and scoring status."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from g2_episode import save_json
from g2_locked_score import collect
from summarize_experiments import paired_tables


def assessment(rows, supervisor):
    if not rows:
        raise ValueError("cannot assess an empty trial set")
    kind = (
        "formal"
        if (
            supervisor.get("data_kind") == "formal"
            and supervisor.get("formal_score_eligible") is True
        )
        else "diagnostic"
    )
    if any(row["data_kind"] != kind for row in rows):
        raise ValueError("trial record kind contradicts supervisor scoring eligibility")
    outcomes = Counter(row["outcome"] for row in rows)
    return dict(
        data_kind=kind,
        formal_score_eligible=kind == "formal",
        trial_records_complete=True,
        trial_count=len(rows),
        task_success_count=outcomes["success"],
        task_success_rate=outcomes["success"] / len(rows),
        all_tasks_successful=outcomes["success"] == len(rows),
        outcomes=dict(outcomes),
        scope="terminal trial records; host stage completion is tracked separately",
    )


def summarize(
    lock,
    digest,
    trials,
    output,
    expert_path,
    supervisor_path,
    legacy_formal_records=False,
):
    supervisor = json.loads(supervisor_path.read_text())
    kind = (
        "formal"
        if (
            supervisor.get("data_kind") == "formal"
            and supervisor.get("formal_score_eligible") is True
        )
        else "diagnostic"
    )
    if legacy_formal_records and kind != "diagnostic":
        raise ValueError(
            "legacy record reclassification requires diagnostic supervision"
        )
    original_run = trials.resolve().parent
    if legacy_formal_records and (
        output.resolve() == original_run or original_run in output.resolve().parents
    ):
        raise ValueError("legacy analysis output must be outside the original run")
    output.mkdir()
    records = output / "records.json"
    rows = collect(
        lock,
        digest,
        trials,
        records,
        data_kind="formal" if legacy_formal_records else kind,
    )
    if legacy_formal_records:
        # Preserve the original files and mark the newly derived summary consistently.
        rows = [dict(row, data_kind="diagnostic") for row in rows]
        save_json(records, rows)
    command = [
        sys.executable,
        str(Path(__file__).with_name("summarize_experiments.py")),
        str(records),
        "--data-kind",
        kind,
        "--output",
        str(output / "transfer"),
    ]
    if kind == "formal":
        command += ["--run-lock", str(lock)]
    subprocess.run(command, check=True)
    ablations = []
    for embodiment in ("aloha-agilex", "arx-x5"):
        ablations.extend(
            dict(row, comparison_id="window-" + embodiment)
            for row in rows
            if row["condition"]["embodiment"] == embodiment
        )
    summary, pairs = paired_tables(ablations)
    save_json(
        output / "window-contrasts.json",
        dict(data_kind=kind, pairs=pairs, summary=summary),
    )
    expert = json.loads(expert_path.read_text())
    passed = (
        expert["completed"]
        and expert["contact_monitor_passed"]
        and expert["native_success_at_expert_end"]["success"]
        and expert["native_success_after_diagnostic_hold"]["success"]
    )
    save_json(
        output / "expert-reference-check.json", dict(passed=passed, expert=expert)
    )
    result = assessment(rows, supervisor)
    result.update(
        expert_diagnostic_passed=passed,
        legacy_formal_labels_reclassified=legacy_formal_records,
    )
    save_json(output / "evaluation-summary.json", result)
    if not passed:
        raise ValueError("expert diagnostic reference failed")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("digest")
    p.add_argument("trials", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("expert", type=Path)
    p.add_argument("--supervisor-report", type=Path)
    p.add_argument("--legacy-formal-records", action="store_true")
    a = p.parse_args()
    supervisor = (
        a.supervisor_report or a.trials.parent / "supervisor/g2_acceptance.json"
    )
    summarize(
        a.lock,
        a.digest,
        a.trials,
        a.output,
        a.expert,
        supervisor,
        a.legacy_formal_records,
    )
