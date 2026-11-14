"""Extract a public, hash-linked record from the private frozen job directory."""

import argparse
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def select(record, *keys):
    return {key: record[key] for key in keys if key in record}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    value = sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        value.update(chunk)
    return value.hexdigest()


def collect(run, accounting, submission):
    if submission["job_id"] != "127408633":
        raise ValueError("unexpected submission binding")
    job, state, start, end, elapsed, exit_code, derived_exit = (
        accounting.read_text().splitlines()[0].split("|")
    )
    if job != submission["job_id"]:
        raise ValueError("accounting ID differs")
    hours, minutes, seconds = map(int, elapsed.split(":"))
    elapsed_s = timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds()
    dates = [
        datetime.fromisoformat(value)
        .replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        .isoformat()
        for value in (start, end)
    ]
    source_files = {}

    def record(path):
        source_files[path] = digest(run / path)
        return run / path

    campaign = read(record("supervisor/campaign.json"))
    record("diagnostic-statistics.json")
    record("model/phases.jsonl")
    graphics = record("graphics/check_status.tsv")
    graphics_checks = [line.split("\t") for line in graphics.read_text().splitlines()]
    if list(campaign["rows"]) != submission["roster"]:
        raise ValueError("actual dispatch differs from the frozen roster")
    trials = []
    for episode, row in campaign["rows"].items():
        entry = {
            "episode_id": episode,
            **select(row, "status", "reason", "process_group_reaped"),
        }
        path = f"trials/{episode}/receipt.json"
        if (run / path).is_file():
            receipt = read(record(path))
            if receipt != row["result"]:
                raise ValueError(f"trial receipt differs from campaign: {episode}")
            entry.update(
                select(
                    receipt, "category", "success", "fresh_reset_verified", "cleanup"
                )
            )
            result = receipt["result"]
            entry.update(
                select(
                    result,
                    "outcome",
                    "executed",
                    "queries",
                    "policy_actions",
                    "recovery_actions",
                )
            )
            entry["elapsed_s"] = receipt["elapsed_s"]
            failure = result.get("failure") or {}
            diagnostic = failure.get("diagnostic") or {}
            entry["failure"] = {
                **select(failure, "phase", "code", "detail"),
                "diagnostic": select(
                    diagnostic,
                    "geometry",
                    "reason",
                    "side",
                    "finger",
                    "depth_m",
                    "allowed_penetration_m",
                    "completed_actions",
                    "contact_tick",
                ),
            }
            if "contact_state" in diagnostic:
                entry["failure"]["contact_state"] = select(
                    diagnostic["contact_state"],
                    "phase",
                    "holders",
                    "contact_fingers",
                    "stable_counts",
                    "saw_shared_hold",
                    "releasing",
                )
            for proof in ("candidate_stop", "contact_stop"):
                if proof in receipt:
                    entry[proof] = select(
                        receipt[proof],
                        "kind",
                        "tick",
                        "state_sha256",
                        "failure_sha256",
                        "terminal_checks_passed",
                        "hold_verified",
                        "no_physics_advance",
                        "hold_complete",
                        "close_complete",
                        "scene_context_closed",
                    )
        elif row["status"] != "not_run":
            raise ValueError(f"missing trial receipt: {episode}")
        trials.append(entry)
    terminal_trial = next(
        row["episode_id"] for row in trials if row["status"] == "actual_safety_stop"
    )
    event_log = record(f"trials/{terminal_trial}/episode/events.jsonl")
    collision_log = record(f"trials/{terminal_trial}/episode/collision-checks.jsonl")
    last_rejection = last_step = None
    with event_log.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("kind") == "original_action_rejection":
                last_rejection = event
            if event.get("kind") == "executed_step":
                last_step = event
    with collision_log.open() as stream:
        for line in stream:
            last_collision_check = json.loads(line)
    original = last_rejection["diagnostic"]
    return {
        "job_id": job,
        "slurm": dict(
            state=state,
            start=dates[0],
            end=dates[1],
            elapsed_s=int(elapsed_s),
            exit_code=exit_code,
            derived_exit_code=derived_exit,
        ),
        "archive_sha256": submission["archive_sha256"],
        "manifest_sha256": submission["manifest_sha256"],
        "commit": submission["commit"],
        **select(
            campaign,
            "model_server_starts",
            "formal_score_eligible",
            "transfer_claim_eligible",
            "observed_gpu_peak_mib",
            "stop_reason",
            "cleanup_failures",
        ),
        "graphics_checks": graphics_checks,
        "startup": [
            dict(episode_id=k, **select(v, "status", "wall_s"))
            for k, v in campaign["startup"].items()
        ],
        "trials": trials,
        "counts": dict(
            planned=len(trials),
            executed=sum("success" in row for row in trials),
            successes=sum(row.get("success") is True for row in trials),
            not_run=sum(row["status"] == "not_run" for row in trials),
        ),
        "terminal_trial_evidence": {
            "episode_id": terminal_trial,
            "last_planning_rejection": {
                **select(last_rejection, "kind", "status"),
                **select(
                    original, "collision_world", "geometry", "stamp", "failed_row"
                ),
            },
            "last_executed_step": select(
                last_step, "tick", "sim_s", "physics_step_count"
            ),
            "last_recorded_collision_check": select(
                last_collision_check, "call", "raw", "blocked", "ignored"
            ),
        },
        "source_sha256": source_files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("accounting_excerpt", type=Path)
    parser.add_argument("submission_receipt", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                args.run_directory,
                args.accounting_excerpt,
                read(args.submission_receipt),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
