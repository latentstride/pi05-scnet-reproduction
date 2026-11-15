"""Score original ARX online receipts and report evidence coverage per trial."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

TERMINAL_RESOURCE = {"TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def events(episode, run):
    path = episode / "events.jsonl"
    full = path.is_file()
    if not full:
        path = run / "events-selected.jsonl"
    if not path.is_file():
        return False, iter(())

    def by_line():
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: malformed physical log") from error
                yield row if full else row["event"]

    return full, by_line()


def z(pose):
    return pose[2][3] if isinstance(pose[2], list) else pose[2]


def trial_metrics(run, episode_id, scheduler_state=None, *, hash_files=False):
    trial = run / "trials" / episode_id
    episode = trial / "episode"
    receipt = read(trial / "receipt.json")
    result = receipt.get("result") or read(episode / "result.json")
    standalone = read(run / "standalone-result.json")
    crash = read(run / "deployment-failure.json")
    readback = read(trial / "initial-readback.json") or read(run / "initial-readback.json")
    full, rows = events(episode, run)
    stage_names = ("FIRST_CONTACT", "BILATERAL_CONTACT", "STABLE_GRASP", "LIFT_50MM", "HANDOVER")
    stages = dict.fromkeys(stage_names)
    query_count, steps, last_tick = 0, 0, -1
    initial = (result.get("initial") or readback).get("objects_task", {}).get("block")
    initial_z = z(initial) if initial is not None else None
    peak_lift = None
    previously_held_left = shared = False
    first_target_contact_after_handover = None
    query_ticks, prefix_boundaries = [], []
    contact_steps, max_fingers = Counter(), Counter()
    row_in_prefix = None
    for event in rows:
        if event.get("kind") == "policy_request":
            query_count += 1
            query_ticks.append(event["stamp"]["tick"])
            if initial_z is None and event["stamp"]["tick"] == 0:
                block = (event.get("objects_task") or {}).get("block")
                initial_z = z(block) if block is not None else None
        if event.get("kind") == "direct_row_planning":
            row_in_prefix = event["row_in_prefix"]
        if event.get("kind") == "direct_prefix_complete":
            prefix_boundaries.append(dict(query=query_count, **{
                key: event[key] for key in ("executed_actions", "discarded_rows", "stamp", "reason")
            }))
        if not event.get("physics_step_count"):
            continue
        tick = event["tick"]
        if tick == last_tick:
            continue
        if tick < last_tick:
            raise ValueError(f"nonmonotonic physical ticks in {run}")
        last_tick = tick
        steps += event["physics_step_count"]
        contact = event.get("contact_state") or {}
        fingers = contact.get("contact_fingers", {})
        for side, ids in fingers.items():
            contact_steps[side] += event["physics_step_count"] if ids else 0
            max_fingers[side] = max(max_fingers[side], len(ids))
        holders = set(contact.get("holders", ()))
        previously_held_left |= holders == {"left"}
        shared |= previously_held_left and holders == {"left", "right"}
        block = contact.get("block_pose_task")
        lift = z(block) - initial_z if block is not None and initial_z is not None else None
        if holders and lift is not None:
            peak_lift = lift if peak_lift is None else max(peak_lift, lift)
        reached = (
            any(fingers.values()), any(len(ids) >= 2 for ids in fingers.values()), bool(holders),
            bool(holders) and lift is not None and lift >= 0.050, shared and holders == {"right"},
        )
        if shared and not holders and first_target_contact_after_handover is None:
            if any(set(pair["body_names"]) == {"box", "target_box"} and pair["points"]
                   for pair in event.get("physical_contacts") or ()):
                first_target_contact_after_handover = tick
        for name, present in zip(stage_names, reached):
            if present and stages[name] is None:
                stages[name] = dict(tick=tick, query=query_count, row_in_prefix=row_in_prefix,
                                    holders=sorted(holders), fingers=fingers)
    if full and "queries" in result and query_count != result["queries"]:
        raise ValueError(f"query count disagrees with original receipt in {run}")
    reached_control = receipt.get("fresh_reset_verified") is True and result.get("queries", 0) > 0
    task_success = result.get("success") if receipt and reached_control else None
    if receipt and result.get("success") is True and receipt.get("success") is True:
        task_success = True
    if (scheduler_state or "").split(" ")[0] in TERMINAL_RESOURCE or standalone.get("outcome") == "RESOURCE_CENSORED":
        outcome = "RESOURCE_CENSORED"
    elif task_success is True:
        outcome = "TASK_SUCCESS"
    elif receipt or standalone or crash or (scheduler_state and scheduler_state.startswith(("FAILED", "CANCELLED"))):
        outcome = "FAILED"
    else:
        outcome = "NO_TERMINAL_EVIDENCE"
    if outcome == "RESOURCE_CENSORED" and task_success is True:
        raise ValueError(f"success and resource censoring conflict in {run}")
    run_files = [path for path in sorted(run.rglob("*")) if path.is_file() and "cache" not in path.relative_to(run).parts]
    hashes = None
    if hash_files:
        hashes = {}
        for path in run_files:
            checksum = sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            hashes[str(path.relative_to(run))] = checksum.hexdigest()
    return dict(
        outcome=outcome, scheduler_state=scheduler_state, original_task_success=task_success,
        stages=stages, full_physical_log=full, logged_physics_steps=steps,
        peak_held_lift_mm=None if peak_lift is None else 1000 * peak_lift,
        first_target_contact_after_handover_tick=first_target_contact_after_handover,
        original_queries=result.get("queries"), original_executed_policy_actions=result.get("policy_actions"),
        query_ticks=query_ticks, prefix_boundaries=prefix_boundaries,
        contact_steps=contact_steps, max_fingers=max_fingers,
        last_planned_row=row_in_prefix, application_outcome=result.get("outcome"),
        application_receipt_category=receipt.get("category"),
        termination=result.get("failure") or crash or standalone.get("stop_reason") or receipt.get("category"),
        rgb_files=sum(path.name.startswith("observation-") for path in run_files),
        action_chunks=sum(path.name.startswith("action-") for path in run_files),
        videos=[str(path.relative_to(run)) for path in run_files if path.suffix == ".mp4" and trial in path.parents],
        sha256=hashes,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("--hash-files", action="store_true")
    args = parser.parse_args()
    receipts = read(args.receipts)
    trials = {}
    for label, job in receipts["candidates"].items():
        if not job.get("job_id"):
            trials[label] = {"outcome": "NOT_SUBMITTED", "reason": job["reason"]}
        else:
            run = args.results / job["result_directory"]
            trials[label] = trial_metrics(run, job["episode"], job.get("slurm_state"),
                                          hash_files=args.hash_files)
    scored = [value for value in trials.values()
              if value["outcome"] in {"TASK_SUCCESS", "FAILED"} and type(value.get("original_task_success")) is bool]
    successes = sum(value["original_task_success"] for value in scored)
    print(json.dumps(dict(trials=trials, observed_task_successes=successes, scored_trials=len(scored),
                         observed_success_rate=successes / len(scored) if scored else None),
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
