"""Derive the author H50 budget review from existing historical logs, offline."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ACTIONS = 800
CAP = 25_000
SECONDS = 1150


def read(path):
    return json.loads(path.read_bytes())


def events_and_sha(path, extract):
    digest = sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            extract(json.loads(line))
    return digest.hexdigest()


def action_marks(event, marks):
    if event.get("kind") == "native_action_request":
        marks.append(
            dict(
                index=event["action_index_in_episode"],
                tick=event["tick"],
                elapsed_s=event["elapsed_s"],
            )
        )


def sample_summary(values):
    return dict(
        n=len(values),
        minimum=min(values),
        mean=float(np.mean(values)),
        p10=float(np.percentile(values, 10)),
        median=float(np.median(values)),
        p90=float(np.percentile(values, 90)),
        maximum=max(values),
    )


def episode(path):
    result = read(path / "result.json")
    marks = []
    digest = events_and_sha(path / "events.jsonl", lambda e: action_marks(e, marks))
    # A subsequent issued command proves its predecessor returned. The last
    # issued command in these historical failed trials is excluded unless the
    # final completed count and total ticks prove it completed.
    complete = result["executed"]
    assert [m["index"] for m in marks] == list(range(len(marks)))
    assert len(marks) == complete + 1 and result["outcome"] == "control_error"
    steps = [b["tick"] - a["tick"] for a, b in zip(marks, marks[1:])]
    assert len(steps) == complete and min(steps) > 0
    return dict(
        source_events_sha256=digest,
        source_result_sha256=sha256((path / "result.json").read_bytes()).hexdigest(),
        action_start_ticks=[m["tick"] for m in marks],
        completed_actions=complete,
        steps_per_completed_action=sample_summary(steps),
        query_count=result["queries"],
        control_s=result["control_s"],
        inference_s=result["inference_s"],
        wall_s=result["elapsed_s"],
        observed_control_s_per_physics_step=result["control_s"]
        / result["control_calls"],
        final_partial_action_steps=result["control_calls"] - marks[-1]["tick"],
        topp_branch="UNKNOWN: historical events do not label individual arm TOPP outcomes",
    )


def service(path):
    stages = defaultdict(list)

    def extract(event):
        if event["event"] == "end":
            stages[event["stage"]].append(event["elapsed_s"])

    digest = events_and_sha(path, extract)
    return dict(
        source_sha256=digest,
        startup_s={k: v[0] for k, v in stages.items() if k != "inference"},
        warm_service_inference_s=sample_summary(stages["inference"]),
    )


def main(first, second, historical_result):
    native = {
        d.parent.name: episode(d)
        for d in sorted((first / "trials").glob("*native-*/episode"))
    }
    assert len(native) == 6
    selected = second / "selected-events.json"
    selection = read(selected)
    marks = []
    for row in selection["selected"]:
        action_marks(row["event"], marks)
    published = read(historical_result)
    assert published["job_id"] == "127192740"
    first_trial = published["first_trial"]
    completed = first_trial["executed"]
    assert completed == 27 and len(marks) == completed + 1
    steps = [b["tick"] - a["tick"] for a, b in zip(marks, marks[1:])]
    assert marks[-1]["tick"] == first_trial["last_physical_tick"] == 3575
    campaign_path = first / "supervisor/campaign.json"
    campaign = read(campaign_path)
    admitted = [
        v["elapsed_s"]
        for v in campaign["startup"].values()
        if v["status"] == "admitted"
    ]
    warm_wall = [v["inference_s"] / v["query_count"] for v in native.values()]
    case_wall = [
        ACTIONS
        * v["steps_per_completed_action"]["mean"]
        * v["observed_control_s_per_physics_step"]
        + (v["wall_s"] - v["control_s"] - v["inference_s"])
        + 16 * warm
        for v, warm in zip(native.values(), warm_wall)
    ]
    return dict(
        author_source=dict(
            revision="fd2e624cc19325cc9b20a620c82d51c0d8e25cc1",
            take_action="envs/_base_task.py:1479-1662",
            success="envs/handover_block.py:111-116",
            rule="No-success command uses at least max(left_steps,right_steps), at most their sum. Both arms falling back together use exactly 50; one fallback uses at least 50. Each arm falls back on TOPP exception or empty output.",
        ),
        job_127147943=dict(
            native=native,
            model=service(first / "model/phases.jsonl"),
            source_campaign_sha256=sha256(campaign_path.read_bytes()).hexdigest(),
            six_dynamic_admission_wall_s=sample_summary(admitted),
            dynamic_admission_sum_s=sum(admitted),
        ),
        job_127192740=dict(
            source_selected_events_sha256=sha256(selected.read_bytes()).hexdigest(),
            source_result_sha256=sha256(historical_result.read_bytes()).hexdigest(),
            action_start_ticks=[m["tick"] for m in marks],
            completed_actions=completed,
            steps_per_completed_action_including_repair=sample_summary(steps),
            historical_repair_and_requery=True,
            topp_branch="UNKNOWN",
            model=service(second / "model/phases.jsonl"),
        ),
        sensitivity_only=dict(
            ceiling=CAP,
            wall_s_ceiling=SECONDS,
            native_actions=ACTIONS,
            queries_at_full_budget=16,
            max_average_substeps_per_command=CAP / ACTIONS,
            always_both_fallback_steps=50 * ACTIONS,
            steps_if_historical_trial_means_held=[
                round(
                    ACTIONS
                    * min(
                        v["steps_per_completed_action"]["mean"] for v in native.values()
                    )
                ),
                round(
                    ACTIONS
                    * max(
                        v["steps_per_completed_action"]["mean"] for v in native.values()
                    )
                ),
            ],
            control_wall_at_25000_steps_if_historical_rate_held=[
                round(
                    CAP
                    * min(
                        v["observed_control_s_per_physics_step"]
                        for v in native.values()
                    ),
                    2,
                ),
                round(
                    CAP
                    * max(
                        v["observed_control_s_per_physics_step"]
                        for v in native.values()
                    ),
                    2,
                ),
            ],
            control_wall_if_historical_mean_and_rate_held=[
                round(
                    min(
                        ACTIONS
                        * v["steps_per_completed_action"]["mean"]
                        * v["observed_control_s_per_physics_step"]
                        for v in native.values()
                    ),
                    2,
                ),
                round(
                    max(
                        ACTIONS
                        * v["steps_per_completed_action"]["mean"]
                        * v["observed_control_s_per_physics_step"]
                        for v in native.values()
                    ),
                    2,
                ),
            ],
            trial_wall_including_historical_residual_and_16_warm_queries_s=[
                round(min(case_wall), 2),
                round(max(case_wall), 2),
            ],
            trial_wall_assumptions=(
                "Per historical native trial: scale complete-action mean and logged control rate to 800, "
                "add the measured trial wall residual and scale its client inference mean to 16 queries. "
                "Separate shared startup and campaign cleanup are excluded."
            ),
            modeled_as_upper_bound=False,
        ),
        feasibility="UNKNOWN for 800 author commands; observed histories and the native fallback do not establish sufficient budget",
        renderer_and_shared_service="Measured historical service phases and admission wall time; standalone fresh rendering, author-state compile overlap, actual author action count and worst-case time are UNKNOWN.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("job-127147943", "job-127192740", "published-result", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(
            main(args.job_127147943, args.job_127192740, args.published_result),
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
