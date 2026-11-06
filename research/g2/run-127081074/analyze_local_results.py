"""Read-only audit of the retrieved G2-E run; never submits or modifies jobs."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text())

events = read(ROOT / "derived/event-summary.json")
evaluation = read(ROOT / "statistics/evaluation-summary.json")
assert evaluation["completed_trials"] == 8
assert evaluation["task_success_count"] == 0
rows = []
blocks = {}
for directory in sorted((ROOT / "trials").iterdir()):
    if not directory.is_dir():
        continue
    result = read(directory / "result.json")
    event = events[directory.name]
    rejection = read(ROOT / "derived" / directory.name / "rejection-context.json")[-1]
    requests = read(ROOT / "derived" / directory.name / "requests.json")
    raw_blocks = []
    for f in sorted(directory.glob("action-*.npy")):
        raw = np.load(f, allow_pickle=False)
        assert raw.shape == (50, 14) and np.isfinite(raw).all()
        first = []
        for column in (6, 13):
            indices = np.flatnonzero(raw[:, column] < 0.5)
            first.append(int(indices[0]) if len(indices) else None)
        prefix = np.load(directory / f.name.replace("action-", "plan-"), allow_pickle=False)
        assert prefix.shape == (result["window"], 14)
        raw_blocks.append(dict(file=f.name, first_close_index=first))
    assert len(raw_blocks) == result["queries"]
    first_closed = {}
    for side, column in (("left", 6), ("right", 13)):
        selected = [r for r in requests if r["normalized_qpos_request"][column] < 0.5]
        first_closed[side] = selected[0]["action_index_in_episode"] if selected else None
    proposed = sum(r["first_close_index"][0] is not None for r in raw_blocks)
    retained = sum(r["first_close_index"][0] is not None and
                   r["first_close_index"][0] < result["window"] for r in raw_blocks)
    blocks[directory.name] = dict(
        window=result["window"], blocks=len(raw_blocks), raw_shape=[50, 14],
        closure_definition="raw gripper < 0.5; indices zero-based",
        left_close_blocks=proposed,
        left_close_inside_requested_prefix=retained,
        left_close_truncated=proposed-retained,
        blocks_detail=raw_blocks)
    rows.append(dict(
        episode_id=directory.name, window=result["window"], success=result["success"],
        executed=result["executed"], queries=result["queries"],
        request_count=len(requests), physics_steps=event["physics_steps"],
        elapsed_s=result["elapsed_s"], failure=result["failure"],
        first_closed_request=first_closed,
        first_holder_ticks=event["first_holder_ticks"],
        left_close_blocks=proposed, left_close_inside_requested_prefix=retained,
        left_close_truncated=proposed-retained,
        tcp_rmse_mm=[1000*x for x in event["tcp_position_rmse_m"]],
        tcp_max_mm=[1000*x for x in event["tcp_position_max_m"]],
        rejection=rejection["rejection"]))
    assert event["physics_steps"] == result["control_calls"]
assert sum(r["executed"] for r in rows) == 512
assert sum(r["physics_steps"] for r in rows) == 42320
assert sum(r["queries"] for r in rows) == 164
summary = dict(
    job_id="127081074", scheduler_state="COMPLETED", exit_code="0:0",
    start_beijing="2026-11-06 22:12:55", end_beijing="2026-11-06 23:28:45",
    elapsed_s=4550, elapsed="01:15:50", planned_trials=8, completed_trials=8,
    task_successes=0, full_physical_sequences=0,
    completed_actions=512, requested_action_calls=sum(r["request_count"] for r in rows),
    policy_queries=164, physics_steps=42320,
    model_server_starts=1, gpu_peak_mib=14382,
    estimated_gpu_cost_yuan_at_previously_verified_3_60_per_hour=4.55,
    billed_cost_verified=False, gpu_operations_by_this_audit=0,
    dynamic_success_admission=False, future_unified_plan_submission_allowed=False,
    rows=rows)
(ROOT / "derived/new-prediction-closure-analysis.json").write_text(
    json.dumps(blocks, indent=2) + "\n")
(ROOT / "audit-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k:v for k,v in summary.items() if k!="rows"}, indent=2))

