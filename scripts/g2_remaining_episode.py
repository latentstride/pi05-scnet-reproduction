"""Bounded diagnostic persistence/cleanup harness for the opt-in CPU-tested coordinator.

Uses injected existing scene/backend and policy; never allocates a resource or
loads a model. Real scenes may only be instantiated by a separately approved
future unified job, never by this module's import.
"""

from dataclasses import asdict
from pathlib import Path
from time import monotonic
import json
import numpy as np
from g2_adapter import G2Error, images_of
from g2_episode import finish_episode, save_json, json_value
from g2_remaining_coordinator import RecoveryBudgetExhausted, RecoveryCoordinator
from robotwin_control_geometry import Status


def run(spec, backend, adapter, policy, output):
    if spec.get("data_kind") not in ("synthetic", "diagnostic") or spec[
        "window"
    ] not in (1, 4):
        raise ValueError("diagnostic original-window scope required")
    action_limit = 300
    if spec.get("task_tcp_projection") is not None:
        from g2_unified_plan import ARX_PROJECTION_ACTION_CAP, ARX_PROJECTION_METHOD

        if spec["task_tcp_projection"] == ARX_PROJECTION_METHOD:
            action_limit = ARX_PROJECTION_ACTION_CAP
    if type(spec["max_actions"]) is not int or not 0 < spec["max_actions"] <= action_limit:
        raise ValueError("policy action cap required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    end = started + float(spec["timeout_s"])
    if not np.isfinite(end) or end <= started:
        raise ValueError("invalid timeout")
    row = dict(
        data_kind=spec["data_kind"],
        formal_score_eligible=False,
        episode_id=spec["episode_id"],
        outcome="running",
        success=False,
        queries=0,
        policy_actions=0,
        recovery_actions=0,
        failure=None,
        condition=spec["condition"],
        window=spec["window"],
        requested=0,
        executed=0,
        inference_s=0.0,
        planning_s=0.0,
        control_s=0.0,
        plan_calls=0,
        plan_failures=0,
        control_calls=0,
        initial=None,
        initial_sim_s=None,
    )
    coordinator = RecoveryCoordinator(
        adapter,
        backend,
        max_recoveries=spec.get("max_recoveries", 4),
        approach_repair=spec.get("approach_repair", False),
        task_tcp_projection=spec.get("task_tcp_projection"),
    )
    if coordinator.task_tcp_projection:
        row["task_tcp_projection"] = coordinator.task_tcp_projection

    def remaining():
        if monotonic() >= end:
            raise G2Error(Status.TIMEOUT, "remaining G2 trial deadline")
        return end - monotonic()

    phase = "reset"
    phase_start = monotonic()
    current_window = spec["window"]

    def transition(next_phase):
        nonlocal phase, phase_start
        now = monotonic()
        key = {
            "inference": "inference_s",
            "planning": "planning_s",
            "control": "control_s",
        }.get(phase)
        if key:
            row[key] += now - phase_start
        phase, phase_start = next_phase, now

    with (output / "events.jsonl").open("x", buffering=1) as stream:

        def event(payload):
            payload = dict(payload)
            kind = payload.get("kind")
            if payload.get("physics_step_count", 0) or (
                "period_s" in payload and "tick" in payload
            ):
                if kind is None:
                    payload["kind"] = "executed_step"
                calls = payload.get("control_call_count")
                if calls is None:
                    row["control_calls"] = None
                elif row["control_calls"] is not None:
                    row["control_calls"] += calls
            if kind == "policy_request":
                transition("inference")
                row["requested"] += current_window
                row["queries"] += 1
                index = row["queries"]
                snap = payload.pop("snapshot")
                request = payload.pop("request")
                if row["initial_sim_s"] is None:
                    row["initial_sim_s"] = snap.stamp.sim_s
                np.savez_compressed(
                    output / f"observation-{index:05}.npz", **images_of(snap)
                )
                payload.update(
                    stamp=snap.stamp.__dict__,
                    state=request["state"],
                    prompt=request["prompt"],
                )
                if coordinator.task_tcp_projection:
                    payload.update(actual=[asdict(arm) for arm in snap.arms],
                                   world_task=snap.world_task, objects_task=snap.objects_task)
            elif kind == "policy_response":
                response = payload.pop("response")
                raw = response.get("actions") if isinstance(response, dict) else None
                if isinstance(raw, np.ndarray) and raw.size <= 100000:
                    np.save(
                        output / f"action-{row['queries']:05}.npy",
                        raw,
                        allow_pickle=False,
                    )
                payload["response_type"] = type(response).__name__
            elif kind == "policy_validated":
                transition("planning")
                row["plan_calls"] += 1
            elif kind == "approved_plan":
                transition("control")
                trajectory = payload.pop("trajectory")
                np.savez_compressed(
                    output / f"plan-{row['queries']:05}.npz",
                    **{k: v for k, v in trajectory.items() if v is not None},
                )
            stream.write(
                json.dumps(
                    dict(phase=phase, elapsed_s=monotonic() - started, **payload),
                    default=json_value,
                    allow_nan=False,
                )
                + "\n"
            )

        save_json(output / "result.json", row)
        try:
            row["initial"] = backend.reset(spec)
            if backend.success():
                raise G2Error(Status.INVALID, "already successful reset")
            while (
                coordinator.policy_actions < spec["max_actions"]
                and row["queries"] < 300
            ):
                remaining()
                transition("observation")
                window = min(
                    spec["window"], spec["max_actions"] - coordinator.policy_actions
                )
                current_window = window
                coordinator.cycle(policy, spec["prompt"], window, remaining, event)
                row.update(
                    executed=coordinator.policy_actions,
                    policy_actions=coordinator.policy_actions,
                    recovery_actions=coordinator.recovery_actions,
                )
                save_json(output / "result.json", row)
                if backend.success():
                    row.update(outcome="success", success=True)
                    break
            if not row["success"]:
                row.update(
                    outcome="task_unsuccessful",
                    failure=dict(
                        phase="termination",
                        code="action_budget_exhausted",
                        detail="policy action or query budget ended before success",
                    ),
                )
        except (Exception, KeyboardInterrupt, SystemExit) as exc:
            row.update(
                outcome="interrupted"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "timeout"
                if isinstance(exc, TimeoutError)
                or isinstance(exc, G2Error)
                and exc.status == Status.TIMEOUT
                else RecoveryBudgetExhausted.reason
                if isinstance(exc, RecoveryBudgetExhausted)
                else {
                    "inference": "policy_failed",
                    "planning": "planning_failed",
                    "control": "control_error",
                    "observation": "adapter_error",
                }.get(phase, "environment_error"),
                failure=dict(
                    phase=phase,
                    code=exc.status.value
                    if isinstance(exc, G2Error)
                    else type(exc).__name__,
                    detail=str(exc),
                    diagnostic=getattr(exc, "diagnostic", {}),
                ),
            )
        finally:
            transition("termination")
            row.update(
                executed=coordinator.policy_actions,
                policy_actions=coordinator.policy_actions,
                recovery_actions=coordinator.recovery_actions,
            )
            finish_episode(backend, output, row, event)
            row["elapsed_s"] = monotonic() - started
            save_json(output / "result.json", row)
    return row
