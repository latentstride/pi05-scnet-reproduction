"""Bounded G2 rollout shared by native, retargeted, and expert conditions."""
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic

import numpy as np

from g2e_protocol import PROTOCOL, APPROACH_REPAIR, NativeApproachBarrier, validate_window
from g2_adapter import G2Error, SIDES, images_of, positive
from pace_runtime import validate_request, validate_response
from robotwin_control_geometry import Status, finite_array


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=json_value, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def save_array(path, writer, *args, **kwargs):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        writer(handle, *args, **kwargs)
    tmp.replace(path)


def accepted(result, event):
    event(dict(status=result.status.value, detail=result.detail, **result.diagnostic))
    if not result.ok:
        raise G2Error(result.status, result.detail, **result.diagnostic)
    return result.payload


def finish_episode(backend, output, record, event):
    """Attempt all cleanup before emitting evidence, including on failed hold."""
    failures = []
    try:
        if not record["success"]:
            try:
                backend.hold()
            except (Exception, KeyboardInterrupt, SystemExit) as error:
                failures.append(dict(kind="hold_failed", detail=repr(error)))
    finally:
        try:
            record["video"] = backend.close(output)
        except (Exception, KeyboardInterrupt, SystemExit) as error:
            failures.append(dict(kind="close_failed", detail=repr(error)))
    if failures:
        record.update(success=False, outcome="environment_error", cleanup_failures=failures)
        if not record.get("failure"):
            first = failures[0]
            record["failure"] = dict(phase="termination", code=first["kind"], detail=first["detail"])
        for failure in failures:
            event(failure)


def native_prefix(chunk, window, limits, max_grip_clip, event):
    targets = chunk[:window].copy()
    for i, row in enumerate(targets.reshape(-1, 2, 7)):
        for side, arm, lim in zip(SIDES, row, limits):
            raw = float(arm[6])
            final = float(np.clip(raw, 0, 1))
            allowed = abs(raw - final) <= max_grip_clip
            event(dict(kind="gripper_target", row=i, side=side, raw=raw,
                       rule="bounded_clip", normalized=final if allowed else None))
            if not allowed or np.any((arm[:6] < lim[:, 0]) | (arm[:6] > lim[:, 1])):
                raise G2Error(Status.LIMIT, "native Aloha target outside configured bounds",
                              geometry="candidate")
            arm[6] = final
    return targets


class ExpertWaypoints:
    """Use indexed Aloha joint targets as geometry. Never infer a recording rate."""
    def __init__(self, rows, source_id):
        rows = np.asarray(rows, dtype=float)
        if not len(rows) or not source_id:
            raise ValueError("expert source and waypoints are required")
        self.rows = finite_array(rows, (len(rows), 14)).copy()
        self.source_id = source_id

    def next(self, start, window):
        rows = self.rows[start:start + window]
        if not len(rows):
            return None, 0
        chunk = np.empty((50, 14), dtype=float)
        chunk[:] = rows[-1]
        chunk[:len(rows)] = rows
        return chunk, len(rows)


def run_episode(spec, backend, policy, output, *, adapter=None, expert=None, allow_native_h50=False):
    """Backend owns simulator reset, state capture, measured execution and hold.

    One execute call applies a native target or a synchronized trajectory prefix.
    Backend logs each physical/control step via the callback before further motion.
    Wall deadlines are cooperative here; a host supervisor must cap native calls.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    window, max_actions = spec["window"], spec["max_actions"]
    validate_window(window, spec.get("execution_protocol"), allow_native_h50=allow_native_h50)
    if type(max_actions) is not int or max_actions < 1:
        raise ValueError("invalid episode bounds")
    if (expert is None) == (policy is None) or (expert is not None and adapter is None):
        raise ValueError("use one policy or expert source; expert requires retiming")
    if spec["data_kind"] not in ("synthetic", "diagnostic"):
        raise ValueError("formal execution requires the future frozen-lock/schema gate")
    t0 = monotonic()
    end = t0 + positive(spec["timeout_s"])
    phase = "reset"
    record = dict(data_kind=spec["data_kind"], formal_score_eligible=False, episode_id=spec["episode_id"], condition=spec["condition"], window=window,
                  success=False, outcome="running", requested=0, executed=0, queries=0,
                  inference_s=0., planning_s=0., control_s=0., plan_calls=0,
                  plan_failures=0, initial=None, failure=None, video=None, control_calls=0)
    save_json(output / "result.json", record)
    with (output / "events.jsonl").open("x", buffering=1) as events:
        control_reports=0
        def event(payload):
            events.write(json.dumps({"phase": phase, "elapsed_s": monotonic() - t0, **payload},
                                    default=json_value, allow_nan=False) + "\n")

        def remaining():
            if monotonic() >= end:
                raise G2Error(Status.TIMEOUT, f"episode timed out in {phase}")
            return end - monotonic()

        def observe():
            snapshot, native = backend.capture()
            if snapshot.stamp.episode != spec["episode_id"]:
                raise G2Error(Status.UNSYNCED, "captured a different episode")
            return snapshot, native

        def step(payload):
            nonlocal control_reports
            control_reports+=1
            calls = payload.get("control_call_count")
            if calls is None:
                record["control_calls"] = None
            elif record["control_calls"] is not None:
                if type(calls) is not int or calls < 0:
                    raise ValueError("invalid backend control call count")
                record["control_calls"] += calls
            event({"kind": "executed_step", **payload})
            remaining()

        prev = None
        pending_native_barrier = None
        try:
            record["initial"] = backend.reset(spec)
            save_json(output / "initial.json", record["initial"])
            if spec.get("execution_protocol") == PROTOCOL:
                if adapter is not None or expert is not None or getattr(backend, "execution_protocol", None) != PROTOCOL:
                    raise G2Error(Status.INVALID, "G2-E native guard is missing")
                record["execution_protocol"] = PROTOCOL
            if backend.success():
                raise G2Error(Status.INVALID, "task already successful on reset")
            if expert is not None:
                event(dict(kind="expert_source", id=expert.source_id,
                           original_sample_period_s="unknown", geometry="expert_qpos"))
            while record["executed"] < max_actions:
                phase = "observation"
                remaining()
                snapshot, native = observe()
                if pending_native_barrier is not None:
                    if (snapshot.stamp.episode != pending_native_barrier.episode
                            or snapshot.stamp.tick != pending_native_barrier.tick
                            or snapshot.stamp.sim_s != pending_native_barrier.sim_s):
                        raise G2Error(Status.UNSYNCED, "bounded approach observation does not match completed physical prefix")
                    pending_native_barrier = None
                if prev is not None and (snapshot.stamp.tick <= prev.tick or snapshot.stamp.sim_s <= prev.sim_s):
                    raise G2Error(Status.UNSYNCED, "policy observation did not advance")
                prev = snapshot.stamp
                i = record["queries"] + 1
                rgb = images_of(snapshot)
                save_array(output / f"observation-{i:05}.npz", np.savez_compressed, **rgb)
                event(dict(kind="snapshot", stamp=asdict(snapshot.stamp),
                           actual=[asdict(arm) for arm in snapshot.arms],
                           world_task=snapshot.world_task, objects_task=snapshot.objects_task,
                           prompt=spec["prompt"], rgb_file=f"observation-{i:05}.npz"))
                if record.get("initial_sim_s") is None:
                    record["initial_sim_s"] = snapshot.stamp.sim_s
                if expert is not None:
                    accepted(adapter.begin_expert(snapshot, remaining()), event)
                    request = None
                else:
                    request = (accepted(adapter.observe(snapshot, spec["prompt"], remaining()), event)
                               if adapter else {"state": np.asarray(native, dtype=np.float32),
                                                "images": rgb, "prompt": spec["prompt"]})
                    validate_request(request)
                    event(dict(kind="policy_input", state=request["state"], prompt=request["prompt"]))
                phase = "inference"
                count = min(window, max_actions - record["executed"])
                record["queries"] += 1
                if expert is not None:
                    chunk, count = expert.next(record["executed"], count)
                    if chunk is None:
                        break
                else:
                    start = monotonic()
                    try:
                        infer = getattr(policy, "infer_raw", policy.infer)
                        response = infer(request)
                        raw = response.get("actions") if isinstance(response, dict) else None
                        if isinstance(raw, np.ndarray) and raw.dtype.kind in "fiu" and raw.size <= 100_000:
                            save_array(output / f"action-{i:05}.npy", np.save, raw, allow_pickle=False)
                        chunk = validate_response(response)
                    finally:
                        record["inference_s"] += monotonic() - start
                remaining()
                if expert is not None:
                    save_array(output / f"action-{i:05}.npy", np.save, chunk[:count], allow_pickle=False)
                    event(dict(kind="expert_window", source_frames=list(range(record["executed"], record["executed"] + count))))
                record["requested"] += count
                phase = "planning"
                record["plan_calls"] += 1
                start = monotonic()
                try:
                    if adapter:
                        plan = accepted(adapter.actions(chunk, snapshot, count, remaining()), event)
                        save_array(output / f"plan-{i:05}.npz", np.savez_compressed,
                                   **{k:v for k,v in asdict(plan).items() if v is not None})
                    else:
                        plan = native_prefix(chunk, count, backend.native_limits, spec["native_grip_max_clip"], event)
                        save_array(output / f"plan-{i:05}.npy", np.save, plan, allow_pickle=False)
                except Exception:
                    record["plan_failures"] += 1
                    raise
                finally:
                    record["planning_s"] += monotonic() - start
                phase = "control"
                start = monotonic()
                try:
                    remaining()
                    before_reports = control_reports
                    try:
                        done = backend.execute(plan, step, remaining)
                    except BaseException as error:
                        if spec.get("execution_protocol") == PROTOCOL:
                            completed = getattr(error, "diagnostic", {}).get("completed_actions", 0)
                            if type(completed) is not int or not 0 <= completed < count:
                                raise ValueError("invalid partial native action receipt") from error
                            record["executed"] += completed
                            event(dict(kind="partial_prefix_abort", completed_actions=completed,
                                       interrupted_action_may_have_advanced=True))
                        if control_reports == before_reports:
                            record["control_calls"] = None
                        raise
                    approved=(len(plan.waypoint_ticks) if adapter and getattr(plan,"contact_profile_id","") else count)
                    reason=None
                    if callable(getattr(backend,"take_replan",None)):
                        reason=backend.take_replan()
                    bounded_replan=(0<done<=approved<=count and
                                    (done==approved or reason is not None)) if type(done) is int else False
                    native_replan = isinstance(reason, NativeApproachBarrier)
                    if native_replan:
                        if not (adapter is None and expert is None
                                and spec.get("execution_protocol") == PROTOCOL
                                and spec.get("free_space_approach_repair") == APPROACH_REPAIR
                                and reason.protocol == APPROACH_REPAIR
                                and type(done) is int and 0 < done <= count
                                and type(reason.consumed_actions) is int and reason.consumed_actions == done
                                and reason.episode == snapshot.stamp.episode
                                and type(reason.tick) is int and reason.tick > snapshot.stamp.tick
                                and np.isfinite(reason.sim_s) and reason.sim_s > snapshot.stamp.sim_s):
                            raise G2Error(Status.UNSYNCED, "invalid native bounded-approach receipt")
                        pending_native_barrier = reason
                    if (type(done) is not int or not 0 <= done <= count
                            or (done<count and not backend.success() and
                                not (native_replan or (adapter and getattr(plan,"contact_profile_id","") and bounded_replan)))):
                        raise G2Error(Status.BACKEND_ERROR, "backend did not complete the approved prefix")
                    if native_replan:
                        reason = dict(reason="bounded_approach_requires_new_observation", **asdict(reason))
                    if reason is not None or approved<count:
                        event(dict(kind="contact_replan",requested_actions=count,approved_actions=approved,
                                   consumed_actions=done,reason=reason or {"reason":"planned_contact_barrier"}))
                    record["executed"] += done
                finally:
                    record["control_s"] += monotonic() - start
                remaining()
                if backend.success():
                    record.update(success=True, outcome="success")
                    break
                save_json(output / "result.json", record)
            if not record["success"]:
                record.update(outcome="task_unsuccessful", failure=dict(
                    phase="termination", code="episode_budget_or_expert_exhausted",
                    detail="success predicate remained false"))
        except (KeyboardInterrupt, SystemExit, Exception) as exc:
            code = exc.status.value if isinstance(exc, G2Error) else type(exc).__name__
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                outcome = "interrupted"
            elif code == Status.TIMEOUT.value or isinstance(exc, TimeoutError):
                outcome = "timeout"
            elif code in (Status.IK_FAILED.value, Status.UNREACHABLE.value):
                outcome = code
            elif code == Status.LIMIT.value:
                outcome = "joint_limit"
            else:
                outcome = {"reset": "environment_error", "inference": "policy_failed",
                           "control": "control_error", "observation": "adapter_error",
                           "planning": "planning_failed"}[phase]
            record.update(outcome=outcome, failure=dict(phase=phase, code=code, detail=str(exc) or code,
                                                       diagnostic=getattr(exc, "diagnostic", {})))
        finally:
            finish_episode(backend, output, record, event)
            record["elapsed_s"] = monotonic() - t0
            save_json(output / "result.json", record)
    return record
