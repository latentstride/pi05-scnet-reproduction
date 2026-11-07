"""Experimental recovery coordinator around unmodified G2 adapter/backend APIs.

No model is queried on fabricated/failed proxy state. Recovery is physical,
individually checked, capped and logged separately from policy action execution.
Only explicitly selected future conditions should construct this coordinator.
"""

from time import monotonic
import numpy as np
from dataclasses import asdict
from g2_adapter import G2Error, deadline, read_snapshot
from g2_remaining_geometry import refined_pair, bounded_progress
from robotwin_control_geometry import Status


class RecoveryBudgetExhausted(G2Error):
    reason = "recovery_budget_exhausted"

    def __init__(self, used, limit):
        super().__init__(
            Status.PLAN_FAILED,
            "trial recovery cap exhausted",
            reason=self.reason,
            recoveries_used=used,
            recovery_limit=limit,
        )


class RecoveryCoordinator:
    def __init__(self, adapter, backend, max_recoveries=4, approach_repair=False):
        if type(max_recoveries) is not int or not 0 <= max_recoveries <= 4:
            raise ValueError("invalid recovery cap")
        self.adapter, self.backend = adapter, backend
        self.max_recoveries = max_recoveries
        self.recoveries = 0
        self.recovery_actions = 0
        self.policy_actions = 0
        self.approach_repair = approach_repair
        self.source_obstructions = {}

    def _free(self):
        tracker = getattr(self.adapter.target_collision, "tracker", None)
        if tracker is not None and (
            tracker.holders
            or tracker.releasing
            or any(getattr(tracker, "contact_fingers", {}).values())
        ):
            raise G2Error(
                Status.COLLISION,
                "generic recovery forbidden during holding/release",
                reason="recovery_unsafe_contact",
            )
        if self.recoveries >= self.max_recoveries:
            raise RecoveryBudgetExhausted(self.recoveries, self.max_recoveries)

    def _start(self, snapshot):
        q, g, frames, _ = read_snapshot(snapshot, self.adapter.arx)
        world = self.adapter.target_collision
        world.refresh()
        if callable(getattr(world, "set_start", None)):
            reader = getattr(self.backend, "_drive_fingers", None)
            world.set_start(g, reader() if callable(reader) else g)
        return q, g, frames, world

    def mapped_observation(self, snapshot, prompt, remaining, log):
        original = self.adapter.observe(snapshot, prompt, remaining())
        if original.ok:
            return snapshot, original.payload
        log(
            dict(
                kind="original_proxy_rejection",
                status=original.status.value,
                diagnostic=original.diagnostic,
            )
        )
        if (
            original.status != Status.COLLISION
            or original.diagnostic.get("collision_world") != "source_collision"
        ):
            raise G2Error(original.status, original.detail, **original.diagnostic)
        if self.approach_repair and original.diagnostic.get("geometry") == "actual":
            raise G2Error(original.status, original.detail, **original.diagnostic)
        self._free()
        obstruction = None
        if self.approach_repair:
            from g2_bounded_approach import source_obstruction

            obstruction = source_obstruction(original.diagnostic)
            if obstruction is not None and obstruction[0] in self.source_obstructions:
                previous = self.source_obstructions[obstruction[0]]
                if obstruction[1] >= previous["penetration_m"]:
                    raise G2Error(
                        Status.PLAN_FAILED,
                        "repeated source obstruction did not improve after physical recovery",
                        geometry="candidate",
                        collision_world="source_collision",
                        reason="repeated_source_obstruction",
                        collision_pairs=obstruction[0],
                        penetration_m=obstruction[1],
                        previous=previous,
                        recoveries_used=self.recoveries,
                        recovery_limit=self.max_recoveries,
                    )
        check = deadline(remaining())
        q, g, frames, world = self._start(snapshot)
        # The last successful source seed is a real achieved reference, not a
        # new substitute observation. Return actual ARX to its tool pose first.
        tools = np.array(
            [
                f @ a.chain.fk(v) @ a.offset()
                for a, f, v in zip(
                    self.adapter.aloha, self.adapter.source_frames, self.adapter.seed
                )
            ]
        )
        try:
            goal, residual, meta = refined_pair(
                self.adapter.arx, frames, tools, q, g, world, check
            )
            path = self.adapter.motion.plan(q, g, [(goal, g)], world, check)
        except G2Error as error:
            error.diagnostic.setdefault("geometry", "candidate")
            raise
        self.recoveries += 1
        log(
            dict(
                kind="physical_proxy_recovery",
                attempt=self.recoveries,
                goal_q=goal,
                target_tools=tools,
                source_seed=self.adapter.seed.copy(),
                residual=residual,
                refinement=meta,
            )
        )
        done = self.backend.execute(path, log, remaining)
        if type(done) is not int or done != 1:
            raise G2Error(Status.BACKEND_ERROR, "proxy recovery prefix not completed")
        self.recovery_actions += done
        if obstruction is not None:
            self.source_obstructions[obstruction[0]] = dict(
                penetration_m=obstruction[1],
                attempt=self.recoveries,
                policy_actions=self.policy_actions,
            )
        fresh, _ = self.backend.capture()
        if (
            fresh.stamp.episode != snapshot.stamp.episode
            or fresh.stamp.tick <= snapshot.stamp.tick
            or fresh.stamp.sim_s <= snapshot.stamp.sim_s
        ):
            raise G2Error(
                Status.UNSYNCED,
                "physical recovery did not produce a fresh measured observation",
            )
        mapped = self.adapter.observe(fresh, prompt, remaining())
        if not mapped.ok:
            raise G2Error(
                mapped.status,
                "proxy still blocked after one physical recovery",
                **mapped.diagnostic,
            )
        return fresh, mapped.payload

    def action_plan(self, chunk, snapshot, window, remaining, log):
        original = self.adapter.actions(chunk, snapshot, window, remaining())
        if original.ok:
            if self.approach_repair:
                from g2_bounded_approach import checked_prefix

                prefix, barrier = checked_prefix(
                    self.adapter, original.payload, snapshot, deadline(remaining())
                )
                if barrier is not None:
                    self._free()
                    if prefix is None:
                        return self._upward_plan(
                            chunk, snapshot, remaining, log, barrier, source_check=True
                        )
                    self.recoveries += 1
                    log(
                        dict(
                            barrier,
                            attempt=self.recoveries,
                            approved_actions=len(prefix.waypoint_ticks),
                        )
                    )
                    return prefix, dict(repaired=True, receipt=barrier)
            return original.payload, dict(repaired=False)
        log(
            dict(
                kind="original_action_rejection",
                status=original.status.value,
                diagnostic=original.diagnostic,
            )
        )
        if (
            original.status != Status.COLLISION
            or original.diagnostic.get("collision_world") != "target_collision"
        ):
            raise G2Error(original.status, original.detail, **original.diagnostic)
        if self.approach_repair and original.diagnostic.get("geometry") == "actual":
            raise G2Error(original.status, original.detail, **original.diagnostic)
        self._free()
        check = deadline(remaining())
        q, g, frames, world = self._start(snapshot)
        # Start from the FIRST failed requested waypoint. Do not silently skip
        # earlier unexecuted goals of a multi-action prefix.
        requests = original.diagnostic.get("requests", [])
        if not requests:
            raise G2Error(
                Status.COLLISION,
                "action rejection has no reconstructable requested endpoint",
            )
        failed_row = requests[-1].get("row")
        if type(failed_row) is not int or not 0 <= failed_row < window:
            raise G2Error(Status.INVALID, "invalid failed prefix row")
        if failed_row > 0:
            prefix = self.adapter.actions(chunk, snapshot, failed_row, remaining())
            if not prefix.ok:
                raise G2Error(prefix.status, prefix.detail, **prefix.diagnostic)
            approved = prefix.payload
            source_barrier = None
            if self.approach_repair:
                from g2_bounded_approach import checked_prefix

                approved, source_barrier = checked_prefix(
                    self.adapter, approved, snapshot, check
                )
                if approved is None:
                    return self._upward_plan(
                        chunk,
                        snapshot,
                        remaining,
                        log,
                        source_barrier,
                        source_check=True,
                    )
            self.recoveries += 1
            receipt = dict(
                kind="safe_prefix_before_rejection",
                approved_actions=len(approved.waypoint_ticks),
                source_barrier=source_barrier,
                failed_row=failed_row,
                requires_new_observation=True,
                attempt=self.recoveries,
            )
            log(receipt)
            return approved, dict(repaired=True, receipt=receipt)
        request = requests[0]
        if self.approach_repair:
            from g2_bounded_approach import finger_start_blocked

            if finger_start_blocked(original.diagnostic):
                return self._upward_plan(
                    chunk,
                    snapshot,
                    remaining,
                    log,
                    original.diagnostic,
                    source_check=True,
                )
        result = bounded_progress(
            self.adapter.arx,
            frames,
            np.asarray(request["task_tools"]),
            q,
            g,
            np.asarray(request["fingers"]),
            world,
            self.adapter.motion,
            check,
        )
        path = result.pop("path")
        self.recoveries += 1
        receipt = dict(
            kind="bounded_policy_progress",
            attempt=self.recoveries,
            policy_actions_in_prefix=1,
            requires_new_observation=True,
            **result,
        )
        log(receipt)
        return path, dict(repaired=True, receipt=receipt)

    def _upward_plan(self, chunk, snapshot, remaining, log, rejection, *, source_check):
        from g2_bounded_approach import upward_endpoint, source_admissible

        self._free()
        q, g, frames, world = self._start(snapshot)
        check = deadline(remaining())
        if not world.measured_clear(q, g):
            raise G2Error(
                Status.COLLISION,
                "measured bounded-approach start is blocked",
                geometry="actual",
                collisions=world.last_collisions,
            )
        row = np.asarray(chunk)[0].reshape(2, 7)
        tools = np.array(
            [
                f @ a.chain.fk(v[:6]) @ a.offset()
                for f, a, v in zip(self.adapter.source_frames, self.adapter.aloha, row)
            ]
        )
        requested = np.array(
            [
                a.grip.target(float(np.clip(v[6], 0, 1)))
                for a, v in zip(self.adapter.arx, row)
            ]
        )
        goal, receipt = upward_endpoint(
            self.adapter.arx, frames, tools, q, requested, world, check
        )
        path = self.adapter.motion.plan(q, g, [(goal, requested)], world, check)
        if source_check:
            predicted = (
                path.predicted_fingers[-1]
                if path.predicted_fingers is not None
                else path.fingers[-1]
            )
            receipt["source_admission"] = source_admissible(
                self.adapter,
                goal,
                predicted,
                frames,
                (len(path.q) - 1) * path.period_s,
                check,
            )
        self.recoveries += 1
        receipt.update(
            attempt=self.recoveries,
            original_rejection=rejection,
            policy_actions_in_prefix=1,
            requires_new_observation=True,
        )
        log(receipt)
        return path, dict(repaired=True, receipt=receipt)

    def cycle(self, policy, prompt, window, remaining, log):
        """One query/execution; runner owns episode persistence, hold and close."""
        snapshot, _ = self.backend.capture()
        snapshot, request = self.mapped_observation(snapshot, prompt, remaining, log)
        from pace_runtime import validate_request, validate_response

        validate_request(request)
        log(dict(kind="policy_request", snapshot=snapshot, request=request))
        response = getattr(policy, "infer_raw", policy.infer)(request)
        log(dict(kind="policy_response", response=response))
        chunk = validate_response(response)
        log(dict(kind="policy_validated"))
        plan, receipt = self.action_plan(chunk, snapshot, window, remaining, log)
        log(dict(kind="approved_plan", trajectory=asdict(plan), receipt=receipt))
        before = getattr(getattr(self.backend, "env", None), "take_action_cnt", None)
        try:
            done = self.backend.execute(plan, log, remaining)
        except G2Error as error:
            after = getattr(getattr(self.backend, "env", None), "take_action_cnt", None)
            completed = (
                after - before if type(before) is int and type(after) is int else None
            )
            if completed is not None and 0 <= completed <= len(plan.waypoint_ticks):
                self.policy_actions += completed
            error.diagnostic.update(
                completed_actions_in_prefix=completed,
                partial_physical_motion_possible=True,
            )
            raise
        if type(done) is not int or not 0 <= done <= len(plan.waypoint_ticks):
            raise G2Error(Status.BACKEND_ERROR, "invalid policy prefix receipt")
        reason = (
            self.backend.take_replan()
            if callable(getattr(self.backend, "take_replan", None))
            else None
        )
        if (
            done < len(plan.waypoint_ticks)
            and not self.backend.success()
            and reason is None
        ):
            raise G2Error(
                Status.BACKEND_ERROR,
                "partial execution has no approved contact transition",
            )
        if done == 0 and not self.backend.success():
            raise G2Error(Status.BACKEND_ERROR, "no policy action progress")
        if reason is not None:
            log(dict(kind="measured_replan", reason=reason, executed_actions=done))
        self.policy_actions += done
        return dict(
            snapshot=snapshot,
            request=request,
            chunk=chunk,
            plan=plan,
            receipt=receipt,
            policy_actions=done,
            total_policy_actions=self.policy_actions,
            recovery_actions=self.recovery_actions,
        )
