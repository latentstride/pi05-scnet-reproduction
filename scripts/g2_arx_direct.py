"""ARX policy execution using only measured target-side motion and contact gates."""

from dataclasses import asdict, dataclass

import numpy as np

from g2_adapter import Adapter, G2Error, Stamp, bounds, deadline, guarded, read_snapshot
from robotwin_control_geometry import Status, finite_array

from g2_arx_direct_profile import PREFIX_CAP


def kinematic_pair(_q, _fingers):
    """The Aloha coordinates represent policy state, not a simulated robot."""
    return True


class DirectAdapter(Adapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, source_collision=kinematic_pair, **kwargs)

    def policy_row(self, row, snapshot, remaining):
        # The row belongs to the last queried block. Its physical start is
        # refreshed independently, without committing a new source IK seed.
        return self._actions(
            np.broadcast_to(row, (50, 14)),
            snapshot,
            1,
            remaining(),
            require_observation=False,
        )

    def expert_row(self, row, snapshot, remaining):
        """Native ARX expert input, isolated from the policy exchange."""

        def plan(log):
            q, fingers, _, _ = read_snapshot(snapshot, self.arx)
            row_array = finite_array(row, (14,)).reshape(2, 7)
            target = row_array[:, :6]
            native = []
            for arm, goal in zip(self.arx, row_array):
                bounds(goal[:6], arm.chain.limits, geometry="candidate")
                if not 0 <= goal[6] <= 1:
                    raise G2Error(
                        Status.LIMIT,
                        "invalid expert gripper value",
                        geometry="candidate",
                    )
                native.append(arm.grip.target(goal[6]))
            world = self.target_collision
            native = np.asarray(native)
            current_tools = world.tools(q)
            _, phase = world.prepare_tools(world.tools(target), native, current_tools)
            if phase:
                target = q
                log["contact_phase"] = phase
            barrier = world.action_barrier(target, native, deadline(remaining()))
            if barrier:
                log["contact_barrier"] = True
            return self.motion.plan(
                q, fingers, [(target, native)], world, deadline(remaining())
            )

        return guarded("expert_plan", snapshot.stamp, plan)


@dataclass(frozen=True)
class Prefix:
    completed: int
    stamp: Stamp
    reason: str


def execute_prefix(backend, snapshot, rows, plan_row, remaining, log, consumed):
    """Plan one row at a time; invalidate the tail when measured contact changes."""
    if not 1 <= len(rows) <= PREFIX_CAP:
        raise ValueError("invalid direct ARX prefix")
    world = backend.collision
    contact_at_start = world.tracker.snapshot()
    in_contact = bool(
        contact_at_start["holders"]
        or contact_at_start["releasing"]
        or any(contact_at_start["contact_fingers"].values())
    )
    for i, row in enumerate(rows):
        if i:
            snapshot = backend.capture_state()
        world.refresh()
        log(dict(kind="direct_row_planning", row_in_prefix=i + 1))
        q, fingers, _, _ = read_snapshot(snapshot, backend.arms)
        if not world.measured_clear(q, fingers):
            raise G2Error(
                Status.COLLISION,
                "measured ARX state blocks the next row",
                geometry="actual",
                collisions=world.last_collisions,
            )
        result = plan_row(row, snapshot, remaining)
        if not result.ok:
            raise G2Error(
                result.status,
                result.detail,
                **dict({"geometry": "candidate"}, **result.diagnostic)
            )
        plan = result.payload
        if len(plan.waypoint_ticks) != 1:
            raise G2Error(Status.BACKEND_ERROR, "one checked waypoint required")
        log(
            dict(
                kind="approved_plan",
                trajectory=asdict(plan),
                receipt=result.diagnostic,
                row_in_prefix=i + 1,
            )
        )
        done = backend.execute(plan, log, remaining, stop_on_contact=True)
        if type(done) is not int or done != 1:
            raise G2Error(Status.BACKEND_ERROR, "invalid direct ARX row receipt")
        consumed()
        reason = backend.take_replan()
        stamp = Stamp(snapshot.stamp.episode, backend.tick, backend.tick * backend.dt)
        if stamp.tick <= snapshot.stamp.tick:
            raise G2Error(Status.UNSYNCED, "ARX execution did not advance")
        if reason:
            return Prefix(i + 1, stamp, reason["reason"])
        if backend.success():
            return Prefix(i + 1, stamp, "task_success")
        contact = world.tracker.snapshot()
        if (
            in_contact
            or result.diagnostic.get("contact_barrier")
            or any(contact["contact_fingers"].values())
        ):
            return Prefix(i + 1, stamp, "contact_requires_observation")
    return Prefix(len(rows), stamp, "prefix_complete")


class DirectController:
    def __init__(self, adapter, backend):
        self.adapter, self.backend = adapter, backend
        self.policy_actions = self.recovery_actions = 0
        self.expected_stamp = None

    def cycle(self, policy, prompt, window, remaining, log):
        from pace_runtime import validate_request, validate_response

        if type(window) is not int or not 1 <= window <= PREFIX_CAP:
            raise ValueError("invalid ARX policy prefix")
        snapshot, _ = self.backend.capture()
        if self.expected_stamp is not None and snapshot.stamp != self.expected_stamp:
            raise G2Error(
                Status.UNSYNCED, "observation does not match the executed prefix"
            )
        if not {"block", "target"} <= (snapshot.objects_task or {}).keys():
            raise G2Error(Status.INVALID, "missing measured block or target pose")
        observed = self.adapter.observe(snapshot, prompt, remaining())
        if not observed.ok:
            raise G2Error(observed.status, observed.detail, **observed.diagnostic)
        request = observed.payload
        validate_request(request)
        log(dict(kind="policy_request", snapshot=snapshot, request=request))
        response = getattr(policy, "infer_raw", policy.infer)(request)
        log(dict(kind="policy_response", response=response))
        remaining()
        chunk = validate_response(response)
        log(dict(kind="policy_validated"))

        def consumed():
            self.policy_actions += 1

        prefix = execute_prefix(
            self.backend,
            snapshot,
            chunk[:window],
            self.adapter.policy_row,
            remaining,
            log,
            consumed,
        )
        self.expected_stamp = prefix.stamp
        log(
            dict(
                kind="direct_prefix_complete",
                executed_actions=prefix.completed,
                stamp=asdict(prefix.stamp),
                discarded_rows=len(chunk) - prefix.completed,
                reason=prefix.reason,
            )
        )
