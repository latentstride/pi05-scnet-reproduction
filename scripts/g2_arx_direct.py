"""ARX policy execution using only measured target-side motion and contact gates."""

from dataclasses import asdict, dataclass

import numpy as np

from g2_adapter import Adapter, G2Error, Stamp, bounds, deadline, guarded, read_snapshot
from robotwin_control_geometry import Status, finite_array

from g2_arx_direct_profile import PREFIX_CAP


def kinematic_pair(_q, _fingers):
    """The Aloha coordinates represent policy state, not a simulated robot."""
    return True


class UnexecutedContactProposal:
    """Defer only movable-block/finger prediction in an unexecuted proposal.

    Other robot/environment collisions stay checked. The distinct profile makes
    this proposal invalid for ARXBackend.execute; a returned segment is always
    replanned against the original world before it can be executed.
    """
    def __init__(self,world):
        self.world=world
        self.profile_id=world.profile_id+':unexecuted_contact_proposal'

    def __getattr__(self,name):return getattr(self.world,name)

    def allows(self,row):
        a=(row['object_name1'],row['link_name1'])
        b=(row['object_name2'],row['link_name2'])
        if row.get('max_penetration_m') is not None and (
            (a[0]==self.world.block_name and b in self.world.allowed_pairs)
            or (b[0]==self.world.block_name and a in self.world.allowed_pairs)):
            return True
        return self.world.allows(row)

    def __call__(self,q,fingers):
        return self.world.base.evaluate(q,fingers,contact=self)

    def planning_geometry(self,q,requested,previous,check):
        check()
        return requested.copy()


class DirectAdapter(Adapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, source_collision=kinematic_pair, **kwargs)

    def policy_row(self, row, snapshot, remaining):
        # The row belongs to the last queried block. Its physical start is
        # refreshed independently, without committing a new source IK seed.
        original = self._actions(
            np.broadcast_to(row, (50, 14)),
            snapshot,
            1,
            remaining(),
            require_observation=False,
        )
        world=self.target_collision
        if (original.ok or original.status!=Status.COLLISION
                or not hasattr(world,'tracker') or not hasattr(world,'base')):
            return original
        contact=world.tracker.snapshot()
        if contact.get('phase')!='approach' or contact['holders'] or contact['releasing']:
            return original
        rejected=[c for verdict in original.diagnostic.get('candidate_verdicts',[])
                  for c in (verdict.get('collisions') or [])]
        rejected+=world.last_collisions
        if not any(c.get('reason')=='predicted finger start intersects task block' for c in rejected):
            return original
        return guarded('action',snapshot.stamp,
            lambda log:self._contact_segment(original,snapshot,remaining,log))

    def _contact_segment(self,original,snapshot,remaining,log):
        check=deadline(remaining())
        world=self.target_collision
        q,g,_,_=read_snapshot(snapshot,self.arx)
        request=original.diagnostic['requests'][-1]
        fingers=finite_array(request['fingers'],(2,2))
        candidates=([request['ik_target']] if 'ik_target' in request else
                    [v['candidate_q'] for v in original.diagnostic.get('candidate_verdicts',[])])
        future=UnexecutedContactProposal(world)
        goal=None
        for candidate in candidates:
            check()
            candidate=finite_array(candidate,(2,6))
            if future(candidate,fingers):
                goal=candidate
                break
        if goal is None:
            raise G2Error(Status.COLLISION,'no non-finger-safe original IK branch',
                          geometry='candidate',collisions=world.last_collisions)
        # No new IK search, TCP offset, finger target or joint path: reuse the
        # original residual-checked candidates and original quintic interpolation.
        nominal=self.motion.plan(q,g,[(goal,fingers)],future,check)
        geometry=g.copy();last=None
        for index in range(1,len(nominal.q)):
            check()
            try:
                geometry=world.planning_geometry(nominal.q[index],nominal.fingers[index],geometry,check)
                if not world(nominal.q[index],geometry):break
            except G2Error as error:
                if error.status!=Status.COLLISION:raise
                break
            last=index
        if last is None:
            raise G2Error(Status.COLLISION,'no safe progress on original contact segment',
                          geometry='candidate',collisions=world.last_collisions)
        plan=self.motion.plan(q,g,[(nominal.q[last],nominal.fingers[last])],world,check)
        log.update(collision_world='target_collision',contact_motion='measured_safe_prefix_v1',
            original_rejection=original.detail,original_request=request,
            original_ik_target=goal.tolist(),segment_ik_target=plan.q[-1].tolist(),
            partial_policy_row=last<len(nominal.q)-1,
            row_duration_limit_s=self.motion.max_ticks*self.motion.period,
            proposal_contact_profile=future.profile_id,executed_contact_profile=plan.contact_profile_id)
        return plan

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
    observed_sides = (set(contact_at_start['holders']) | set(contact_at_start['releasing'])
                      | {side for side, fingers in contact_at_start['contact_fingers'].items() if fingers})
    for i, row in enumerate(rows):
        row_start = backend.tick
        segment = 0
        row_limit = None
        while True:
            if i or segment:
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
                    row_in_prefix=i + 1, segment_in_row=segment + 1,
                    counts_as_policy_action=segment == 0,
                )
            )
            # A query made after measured contact already observes that phase.
            # Keep checking every row/physics step, but do not repeatedly retire
            # its closure tail for the same contact context. Holder transitions
            # and all actual safety checks remain active inside the backend.
            row_limit = result.diagnostic.get('row_duration_limit_s', row_limit)
            if row_limit is not None and hasattr(plan, 'q'):
                duration = (len(plan.q)-1)*plan.period_s
                if (backend.tick-row_start)*backend.dt+duration > row_limit+1e-12:
                    raise G2Error(Status.PLAN_FAILED, 'original row motion duration exhausted', geometry='candidate')
            done = backend.execute(plan, log, remaining, stop_on_contact=not in_contact)
            if type(done) is not int or done != 1:
                raise G2Error(Status.BACKEND_ERROR, "invalid direct ARX row receipt")
            if segment == 0:
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
            if (set(contact['holders']) != set(contact_at_start['holders'])
                    or set(contact['releasing']) != set(contact_at_start['releasing'])):
                return Prefix(i + 1, stamp, "measured_contact_transition")
            touched_sides = {side for side, fingers in contact['contact_fingers'].items() if fingers}
            if touched_sides - observed_sides:
                return Prefix(i + 1, stamp, "contact_requires_observation")
            if not result.diagnostic.get('partial_policy_row'):
                break
            if row_limit is None or (backend.tick-row_start)*backend.dt >= row_limit-1e-12:
                raise G2Error(Status.PLAN_FAILED, 'original row motion duration exhausted', geometry='candidate')
            segment += 1
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
