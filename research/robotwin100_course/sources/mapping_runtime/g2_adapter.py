"""G2 bimanual TCP retargeting. No simulator, CUDA or model imports."""
from dataclasses import dataclass
from itertools import product
from time import monotonic

import numpy as np

from g2_kinematics import BACKEND, pose_error
from robotwin_control_geometry import Status, checked_pose, finite_array, inverse_pose, select_joints

SIDES = ("left", "right")
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class G2Error(RuntimeError):
    def __init__(self, status, detail, **diagnostic):
        super().__init__(detail)
        self.status = status
        self.diagnostic = diagnostic


def deadline(seconds):
    end = monotonic() + positive(seconds)

    def check():
        if monotonic() >= end:
            raise G2Error(Status.TIMEOUT, "adapter deadline exceeded")

    return check


def positive(value):
    x = finite_array(value, ()).item()
    if x <= 0:
        raise ValueError("expected positive finite scalar")
    return x


def bounds(q, limits, **diagnostic):
    if np.any((q < limits[:, 0]) | (q > limits[:, 1])):
        raise G2Error(Status.LIMIT, "arm joint outside limits", q=q.tolist(), limits=limits.tolist(), **diagnostic)


@dataclass(frozen=True)
class Stamp:
    episode: str
    tick: int
    sim_s: float


@dataclass(frozen=True)
class ArmSample:
    stamp: Stamp
    instance: str
    names: tuple
    qpos: np.ndarray
    world_base: np.ndarray


@dataclass(frozen=True)
class Snapshot:
    stamp: Stamp
    arms: tuple
    images: dict   # camera name -> (Stamp, uint8 CHW)
    world_task: np.ndarray
    objects_task: dict = None  # absent when a backend has no task actors


@dataclass(frozen=True)
class Grip:
    names: tuple
    closed: np.ndarray    # native positions of both fingers
    opened: np.ndarray
    limits: np.ndarray
    source: str
    tolerance: float = 0.001  # native coordinate tolerance, m
    state_mode: str = "strict_mimic"
    state_source: str = ""

    def __post_init__(self):
        lo, hi = finite_array(self.limits, (2, 2)).T
        closed, opened = finite_array(self.closed, (2,)), finite_array(self.opened, (2,))
        if (not self.source or len(self.names) != 2 or len(set(self.names)) != 2
                or np.any(lo >= hi) or np.any(np.minimum(closed, opened) < lo)
                or np.any(np.maximum(closed, opened) > hi) or np.any(closed == opened)):
            raise ValueError("invalid finger calibration or limit interval")
        positive(self.tolerance)
        if self.state_mode not in ("strict_mimic","independent_aperture_v1"):
            raise ValueError("unknown gripper state mapping")
        if self.state_mode=="independent_aperture_v1" and (not self.state_source or np.any(opened<=closed)):
            raise ValueError("independent aperture needs audited outward axes and positive travel")

    def read(self, native):
        return self.measure(native)["normalized"]

    def measure(self,native):
        native = finite_array(native, (2,))
        lo, hi = np.asarray(self.limits).T
        if np.any((native < lo - self.tolerance) | (native > hi + self.tolerance)):
            raise G2Error(Status.LIMIT, "measured finger outside limit", native=native.tolist())
        opened, closed = np.asarray(self.opened), np.asarray(self.closed)
        if np.any((native<closed-self.tolerance)|(native>opened+self.tolerance)) and self.state_mode=="independent_aperture_v1":
            raise G2Error(Status.LIMIT,"measured independent finger outside calibrated travel",native=native.tolist())
        fractions = (native - closed) / (opened - closed)
        if self.state_mode=="strict_mimic" and abs(fractions[0] - fractions[1]) > self.tolerance / max(abs(opened - closed)):
            raise G2Error(Status.INVALID, "finger mimic mismatch", native=native.tolist())
        raw=(float(np.sum(native-closed)/np.sum(opened-closed))
             if self.state_mode=="independent_aperture_v1" else float(fractions.mean()))
        normal=float(np.clip(raw,0,1))
        if self.state_mode=="strict_mimic" and np.any(np.abs(self.target(normal) - native) > self.tolerance):
            raise G2Error(Status.LIMIT, "measured grip outside calibrated travel", native=native.tolist())
        residual=native-self.target(normal)
        return dict(normalized=normal,raw_normalized=raw,state_mode=self.state_mode,
                    state_source=self.state_source,native_m=native.tolist(),
                    equivalent_symmetric_native_m=self.target(normal).tolist(),
                    reduction_residual_m=residual.tolist(),
                    aperture_increase_m=float(np.sum(native-closed)),
                    center_shift_from_symmetric_m=float((residual[0]-residual[1])/2),
                    scalar_clipped=raw!=normal)

    def target(self, normal):
        return np.asarray(self.closed) + normal * (np.asarray(self.opened) - self.closed)


@dataclass(frozen=True)
class Arm:
    chain: object
    instance: str
    child_joint: np.ndarray
    joint_tcp: np.ndarray
    tcp_tool: np.ndarray
    grip: Grip
    calibration_id: str

    def offset(self):
        if not self.calibration_id or self.tcp_tool is None or self.grip is None:
            raise G2Error(Status.UNCALIBRATED, "missing TCP or finger calibration")
        return checked_pose(self.child_joint) @ checked_pose(self.joint_tcp) @ checked_pose(self.tcp_tool)


@dataclass(frozen=True)
class Result:
    status: Status
    detail: str
    diagnostic: dict
    payload: object = None

    @property
    def ok(self):
        return self.status == Status.OK


def guarded(phase, stamp, work):
    t0 = monotonic()
    log = {"phase": phase, "stamp": vars(stamp), "ik_backend": BACKEND}
    try:
        payload = work(log)
        status, detail = Status.OK, ""
    except G2Error as error:
        status, detail, payload = error.status, str(error), None
        log.update(error.diagnostic)
    except (ValueError, TypeError, KeyError) as error:
        status, detail, payload = Status.INVALID, str(error), None
    except Exception as error:
        status, detail, payload = Status.BACKEND_ERROR, repr(error), None
    log["wall_s"] = monotonic() - t0
    return Result(status, detail, log, payload)


def read_snapshot(snapshot, arms):
    if len(snapshot.arms) != 2 or len(arms) != 2:
        raise ValueError("expected two arms in left/right order")
    stamp = snapshot.stamp
    if not stamp.episode or type(stamp.tick) is not int or stamp.tick < 0 or not np.isfinite(stamp.sim_s) or stamp.sim_s < 0:
        raise ValueError("invalid snapshot stamp")
    task_world = inverse_pose(snapshot.world_task)
    q, fingers, frames, tools = [], [], [], []
    for sample, arm in zip(snapshot.arms, arms):
        if sample.stamp != stamp:
            raise G2Error(Status.UNSYNCED, "robot states do not share a physics tick")
        if sample.instance != arm.instance:
            raise ValueError("unexpected articulation instance")
        offset = arm.offset()
        q_i = select_joints(sample.names, sample.qpos, arm.chain.names)
        bounds(q_i, arm.chain.limits, geometry="actual")
        native = select_joints(sample.names, sample.qpos, arm.grip.names)
        fingers.append(native)
        frames.append(task_world @ checked_pose(sample.world_base))
        q.append(q_i)
        tools.append(frames[-1] @ arm.chain.fk(q_i) @ offset)
    return np.array(q), np.array(fingers), frames, np.array(tools)


def images_of(snapshot):
    if set(snapshot.images) != set(CAMERAS):
        raise ValueError("expected three named PACE cameras")
    out = {}
    for name in CAMERAS:
        stamp, image = snapshot.images[name]
        if stamp != snapshot.stamp:
            raise G2Error(Status.UNSYNCED, f"{name} has a stale stamp")
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3
                or not all(0 < x <= 1024 for x in image.shape[1:])):
            raise ValueError(f"invalid RGB camera {name}")
        out[name] = np.ascontiguousarray(image).copy()
    return out


def tool_ik(arm, frame, tool, seed, check):
    """Verify the task tool, budgeting link error and the offset's angular lever arm."""
    offset = arm.offset()
    lever = float(np.linalg.norm(offset[:3, 3]))
    angle_tol = min(np.deg2rad(.05), 2 * np.arcsin(min(1., .00025 / (2 * max(lever, 1e-12)))))
    link = inverse_pose(frame) @ checked_pose(tool) @ inverse_pose(offset)
    stream = arm.chain.ik(link, seed, check, pos_tol=.00025, ang_tol=angle_tol)
    while True:
        try:
            q, terminal = next(stream)
        except StopIteration as end:
            return end.value
        error = pose_error(tool, frame @ arm.chain.fk(q) @ offset)
        position = float(np.linalg.norm(error[:3]))
        angle = float(np.rad2deg(np.linalg.norm(error[3:])))
        if position <= .0005 and angle <= .05:
            yield q, dict(terminal, position_m=position, angle_deg=angle, terminal_link_residual=terminal)


def solve_pair(arms, frames, tools, seed, fingers, collision, check, max_delta=None):
    def accepted(q):
        predict=getattr(collision,"ik_clear",None)
        return predict(q,fingers,check) if callable(predict) else clear(collision,q,fingers)
    streams, choices = [], []
    for side, arm, frame, tool, q0 in zip(SIDES, arms, frames, tools, seed):
        solutions = tool_ik(arm, frame, tool, q0, check)
        try:
            first = next(solutions)
        except StopIteration as end:
            raise G2Error(Status.IK_FAILED, f"{side} has no accepted IK solution", side=side, best=end.value)
        streams.append(solutions)
        choices.append([first])
    q = np.array([items[0][0] for items in choices])
    if (max_delta is None or np.all(abs(q - seed) <= max_delta + 1e-9)) and accepted(q):
        check()
        return q, [items[0][1] for items in choices]
    for items, stream in zip(choices, streams):
        items.extend(stream)
    limit_rejected = 0
    collision_rejected = 0
    collision_examples = []
    candidate_verdicts = []
    ranked = sorted(product(*choices), key=lambda pair: np.linalg.norm(np.array([x[0] for x in pair]) - seed))
    for pair in ranked:
        check()
        q = np.array([x[0] for x in pair])
        if max_delta is not None and np.any(abs(q - seed) > max_delta + 1e-9):
            limit_rejected += 1
            candidate_verdicts.append(dict(candidate_q=q.tolist(), reason="continuity_limit"))
            continue
        if accepted(q):
            return q, [x[1] for x in pair]
        collision_rejected += 1
        candidate_verdicts.append(dict(candidate_q=q.tolist(), reason="collision",
                                      collisions=getattr(collision,"last_collisions",None)))
        if len(collision_examples) < 8:
            collision_examples.append({"candidate_q":q.tolist(),
                                       "collisions":getattr(collision,"last_collisions",None)})
    if collision_rejected:
        raise G2Error(Status.COLLISION, "all temporally admissible IK pairs collide",
                      rejected_pairs=collision_rejected, collision_examples=collision_examples,
                      candidate_verdicts=candidate_verdicts, continuity_rejected=limit_rejected,
                      ik_solutions_per_side=[len(x) for x in choices])
    raise G2Error(Status.VELOCITY, "IK branch change exceeds continuity limits", rejected_pairs=limit_rejected, candidate_verdicts=candidate_verdicts,
                  ik_solutions_per_side=[len(x) for x in choices])


def clear(collision, q, fingers):
    value = collision(q.copy(), fingers.copy())
    if not isinstance(value, (bool, np.bool_)):
        raise G2Error(Status.BACKEND_ERROR, "collision checker did not return a verdict")
    return value


class Adapter:
    def __init__(self, aloha, arx, source_world_base, source_world_task, source_seed,
                 source_speed, source_collision, target_collision, motion,
                 grip_rule="reject", grip_max_clip=0.0):
        self.aloha, self.arx = tuple(aloha), tuple(arx)
        if len(self.aloha) != 2 or len(self.arx) != 2:
            raise ValueError("expected exactly two arms per embodiment")
        self.source_frames = [inverse_pose(source_world_task) @ checked_pose(base) for base in source_world_base]
        if len(self.source_frames) != 2:
            raise ValueError("missing source base calibration")
        self.seed = finite_array(source_seed, (2, 6)).copy()
        for arm, q in zip(self.aloha, self.seed):
            bounds(q, arm.chain.limits)
        self.source_speed = finite_array(source_speed, (2, 6))
        if np.any(self.source_speed <= 0) or not callable(source_collision) or not callable(target_collision):
            raise ValueError("missing source velocity or collision checks")
        self.source_collision, self.target_collision = source_collision, target_collision
        self.motion = motion
        if (grip_rule not in ("clip", "reject") or not np.isfinite(grip_max_clip)
                or not 0 <= grip_max_clip <= 1):
            raise ValueError("invalid normalized gripper policy")
        self.grip_rule, self.grip_max_clip = grip_rule, grip_max_clip
        self.last_stamp = None

    def observe(self, snapshot, prompt, budget_s):
        def work(log):
            check = deadline(budget_s)
            log["calibration"] = [a.calibration_id for a in self.aloha + self.arx]
            _, native, _, tools = read_snapshot(snapshot, self.arx)
            rgb = images_of(snapshot)
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2048:
                raise ValueError("invalid task prompt")
            stamp = snapshot.stamp
            delta = None
            if self.last_stamp is not None:
                dt = stamp.sim_s - self.last_stamp.sim_s
                if stamp.episode != self.last_stamp.episode or stamp.tick <= self.last_stamp.tick or dt <= 0:
                    raise G2Error(Status.UNSYNCED, "non-increasing time or cross-episode seed reuse")
                delta = self.source_speed * dt
            grip_state=[arm.grip.measure(value) for arm,value in zip(self.arx,native)]
            grip=[state["normalized"] for state in grip_state]
            source_fingers = np.array([arm.grip.target(g) for arm, g in zip(self.aloha, grip)])
            if callable(getattr(self.source_collision,"set_start",None)):
                self.source_collision.set_start(source_fingers)
            log.update(collision_world="source_collision", arx_tools_task=tools.tolist(),
                       ik_seed=self.seed.tolist(), source_fingers=source_fingers.tolist(),
                       measured_target_fingers=native.tolist(),
                       continuity_delta=None if delta is None else delta.tolist())
            q, residual = solve_pair(self.aloha, self.source_frames, tools, self.seed,
                                     source_fingers, self.source_collision, check, delta)
            log.update(ik=residual, arx_tools_task=tools.tolist(), grip_fraction=grip,grip_measurement=grip_state)
            check()
            self.seed, self.last_stamp = q, stamp
            return {"state": np.column_stack((q, grip)).astype(np.float32).ravel(), "images": rgb, "prompt": prompt}

        return guarded("observation", snapshot.stamp, work)

    def begin_expert(self, snapshot, budget_s):
        """Validate an expert control start without requiring a policy proxy IK."""
        def work(log):
            check = deadline(budget_s)
            q, native, _, _ = read_snapshot(snapshot, self.arx)
            images_of(snapshot)
            log["grip_measurement"]=[arm.grip.measure(value) for arm,value in zip(self.arx,native)]
            if self.last_stamp is not None:
                if (snapshot.stamp.episode != self.last_stamp.episode
                        or snapshot.stamp.tick <= self.last_stamp.tick
                        or snapshot.stamp.sim_s <= self.last_stamp.sim_s):
                    raise G2Error(Status.UNSYNCED, "expert observations did not advance")
            check()
            self.last_stamp = snapshot.stamp
            log["proxy_state_required"] = False
            return {"actual_arm": q}
        return guarded("expert_observation", snapshot.stamp, work)

    def actions(self, chunk, snapshot, window, budget_s):
        def work(log):
            check = deadline(budget_s)
            if snapshot.stamp != self.last_stamp:
                raise G2Error(Status.UNSYNCED, "actions must start from the mapped observation")
            chunk_array = finite_array(chunk, (50, 14))
            if type(window) is not int or not 1 <= window <= 4:
                raise ValueError("invalid action prefix")
            q, fingers, frames, _ = read_snapshot(snapshot, self.arx)
            for arm, value in zip(self.arx, fingers):
                bounds(value, np.asarray(arm.grip.limits), geometry="actual")
            log.update(collision_world="target_collision", measured_start_q=q.tolist(),
                       measured_start_fingers=fingers.tolist())
            grip_log = log["grippers"] = []
            requests = log["requests"] = []
            targets = []
            seed = q
            for i, row in enumerate(chunk_array[:window]):
                tools, native = [], []
                for side, arm, target_arm, frame, values in zip(
                        SIDES, self.aloha, self.arx, self.source_frames, row.reshape(2, 7)):
                    raw = float(values[6])
                    g = float(np.clip(raw, 0, 1))
                    entry = dict(row=i, side=side, raw=raw, rule=self.grip_rule, normalized=None, native=None)
                    grip_log.append(entry)
                    if raw != g and (self.grip_rule == "reject" or abs(raw - g) > self.grip_max_clip):
                        raise G2Error(Status.LIMIT, "normalized grip outside allowed envelope", side=side, row=i,
                                      geometry="candidate")
                    goal_fingers = target_arm.grip.target(g)
                    entry.update(normalized=g, native=goal_fingers.tolist())
                    bounds(values[:6], arm.chain.limits, geometry="candidate")
                    tools.append(frame @ arm.chain.fk(values[:6]) @ arm.offset())
                    native.append(goal_fingers)
                request = dict(row=i, task_tools=np.asarray(tools).tolist(),
                               ik_start=seed.tolist(), fingers=np.asarray(native).tolist())
                requests.append(request)
                prepare=getattr(self.target_collision,"prepare_tools",None)
                if callable(prepare):
                    current_tools=[frame@arm.chain.fk(values)@arm.offset()
                                   for frame,arm,values in zip(frames,self.arx,seed)]
                    tools,phase=prepare(tools,np.asarray(native),current_tools)
                    if phase:
                        request.update(contact_phase=phase,execution_tools=np.asarray(tools).tolist())
                goal, residual = solve_pair(self.arx, frames, tools, seed, np.array(native),
                                             self.target_collision, check)
                request.update(ik_target=goal.tolist(), residual=residual)
                targets.append((goal, np.array(native)))
                seed = goal
                barrier=getattr(self.target_collision,"action_barrier",None)
                if callable(barrier) and barrier(goal,np.array(native),check):
                    log["contact_barrier"]=dict(row=i,requested_window=window,
                                               planned_actions=len(targets),reason="contact_or_release_requires_observation")
                    break
            log["motion"] = {"method": self.motion.method, "limits_source": self.motion.limits_source,
                             "collision_id": self.motion.collision_id, "control_period_s": self.motion.period}
            return self.motion.plan(q, fingers, targets, self.target_collision, check)

        return guarded("action", snapshot.stamp, work)
