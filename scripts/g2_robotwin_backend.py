"""Audited ARX joint-control bridge. Real SAPIEN dynamics require later GPU acceptance."""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np

from g2_adapter import G2Error, SIDES, Stamp, Snapshot, read_snapshot
from g2_capture import capture_snapshot, capture_arms
from g2_contact import contact_failure_kind, finger_force_limits
from g2_finger_force import MODE as CENTERED_FORCE,centered_finger_force
from g2_kinematics import pose_error, rotation_vector
from g2_motion import Trajectory
from robotwin_control_geometry import Status, checked_pose, finite_array, pose_from_wxyz, inverse_pose


def pose7(matrix):
    matrix = checked_pose(matrix)
    axis_angle = rotation_vector(matrix[:3, :3])
    angle = np.linalg.norm(axis_angle)
    xyz = axis_angle * (np.sin(angle / 2) / angle) if angle > 1e-12 else axis_angle / 2
    return np.r_[matrix[:3, 3], np.cos(angle / 2), xyz].tolist()


def arx_settings(native, asset_dir, expected_config_sha256, distance_m):
    import yaml
    asset_dir = Path(asset_dir)
    raw = (asset_dir / "config.yml").read_bytes()
    if sha256(raw).hexdigest() != expected_config_sha256:
        raise ValueError("ARX embodiment configuration drift")
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("invalid ARX base spacing")
    config = yaml.safe_load(raw)
    if config["dual_arm"]:
        raise ValueError("expected two single-arm ARX instances")
    settings = deepcopy(native)
    settings.update(embodiment=["ARX-X5", "ARX-X5"], dual_arm_embodied=False,
                    embodiment_dis=distance_m, left_robot_file=str(asset_dir),
                    right_robot_file=str(asset_dir), left_embodiment_config=config,
                    right_embodiment_config=deepcopy(config), collect_data=False,
                    save_data=False, render_freq=0, eval_video_log=False)
    return settings


@contextmanager
def joint_space_setup(robot_class, grippers=None):
    """Scope the known unconditional warmup bypass to an explicit ARX setup.

    This installs no collision exemptions and does not alter fixed upstream files.
    ARX commands must use ARXBackend and JointCollisionWorld afterwards.
    """
    original = robot_class.set_planner
    original_gripper = getattr(robot_class, "set_gripper", None)
    class DisabledPlanner:
        def plan_grippers(self, now_val, target_val):
            now_val = finite_array(now_val, ()).item()
            target_val = finite_array(target_val, ()).item()
            if not 0 <= now_val <= 1 or not 0 <= target_val <= 1:
                raise ValueError("invalid bootstrap gripper command")
            return dict(num_step=200, per_step=(target_val-now_val)/200,
                        result=np.linspace(now_val, target_val, 200))
        def __getattr__(self, name):
            raise RuntimeError("CuRobo path disabled in explicit G2 joint-space mode")
    def configure(robot, scene=None):
        if robot.is_dual_arm:
            raise ValueError("joint-space setup override is restricted to two-instance ARX")
        robot.communication_flag = False
        robot.left_planner = robot.right_planner = DisabledPlanner()
        robot.g2_planner_mode = "g2_joint_space_cpu_fcl_v1"
    def calibrated_gripper(robot, value, side, gripper_eps=.1):
        if side not in SIDES:
            raise ValueError("invalid side")
        value = finite_array(value, ()).item()
        if not 0 <= value <= 1:
            raise ValueError("bootstrap normalized gripper outside [0,1]")
        calibration = grippers[SIDES.index(side)]
        expected = calibration.target(value)
        joints = {joint[0].get_name(): joint[0] for joint in getattr(robot, side + "_gripper")}
        for name, native in zip(calibration.names, expected):
            joints[name].set_drive_target(float(native))
            joints[name].set_drive_velocity_target(0.)
        setattr(robot, side + "_gripper_val", value)
    robot_class.set_planner = configure
    if grippers is not None:
        if len(grippers) != 2:
            robot_class.set_planner = original
            raise ValueError("both bootstrap finger calibrations are required")
        robot_class.set_gripper = calibrated_gripper
    try:
        yield
    finally:
        robot_class.set_planner = original
        if grippers is not None:
            if original_gripper is None:
                del robot_class.set_gripper
            else:
                robot_class.set_gripper = original_gripper


class ARXBackend:
    """Own one pre-reset scene and advance both arms exactly once per physics tick.

    Scene creation/reset and contact calibration occur before constructing this
    object. The initial scene is consumed once by reset(), preventing stale reuse.
    """
    def __init__(self, env, arms, collision, world_task, physics_dt_s, initial,
                 tracking_position_limit_m, tracking_angle_limit_deg, video, contact_drive=None):
        self.env, self.arms, self.collision = env, tuple(arms), collision
        if len(self.arms) != 2 or env.robot.left_entity is env.robot.right_entity:
            raise ValueError("expected two actual ARX articulations")
        if not np.isfinite(physics_dt_s) or physics_dt_s <= 0:
            raise ValueError("invalid physics period")
        if abs(env.scene.get_timestep() - physics_dt_s) > 1e-9:
            raise ValueError("physics period differs from scene")
        if not callable(getattr(collision, "refresh", None)):
            raise ValueError("requires a refreshed real collision-world backend")
        self.dt = physics_dt_s
        self.world_task = checked_pose(world_task)
        self.initial = deepcopy(initial)
        self.position_tol = tracking_position_limit_m
        self.angle_tol = tracking_angle_limit_deg
        if not (np.isfinite(self.position_tol) and np.isfinite(self.angle_tol)
                and self.position_tol > 0 and self.angle_tol > 0):
            raise ValueError("invalid tracking acceptance limits")
        self.video, self.tick, self.consumed = video, 0, False
        self.episode = None
        self.last_capture = None
        self.completed = 0
        self.replan_reason=None
        self.contact_drive=deepcopy(contact_drive)
        self.applied_finger_force=np.zeros((2,2))
        if contact_drive is not None:
            if contact_drive.get('mode') not in (None,CENTERED_FORCE):raise ValueError('unknown finger force controller')
            if not callable(getattr(collision,"planning_geometry",None)) or not contact_drive.get("source"):
                raise ValueError("contact drive requires collision prediction and parameter provenance")
            for name in ("stiffness","damping"):
                if not np.isfinite(contact_drive[name]) or contact_drive[name]<=0:
                    raise ValueError("invalid finger drive property")
            caps=finger_force_limits(contact_drive["force_limit_n"])
            for cap,(_,_,grip) in zip(caps,self._joints()):
                for joint in grip:
                    centered=contact_drive.get('mode')==CENTERED_FORCE
                    joint.set_drive_property(stiffness=0. if centered else contact_drive["stiffness"],damping=0. if centered else contact_drive["damping"],
                                             force_limit=0. if centered else float(cap),mode="force")

    def _apply_forces(self,side,entity,grip):
        force=np.asarray(entity.compute_passive_force(gravity=True,coriolis_and_centrifugal=True)).copy()
        if self.contact_drive is not None and self.contact_drive.get('mode')==CENTERED_FORCE:
            names=[j.get_name() for j in entity.get_active_joints()]
            indices=[names.index(j.get_name()) for j in grip]
            measured=np.asarray(entity.get_qpos())[indices];velocity=np.asarray(entity.get_qvel())[indices]
            target=np.array([float(j.get_drive_target()[0]) for j in grip])
            applied=centered_finger_force(measured,velocity,target,np.zeros(2),self.contact_drive['stiffness'],
                                         self.contact_drive['damping'],finger_force_limits(self.contact_drive['force_limit_n'])[side])
            force[indices]+=applied;self.applied_finger_force[side]=applied
        entity.set_qf(force)

    def take_replan(self):
        reason,self.replan_reason=self.replan_reason,None
        return reason

    def _drive_fingers(self):
        return np.array([[float(joint.get_drive_target()[0]) for joint in grip]
                         for _,_,grip in self._joints()])

    def _contact_observation(self,snapshot):
        q,native,_,tools=read_snapshot(snapshot,self.arms)
        block=getattr(self.env.box,"actor",self.env.box)
        pose=block.get_pose()
        block_task=inverse_pose(self.world_task)@pose_from_wxyz(pose.p,pose.q)
        state=self.collision.tracker.observe(snapshot.stamp,self.env.scene.get_contacts(),tools,block_task)
        self.collision.set_start(native,self._drive_fingers())
        return state

    def reset(self, spec):
        if self.consumed:
            raise RuntimeError("ARXBackend requires a new scene/reset per episode")
        self.consumed = True
        self.episode = spec["episode_id"]
        snapshot, _ = self.capture()
        q, native, _, _ = read_snapshot(snapshot, self.arms)
        actual_bases = {side: pose7(arm.world_base) for side, arm in zip(SIDES, snapshot.arms)}
        for side in SIDES:
            expected = np.asarray(self.initial["robot_base_world"][side])
            actual = np.asarray(actual_bases[side])
            if np.linalg.norm(pose_error(pose_from_wxyz(expected[:3], expected[3:]),
                                         pose_from_wxyz(actual[:3], actual[3:]))) > 1e-6:
                raise ValueError("reset metadata does not match actual base snapshot")
        for side, arm, q_i, finger_i in zip(SIDES, self.arms, q, native):
            if not np.allclose(q_i, self.initial["actual_arm_qpos_rad"][side], atol=1e-6, rtol=0):
                raise ValueError("reset metadata does not match actual arm state")
            if abs(arm.grip.read(finger_i) - self.initial["actual_gripper"][side]) > .001:
                raise ValueError("reset metadata does not match actual finger state")
        self.collision.refresh()
        if not self.collision(q, native):
            raise G2Error(Status.COLLISION, "actual reset state collides")
        return deepcopy(self.initial)

    def capture(self):
        stamp = Stamp(self.episode, self.tick, self.tick * self.dt)
        snapshot, observation = capture_snapshot(self.env, stamp, self.world_task)
        self.last_capture = snapshot
        if self.contact_drive is not None:
            self._contact_observation(snapshot)
        self.video.frame(observation)
        return snapshot, None

    def capture_state(self):
        """Refresh the physical start of an already queried action block."""
        stamp = Stamp(self.episode, self.tick, self.tick * self.dt)
        snapshot = Snapshot(stamp, capture_arms(self.env.robot, stamp), {}, self.world_task)
        if self.contact_drive is not None:
            self._contact_observation(snapshot)
        self.last_capture = snapshot
        return snapshot

    def success(self):
        # Same frozen task predicate for both robots, including its gripper rule.
        return bool(self.env.check_success())

    def _joints(self):
        robot = self.env.robot
        pairs = []
        for side, arm in zip(SIDES, self.arms):
            entity = getattr(robot, side + "_entity")
            lookup = {joint.get_name(): joint for joint in entity.get_active_joints()}
            pairs.append((entity, [lookup[name] for name in arm.chain.names],
                          [lookup[name] for name in arm.grip.names]))
        return pairs

    def hold(self):
        if self.episode is None:
            return
        stamp = Stamp(self.episode, self.tick, self.tick*self.dt)
        snapshot = Snapshot(stamp, capture_arms(self.env.robot, stamp), {}, self.world_task)
        q, fingers, _, _ = read_snapshot(snapshot, self.arms)
        if self.contact_drive is not None and self.collision.tracker.holders:
            drives=self._drive_fingers()
            for i,side in enumerate(SIDES):
                if side in self.collision.tracker.holders:
                    fingers[i]=drives[i]
        for side,((entity, joints, grip), values, native) in enumerate(zip(self._joints(), q, fingers)):
            for joint, value in zip(joints + grip, np.r_[values, native]):
                joint.set_drive_target(float(value))
                joint.set_drive_velocity_target(0.)
            self._apply_forces(side,entity,grip)

    def execute(self, plan, log, remaining, *, stop_on_success=True, stop_on_contact=False):
        self.replan_reason=None
        if not isinstance(plan, Trajectory) or self.last_capture is None:
            raise ValueError("ARXBackend only accepts synchronized G2 trajectories")
        count = len(plan.q)
        if count < 2 or not np.isfinite(plan.period_s) or plan.period_s <= 0:
            raise ValueError("empty or invalid trajectory timing")
        finite_array(plan.q, (count, 2, 6))
        finite_array(plan.velocity, (count, 2, 6))
        finite_array(plan.fingers, (count, 2, 2))
        if self.contact_drive is not None:
            if plan.contact_profile_id!=self.collision.profile_id or plan.predicted_fingers is None:
                raise ValueError("contact execution requires a matching predicted geometry trajectory")
            finite_array(plan.predicted_fingers,(count,2,2))
        elif plan.predicted_fingers is not None or plan.contact_profile_id:
            raise ValueError("contact trajectory requires explicit force-limited execution")
        events = plan.waypoint_ticks
        if (not events or any(type(tick) is not int for tick in events)
                or list(events) != sorted(set(events)) or events[0] < 1 or events[-1] != count - 1):
            raise ValueError("invalid trajectory waypoint events")
        for side, arm in enumerate(self.arms):
            limits = arm.chain.limits
            if np.any((plan.q[:, side] < limits[:, 0]) | (plan.q[:, side] > limits[:, 1])):
                raise G2Error(Status.LIMIT, "trajectory exceeds arm limits", geometry="candidate")
            grip_limits = arm.grip.limits
            if np.any((plan.fingers[:, side] < grip_limits[:, 0]) | (plan.fingers[:, side] > grip_limits[:, 1])):
                raise G2Error(Status.LIMIT, "trajectory exceeds finger limits", geometry="candidate")
            if plan.predicted_fingers is not None and np.any(
                    (plan.predicted_fingers[:,side]<grip_limits[:,0]) |
                    (plan.predicted_fingers[:,side]>grip_limits[:,1])):
                raise G2Error(Status.LIMIT,"predicted contact geometry exceeds finger limits", geometry="candidate")
        ratio = plan.period_s / self.dt
        substeps = round(ratio)
        if substeps < 1 or abs(substeps - ratio) > 1e-6:
            raise ValueError("trajectory period must be an integer multiple of physics dt")
        stamp = Stamp(self.episode,self.tick,self.tick*self.dt)
        current = Snapshot(stamp,capture_arms(self.env.robot,stamp),{},self.world_task)
        if any(not np.allclose(a.world_base,b.world_base,atol=1e-8,rtol=0)
               for a,b in zip(current.arms,self.last_capture.arms)):
            raise G2Error(Status.UNSYNCED,"robot base changed after observation")
        q0, g0, _, _ = read_snapshot(current, self.arms)
        initial_geometry=plan.predicted_fingers[0] if self.contact_drive is not None else plan.fingers[0]
        if not np.allclose(q0, plan.q[0], atol=1e-6, rtol=0) or not np.allclose(g0, initial_geometry, atol=1e-6, rtol=0):
            raise G2Error(Status.UNSYNCED, "trajectory starts from a different measured state")
        if self.contact_drive is not None and not np.allclose(self._drive_fingers(),plan.fingers[0],atol=1e-6,rtol=0):
            raise G2Error(Status.UNSYNCED,"finger drives changed after observation")
        self.collision.refresh()
        entities = self._joints()
        done = 0
        for index in range(1, len(plan.q)):
            remaining()
            geometry=plan.fingers[index]
            if self.contact_drive is not None:
                self.collision.refresh()
                self.collision.tracker.guard_opening(self._drive_fingers(),plan.fingers[index],self.arms,self.collision.giver)
                geometry=self.collision.planning_geometry(plan.q[index],plan.fingers[index],g0,remaining)
            if not self.collision(plan.q[index], geometry):
                collisions=self.collision.last_collisions
                log(dict(kind="planning_gate_rejection",tick=self.tick,trajectory_tick=index,
                         sim_s=self.tick*self.dt,target_arm=plan.q[index],target_fingers=plan.fingers[index],
                         predicted_fingers=geometry,collisions=collisions,
                         control_call_count=0,physics_step_count=0))
                raise G2Error(Status.COLLISION, "scene changed after planning", geometry="candidate",
                              tick=index,collisions=collisions)
            before_holders=set(self.collision.tracker.holders) if self.contact_drive is not None else set()
            before_contact = (
                any(self.collision.tracker.contact_fingers.values())
                if stop_on_contact and self.contact_drive is not None else False
            )
            if self.contact_drive is not None:
                self.collision.tracker.authorize_opening(self._drive_fingers(),plan.fingers[index],self.arms,self.collision.giver)
            # Submit both arms and all native finger drives before any scene.step().
            for side, arm, (entity, joints, grip), q, v, g in zip(SIDES, self.arms, entities, plan.q[index], plan.velocity[index], plan.fingers[index]):
                for joint, value, speed in zip(joints, q, v):
                    joint.set_drive_target(float(value))
                    joint.set_drive_velocity_target(float(speed))
                for joint, value in zip(grip, g):
                    joint.set_drive_target(float(value))
                    joint.set_drive_velocity_target(0.)
                setattr(self.env.robot, side + "_gripper_val", arm.grip.read(g))
                entity.set_qf(entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            for substep in range(substeps):
                remaining()
                if self.contact_drive is not None and self.contact_drive.get('mode')==CENTERED_FORCE:
                    for si,(entity,_,grip) in enumerate(entities):self._apply_forces(si,entity,grip)
                self.env.scene.step()
                self.tick += 1
                stamp = Stamp(self.episode, self.tick, self.tick * self.dt)
                snapshot = Snapshot(stamp, capture_arms(self.env.robot, stamp), {}, self.world_task)
                q, native, frames, actual = read_snapshot(snapshot, self.arms)
                contact_state=None
                contact_error=None
                if self.contact_drive is not None:
                    try:
                        contact_state=self._contact_observation(snapshot)
                        self.collision.refresh()
                        if not self.collision.measured_clear(q,native):
                            raise G2Error(Status.COLLISION,"measured state has non-permitted contact",
                                          geometry="actual", collisions=self.collision.last_collisions)
                    except G2Error as error:
                        log(dict(tick=self.tick,sim_s=self.tick*self.dt,kind="contact_failure",
                                 actual_arm=q,actual_fingers=native,target_fingers=plan.fingers[index],
                                 contact_state=self.collision.tracker.snapshot(),detail=str(error),
                                 diagnostic=error.diagnostic,control_call_count=1 if substep==0 else 0,physics_step_count=1))
                        kind=contact_failure_kind(error,self.collision.tracker,stamp)
                        if kind is None:
                            raise
                        for arm, fingers in zip(self.arms, native):
                            arm.grip.measure(fingers)
                        self.collision.verify_contact_stop(q,native)
                        contact_error=error
                desired = np.array([frame @ arm.chain.fk(target) @ arm.offset()
                                    for frame, arm, target in zip(frames, self.arms, plan.q[index])])
                residual = [pose_error(goal, real) for goal, real in zip(desired, actual)]
                p = [float(np.linalg.norm(x[:3])) for x in residual]
                a = [float(np.rad2deg(np.linalg.norm(x[3:]))) for x in residual]
                log(dict(tick=self.tick, sim_s=self.tick*self.dt, period_s=self.dt,
                         actual_arm=q, target_arm=plan.q[index], actual_fingers=native,
                         target_fingers=plan.fingers[index], actual_tools_task=actual,
                         predicted_fingers=geometry,contact_state=contact_state,
                         finger_drive_properties=self.contact_drive,
                         applied_finger_force_n=self.applied_finger_force.copy() if self.contact_drive and self.contact_drive.get('mode')==CENTERED_FORCE else None,
                         target_tools_task=desired, position_error_m=p, orientation_error_deg=a,
                         control_call_count=1 if substep == 0 and contact_error is None else 0,
                         physics_step_count=1 if contact_error is None else 0))
                if max(p) > self.position_tol or max(a) > self.angle_tol:
                    raise G2Error(Status.BACKEND_ERROR, "measured TCP tracking exceeded acceptance bounds",
                                  geometry="actual", position_error_m=p, orientation_error_deg=a)
                if contact_error is not None:
                    remaining()
                    contact_error.diagnostic.update(terminal_checks_passed=True,contact_loss=kind,contact_tick=self.tick)
                    raise contact_error
                g0=native
                if self.contact_drive is not None and set(self.collision.tracker.holders)!=before_holders:
                    self.replan_reason=dict(reason="measured_contact_transition",tick=self.tick,
                                            before=sorted(before_holders),after=sorted(self.collision.tracker.holders),
                                            interrupted_waypoint=True)
                    self.env.take_action_cnt+=1
                    self.completed+=1
                    return done+1
                if (stop_on_contact and self.contact_drive is not None and not before_contact
                        and any(self.collision.tracker.contact_fingers.values())):
                    self.replan_reason = dict(reason="measured_first_contact", tick=self.tick,
                                              interrupted_waypoint=True)
                    self.env.take_action_cnt += 1
                    self.completed += 1
                    return done + 1
                if stop_on_success and self.success():
                    # A successful partial waypoint still consumes its current policy action.
                    return done + 1
            if index in plan.waypoint_ticks:
                done += 1
                self.env.take_action_cnt += 1
                self.completed += 1
        return done

    def close(self, output):
        try:
            if self.episode is not None:
                self.video.frame(self.env.get_obs())
            return str(self.video.path)
        finally:
            try:
                self.video.close()
            finally:
                self.env.close_env(clear_cache=True)
