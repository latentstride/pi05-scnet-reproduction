"""Full native Aloha qpos episodes with per-physics-step evidence."""
from copy import deepcopy

import numpy as np

from g2_adapter import G2Error, Snapshot, SIDES, Stamp, read_snapshot
from g2_capture import capture_arms, capture_snapshot
from g2_initial import compare_initial, capture_initial
from g2_kinematics import pose_error
from pace_client_smoke import NativeState, encode
from robotwin_control_geometry import Status, finite_array


class SceneClock:
    """Forward the actual scene interface and count only real step() calls."""
    def __init__(self, scene, after_step, before_step=None):
        self.actual, self.after_step = scene, after_step
        self.before_step = before_step

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def step(self, *args, **kwargs):
        if self.before_step is not None:
            self.before_step()
        value = self.actual.step(*args, **kwargs)
        self.after_step()
        return value


class NativeBackend:
    def __init__(self, env, arms, world_task, initial, video, guard=None):
        if env.robot.left_entity is not env.robot.right_entity:
            raise ValueError("native Aloha must use its shared articulation")
        self.env, self.arms, self.world_task = env, tuple(arms), world_task
        self.initial, self.video = deepcopy(initial), video
        self.guard = guard
        self.execution_protocol = guard.protocol if guard is not None else None
        self.action_index = None
        self.dt = env.scene.get_timestep()
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("invalid native scene timestep")
        self.tick, self.episode, self.consumed = 0, None, False
        self.native = NativeState(env.robot)
        self.native_limits = [arm.chain.limits for arm in self.arms]
        self.log, self.remaining = None, None
        self.actual_scene = env.scene
        env.scene = SceneClock(env.scene, self._stepped, self._before_step if guard is not None else None)

    def reset(self, spec):
        if self.consumed:
            raise RuntimeError("native backend requires a fresh reset scene")
        self.consumed = True
        self.episode = spec["episode_id"]
        self.window = spec["window"]
        self.env.step_lim = min(self.env.step_lim, spec["max_actions"])
        actual = capture_initial(self.env, self.arms, self.world_task, self.episode,
                                 self.initial["reset_fixture_sha256"], self.initial["workspace_check_id"],
                                 self.initial["common_workspace_pass"])
        mismatch = compare_initial(actual, self.initial)
        if mismatch:
            raise ValueError("native reset snapshot changed: " + ", ".join(mismatch))
        return actual

    def capture(self):
        snapshot, obs = capture_snapshot(self.env, Stamp(self.episode, self.tick, self.tick*self.dt), self.world_task)
        request = encode(obs, self.env.get_instruction())
        self.native.measured(obs, request["state"])
        if self.guard is None:
            self.video.frame(obs)
        return snapshot, request["state"]

    def success(self):
        return bool(self.env.check_success())

    def _drive(self):
        values=[]
        for side, arm in zip(SIDES, self.arms):
            entity=getattr(self.env.robot, side+"_entity")
            joints={j.get_name():j for j in entity.get_active_joints()}
            values.append([float(joints[n].get_drive_target()[0]) for n in arm.chain.names+arm.grip.names])
        values=np.asarray(values)
        return values[:,:6], values[:,6:]

    def _before_step(self):
        if self.log is None:
            raise G2Error(Status.BACKEND_ERROR, "unguarded scene advance outside native execution")
        if self.tick >= getattr(self.guard,"max_physics_steps",25000):
            raise G2Error(Status.TIMEOUT,"G2-E physical-step ceiling reached before advance")
        stamp=Stamp(self.episode,self.tick,self.tick*self.dt)
        snap=Snapshot(stamp,capture_arms(self.env.robot,stamp),{},self.world_task)
        q,fingers,_,_=read_snapshot(snap,self.arms)
        targets,drive=self._drive()
        self.guard.candidate(q,fingers,targets,drive,self.remaining)

    def _stepped(self):
        self.tick += 1
        if self.log is None:
            return
        stamp = Stamp(self.episode, self.tick, self.tick*self.dt)
        snapshot = Snapshot(stamp, capture_arms(self.env.robot, stamp), {}, self.world_task)
        try:
            q, fingers, frames, actual = read_snapshot(snapshot, self.arms)
        except G2Error:
            if self.guard is not None:
                self.log(dict(kind="executed_step",tick=self.tick,sim_s=self.tick*self.dt,period_s=self.dt,
                              actual_named=[dict(names=s.names,qpos=s.qpos) for s in snapshot.arms],
                              measurement_rejected=True,control_call_count=1,physics_step_count=1))
            raise
        targets = []
        for side, arm in zip(SIDES, self.arms):
            entity = getattr(self.env.robot, side + "_entity")
            lookup = {joint.get_name(): joint for joint in entity.get_active_joints()}
            targets.append([float(lookup[name].get_drive_target()[0]) for name in arm.chain.names])
        target_tools = [frame @ arm.chain.fk(target) @ arm.offset()
                        for frame, arm, target in zip(frames, self.arms, targets)]
        residual = [pose_error(goal, real) for goal, real in zip(target_tools, actual)]
        self.log(dict(tick=self.tick, sim_s=self.tick*self.dt, period_s=self.dt,
                      actual_arm=q, actual_fingers=fingers, target_arm=targets,
                      target_tools_task=target_tools, actual_tools_task=actual,
                      position_error_m=[float(np.linalg.norm(x[:3])) for x in residual],
                      orientation_error_deg=[float(np.rad2deg(np.linalg.norm(x[3:]))) for x in residual],
                      control_call_count=1, control_call_unit="native_control_loop_iteration",
                      physics_step_count=1))
        if self.guard is not None:
            from robotwin_control_geometry import pose_from_wxyz
            block=getattr(self.env.box,"actor",self.env.box)
            pose=block.get_pose()
            block_pose=np.linalg.inv(self.world_task)@pose_from_wxyz(pose.p,pose.q)
            contacts=self.actual_scene.get_contacts()
            self.log(dict(kind="native_contact_sample",tick=self.tick,action_index=self.action_index,
                          block_pose_task=block_pose,target_fingers_drive_m=self._drive()[1],
                          contacts=[dict(body_ids=[b.entity.per_scene_id for b in c.bodies],
                              body_names=[b.entity.name for b in c.bodies],
                              points=[dict(position=p.position,separation=float(p.separation),
                                           normal=p.normal,impulse=p.impulse) for p in c.points]) for c in contacts],
                          control_call_count=0,physics_step_count=0))
            self.guard.measured(stamp,q,fingers,actual,block_pose,contacts)
            self.log(dict(kind="native_guard_state",tick=self.tick, action_index=self.action_index,
                          control_call_count=0, physics_step_count=0, **self.guard.metrics()))
            self.guard.tracking(target_tools,actual)
        self.remaining()

    def execute(self, targets, log, remaining):
        array = np.asarray(targets)
        if array.ndim != 2 or not 1 <= len(array) <= (getattr(self,"window",0) if self.guard is not None else 4):
            raise ValueError("native action prefix outside admitted horizon")
        finite_array(array, (len(array), 14))
        for side,arm in enumerate(self.arms):
            values=array.reshape(-1,2,7)[:,side]
            if (np.any((values[:,:6]<arm.chain.limits[:,0]) | (values[:,:6]>arm.chain.limits[:,1]))
                    or np.any((values[:,6]<0) | (values[:,6]>1))):
                raise G2Error(Status.LIMIT,"native target outside calibrated bounds")
        self.log, self.remaining = log, remaining
        done = 0
        if self.guard is not None and self.guard.previous_drive is None:
            self.guard.previous_drive = self._drive()[1].copy()
        try:
            for self.action_index,row in enumerate(array):
                remaining()
                before = self.env.take_action_cnt
                if self.guard is not None:
                    log(dict(kind="native_action_request",action_index_in_prefix=self.action_index,
                             action_index_in_episode=before,normalized_qpos_request=row,
                             tick=self.tick,control_call_count=0,physics_step_count=0))
                self.env.take_action(row.copy(), action_type="qpos")
                if self.env.take_action_cnt != before + 1:
                    raise G2Error(Status.BACKEND_ERROR, "native qpos action was not consumed")
                done += 1
                if self.guard is not None:
                    self.video.frame(self.env.get_obs())
                if self.success():
                    break
        except BaseException as error:
            if self.guard is None:
                raise
            diagnostic=dict(getattr(error,"diagnostic",{}),completed_actions=done,
                            interrupted_action_index=self.action_index)
            log(dict(kind="native_guard_rejection",tick=self.tick,control_call_count=0,
                     physics_step_count=0,error=str(error),**diagnostic))
            if isinstance(error,(KeyboardInterrupt,SystemExit)):
                error.diagnostic=diagnostic
                raise
            raise G2Error(getattr(error,"status",Status.BACKEND_ERROR),str(error),**diagnostic) from error
        finally:
            self.log, self.remaining = None, None
        return done

    def hold(self):
        if self.episode is None:
            return
        states = capture_arms(self.env.robot, Stamp(self.episode, self.tick, self.tick*self.dt))
        for sample, arm, side in zip(states, self.arms, SIDES):
            entity = getattr(self.env.robot, side + "_entity")
            actual = dict(zip(sample.names, sample.qpos))
            lookup = {joint.get_name(): joint for joint in entity.get_active_joints()}
            for name in arm.chain.names + arm.grip.names:
                lookup[name].set_drive_target(float(actual[name]))
                lookup[name].set_drive_velocity_target(0.)
        entity = self.env.robot.left_entity
        entity.set_qf(entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))

    def close(self, output):
        self.env.scene = self.actual_scene
        try:
            # Always retain the final physical state, including early success.
            if self.episode is not None:
                self.video.frame(self.env.get_obs())
            self.video.close()
            return str(self.video.path)
        finally:
            self.env.close_env(clear_cache=True)
