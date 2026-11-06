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
    def __init__(self, scene, after_step):
        self.actual, self.after_step = scene, after_step

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def step(self, *args, **kwargs):
        value = self.actual.step(*args, **kwargs)
        self.after_step()
        return value


class NativeBackend:
    def __init__(self, env, arms, world_task, initial, video):
        if env.robot.left_entity is not env.robot.right_entity:
            raise ValueError("native Aloha must use its shared articulation")
        self.env, self.arms, self.world_task = env, tuple(arms), world_task
        self.initial, self.video = deepcopy(initial), video
        self.dt = env.scene.get_timestep()
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("invalid native scene timestep")
        self.tick, self.episode, self.consumed = 0, None, False
        self.native = NativeState(env.robot)
        self.native_limits = [arm.chain.limits for arm in self.arms]
        self.log, self.remaining = None, None
        self.actual_scene = env.scene
        env.scene = SceneClock(env.scene, self._stepped)

    def reset(self, spec):
        if self.consumed:
            raise RuntimeError("native backend requires a fresh reset scene")
        self.consumed = True
        self.episode = spec["episode_id"]
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
        self.video.frame(obs)
        return snapshot, request["state"]

    def success(self):
        return bool(self.env.check_success())

    def _stepped(self):
        self.tick += 1
        if self.log is None:
            return
        stamp = Stamp(self.episode, self.tick, self.tick*self.dt)
        snapshot = Snapshot(stamp, capture_arms(self.env.robot, stamp), {}, self.world_task)
        q, fingers, frames, actual = read_snapshot(snapshot, self.arms)
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
        self.remaining()

    def execute(self, targets, log, remaining):
        array = np.asarray(targets)
        if array.ndim != 2 or not 1 <= len(array) <= 4:
            raise ValueError("native action prefix must contain 1..4 rows")
        finite_array(array, (len(array), 14))
        for side,arm in enumerate(self.arms):
            values=array.reshape(-1,2,7)[:,side]
            if (np.any((values[:,:6]<arm.chain.limits[:,0]) | (values[:,:6]>arm.chain.limits[:,1]))
                    or np.any((values[:,6]<0) | (values[:,6]>1))):
                raise G2Error(Status.LIMIT,"native target outside calibrated bounds")
        self.log, self.remaining = log, remaining
        done = 0
        try:
            for row in array:
                remaining()
                before = self.env.take_action_cnt
                self.env.take_action(row.copy(), action_type="qpos")
                if self.env.take_action_cnt != before + 1:
                    raise G2Error(Status.BACKEND_ERROR, "native qpos action was not consumed")
                done += 1
                if self.success():
                    break
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
