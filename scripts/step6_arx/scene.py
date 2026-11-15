"""Bind the PR 5 target-side guards to a new, native ARX RoboTwin reset."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
import json
import random
import time

import numpy as np

from g2_adapter import G2Error, SIDES, Snapshot, Stamp, bounds, read_snapshot
from g2_assets import load_arms
from g2_capture import capture_arms, capture_task_objects
from g2_collision import JointCollisionWorld, apply_audited_arx_physics, audited_arx_self_profile
from g2_contact import ContactTracker, ContactWorld, contact_profile
from g2_initial import capture_initial, digest_json, matrix7, raw_actor
from g2_motion import QuinticMotion
from g2_robotwin_backend import ARXBackend, arx_settings, joint_space_setup
from g2_task_runtime import pace_context
from pace_client_smoke import Video, load_native_task
from pace_diagnostics import Diagnostics
from pace_runtime import PACE_REVISION
from robotwin_control_geometry import Status

from .protocol import CAMERAS, PROMPT, SHAPE, request_from


class CamerasUnavailable(RuntimeError):
    pass


def pose(actor):
    p = actor.get_pose()
    return np.r_[p.p, p.q]


class NativeScene(ARXBackend):
    """Only changes observation format. All control, reset admission and hold are inherited."""

    def __init__(self, *args, motion, headless, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = motion
        self.headless = headless
        self.observation = None
        self.camera_config = None

    def state(self):
        stamp = Stamp(self.episode, self.tick, self.tick * self.dt)
        return Snapshot(stamp, capture_arms(self.env.robot, stamp), {}, self.world_task,
                        capture_task_objects(self.env, self.world_task))

    def command(self):
        actual_drive = [[float(j.get_drive_target()[0]) for j in arm]
                        for _, arm, _ in self._joints()]
        robot = self.env.robot
        for side, (_, arm, _) in zip(SIDES, self._joints(), strict=True):
            if [j.get_name() for j in arm] != [j.get_name() for j in getattr(robot, side + "_arm_joints")]:
                raise ValueError("the native sampler and PR 5 arm order differ")
        return np.column_stack((actual_drive, [robot.get_left_gripper_val(),
                                               robot.get_right_gripper_val()])).ravel()

    def sample(self):
        if not self.headless:
            try:
                return self.env.get_obs()
            except Exception as error:
                raise CamerasUnavailable(f"native camera sampling failed: {error}") from error
        # Explicit wire-only fixture. Never presented as rendered evidence or sent to a model.
        robot = self.env.robot
        state = np.array(robot.get_left_arm_jointState() + robot.get_right_arm_jointState())
        images = {}
        for i, name in enumerate(CAMERAS):
            rgb = np.empty(SHAPE, dtype=np.uint8)
            rgb[:] = [10 + i, 30 + self.tick % 100, 100 + i]
            images[name] = {"rgb": rgb}
        return {"joint_action": {"vector": state}, "observation": images}

    def capture(self):
        snapshot = self.state()
        self.observation = self.sample()
        self.sampled_ns = time.time_ns()
        if self.state().stamp != snapshot.stamp:
            raise G2Error(Status.UNSYNCED, "physics advanced during observation")
        self.last_capture = snapshot
        self._contact_observation(snapshot)
        if not self.headless:
            self.camera_config = {}
            for name in CAMERAS:
                metadata = self.observation["observation"][name]
                self.camera_config[name] = {}
                for key in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl"):
                    matrix = np.asarray(metadata[key], dtype=float)
                    if matrix.size == 0 or not np.isfinite(matrix).all():
                        raise CamerasUnavailable(f"invalid camera calibration: {name}/{key}")
                    self.camera_config[name][key] = matrix.tolist()
            self.video.frame(self.observation)
        return snapshot, None

    def request(self, query):
        # reset() already captured frame zero at precisely this tick.
        if self.last_capture.stamp.tick != self.tick:
            self.capture()
        return request_from(self.observation, query, self.command())

    def initial_extras(self):
        block = raw_actor(self.env.box)
        body = self.block_body
        return dict(camera_config=self.camera_config, command=self.command().tolist(),
                    fingers_m=self._drive_fingers().tolist(),
                    qpos=[np.asarray(entity.get_qpos()).tolist() for entity, _, _ in self._joints()],
                    qvel=[np.asarray(entity.get_qvel()).tolist() for entity, _, _ in self._joints()],
                    block_world=pose(block).tolist(), block_linear_velocity=body.linear_velocity.tolist(),
                    block_angular_velocity=body.angular_velocity.tolist(), mass_kg=float(body.mass),
                    dt=self.dt, control_period=self.motion.period)

    def observe_physics(self, snapshot):
        q, fingers, _, tools = read_snapshot(snapshot, self.arms)
        tracker = self.collision.tracker
        block = raw_actor(self.env.box)
        target = raw_actor(self.env.target_box)
        impulse_min = tracker.limits.minimum_impulse
        contacts = []
        robot_links = {link.entity.per_scene_id: (side, link.name)
                       for side, (entity, _, _) in zip(SIDES, self._joints())
                       for link in entity.get_links()}
        for contact in self.env.scene.get_contacts():
            a, b = (c.entity for c in contact.bodies)
            if any(entity.per_scene_id in robot_links or entity.per_scene_id == block.per_scene_id
                   for entity in (a, b)):
                impulses = [float(np.linalg.norm(point.impulse)) for point in contact.points]
                contacts.append(dict(entities=[a.name, b.name],
                                     roles=[robot_links.get(entity.per_scene_id) for entity in (a, b)],
                                     involves_block=block.per_scene_id in (a.per_scene_id, b.per_scene_id),
                                     max_impulse=max(impulses, default=0.),
                                     active=any(i >= impulse_min for i in impulses)))
        measured = [arm.grip.read(g) for arm, g in zip(self.arms, fingers)]
        return dict(tick=snapshot.stamp.tick, sim_s=snapshot.stamp.sim_s, actual_qpos=q,
                    actual_fingers_m=fingers, actual_gripper_aperture=measured,
                    arm_and_gripper_command=self.command(), finger_drive_targets_m=self._drive_fingers(),
                    tcp_world=self.world_task @ tools, block_pose_world=pose(block),
                    allowed_block_contact_roles=[(side, entity.find_joint_by_name(name).child_link.name)
                                                 for side, (entity, _, _), arm in zip(SIDES, self._joints(), self.arms)
                                                 for name in arm.grip.names],
                    target_pose_world=pose(target),
                    box_functional_point_world=self.env.box.get_functional_point(0, "pose").p,
                    target_functional_point_world=self.env.target_box.get_functional_point(1, "pose").p,
                    contacts=contacts, contact_state=tracker.snapshot(),
                    native_success=self.success())

    def execute_native_row(self, row, remaining, log):
        """Map native units and delegate to the existing target-side gate and executor."""
        snapshot = self.state()
        q, fingers, _, _ = read_snapshot(snapshot, self.arms)
        self.collision.refresh()
        if not self.collision.measured_clear(q, fingers):
            raise G2Error(Status.COLLISION, "measured ARX state blocks the next row",
                          geometry="actual", collisions=self.collision.last_collisions)
        goals = row.reshape(2, 7)
        native = []
        for arm, goal in zip(self.arms, goals, strict=True):
            bounds(goal[:6], arm.chain.limits, geometry="candidate")
            if not 0 <= goal[6] <= 1:
                raise G2Error(Status.LIMIT, "native ARX gripper target outside [0,1]",
                              geometry="candidate")
            native.append(arm.grip.target(goal[6]))
        native = np.asarray(native)
        target = goals[:, :6]
        current_tools = self.collision.tools(q)
        _, phase = self.collision.prepare_tools(self.collision.tools(target), native, current_tools)
        if phase:
            target = q  # Existing donor release barrier, before withdrawing the arm.
        barrier = self.collision.action_barrier(target, native, remaining)
        try:
            plan = self.motion.plan(q, fingers, [(target, native)], self.collision, remaining)
        except G2Error as error:
            error.diagnostic.setdefault("collisions", self.collision.last_collisions)
            raise
        log(dict(kind="approved_row", input=row, joint_goal=target, finger_goal_m=native,
                 planned_control_ticks=plan.waypoint_ticks[0], contact_barrier=barrier, phase=phase))
        if self.execute(plan, log, remaining) != 1 or self.tick <= snapshot.stamp.tick:
            raise G2Error(Status.UNSYNCED, "native row returned without a completed execution receipt")
        return self.take_replan() or ({"reason": "contact_or_release_requires_observation"} if barrier else None)

    def close(self, output):
        if not self.headless:
            return super().close(output)
        self.env.close_env(clear_cache=False)
        return None


@contextmanager
def open_scene(args, campaign, safety, seed, out, episode):
    import sapien
    from mplib.sapien_utils.conversion import convert_object_name

    assets = args.pace_root / "assets"
    audit = json.loads(args.asset_audit.read_text())
    row, = [x for x in audit["robots"] if x["robot"] == "ARX-X5"]
    arms = load_arms(assets, args.asset_audit, "ARX-X5", safety["calibration"])
    world_task = matrix7(safety["world_task"])
    env = backend = None
    with pace_context(args.pace_root):
        try:
            from pace.robotwin import load_environment

            if args.headless:
                import faulthandler
                with (out / "simulator-imports.jsonl").open("w") as events, (out / "simulator-imports.log").open("w") as stacks:
                    diagnostics = Diagnostics(events, stacks, {}, 45)
                    try:
                        with diagnostics.phase("load_environment"):
                            env, settings = load_native_task(load_environment, diagnostics, True)
                    finally:
                        faulthandler.disable()
            else:
                env, settings = load_environment(campaign["task"])
            from envs.robot.robot import Robot
            settings = arx_settings(settings, assets / "embodiments/ARX-X5",
                                    row["config_sha256"], safety["arx_base_distance_m"])
            settings.pop("eval_video_save_dir", None)
            settings["data_type"] = dict(settings["data_type"], rgb=True, qpos=True)
            settings["domain_randomization"] = dict(settings["domain_randomization"])
            if any(settings["domain_randomization"].get(k) for k in
                   ("random_background", "cluttered_table", "random_head_camera_dis",
                    "random_table_height", "random_light", "random_embodiment")):
                raise ValueError("paired protocol requires RoboTwin clean scene settings")
            if args.headless:
                from .headless import presentation_off
                presentation = presentation_off(env, assets / row["urdf_member"], row["urdf_sha256"],
                                                out / "arx-collision-only.urdf")
            else:
                presentation = nullcontext()
            random.seed(seed)
            with presentation, joint_space_setup(Robot, [arm.grip for arm in arms]):
                env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **settings)
            env.set_instruction(PROMPT)
            actual = capture_arms(env.robot, Stamp(episode, 0, 0.))
            arms = tuple(replace(arm, instance=sample.instance) for arm, sample in zip(arms, actual))
            initial = capture_initial(env, arms, world_task, episode, digest_json({"scene_seed": seed}),
                                      "step6-seeded-reset")
            entities = [env.robot.left_entity, env.robot.right_entity]
            physical_filter = apply_audited_arx_physics(env.scene, entities, assets)
            collision = JointCollisionWorld.from_scene(env.scene, entities,
                                                       [a.chain.names for a in arms],
                                                       [a.grip.names for a in arms])
            collision.use_self_profile(audited_arx_self_profile(assets))
            collision.audit_path = out / "collision-checks.jsonl"
            block = raw_actor(env.box)
            body = block.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            native_mass = float(body.mass)
            # Fail closed if the native RoboTwin block does not match the audited contact profile.
            contact_limits, drive = contact_profile(safety, native_mass)
            tracker = ContactTracker.from_scene(env.robot, block, arms, contact_limits)
            actors = env.scene.get_all_actors()
            frames = [np.linalg.inv(world_task) @ arm.world_base for arm in actual]
            collision = ContactWorld(
                collision, tracker, arms, frames, convert_object_name(block),
                [convert_object_name(a) for a in actors],
                [[entity.find_joint_by_name(n).child_link.name for n in arm.grip.names]
                 for entity, arm in zip(entities, arms)],
                [convert_object_name(a) for a in actors if a.name in ("table", "target_box")],
                world_task=world_task)
            collision.giver = safety["contact_profile"]["giver"]
            limits = safety["control_limits"]
            motion = QuinticMotion(safety["control_period_s"], limits["arm_velocity_rad_s"],
                                   limits["arm_acceleration_rad_s2"], limits["finger_velocity_m_s"],
                                   limits["finger_acceleration_m_s2"], safety["collision_joint_step_rad"],
                                   safety["collision_finger_step_m"], limits["source"],
                                   "sapien3.0.0b1_mplib0.2.1_real_scene_FCL",
                                   max_ticks=safety["max_plan_ticks"])
            backend = NativeScene(env, arms, collision, world_task, env.scene.get_timestep(), initial,
                                  safety["tracking_position_limit_m"], safety["tracking_angle_limit_deg"],
                                  Video(out / "head.mp4"), drive, motion=motion, headless=args.headless)
            backend.block_body = body
            yield backend, dict(pace_revision=PACE_REVISION, arx_config_sha256=row["config_sha256"],
                                arx_urdf_sha256=row["urdf_sha256"], physx_filter=physical_filter,
                                safety_profile_sha256=digest_json(safety), native_block_mass_kg=native_mass,
                                contact_profile_id=collision.profile_id,
                                versions={key: version(key) for key in ("sapien", "mplib")},
                                task_file_sha256=sha256((args.pace_root / "envs/handover_block.py")
                                                        .read_bytes()).hexdigest())
        finally:
            if env is not None:
                if backend is None:
                    # Partial setup may not have created a scene.
                    if getattr(env, "scene", None) is not None:
                        env.close_env(clear_cache=not args.headless)
                else:
                    backend.close(out)
