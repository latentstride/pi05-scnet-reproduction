"""Real PACE scene/session construction. This module does not allocate resources."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np

from g2e_protocol import PROTOCOL, CONTACT_MODE_NATIVE, validate_window
from g2_adapter import Adapter, SIDES, Stamp
from g2_assets import load_arms
from g2_capture import capture_arms
from g2_collision import JointCollisionWorld, audited_arx_self_profile, apply_audited_arx_physics
from g2_contact import CONTACT_MODE, ContactTracker, ContactWorld, contact_profile
from g2_episode import save_json
from g2_initial import apply_fixture, capture_initial, digest_json, matrix7
from g2_motion import QuinticMotion
from g2_native_backend import NativeBackend
from g2_proxy_collision import ProxyCollisionWorld
from g2_robotwin_backend import ARXBackend, arx_settings, joint_space_setup, pose7
from pace_client_smoke import Video
from pace_runtime import PACE_REVISION, TASK, require_checkout


def verify_runtime(pace_root, audit):
    require_checkout(pace_root, PACE_REVISION)
    for name, expected in audit.items():
        if name.startswith(("envs/", "task_config/")):
            path = pace_root / name
            if sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"PACE runtime source mismatch: {name}")


def validate_config(config, *, allow_native_h50=False):
    from g2e_protocol import APPROACH_REPAIR
    repair = config.get("free_space_approach_repair")
    if repair is not None and (repair != APPROACH_REPAIR
            or config.get("embodiment") not in ("aloha-agilex", "ARX-X5")):
        raise ValueError("unknown bounded approach profile")
    if repair is not None and ((config["embodiment"] == "aloha-agilex" and
            (config.get("execution_protocol") != PROTOCOL or config.get("contact_mode") != CONTACT_MODE_NATIVE))
            or (config["embodiment"] == "ARX-X5" and config.get("contact_mode") != CONTACT_MODE)):
        raise ValueError("bounded approach requires its existing measured-contact guard")
    from g2e_protocol import PRELOAD_ESTABLISHMENT, PRELOAD_PROTOCOL
    establishment = config.get("native_preload_establishment")
    if establishment is not None and (establishment != PRELOAD_ESTABLISHMENT
            or config.get("native_preload_control") != PRELOAD_PROTOCOL):
        raise ValueError("contact preload establishment requires the reviewed native hold profile")
    preload = config.get("native_preload_control")
    if preload is not None:
        from g2_native_preload import PRELOAD_PROTOCOL
        if (preload != PRELOAD_PROTOCOL or config.get("native_release_semantics") != "physical_aperture_v1"
                or config.get("contact_mode") != CONTACT_MODE_NATIVE
                or config.get("embodiment") != "aloha-agilex"):
            raise ValueError("native preload candidate requires guarded Aloha physical-aperture semantics")
    semantics = config.get("native_release_semantics")
    if semantics is not None and (semantics != "physical_aperture_v1"
            or config["contact_mode"] != CONTACT_MODE_NATIVE):
        raise ValueError("native release semantics require guarded Aloha")

    if config["data_kind"] != "diagnostic" or config["task"] != TASK:
        raise ValueError("real scene entry currently accepts diagnostic handover_block only")
    if config["contact_mode"]==CONTACT_MODE_NATIVE:
        if config.get("execution_protocol")!=PROTOCOL or config["embodiment"]!="aloha-agilex":
            raise ValueError("native contact guard requires the G2-E native protocol")
        seed=config.get("policy_rng_seed")
        if type(seed) is not int or not 0<=seed<2**32:
            raise ValueError("G2-E requires an explicit uint32 policy RNG seed")
    elif config.get("execution_protocol")==PROTOCOL:
        raise ValueError("G2-E requires its explicit measured-contact mode")
    elif config["contact_mode"]==CONTACT_MODE:
        contact_profile(config)
    elif config["contact_mode"] != "strict_no_contact_exemptions":
        raise ValueError("contact permission/held-object profiles have not passed acceptance")
    if not config["control_limits"]["source"]:
        raise ValueError("control bounds require an explicit source")
    if not np.allclose(matrix7(config["world_task"]), matrix7(config["source_world_task"]), atol=1e-9, rtol=0):
        raise ValueError("proxy collision scene currently requires identical source/target task frames")
    validate_window(config["window"],config.get("execution_protocol"),config["embodiment"], allow_native_h50=allow_native_h50)
    if type(config["max_actions"]) is not int:
        raise ValueError("maximum actions must be an integer")
    for name in ("max_actions", "timeout_s"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError("invalid episode budget")


def inspect_assets(config, assets_root, asset_audit, *, native_only=False):
    robots = ("aloha-agilex",) if native_only else ("aloha-agilex", "ARX-X5")
    for robot in robots:
        load_arms(assets_root,asset_audit,robot,config["calibration"][robot])
    limits=config["control_limits"]
    QuinticMotion(config["control_period_s"],limits["arm_velocity_rad_s"],limits["arm_acceleration_rad_s2"],
                  limits["finger_velocity_m_s"],limits["finger_acceleration_m_s2"],
                  config["collision_joint_step_rad"],config["collision_finger_step_m"],
                  limits["source"],"preflight-only",max_ticks=config["max_plan_ticks"])
    profile = None if native_only else audited_arx_self_profile(assets_root)["sha256"]
    return {"calibration_assets_verified":True,"self_collision_policy_sha256":profile,"contact_dynamics_verified":False}


@contextmanager
def pace_context(pace_root):
    previous_cwd, previous_path = Path.cwd(), sys.path[:]
    try:
        os.chdir(pace_root)
        sys.path[:0] = [str(pace_root), str(pace_root / "pace/src"), str(pace_root / "description/utils")]
        yield
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


class RuntimeSession:
    """A single new scene, owned by an episode; factories may be injected for CPU tests."""
    def __init__(self, factory):
        self.factory, self.context, self.backend, self.adapter = factory, None, None, None

    def reset(self, spec):
        if self.backend is not None:
            raise RuntimeError("one reset/session per episode")
        context = self.factory(spec)
        self.backend, self.adapter = context.__enter__()
        self.context = context
        return self.backend.reset(spec)

    @property
    def execution_protocol(self):
        return getattr(self.backend,"execution_protocol",None)

    @property
    def native_limits(self):
        return self.backend.native_limits

    def capture(self):
        return self.backend.capture()

    def execute(self, *args):
        return self.backend.execute(*args)

    def take_replan(self):
        take=getattr(self.backend,"take_replan",None)
        return take() if callable(take) else None

    def success(self):
        return self.backend.success() if self.backend is not None else False

    def hold(self):
        if self.backend is not None:
            self.backend.hold()

    def close(self, output):
        context, self.context = self.context, None
        if context is None:
            return
        try:
            return self.backend.close(output)
        finally:
            context.__exit__(None, None, None)


class SessionAdapter:
    def __init__(self, session):
        self.session = session

    def observe(self, *args):
        return self.session.adapter.observe(*args)

    def begin_expert(self, *args):
        return self.session.adapter.begin_expert(*args)

    def actions(self, *args):
        return self.session.adapter.actions(*args)


@contextmanager
def real_scene(config, spec, pace_root, assets_root, asset_audit, source_audit, output, *, allow_native_h50=False):
    """Create the real renderer/scene only inside an already authorized allocation."""
    validate_config(config, allow_native_h50=allow_native_h50)
    verify_runtime(pace_root, source_audit)
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NAME") != "cpf-pace-g2":
        raise RuntimeError("real scene initialization requires the existing G2 allocation")
    embodiment = config["embodiment"]
    if embodiment not in ("aloha-agilex", "ARX-X5"):
        raise ValueError("unknown embodiment")
    world_task = matrix7(config["world_task"])
    calibration = deepcopy(config["calibration"])
    # Loading hash-checked kinematics before renderer creation catches missing calibration early.
    arms = load_arms(assets_root, asset_audit, embodiment, calibration[embodiment])
    output = Path(output).resolve()
    env = None
    handed_off = False
    with pace_context(pace_root):
        try:
            from pace.robotwin import load_environment, restore_cached_scene_state
            from envs.robot.robot import Robot
            import sapien

            env, settings = load_environment(TASK)
            settings.update(collect_data=False, save_data=False, render_freq=0, eval_video_log=False)
            settings.pop("eval_video_save_dir", None)
            if embodiment == "ARX-X5":
                audit_rows = json.loads(Path(asset_audit).read_text())["robots"]
                row, = [value for value in audit_rows if value["robot"] == embodiment]
                settings = arx_settings(settings, assets_root / "embodiments/ARX-X5",
                                        row["config_sha256"], config["arx_base_distance_m"])
                with joint_space_setup(Robot, [arm.grip for arm in arms]):
                    env.setup_demo(now_ep_num=0, seed=config["scene_seed"], is_test=True, **settings)
            else:
                env.setup_demo(now_ep_num=0, seed=config["scene_seed"], is_test=True, **settings)
            restore_cached_scene_state(TASK, env, config.get("episode_info", {}))
            env.set_instruction(spec["prompt"])
            actual = capture_arms(env.robot, Stamp(spec["episode_id"], 0, 0.))
            arms = tuple(replace(arm, instance=sample.instance) for arm, sample in zip(arms, actual))
            fixture = config.get("reset_fixture")
            if fixture is not None:
                apply_fixture(env, arms, fixture, embodiment, sapien.Pose)
                actual=capture_arms(env.robot,Stamp(spec["episode_id"],0,0.))
            initial = capture_initial(env, arms, world_task, spec["episode_id"],
                                      digest_json(fixture if fixture is not None else {"seed": config["scene_seed"]}),
                                      "diagnostic-unverified-workspace", workspace_pass=False)
            setup = dict(embodiment=embodiment, scene_seed=config["scene_seed"],
                         free_space_approach_repair=config.get("free_space_approach_repair", "disabled"),
                         config_sha256=digest_json(config), actual_initial=initial,
                         contact_mode=config["contact_mode"], formal_score_eligible=False,
                         calibration_ids=[arm.calibration_id for arm in arms])
            save_json(output / "setup.json", setup)
            video = Video(output / "head.mp4")
            if embodiment == "aloha-agilex":
                guard = None
                if config.get("execution_protocol") == PROTOCOL:
                    from g2e_native_guard import NativeGuard, NativeReleaseTracker
                    from g2_contact import ContactLimits
                    from mplib.sapien_utils.conversion import convert_object_name
                    entities=[env.robot.left_entity,env.robot.right_entity]
                    collision=JointCollisionWorld.from_scene(env.scene,entities,
                        [a.chain.names for a in arms],[a.grip.names for a in arms])
                    block=getattr(env.box,"actor",env.box)
                    tracker_type = (NativeReleaseTracker if config.get("native_release_semantics")
                                    else ContactTracker)
                    tracker=tracker_type.from_scene(env.robot,block,arms,ContactLimits())
                    actors=env.scene.get_all_actors()
                    collision=ContactWorld(collision,tracker,arms,
                        [np.linalg.inv(world_task)@sample.world_base for sample in actual],
                        convert_object_name(block),[convert_object_name(a) for a in actors],
                        [[entity.find_joint_by_name(n).child_link.name for n in arm.grip.names]
                         for entity,arm in zip(entities,arms)],
                        [convert_object_name(a) for a in actors if a.name in ("table","target_box")],world_task=world_task)
                    guard=NativeGuard(collision,arms,config["collision_joint_step_rad"],
                                      config["tracking_position_limit_m"],config["tracking_angle_limit_deg"])
                    setup.update(execution_protocol=PROTOCOL,contact_limits_id=tracker.limits.profile_id,
                                 native_release_semantics=config.get("native_release_semantics", "legacy_drive_delta"),
                                 native_gripper_scale_m={side:getattr(env.robot,side+"_gripper_scale") for side in SIDES})
                    save_json(output/"setup.json",setup)
                setup["native_preload_control"] = config.get("native_preload_control", "disabled")
                save_json(output/"setup.json", setup)
                setup["native_preload_establishment"] = config.get("native_preload_establishment", "disabled")
                save_json(output/"setup.json", setup)
                backend = NativeBackend(env, arms, world_task, initial, video,guard=guard,
                                        preload_hold=config.get("native_preload_control") is not None,
                                        contact_preload=config.get("native_preload_establishment") is not None,
                                        approach_repair=config.get("free_space_approach_repair") is not None)
                if guard is not None: backend.collision=guard.collision
                adapter = None
            else:
                entities = [env.robot.left_entity, env.robot.right_entity]
                setup['physics_self_collision_profile']=apply_audited_arx_physics(env.scene,entities,assets_root)
                collision = JointCollisionWorld.from_scene(env.scene, entities,
                                                            [arm.chain.names for arm in arms],
                                                            [arm.grip.names for arm in arms])
                collision.use_self_profile(audited_arx_self_profile(assets_root))
                collision.audit_path=output/"collision-checks.jsonl"
                setup["arx_self_collision_profile"]=collision.self_profile
                save_json(output/"setup.json",setup)
                source_arms = load_arms(assets_root, asset_audit, "aloha-agilex", calibration["aloha-agilex"])
                bases = [matrix7(value) for value in config["source_base_world"]]
                if not np.allclose(bases[0], bases[1], atol=1e-9, rtol=0):
                    raise ValueError("Aloha proxy requires its shared root")
                source_row, = [row for row in json.loads(Path(asset_audit).read_text())["robots"]
                                if row["robot"] == "aloha-agilex"]
                base_pose = pose7(bases[0])
                proxy = ProxyCollisionWorld(env.scene, assets_root / source_row["urdf_member"],
                                            source_row["urdf_sha256"], sapien.Pose(base_pose[:3], base_pose[3:]),
                                            source_arms, output / "proxy-collision.urdf")
                proxy.checker.audit_path=output/"proxy-collision-checks.jsonl"
                contact_drive=None
                if config["contact_mode"]==CONTACT_MODE:
                    from mplib.sapien_utils.conversion import convert_object_name
                    block=getattr(env.box,"actor",env.box)
                    block_body=block.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                    if block_body is None:raise ValueError("task block has no dynamic mass")
                    actual_mass=float(block_body.mass)
                    contact_limits,contact_drive=contact_profile(config,actual_mass)
                    tracker=ContactTracker.from_scene(env.robot,block,arms,contact_limits)
                    objects=env.scene.get_all_actors()
                    environment=[convert_object_name(actor) for actor in objects]
                    supports=[convert_object_name(actor) for actor in objects
                              if actor.name in ("table","target_box")]
                    frames=[np.linalg.inv(world_task)@sample.world_base for sample in actual]
                    collision=ContactWorld(collision,tracker,arms,frames,convert_object_name(block),environment,
                                           [[entity.find_joint_by_name(name).child_link.name for name in arm.grip.names]
                                            for entity,arm in zip(entities,arms)],supports,world_task=world_task)
                    collision.giver=config["contact_profile"]["giver"]
                    source_collision=ContactWorld(proxy.checker,tracker,source_arms,
                                                  [np.linalg.inv(world_task)@base for base in bases],
                                                  convert_object_name(block),environment,
                                                  [[proxy.entity.find_joint_by_name(name).child_link.name for name in arm.grip.names]
                                                   for arm in source_arms],supports,world_task=world_task)
                    source_collision.refresh=proxy.refresh
                    setup.update(contact_profile=config["contact_profile"],contact_profile_id=collision.profile_id,
                                 grip_state_semantics=[arm.grip.state_mode for arm in arms],actual_block_mass_kg=actual_mass)
                    save_json(output/"setup.json",setup)
                else:
                    source_collision=proxy
                limits = config["control_limits"]
                motion = QuinticMotion(config["control_period_s"], limits["arm_velocity_rad_s"],
                                       limits["arm_acceleration_rad_s2"], limits["finger_velocity_m_s"],
                                       limits["finger_acceleration_m_s2"], config["collision_joint_step_rad"],
                                       config["collision_finger_step_m"], limits["source"],
                                       "sapien3.0.0b1_mplib0.2.1_real_scene_FCL", max_ticks=config["max_plan_ticks"])
                adapter = Adapter(source_arms, arms, bases, matrix7(config["source_world_task"]),
                                  config["source_proxy_seed_rad"], config["source_proxy_velocity_rad_s"],
                                  source_collision, collision, motion, grip_rule=config["grip_rule"],
                                  grip_max_clip=config["grip_max_clip"])
                backend = ARXBackend(env, arms, collision, world_task, env.scene.get_timestep(), initial,
                                     config["tracking_position_limit_m"], config["tracking_angle_limit_deg"], video,contact_drive)
                # Refresh both physical and virtual collision worlds before every observation.
                original_capture = backend.capture
                def capture():
                    proxy.refresh()
                    collision.refresh()
                    return original_capture()
                backend.capture = capture
            handed_off = True
            yield backend, adapter
        finally:
            if env is not None and not handed_off:
                try:
                    env.close_env(clear_cache=True)
                except Exception as error:
                    save_json(output / "setup-cleanup-error.json", {"error":repr(error)})
                    raise
