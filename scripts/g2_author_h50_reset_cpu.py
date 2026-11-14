"""Read the author's scene 100000 in CPU PhysX without a renderer or model."""

import argparse
import ast
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from g2_adapter import Snapshot, Stamp, read_snapshot
from g2_assets import load_arms
from g2_capture import capture_arms
from g2_collision import JointCollisionWorld
from g2_contact import ContactLimits, ContactWorld
from g2_initial import capture_initial, digest_json
from g2_proxy_collision import collision_only_urdf

PACE = "fd2e624cc19325cc9b20a620c82d51c0d8e25cc1"
TASK = "handover_block"
SEED = 100000
SIDES = ("left", "right")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def native_method(path, cls, name):
    tree = ast.parse(path.read_text())
    owner = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == cls
    )
    return next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def compiled(node, path, scope):
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[node.name]


def physical_factory(module, name):
    """Remove only the verified create_actor.py render statements."""
    path = Path(module.__file__)
    fn = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    kept = []
    for stmt in fn.body:
        if (
            isinstance(stmt, ast.If)
            and ast.unparse(stmt.test) == "texture_id is not None"
        ):
            continue
        if (
            isinstance(stmt, ast.Assign)
            and ast.unparse(stmt.targets[0]) == "render_component"
        ):
            continue
        if isinstance(stmt, ast.Expr) and ast.unparse(stmt.value).startswith(
            (
                "render_component.attach(",
                "entity.add_component(render_component)",
                "builder.add_box_visual(",
            )
        ):
            continue
        if name == "create_table" and isinstance(stmt, ast.For):
            for inner in stmt.body:
                if isinstance(inner, ast.For):
                    inner.body = [
                        item
                        for item in inner.body
                        if not (
                            isinstance(item, ast.Expr)
                            and ast.unparse(item.value).startswith(
                                "builder.add_box_visual("
                            )
                        )
                    ]
        kept.append(stmt)
    fn.body = kept
    assert "sapien.render" not in ast.unparse(
        fn
    ) and "add_box_visual" not in ast.unparse(fn)
    return compiled(fn, path, vars(module).copy())


def run(args):
    import sapien
    from mplib.sapien_utils.conversion import convert_object_name

    root, release, out = (
        getattr(args, key).resolve() for key in ("pace", "release", "output")
    )
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "pace/src"), str(root / "description/utils")]
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == PACE
    )
    source_paths = (
        "envs/_base_task.py",
        "envs/handover_block.py",
        "envs/camera/camera.py",
        "envs/robot/robot.py",
        "envs/robot/planner.py",
        "envs/utils/create_actor.py",
        "envs/utils/actor_utils.py",
        "envs/utils/rand_create_actor.py",
        "pace/src/pace/robotwin.py",
        "pace/configs/expert_scenes.json",
        "description/utils/generate_episode_instructions.py",
        "description/task_instruction/handover_block.json",
        "task_config/demo_clean.yml",
        "task_config/_camera_config.yml",
        "task_config/_eval_step_limit.yml",
    )
    for path in source_paths:
        assert (root / path).read_bytes() == subprocess.check_output(
            ["git", "show", f"{PACE}:{path}"]
        ), path
    from pace.robotwin import (
        load_cached_scenes,
        load_environment,
        restore_cached_scene_state,
        sample_instruction,
    )

    planner = root / "envs/robot/planner.py"

    class ResetGripper:
        def __init__(self, *_args, **_kwargs):
            pass

        plan_grippers = compiled(
            native_method(planner, "CuroboPlanner", "plan_grippers"),
            planner,
            {"np": np},
        )

    # Loading the original optional CuRobo module evaluates CUDA defaults at
    # import time. In this process compile its unchanged CPU MPLib path and
    # expose only the original NumPy gripper interpolation for reset.
    planner_module = ModuleType("envs.robot.planner")
    planner_module.__file__ = str(planner)
    tree = ast.parse(planner.read_text())
    (cuda_block,) = [
        node
        for node in tree.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(item, ast.ClassDef) and item.name == "CuroboPlanner"
            for item in node.body
        )
    ]
    tree.body.remove(cuda_block)
    exec(compile(tree, str(planner), "exec"), vars(planner_module))
    planner_module.CuroboPlanner = ResetGripper
    sys.modules[planner_module.__name__] = planner_module
    print("CPU_SOURCE_AND_PLANNERS_VERIFIED", flush=True)

    from envs._base_task import Base_Task
    from envs.camera import Camera
    from g2e_native_guard import NativeReleaseTracker

    cached, cache = load_cached_scenes(TASK, 0, 1)
    assert cached == [{"scene_seed": SEED, "episode_info": {}}]
    audit = release / "asset-audit.json"
    assets = root / "assets"
    (row,) = [
        item
        for item in json.loads(audit.read_text())["robots"]
        if item["robot"] == "aloha-agilex"
    ]
    config = json.loads((release / "configs/unified-seed0-native-w50.json").read_text())
    arms = load_arms(
        assets, audit, "aloha-agilex", config["calibration"]["aloha-agilex"]
    )
    urdf = assets / row["urdf_member"]
    collision_urdf = collision_only_urdf(
        urdf, row["urdf_sha256"], out / "aloha-collision-only.urdf"
    )

    def setup_scene(env):
        env.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        env.scene.set_timestep(1 / 250)
        env.scene.add_ground(0, render=False)
        env.scene.default_physical_material = env.scene.create_physical_material(
            0.5, 0.5, 0
        )

    def load_camera(env, **settings):
        env.cameras = Camera(
            bias=env.table_z_bias,
            random_head_camera_dis=env.random_head_camera_dis,
            **settings,
        )

        def add_camera(**settings):
            camera = SimpleNamespace(settings=settings, pose=None)
            camera.entity = SimpleNamespace(
                set_pose=lambda pose: setattr(camera, "pose", pose)
            )
            return camera

        # Author code consumes RNG even for zero camera displacement.
        env.cameras.load_camera(SimpleNamespace(add_camera=add_camera))
        env.scene.step()

    def pose7(pose):
        return np.r_[pose.p, pose.q].tolist()

    author_actor = import_module("envs.utils.create_actor")
    base = import_module("envs._base_task")
    env, settings = load_environment(TASK)
    assert settings["embodiment"] == ["aloha-agilex"]
    assert (
        settings["camera"]["head_camera_type"]
        == settings["camera"]["wrist_camera_type"]
        == "D435"
    )
    assert not any(
        settings["domain_randomization"][key]
        for key in (
            "random_background",
            "cluttered_table",
            "random_head_camera_dis",
            "random_table_height",
            "random_light",
        )
    )
    authored_settings = {
        key: settings[key]
        for key in ("camera", "domain_randomization", "embodiment", "data_type")
    }
    for side in SIDES:
        settings[f"{side}_embodiment_config"]["urdf_path"] = str(collision_urdf)
    assert env.__class__.__module__ == "envs.handover_block"
    random.seed(SEED)
    with (
        patch.object(Base_Task, "setup_scene", setup_scene),
        patch.object(Base_Task, "load_camera", load_camera),
        patch.object(
            author_actor,
            "create_entity_box",
            physical_factory(author_actor, "create_entity_box"),
        ),
        patch.object(
            base, "create_table", physical_factory(author_actor, "create_table")
        ),
    ):
        try:
            print("CPU_AUTHOR_SETUP_BEGIN", flush=True)
            env.setup_demo(now_ep_num=0, seed=SEED, is_test=True, **settings)
            print("CPU_AUTHOR_SETUP_COMPLETE", flush=True)
            restore_cached_scene_state(TASK, env, cached[0]["episode_info"])
            env.set_instruction(
                sample_instruction(
                    TASK, {"info": cached[0]["episode_info"]}, 100, "unseen"
                )
            )
            assert (
                env.step_lim == 800
                and env.take_action_cnt == 0
                and not env.check_success()
            )
            stamp = Stamp("author-h50-cpu-reset", 0, 0.0)
            samples = capture_arms(env.robot, stamp)
            arms = tuple(
                replace(arm, instance=sample.instance)
                for arm, sample in zip(arms, samples)
            )
            world_task = np.eye(4)
            q, fingers, frames, tools = read_snapshot(
                Snapshot(stamp, samples, {}, world_task), arms
            )
            grip = [arm.grip.measure(v) for arm, v in zip(arms, fingers)]
            entities = [env.robot.left_entity, env.robot.right_entity]
            assert entities[0] is entities[1]
            scene = env.scene
            block = env.box.actor
            mass = float(
                block.find_component_by_type(
                    sapien.physx.PhysxRigidDynamicComponent
                ).mass
            )
            assert abs(mass - 0.01) <= 1e-8
            actors = scene.get_all_actors()
            checker = JointCollisionWorld.from_scene(
                scene,
                entities,
                [a.chain.names for a in arms],
                [a.grip.names for a in arms],
            )
            tracker = NativeReleaseTracker.from_scene(
                env.robot, block, arms, ContactLimits()
            )
            world = ContactWorld(
                checker,
                tracker,
                arms,
                frames,
                convert_object_name(block),
                [convert_object_name(a) for a in actors],
                [
                    [
                        entity.find_joint_by_name(n).child_link.name
                        for n in arm.grip.names
                    ]
                    for entity, arm in zip(entities, arms)
                ],
                [
                    convert_object_name(a)
                    for a in actors
                    if a.name in ("table", "target_box")
                ],
                world_task=world_task,
            )
            world.refresh()
            clear = bool(world.measured_clear(q, fingers))
            stopped_clear = bool(world.measured_clear(q, fingers, check_unheld=True))
            initial = capture_initial(
                env,
                arms,
                world_task,
                stamp.episode,
                digest_json(cached[0]),
                "author-cpu-measured",
                False,
            )
            env.cameras.update_wrist_camera(
                env.robot.left_camera.get_pose(), env.robot.right_camera.get_pose()
            )
            cameras = [
                env.cameras.left_camera,
                env.cameras.right_camera,
                env.cameras.static_camera_list[env.cameras.head_camera_id],
            ]
            camera_records = [
                dict(camera.settings, world_pose=pose7(camera.pose))
                for camera in cameras
            ]
            assert all((c["width"], c["height"]) == (320, 240) for c in camera_records)
            state = (
                env.robot.get_left_arm_jointState()
                + env.robot.get_right_arm_jointState()
            )
            result = dict(
                scope="Real CPU PhysX physical readback from author setup_demo with headless adapters; no images",
                cache={"sha256": cache["sha256"], "shard": 0, "index": 0, **cached[0]},
                checkpoint_loaded=False,
                gpu_used=False,
                model_requests=0,
                source_revision=PACE,
                source_sha256={path: digest(root / path) for path in source_paths},
                asset_sha256={
                    row["urdf_member"]: digest(urdf),
                    row["archive_config_member"]: digest(
                        assets / row["archive_config_member"]
                    ),
                },
                timestep_s=scene.get_timestep(),
                sapien=version("sapien"),
                mplib=version("mplib"),
                authored_settings=authored_settings,
                reconstruction_python_seed=SEED,
                instruction=env.get_instruction(),
                task_success_at_reset=False,
                task_action_count=0,
                initial=initial,
                block_mass_kg=mass,
                normalized_gripper_measurements=grip,
                actual_fingers_m=fingers.tolist(),
                measured_tcp_world=tools.tolist(),
                model_state_from_native_getters=state,
                native_gripper_scale_m=[
                    env.robot.left_gripper_scale,
                    env.robot.right_gripper_scale,
                ],
                measured_fcl_clear=clear,
                stopped_geometry_clear_including_unheld_block=stopped_clear,
                fcl_collisions=world.last_collisions,
                cameras=camera_records,
                real_rgb_available=False,
                headless_substitutions=[
                    "CPU scene without visual systems",
                    "same camera setup with metadata-only camera handles",
                    "visual-only actor/URDF data omitted",
                    "CuRobo warmup replaced by author gripper method",
                ],
            )
        finally:
            env.close_env(clear_cache=False)
    result["author_close_env_returned"] = True
    (out / "readback.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print("CPU_AUTHOR_RESET_COMPLETE", clear, stopped_clear, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pace", "release", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
