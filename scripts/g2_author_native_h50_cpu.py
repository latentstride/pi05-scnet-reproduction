"""Run the isolated H50 code on CPU PhysX and original TOPP with fixed chunks."""

import argparse
import ast
from contextlib import contextmanager, nullcontext
from functools import wraps
from importlib import import_module
import json
import os
from pathlib import Path
import sys
from time import monotonic
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from g2_adapter import G2Error
from g2_author_h50_reset_cpu import compiled, native_method, physical_factory
from g2_author_native_h50 import run_diagnostic
from g2_proxy_collision import collision_only_urdf
from robotwin_control_geometry import Status


class Headless:
    """Reuse the preflight's CPU-only physical/visual separation."""

    def __init__(self, output):
        self.output = output

    @contextmanager
    def __call__(self, root, audit):
        import sapien

        path = root / "envs/robot/planner.py"
        planner = ModuleType("envs.robot.planner")
        planner.__file__ = str(path)
        tree = ast.parse(path.read_text())
        (cuda_block,) = [
            n
            for n in tree.body
            if isinstance(n, ast.Try)
            and any(
                isinstance(x, ast.ClassDef) and x.name == "CuroboPlanner"
                for x in n.body
            )
        ]
        tree.body.remove(cuda_block)
        exec(compile(tree, str(path), "exec"), vars(planner))

        class ResetGripper:
            def __init__(self, *_args, **_kwargs):
                pass

            plan_grippers = compiled(
                native_method(path, "CuroboPlanner", "plan_grippers"), path, {"np": np}
            )

        planner.CuroboPlanner = ResetGripper
        sys.modules[planner.__name__] = planner
        base = import_module("envs._base_task")
        actor = import_module("envs.utils.create_actor")
        camera_type = import_module("envs.camera").Camera
        self.audit = json.loads(audit.read_text())
        self.root = root

        def setup_scene(env):
            env.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
            env.scene.set_timestep(1 / 250)
            env.scene.add_ground(0, render=False)
            env.scene.default_physical_material = env.scene.create_physical_material(
                0.5, 0.5, 0
            )

        def load_camera(env, **settings):
            env.cameras = camera_type(
                bias=env.table_z_bias,
                random_head_camera_dis=env.random_head_camera_dis,
                **settings,
            )

            def add_camera(**_settings):
                return SimpleNamespace(entity=SimpleNamespace(set_pose=lambda _: None))

            env.cameras.load_camera(SimpleNamespace(add_camera=add_camera))
            env.scene.step()

        def get_obs(env):
            state = np.array(
                env.robot.get_left_arm_jointState()
                + env.robot.get_right_arm_jointState()
            )
            return dict(
                joint_action={"vector": state},
                observation={
                    key: {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)}
                    for key in ("head_camera", "left_camera", "right_camera")
                },
            )

        with (
            patch.object(base.Base_Task, "setup_scene", setup_scene),
            patch.object(base.Base_Task, "load_camera", load_camera),
            patch.object(base.Base_Task, "_update_render", lambda _: None),
            patch.object(base.Base_Task, "get_obs", get_obs),
            patch.object(
                actor, "create_entity_box", physical_factory(actor, "create_entity_box")
            ),
            patch.object(base, "create_table", physical_factory(actor, "create_table")),
        ):
            yield

    def settings(self, settings):
        row = next(
            item for item in self.audit["robots"] if item["robot"] == "aloha-agilex"
        )
        urdf = self.root / "assets" / row["urdf_member"]
        collision = collision_only_urdf(
            urdf, row["urdf_sha256"], self.output / "aloha-collision.urdf"
        )
        for side in ("left", "right"):
            settings[f"{side}_embodiment_config"]["urdf_path"] = str(collision)


class FixedPolicy:
    def __init__(self):
        self.queries = 0

    def infer(self, request):
        rows = np.tile(request["state"], (50, 1))
        phase = np.arange(1, 51) + self.queries * 50
        rows[:, 0] = np.sin(phase / 5) * 0.015
        rows[:, 7] = np.sin(phase / 6) * 0.012
        rows[:, 6] = np.where(phase % 2, 1.001, 0.98)
        rows[:, 13] = 0.99
        self.queries += 1
        return {"actions": rows}


def fault(mode, backend):
    print("CPU_PHYSX_RESET_ADMITTED", mode, flush=True)
    for side in ("left", "right"):
        planner = getattr(backend.env.robot, f"{side}_mplib_planner")

        def traced(fn, side):
            @wraps(fn)
            def topp(*args, **kwargs):
                try:
                    result = fn(*args, **kwargs)
                except Exception as error:
                    backend.log(
                        dict(
                            kind="cpu_topp",
                            side=side,
                            index=backend.action_index,
                            status="fallback_exception",
                            reason=type(error).__name__,
                        )
                    )
                    raise
                steps = int(result[1].shape[0])
                backend.log(
                    dict(
                        kind="cpu_topp",
                        side=side,
                        index=backend.action_index,
                        status="trajectory" if steps else "fallback_zero",
                        steps=steps,
                    )
                )
                return result

            return topp

        planner.TOPP = traced(planner.TOPP, side)
    if mode in ("mid", "tick-zero", "missing"):
        original = backend.guard.candidate

        def candidate(*args):
            original(*args)
            if backend.action_index == (51 if mode == "mid" else 0):
                if mode == "missing":
                    # Explicitly simulate unavailability. The production read
                    # still runs against the true physical scene in other modes.
                    backend.guard.collision.tracker.last_stamp = None
                raise G2Error(
                    Status.COLLISION,
                    "CPU injected candidate rejection",
                    geometry="candidate",
                )

        backend.guard.candidate = candidate
    elif mode == "actual":
        original = backend.guard.measured

        def measured(*args):
            original(*args)
            raise G2Error(
                Status.COLLISION,
                "CPU injected real-danger classification",
                geometry="actual",
            )

        backend.guard.measured = measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pace", "release", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--case", choices=("mid", "tick-zero", "missing", "actual"), required=True
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU probe requires hidden CUDA devices")
    receipt = run_diagnostic(
        args.pace.resolve(),
        args.release.resolve(),
        args.reference.resolve(),
        args.output.resolve(),
        "cpu-" + args.case,
        lambda: nullcontext(FixedPolicy()),
        deadline=monotonic() + 1080,
        configure_cpu=Headless(args.output.resolve()),
        inject=lambda backend: fault(args.case, backend),
    )
    print(
        "CPU_PHYSX_RESULT",
        args.case,
        receipt["category"],
        receipt.get("completed_actions"),
        receipt.get("physics_steps"),
        flush=True,
    )
    expected = {
        "mid": "controlled_candidate_stop",
        "tick-zero": "controlled_candidate_stop",
        "missing": "infrastructure_failure",
        "actual": "actual_safety_stop",
    }
    if receipt["category"] != expected[args.case]:
        raise AssertionError(receipt)


if __name__ == "__main__":
    main()
