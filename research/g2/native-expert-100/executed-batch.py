"""Prepare, replay and audit native cached handover_block trajectories on CPU."""
import argparse
import ast
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zlib

import numpy as np

ROBOTS = ("aloha-agilex", "arx-x5")
REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
SOURCE = "ea8b21121ebb3cd201ff5b3fe361944ac94eda3f"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class EvidenceError(ValueError):
    def __init__(self, category, detail):
        super().__init__(detail)
        self.category = category


def prepare(args):
    """Read existing HDF5 targets only. Preserve the original file hash."""
    import h5py
    from g2_audit_expert_metadata import NumpyOnlyUnpickler, spawn, timeline
    from g2_expert_dense import dense

    manifests = [read(path) for path in args.manifest]
    roster = {}
    for manifest in manifests:
        if manifest["revision"] != REVISION:
            raise ValueError("HDF5 manifest revision differs")
        for row in manifest["records"]:
            if row.get("crc_verified") and row.get("sha256"):
                key = row["robot"], row["episode"]
                if key in roster and roster[key]["sha256"] != row["sha256"]:
                    raise ValueError("conflicting HDF5 manifests")
                roster[key] = row
    for robot in ROBOTS:
        cache = args.cache / robot
        cache_manifest = read(cache / "manifest.json") if (cache / "manifest.json").exists() else {}
        for episode in args.episodes:
            out = args.output / robot / f"episode{episode:02}"
            out.mkdir(parents=True, exist_ok=True)
            if (out / "input.json").is_file():
                continue
            meta = dict(robot=robot, episode=episode, dataset_revision=REVISION)
            category = "RAW_DATA_MISSING_OR_CORRUPT"
            try:
                hdf5 = args.raw / robot / f"episode{episode}.hdf5"
                origin = roster[robot, episode]
                meta["hdf5_sha256"] = sha(hdf5)
                if meta["hdf5_sha256"] != origin["sha256"]:
                    raise ValueError("existing HDF5 differs from verified archive member")
                meta["hdf5_member"] = origin["member"]
                category = "CONTROL_CACHE_OR_TIMING"
                if cache_manifest.get("revision") != REVISION:
                    raise ValueError("missing fixed planner and seed manifest")
                planner = cache / f"episode{episode}.pkl"
                entry = cache_manifest["members"][planner.name]
                meta["pkl_sha256"] = sha(planner)
                if meta["pkl_sha256"] != entry["sha256"] or f"{zlib.crc32(planner.read_bytes()):08x}" != entry["zip_crc32"]:
                    raise ValueError("planner member hash or ZIP CRC differs")
                meta["pkl_member"] = entry["member"]
                seed_file = cache / "seed.txt"
                meta["seed_file_sha256"] = sha(seed_file)
                if meta["seed_file_sha256"] != cache_manifest["members"]["seed.txt"]["sha256"]:
                    raise ValueError("seed list digest differs")
                meta["seed"] = int(seed_file.read_text().split()[episode])
                with planner.open("rb") as stream:
                    paths = NumpyOnlyUnpickler(stream).load()
                if set(paths) != {"left_joint_path", "right_joint_path"} or [len(paths[k]) for k in ("left_joint_path", "right_joint_path")] != [7, 4]:
                    raise ValueError("planner cache differs from handover stage structure")
                for path in paths["left_joint_path"] + paths["right_joint_path"]:
                    q, v = np.asarray(path["position"]), np.asarray(path["velocity"])
                    if (path["status"] != "Success" or q.ndim != 2 or q.shape[1] != 6 or not len(q)
                            or q.shape != v.shape or not np.isfinite(q).all() or not np.isfinite(v).all()):
                        raise ValueError("invalid planner position/velocity array")
                with h5py.File(hdf5) as file:
                    targets = file["joint_action/vector"][:]
                    parts = [file[f"joint_action/{side}_{field}"][:] for side in ("left", "right") for field in ("arm", "gripper")]
                    if not np.array_equal(targets, np.column_stack(parts)):
                        raise ValueError("recorded joint group order differs from state14")
                    ee = np.stack([file[f"endpose/{side}_endpose"][:] for side in ("left", "right")], axis=1)
                rebuilt, ticks, segments = timeline(paths, targets[0])
                controls, _ = dense(paths, targets[0])
                if not np.array_equal(rebuilt, targets) or not np.array_equal(controls["target"][ticks], targets):
                    raise ValueError("cached controls do not reproduce every recorded command")
                # Keep precisely the updates from the saved planner. Other joints retain drive targets.
                arm_active = np.zeros((len(controls["target"]), 2), dtype=bool)
                for s in segments:
                    kind, first, last = s["kind"], s["first_tick"], s["last_tick"]
                    if kind in ("L", "R"):
                        arm_active[first + 1:last + 1, int(kind == "R")] = True
                    elif kind == "B":
                        left, right = [len(paths[f"{side}_joint_path"][i]["position"]) for side, i in zip(("left", "right"), s["path_index"])]
                        li = ri = 0
                        for tick in range(first + 1, last + 1):
                            if li < left and li / left <= ri / right:
                                arm_active[tick, 0] = True
                                li += 1
                            if ri < right and ri / right <= li / left:
                                arm_active[tick, 1] = True
                                ri += 1
                        if (li, ri) != (left, right):
                            raise ValueError("joint-path synchronization did not consume cache")
                np.savez_compressed(out / "controls.npz", **controls, arm_active=arm_active,
                                    frame_ticks=ticks, expected_endpose=ee)
                meta.update(prepared=True, frames=len(ticks), steps=int(ticks[-1]), segments=segments,
                            physics_dt_s=.004, time_semantics="derived from cached control schedule",
                            source_revision=SOURCE, exact_recorded_targets=True,
                            controls_sha256=sha(out / "controls.npz"),
                            spawn_pose=spawn(meta["seed"], cameras=2 if robot == ROBOTS[0] else 1))
            except (OSError, ValueError, KeyError, EOFError) as exc:
                meta.update(prepared=False, category=category, detail=str(exc))
            write(out / "input.json", meta)
            print(robot, episode, meta.get("steps", meta.get("category")), flush=True)


def original_methods(path, class_name, names):
    tree = ast.parse(path.read_text())
    cls, = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in methods} != set(names):
        raise ValueError(f"missing expected official methods in {path.name}")
    namespace = {"np": np}
    import sapien
    namespace["sapien"] = sapien
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def physics(args):
    import sapien
    import yaml
    from g2_proxy_collision import collision_only_urdf
    from robotwin_control_geometry import pose_from_wxyz

    meta = read(args.input / "input.json")
    source_manifest = read(args.source / "manifest.json")
    if source_manifest["revision"] != SOURCE:
        raise ValueError("wrong pinned reference code")
    recorded_source = {item["path"]: item.get("sha256") for item in source_manifest["files"]}
    task_path = args.source / "envs/handover_block.py"
    robot_path = args.source / "envs/robot/robot.py"
    reference_paths = ["envs/handover_block.py", "envs/robot/robot.py", "envs/_base_task.py",
                       "envs/utils/create_actor.py", "envs/utils/actor_utils.py",
                       "envs/utils/rand_create_actor.py", "envs/camera/camera.py", "envs/robot/planner.py"]
    source_hashes = {name: sha(args.source / name) for name in reference_paths}
    if source_hashes != {name: recorded_source[name] for name in reference_paths}:
        raise ValueError("official source differs from the recorded manifest")
    check_success = original_methods(task_path, "handover_block", ["check_success"])["check_success"]
    methods = original_methods(robot_path, "Robot", ["_entity_qf", "set_arm_joints", "set_gripper",
                               "get_normal_real_gripper_val", "is_right_gripper_open"])
    with np.load(args.input / "controls.npz", allow_pickle=False) as file:
        data = dict(file)
    if sha(args.input / "controls.npz") != meta["controls_sha256"]:
        raise ValueError("prepared control array digest differs")
    audit = read(args.audit)
    name = "ARX-X5" if meta["robot"] == "arx-x5" else meta["robot"]
    entry, = [item for item in audit["robots"] if item["robot"] == name]
    config_path = args.assets / entry["archive_config_member"]
    if sha(config_path) != entry["config_sha256"]:
        raise ValueError("embodiment configuration hash differs")
    cfg = yaml.safe_load(config_path.read_text())
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(.004)
    material = scene.create_physical_material(.5, .5, 0.)
    scene.default_physical_material = material
    scene.add_ground(0, render=False)
    wall = scene.create_actor_builder()
    wall.add_box_collision(half_size=[3., .6, 1.5], material=material)
    wall.build_static(name="wall").set_pose(sapien.Pose([0., 1., 1.5]))
    builder = scene.create_actor_builder()
    builder.add_box_collision(pose=sapien.Pose([0., 0., -.025]), half_size=[.6, .35, .025], material=material)
    for x in (-.55, .55):
        for y in (-.30, .30):
            builder.add_box_collision(pose=sapien.Pose([x, y, -.372]), half_size=[.05, .05, .368])
    builder.build_static(name="table").set_pose(sapien.Pose([0., 0., .74]))

    urdf = collision_only_urdf(args.assets / entry["urdf_member"], entry["urdf_sha256"], args.output / "robot.urdf")
    entities = []
    roots = []
    dual = cfg["dual_arm"]
    sides = ("left", "right")
    for index in range(1 if dual else 2):
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        entity = loader.load(str(urdf))
        entity.set_name("native_aloha" if dual else f"native_arx_{sides[index]}")
        root = list(cfg["robot_pose"][0])
        if not dual:
            root[0] += (-.3, .3)[index]
        entity.set_root_pose(sapien.Pose(root[:3], root[3:]))
        entities.append(entity)
        roots.append(root)
        for link in entity.get_links():
            link.set_mass(1.)
        for joint in entity.get_active_joints():
            joint.set_drive_property(stiffness=cfg.get("joint_stiffness", 1000),
                                     damping=cfg.get("joint_damping", 200))
    if dual:
        entities *= 2
    robot_type = type("NativeRobotMethods", (), methods)
    robot = robot_type()
    ends, selectors, fingers, groups = [], [], {}, []
    for index, (side, entity) in enumerate(zip(sides, entities)):
        arm_names = cfg["arm_joints_name"][index]
        gripper = cfg["gripper_name"][index]
        grip_names = [(gripper["base"], 1., 0.), *gripper["mimic"]]
        grip = [(entity.find_joint_by_name(key), scale, offset) for key, scale, offset in grip_names]
        arm = [entity.find_joint_by_name(key) for key in arm_names]
        setattr(robot, f"{side}_entity", entity)
        setattr(robot, f"{side}_arm_joints", arm)
        setattr(robot, f"{side}_gripper", grip)
        setattr(robot, f"{side}_gripper_val", 0.)
        setattr(robot, f"{side}_gripper_scale", cfg["gripper_scale"])
        joints = [*arm, *(item[0] for item in grip)]
        groups.extend(joints)
        all_names = [joint.name for joint in entity.get_active_joints()]
        selectors.append([all_names.index(joint.name) for joint in joints])
        ends.append(entity.find_joint_by_name(cfg["ee_joints"][index]))
        for bit, (joint, _, _) in enumerate(grip):
            joint.set_drive_property(stiffness=cfg.get("gripper_stiffness", 1000),
                                     damping=cfg.get("gripper_damping", 200))
            fingers[joint.child_link.entity.per_scene_id] = 1 << (index * 2 + bit)
    scene.step()  # native camera setup advances one step
    for index, side in enumerate(sides):
        for joint, value in zip(getattr(robot, f"{side}_arm_joints"), cfg["homestate"][index]):
            joint.set_drive_target(value)
    for value in np.r_[np.linspace(0., 1., 200), np.ones(100)]:
        for side in sides:
            robot.set_gripper(value, side, .005)
        scene.step()

    objects = {}
    for key, half, moving in (("box", [.03, .03, .1], True), ("target_box", [.05, .05, .005], False)):
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=half, material=material)
        actor = builder.build(name=key) if moving else builder.build_static(name=key)
        pose = meta["spawn_pose"][key]
        actor.set_pose(sapien.Pose(pose["position_m"], pose["quaternion_wxyz"]))
        objects[key] = actor
    block, target = objects["box"], objects["target_box"]
    body = block.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    body.mass = .01  # Wrapper's original default, verified in the pinned actor source.
    block_id, target_id = block.per_scene_id, target.per_scene_id
    metric_rotation = np.array(cfg["global_trans_matrix"]) @ np.array(cfg["delta_matrix"])
    for _ in range(2500):
        scene.step()
    def pose7(actor):
        return np.r_[actor.pose.p, actor.pose.q].astype(float)
    def ee_pose():
        result = []
        for joint in ends:
            matrix = joint.global_pose.to_transformation_matrix().astype(float)
            matrix[:3, :3] = matrix[:3, :3] @ metric_rotation
            matrix[:3, 3] += matrix[:3, :3] @ [cfg["gripper_bias"] - .12, 0, 0]
            result.append(matrix)
        return np.array(result)
    expected = np.array([[pose_from_wxyz(row[:3], row[3:]) for row in pair] for pair in data["expected_endpose"]])
    start_error = np.linalg.norm(ee_pose()[:, :3, 3] - expected[0, :, :3, 3], axis=1)
    begin = dict(roots=roots, block_pose=pose7(block).tolist(), target_pose=pose7(target).tolist(),
                 actual_qpos=[e.get_qpos().tolist() for e in entities], initial_ee_error_m=start_error.tolist())
    write(args.output / "initial.json", begin)
    if max(start_error) >= .001:
        raise EvidenceError("SCENE_INITIALIZATION", f"original initial EE differs by {start_error.tolist()} m")
    actors = SimpleNamespace(
        box=SimpleNamespace(get_functional_point=lambda *_: block.pose * sapien.Pose([0, 0, -.1])),
        target_box=SimpleNamespace(get_functional_point=lambda *_: target.pose * sapien.Pose([0, 0, .005])),
        is_right_gripper_open=robot.is_right_gripper_open)
    actual, drive, block_pose, endpose, contact, requested = [], [], [], [], [], []
    first = dict(left_bilateral_lift=None, both_bilateral=None, right_bilateral_after_shared=None, target_contact=None)
    base_height = float(block.pose.p[2])
    shared = False
    for tick in range(meta["steps"] + 1):
        if tick:
            q = data["target"][tick]
            for i, side in enumerate(sides):
                if data["arm_active"][tick, i]:
                    robot.set_arm_joints(q[i*7:i*7+6], data["arm_velocity"][tick, i], side)
                if data["gripper_active"][tick, i]:
                    robot.set_gripper(q[i*7+6], side, data["gripper_eps"][tick, i])
            scene.step()
        actual.append(np.concatenate([e.get_qpos()[idx] for e, idx in zip(entities, selectors)]))
        drive.append([float(j.get_drive_target()[0]) for j in groups])
        block_pose.append(pose7(block))
        endpose.append(ee_pose())
        requested.append([float(robot.left_gripper_val), float(robot.right_gripper_val)])
        mask = 0
        depth = 0.
        impulses = np.zeros(4)
        for pair in scene.get_contacts():
            one, two = (item.entity.per_scene_id for item in pair.bodies)
            if block_id not in (one, two):
                continue
            other = two if one == block_id else one
            impulse = sum(float(np.linalg.norm(point.impulse)) for point in pair.points)
            depth = max(depth, max((-float(point.separation) for point in pair.points), default=0.))
            if impulse > 0.:
                if other == target_id:
                    mask |= 16
                elif other in fingers:
                    bit = fingers[other]
                    mask |= bit
                    impulses[bit.bit_length()-1] += impulse
        contact.append([mask, depth, *impulses])
        if mask & 3 == 3 and block_pose[-1][2] - base_height >= .02 and first["left_bilateral_lift"] is None:
            first["left_bilateral_lift"] = tick
        if mask & 15 == 15:
            shared = True
            if first["both_bilateral"] is None:
                first["both_bilateral"] = tick
        if shared and mask & 15 == 12 and first["right_bilateral_after_shared"] is None:
            first["right_bilateral_after_shared"] = tick
        if mask & 16 and first["target_contact"] is None:
            first["target_contact"] = tick
    arrays = dict(actual_qpos=np.array(actual), native_drive=np.array(drive),
                  block_pose=np.array(block_pose), actual_endpose_matrix=np.array(endpose),
                  contacts=np.array(contact), normalized_gripper_request=np.array(requested),
                  frame_ticks=data["frame_ticks"], joint_names=np.array([f"{side}:{joint.name}"
                      for side, group in zip(sides, (groups[:8], groups[8:])) for joint in group]))
    if not all(np.isfinite(value).all() for key, value in arrays.items() if key != "joint_names"):
        raise EvidenceError("SIMULATION_OR_DYNAMICS", "non-finite physical result")
    np.savez_compressed(args.output / "physics-trace.npz", **arrays)
    observed = arrays["actual_endpose_matrix"][data["frame_ticks"]]
    errors = np.linalg.norm(observed[:, :, :3, 3] - expected[:, :, :3, 3], axis=2)
    rot = np.swapaxes(expected[:, :, :3, :3], -1, -2) @ observed[:, :, :3, :3]
    angle = np.rad2deg(np.arccos(np.clip((np.trace(rot, axis1=-2, axis2=-1)-1)/2, -1, 1)))
    bottom = actors.box.get_functional_point(0, "pose").p
    top = actors.target_box.get_functional_point(1, "pose").p
    success = bool(check_success(actors))
    return dict(native_success=success, physical_steps=tick, expected_steps=meta["steps"],
                source_hashes=source_hashes, simulator=version("sapien"), numpy=version("numpy"),
                control="original native cached drives", physics_system="PhysxCpuSystem",
                flags=dict(renderer=False, gpu=False, policy=False, cross_embodiment=False),
                physics_config=dict(dt_s=.004, material=[.5,.5,0.], mass_kg=float(body.mass),
                                    link_mass_kg=1., robot_config_sha256=entry["config_sha256"],
                                    urdf_sha256=entry["urdf_sha256"], native_gripper_scale=cfg["gripper_scale"],
                                    initial_seed=meta["seed"], camera_rng_draws=2 if dual else 1,
                                    initialization_steps=2801, arx_base_separation_m=None if dual else .6,
                                    collision="native URDF defaults and task geometry"),
                initialization=begin, final=dict(block_pose=pose7(block).tolist(), target_pose=pose7(target).tolist(),
                                        qpos=arrays["actual_qpos"][-1].tolist(), block_velocity=np.r_[body.linear_velocity,body.angular_velocity].tolist(),
                                        functional_delta_m=(bottom-top).tolist(), right_gripper_request=requested[-1][1]),
                contacts=first, contact_trace_columns=["finger_and_tray_mask", "block_max_depth_m",
                                            "left_0_Ns","left_1_Ns","right_0_Ns","right_1_Ns"],
                ee_error=dict(max_m=float(errors.max()), max_deg=float(angle.max()), first_frame_m=start_error.tolist(),
                              within_prior_tolerance=bool(errors.max() < .001 and angle.max() < .1)),
                physics_trace_sha256=sha(args.output / "physics-trace.npz"))


def worker(args):
    args.output.mkdir(parents=True, exist_ok=True)
    meta = read(args.input / "input.json")
    receipt = dict(input=meta, physical_steps=0, native_success=None)
    try:
        if not meta["prepared"]:
            raise EvidenceError(meta["category"], meta["detail"])
        receipt.update(physics(args))
        receipt["status"] = "SUCCESS" if receipt["native_success"] else "TASK_FAILED"
        receipt["category"] = None if receipt["native_success"] else "NORMAL_TASK_FAILURE"
    except EvidenceError as exc:
        receipt.update(status="NOT_REPRODUCIBLE", category=exc.category, detail=str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        receipt.update(status="NOT_REPRODUCIBLE", category="SIMULATION_OR_DYNAMICS",
                       detail=f"{type(exc).__name__}: {exc}")
    write(args.output / "receipt.json", receipt)
    print(meta["robot"], meta["episode"], receipt["status"], receipt["physical_steps"], flush=True)


def batch(args):
    """Each isolated process writes its own receipt. Existing attempts remain intact."""
    for robot in args.robots:
        for episode in args.episodes:
            name = Path(robot) / f"episode{episode:02}"
            data = args.input / name
            out = args.output / name
            out.mkdir(parents=True, exist_ok=True)
            receipt = out / "receipt.json"
            if receipt.exists():
                continue
            attempt = out / "attempt"
            attempt.mkdir()
            try:
                with (attempt / "worker.log").open("wb") as log:
                    result = subprocess.run([sys.executable, "-B", __file__, "worker", str(data), str(attempt),
                            str(args.assets), str(args.audit), str(args.source)], stdout=log,
                            stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
                if result.returncode or not (attempt / "receipt.json").is_file():
                    raise RuntimeError(f"worker exited {result.returncode} without a complete result")
                (attempt / "receipt.json").replace(receipt)
            except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
                meta = read(data / "input.json") if (data / "input.json").is_file() else dict(robot=robot, episode=episode)
                write(receipt, dict(input=meta, physical_steps=0, native_success=None,
                     status="NOT_REPRODUCIBLE", category="SIMULATION_OR_DYNAMICS", detail=str(exc)))
            print(robot, episode, read(receipt)["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for key in ("raw", "cache", "output"):
        prep.add_argument(key, type=Path)
    prep.add_argument("--manifest", nargs="+", type=Path, required=True)
    prep.add_argument("--episodes", nargs="+", type=int, default=list(range(50)))
    for mode in ("worker", "batch"):
        p = sub.add_parser(mode)
        for key in ("input", "output", "assets", "audit", "source"):
            p.add_argument(key, type=Path)
        if mode == "batch":
            p.add_argument("--robots", nargs="+", choices=ROBOTS, default=ROBOTS)
            p.add_argument("--episodes", nargs="+", type=int, default=list(range(50)))
            p.add_argument("--timeout", type=float, default=150.)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
