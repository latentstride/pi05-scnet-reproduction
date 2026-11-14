"""Rebuild verified request transforms on CPU using archived GPU exchanges."""
import argparse
import ast
from collections import Counter
from dataclasses import dataclass, replace
from io import BytesIO
from itertools import product
import json
import logging
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import yaml
from PIL import Image

from archive import JOBS, Trial, digest, read
from g2_adapter import Adapter, ArmSample, Snapshot, Stamp, pose_error
from g2_arx_direct import kinematic_pair
from g2_assets import load_arms
from g2_capture import capture_snapshot
from pace_runtime import CAMERAS, msgpack_numpy, validate_request
from robotwin_control_geometry import pose_from_wxyz
from verify_robotwin_fk import Chain

ROOT = Path(__file__).resolve().parents[3]
NAMES = {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}
MODEL_NAMES = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
JOINT_MASK = np.array([True]*6 + [False] + [True]*6 + [False])
EXTRA_SOURCES = {
    "openpi-image_tools.py": ("Openpi-PACE", "f4e52e6de4cc487a11585a3a7414d610b2883dae",
                               "packages/openpi-client/src/openpi_client/image_tools.py"),
    "openpi-model.py": ("Openpi-PACE", "f4e52e6de4cc487a11585a3a7414d610b2883dae",
                         "src/openpi/models/model.py"),
    "openpi-tokenizer.py": ("Openpi-PACE", "f4e52e6de4cc487a11585a3a7414d610b2883dae",
                             "src/openpi/models/tokenizer.py"),
    "pace-camera.py": ("PACE", "fd2e624cc19325cc9b20a620c82d51c0d8e25cc1", "envs/camera/camera.py"),
}
OK = "INPUT_ADAPTER_VERIFIED"
UNKNOWN = "INSUFFICIENT_EVIDENCE"


def components(file, names, namespace):
    tree = ast.parse(file.read_text())
    definitions = [node for node in tree.body if
                   isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in definitions} != set(names):
        raise ValueError(f"missing pinned definitions in {file.name}")
    for node in definitions:
        if isinstance(node, ast.ClassDef):
            node.bases = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    program = ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[]))
    exec(compile(program, str(file), "exec"), namespace)
    return SimpleNamespace(**namespace)


def matrix(pose):
    return pose_from_wxyz(pose[:3], pose[3:])


def verify_sources(source):
    locked = read(ROOT / "research/g2/protocol-audit/source-lock.json")["files"]
    index = {}
    for file in sorted(p for p in source.iterdir() if p.suffix in {".py", ".yml"}):
        project, name = file.name.split("-", 1)
        prefix = "runtime/" if project == "openpi" else "pace/"
        entries = [e for e in locked if e["file"].startswith(prefix) and e["file"].endswith("/" + name)]
        item = {"sha256": digest(file.read_bytes()), "pinned_source_lock_match": None}
        if len(entries) == 1:
            item.update(url=entries[0]["url"], pinned_source_lock_match=item["sha256"] == entries[0]["sha256"])
            if not item["pinned_source_lock_match"]:
                raise ValueError(f"pinned checkout differs from source lock: {file.name}")
        elif file.name in EXTRA_SOURCES:
            project, rev, relative = EXTRA_SOURCES[file.name]
            item["url"] = f"https://raw.githubusercontent.com/niejunnan25/{project}/{rev}/{relative}"
        else:
            continue
        index[file.name] = item
    return index


def camera_stats(array):
    red, green, blue = array.astype(float)
    def mask(primary, a, b):
        return (primary > 100) & (primary > 1.7*a) & (primary > 1.7*b)
    red_mask = mask(red, green, blue)
    y, x = np.where(red_mask)
    return dict(shape=list(array.shape), dtype=str(array.dtype), array_sha256=digest(array.tobytes()),
                mean_rgb=array.mean(axis=(1, 2)).tolist(), red_pixels=int(red_mask.sum()),
                blue_pixels=int(mask(blue, green, red).sum()),
                red_bounds_xyxy=[int(x.min()), int(y.min()), int(x.max()), int(y.max())] if len(x) else None)


def camera_projection(event, images, camera, cfg, fovy):
    """Project the measured object through the fixed renderer mounts into archived images."""
    pose = np.array(event["world_task"]) @ np.array(event["objects_task"]["block"])
    corners = np.array(list(product((-.03, .03), (-.03, .03), (-.1, .1))))
    world = corners @ pose[:3, :3].T + pose[:3, 3]
    matrices = {}
    head, = (c for c in cfg["static_camera_list"] if c["name"] == "head_camera")
    static = np.eye(4)
    forward, left = (np.array(head[k])/np.linalg.norm(head[k]) for k in ("forward", "left"))
    static[:3, :3] = np.stack((forward, left, np.cross(forward, left)), axis=1)
    static[:3, 3] = head["position"]
    matrices["cam_high"] = static
    for key, arm in zip(CAMERAS[1:], event["actual"]):
        matrices[key] = np.array(arm["world_base"]) @ camera.fk(dict(zip(arm["names"], arm["qpos"])))
    values = {}
    for key, matrix in matrices.items():
        height, width = images[key].shape[1:]
        focal = (height / 2) / np.tan(np.deg2rad(fovy / 2))
        xyz = (world - matrix[:3, 3]) @ matrix[:3, :3]
        xy = np.column_stack((width / 2 - focal * xyz[:,1]/xyz[:,0],
                              height / 2 - focal * xyz[:,2]/xyz[:,0]))
        bounds = np.r_[xy.min(0), xy.max(0)]
        visible = (xyz[:,0].min() > 0 and bounds[0] < width and bounds[2] > 0
                   and bounds[1] < height and bounds[3] > 0)
        observed = camera_stats(images[key])["red_bounds_xyxy"]
        clipped = np.clip(bounds, 0, [width-1,height-1]*2)
        edge_error = float(np.max(np.abs(clipped-observed))) if observed else None
        assert visible == (observed is not None)
        if edge_error is not None:
            assert edge_error <= 3.
        values[key] = dict(camera_world=matrix.tolist(), predicted_visible=bool(visible),
                           projected_full_bounds_xyxy=bounds.tolist(),
                           projected_clipped_bounds_xyxy=clipped.tolist() if visible else None,
                           red_bounds_max_error_px=edge_error, fovy_deg=fovy)
    return values


def capture_fixture(images, state, prompt, capture_rgb, official):
    # Re-encode the historical arrays to check the two clients' deterministic layout.
    fixture = dict(observation={source: {"rgb": images[key].transpose(1, 2, 0)}
                   for key, source in NAMES.items()}, joint_action={"vector": state})
    ref = official.encode_observation(fixture, prompt)
    capture = capture_rgb(fixture)
    for key in NAMES:
        np.testing.assert_array_equal(ref["images"][key], images[key])
        np.testing.assert_array_equal(capture.images[key][1], images[key])
    np.testing.assert_array_equal(ref["state"], state)
    return ref


def audit_identity(trial):
    checkpoint = read(ROOT / "pace_download_manifest.json")
    model = json.loads(trial.verified(trial.root / "model/startup.json"))
    launch = json.loads(trial.verified(trial.root / "effective-launch.json"))
    auth = json.loads(trial.verified(trial.root / "authorization-receipt.json"))
    setup = json.loads(trial.verified(trial.episode / "setup.json"))
    rng = json.loads(trial.verified(trial.episode.parent / "rng-reset.json"))
    config = trial.config
    assert model["status"] == "ready" and model["checkpoint_revision"] == checkpoint["revision"]
    assert len(model["checkpoint_files"]) == len(checkpoint["files"]) == 28
    for item in checkpoint["files"]:
        assert model["checkpoint_files"][item["path"]][::2] == [item["bytes"], item["sha256"]]
    assert setup["config_sha256"] == digest(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    assert rng == dict(command="g2e_rng_ready", seed=config["policy_rng_seed"],
                       episode_id=trial.episode.parent.name)
    assert any(value.endswith("/g2e_model_entry.py") for value in launch["model"])
    assert any(value.endswith("Openpi-PACE-f4e52e6de4cc487a11585a3a7414d610b2883dae") for value in launch["model"])
    changes = {}
    for name in ("g2_adapter", "g2_capture", "g2_task_runtime", "g2_arx_direct", "g2_remaining_episode",
                 "g2_robotwin_backend", "g2_assets", "g2_contact", "g2_motion", "g2e_model_entry",
                 "pace_runtime", "pace_model_smoke"):
        path = f"scripts/{name}.py"
        prior = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{auth['commit']}:{path}"])
        actual = trial.release[path]
        current = digest((ROOT / path).read_bytes())
        assert actual == digest(prior), path
        changes[path] = dict(release=actual, current=current, identical=current == actual)
    return dict(status=OK, job_commit=auth["commit"], checkpoint=checkpoint["revision"],
                model_file_checks=28, norm_stats=checkpoint["norm_stats_sha256"],
                tokenizer=next(i["sha256"] for i in checkpoint["files"] if i["path"]=="tokenizer.model"),
                release_trial_config_sha256=trial.config_sha, runtime_config_canonical_sha256=setup["config_sha256"],
                selection_spec_sha256=auth["configuration_sha256"], archive_sha256=auth["archive_sha256"],
                scene_seed=config["scene_seed"], policy_rng_seed=rng["seed"], runtime_files=changes)


def audit_trial(trial, assets, audit, source, bundle, tokenizer):
    cfg = trial.config
    target = load_arms(assets, audit, "ARX-X5", cfg["calibration"]["ARX-X5"])
    aloha = load_arms(assets, audit, "aloha-agilex", cfg["calibration"]["aloha-agilex"])
    row, = [r for r in read(audit)["robots"] if r["robot"] == "ARX-X5"]
    camera = Chain(assets / row["urdf_member"], "camera_joint")
    camera_settings = yaml.safe_load((assets / row["archive_config_member"]).read_text())
    fovy = yaml.safe_load((source / "pace-_camera_config.yml").read_text())["D435"]["fovy"]
    adapter = None
    queries, prefixes, plans = [], [], Counter()
    checked_files = {}
    angles, positions, mismatches = [], [], []
    normal = bundle.transforms.Normalize(None, use_quantiles=True)
    denormal = bundle.transforms.Unnormalize(None, use_quantiles=True)
    out_spec = bundle.transforms.AbsoluteActions(JOINT_MASK)
    in_spec = bundle.transforms.DeltaActions(JOINT_MASK)
    for line, event in trial.events():
        kind = event["kind"]
        if kind == "policy_request":
            index = len(queries) + 1
            stamp = Stamp(**event["stamp"])
            arms = tuple(ArmSample(stamp, a["instance"], tuple(a["names"]), np.array(a["qpos"]),
                                   np.array(a["world_base"])) for a in event["actual"])
            if adapter is None:
                target = tuple(replace(a, instance=s.instance) for a, s in zip(target, arms))
                adapter = Adapter(aloha, target, [matrix(p) for p in cfg["source_base_world"]],
                    matrix(cfg["source_world_task"]), cfg["source_proxy_seed_rad"],
                    cfg["source_proxy_velocity_rad_s"], kinematic_pair, kinematic_pair, None,
                    cfg["grip_rule"], cfg["grip_max_clip"])
            rgb_path = trial.episode / f"observation-{index:05}.npz"
            raw_rgb = trial.verified(rgb_path)
            with np.load(BytesIO(raw_rgb), allow_pickle=False) as archive:
                images = dict(archive)
            state = np.array(event["state"], dtype=np.float32)
            assert event["prompt"] == cfg["prompt"]
            assert all(a["stamp"] == event["stamp"] for a in event["actual"])
            assert set(images)==set(CAMERAS)
            assert all(x.shape==(3,240,320) and x.dtype==np.uint8 for x in images.values())
            if index==1:
                assert trial.verified(trial.episode/"initial-rgb.npz") == raw_rgb
            snap = Snapshot(stamp, arms, {key: (stamp, image) for key, image in images.items()},
                            np.array(event["world_task"]))
            mapped = adapter.observe(snap,event["prompt"],15)
            if not mapped.ok:
                raise ValueError(mapped.detail)
            error = np.max(np.abs(mapped.payload["state"].astype(float)-state))
            mismatches.append(float(error))
            assert error <= 2e-6
            def captured(fixture):
                from unittest.mock import patch
                with patch("g2_capture.capture_arms", return_value=arms):
                    return capture_snapshot(SimpleNamespace(robot=None,get_obs=lambda: fixture), stamp, snap.world_task)[0]
            request = capture_fixture(images,state,event["prompt"],captured,bundle.official)
            validate_request(request)
            packed = msgpack_numpy.packb(request)
            unpacked = msgpack_numpy.unpackb(packed)
            np.testing.assert_array_equal(unpacked["state"],state)
            for key in NAMES:
                np.testing.assert_array_equal(unpacked["images"][key],images[key])
            transformed = bundle.aloha.AlohaInputs(adapt_to_pi=False)(unpacked)
            assert unpacked["prompt"]==transformed["prompt"]==event["prompt"]
            np.testing.assert_array_equal(transformed["state"],state)
            for destination,key in zip(MODEL_NAMES,CAMERAS):
                np.testing.assert_array_equal(transformed["image"][destination],images[key].transpose(1,2,0))
                assert transformed["image_mask"][destination]
            normalized = normal._normalize_quantile(state,bundle.stats["state"])
            reconstructed = denormal._unnormalize_quantile(normalized,bundle.stats["state"])
            np.testing.assert_allclose(reconstructed,state,rtol=0,atol=1e-12)
            camera_matches = camera_projection(event,images,camera,camera_settings,fovy)
            scaled = bundle.transforms.ResizeImages(224,224)(transformed)
            for image in scaled["image"].values():
                assert image.shape==(224,224,3) and image.dtype==np.uint8
                assert not image[:28].any() and not image[196:].any()
            bins = np.digitize(normalized,np.linspace(-1,1,257)[:-1])-1
            tokens = None
            if tokenizer:
                tokens, mask = bundle.tokenizer.PaligemmaTokenizer.tokenize(tokenizer,event["prompt"],normalized)
                assert len(tokens)==len(mask)==200
                tokens = tokens[mask].tolist()
            action = trial.episode/f"action-{index:05}.npy"
            action_bytes = trial.verified(action)
            chunk = np.load(BytesIO(action_bytes),allow_pickle=False)
            assert chunk.shape==(50,14) and np.isfinite(chunk).all()
            local = in_spec(dict(state=state,actions=chunk.copy()))["actions"]
            raw = normal._normalize_quantile(local,bundle.stats["actions"])
            restored = denormal._unnormalize_quantile(raw,bundle.stats["actions"])
            restored = out_spec(dict(state=state,actions=restored))["actions"]
            restored = bundle.aloha.AlohaOutputs(adapt_to_pi=False)(dict(actions=restored))["actions"]
            np.testing.assert_allclose(restored,chunk,rtol=0,atol=3e-15)
            np.testing.assert_array_equal(local[:,~JOINT_MASK],chunk[:,~JOINT_MASK])
            tcp = np.asarray(mapped.diagnostic["arx_tools_task"])
            projected = [frame@arm.chain.fk(q[:6])@arm.offset() for frame,arm,q in
                         zip(adapter.source_frames,adapter.aloha,state.reshape(2,7))]
            pe = np.array([pose_error(reference,value) for reference,value in zip(tcp,projected)])
            positions.extend((1000*np.linalg.norm(pe[:,:3],axis=1)).tolist())
            angles.extend(np.rad2deg(np.linalg.norm(pe[:,3:],axis=1)).tolist())
            queries.append(dict(query=index,event_line=line,tick=stamp.tick,prompt=event["prompt"],
                actual_articulations=[dict(instance=a.instance,names=a.names,qpos=a.qpos.tolist(),
                                           world_base=a.world_base.tolist()) for a in arms],
                world_task=event["world_task"], block_pose_task=event.get("objects_task",{}).get("block"),
                state=state.tolist(),state_cpu_reconstructed=mapped.payload["state"].tolist(),
                state_max_error_rad_or_ratio=float(error),arx_tcp=tcp.tolist(),
                state_normalized_cpu=normalized.tolist(),state_bins_cpu=bins.tolist(),token_ids_cpu=tokens,
                gripper_measurement=mapped.diagnostic["grip_measurement"],
                cameras={key:dict(camera_stats(value),source=NAMES[key],
                         model_field=target_field,projection=camera_matches[key],
                         server_resized_sha256=digest(scaled["image"][target_field].tobytes()),
                         server_pixels_cpu_sha256=digest((scaled["image"][target_field].astype(np.float32)
                                                           /255.*2.-1.).tobytes()))
                         for (key,value),target_field in zip(((key,images[key]) for key in CAMERAS),MODEL_NAMES)},
                archived_type={"state":"float32","images":"uint8 CHW","prompt":"str"},
                observation_npz_sha256=digest(raw_rgb),client_payload_reconstructed_sha256=digest(packed),
                action_np_sha256=digest(action_bytes),action_dtype=str(chunk.dtype),
                action_first_row=chunk[0].tolist(),action_last_row=chunk[-1].tolist(),
                renormalized_delta_first_row_cpu=raw[0].tolist(),
                output_roundtrip_max_error=float(np.max(np.abs(restored-chunk)))))
            checked_files[index]=chunk
        elif kind=="approved_plan":
            index=len(queries)
            number=event["row_in_prefix"]
            key=event["receipt"]["requests"][0]
            row=checked_files[index][number-1].reshape(2,7)
            reconstructed=np.array([frame@arm.chain.fk(v[:6])@arm.offset() for
                          frame,arm,v in zip(adapter.source_frames,aloha,row)])
            np.testing.assert_allclose(reconstructed,key["task_tools"],atol=1e-10,rtol=0)
            np.testing.assert_allclose(np.array([a.grip.target(float(np.clip(v[6],0,1))) for
                                                   a,v in zip(target,row)]),key["fingers"],atol=1e-12,rtol=0)
            grippers=event["receipt"]["grippers"]
            np.testing.assert_array_equal([g["raw"] for g in grippers],row[:,6])
            plans[index]+=1
        elif kind=="direct_prefix_complete":
            assert event["executed_actions"]+event["discarded_rows"]==50
            prefixes.append(dict(query=len(queries),**{k:event[k] for k in
                           ("stamp","executed_actions","discarded_rows","reason")}))
    for prefix in prefixes:
        if prefix["query"] < len(queries):
            assert prefix["stamp"]["tick"]==queries[prefix["query"]]["tick"]
    identity = audit_identity(trial)
    return dict(identity=identity,requests=queries,approved_rows_by_query=dict(plans),prefixes=prefixes,
                numeric_checks=dict(state_max_error_rad_or_ratio=max(mismatches),
                                    state_tcp_ik_max_mm=max(positions),state_tcp_ik_max_deg=max(angles),
                                    requests=len(queries),approved_action_rows=sum(plans.values()),
                                    camera_visible_max_error_px=max(
                                        value["projection"]["red_bounds_max_error_px"] for q in queries
                                        for value in q["cameras"].values()
                                        if value["projection"]["red_bounds_max_error_px"] is not None)))


def transforms(source, model_files):
    checkpoint = read(ROOT/"pace_download_manifest.json")
    stat_file = ROOT/"research/gate25/pace_handover_block_assets_robotwin_norm_stats.json"
    assert digest(stat_file.read_bytes())==checkpoint["norm_stats_sha256"]
    stats = {k:SimpleNamespace(**{name:np.array(value) for name,value in values.items()})
             for k,values in read(stat_file)["norm_stats"].items()}
    assert all(value.q01.shape==value.q99.shape==(14,) for value in stats.values())
    configs=(source/"openpi-configs.py").read_text()
    training=(source/"openpi-config.py").read_text()
    policy=(source/"openpi-policy_config.py").read_text()
    assert all(value in configs for value in ("adapt_to_pi=False","use_delta_joint_actions=True",
                                             'asset_id="robotwin"',"pi05=True, action_horizon=50"))
    assert all(value in training for value in ("use_quantile_norm=model_config.model_type != ModelType.PI0",
        "make_bool_mask(6, -1, 6, -1)", "ResizeImages(224, 224)", "discrete_state_input=model_config.discrete_state_input"))
    assert policy.count("transforms.Normalize(")==policy.count("transforms.Unnormalize(")==1
    assert '_checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)' in policy
    assert policy.index("transforms.Unnormalize(")<policy.index("*data_config.data_transforms.outputs")
    policy_class=(source/"openpi-policy.py").read_text()
    assert policy_class.count("self._output_transform(outputs)")==policy_class.count("self._input_transform(inputs)")==1
    image_tools=components(source/"openpi-image_tools.py",["resize_with_pad","_resize_with_pad_pil"],
                           {"np":np,"Image":Image})
    transform=components(source/"openpi-transforms.py",["Normalize","Unnormalize","DeltaActions",
                        "AbsoluteActions","ResizeImages","pad_to_dim"],
                        {"np":np,"dataclasses":SimpleNamespace(dataclass=dataclass),"image_tools":image_tools})
    def rearrange(array, pattern):
        assert pattern=="c h w -> h w c"
        return array.transpose(1,2,0)
    aloha=components(source/"openpi-aloha_policy.py",["AlohaInputs","AlohaOutputs","_decode_aloha",
                     "_decode_state","_encode_actions"],dict(np=np,dataclasses=SimpleNamespace(dataclass=dataclass),
                                         ClassVar=ClassVar,einops=SimpleNamespace(rearrange=rearrange)))
    tokenizer=None
    if model_files and (model_files/"tokenizer.model").exists():
        import sentencepiece
        raw=(model_files/"tokenizer.model").read_bytes()
        assert digest(raw)==next(i["sha256"] for i in checkpoint["files"] if i["path"]=="tokenizer.model")
        tokenizer=SimpleNamespace(_max_len=200,_tokenizer=sentencepiece.SentencePieceProcessor(model_proto=raw))
    token_methods=components(source/"openpi-tokenizer.py", ["PaligemmaTokenizer"],dict(np=np,logging=logging))
    return SimpleNamespace(aloha=aloha,transforms=transform,official=components(source/"pace-rollout.py",
        ["encode_observation"],{}),stats=stats,tokenizer=token_methods), tokenizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("archive","assets","asset_audit","pinned_sources","output"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--model-files",type=Path)
    args=parser.parse_args()
    sources=verify_sources(args.pinned_sources)
    bundle,tokenizer=transforms(args.pinned_sources,args.model_files)
    jobs={job:audit_trial(Trial(args.archive,job),args.assets,args.asset_audit,
                         args.pinned_sources,bundle,tokenizer) for job in JOBS}
    def verdict(path, evidence, status=OK):
        return dict(status=status, code=path, evidence=evidence)
    report=dict(overall_observable_input_status=OK, full_historical_model_tensor_status=UNKNOWN,
        jobs=jobs,pinned_source_hashes=sources,
        checks={
            "camera_order_orientation_rgb": verdict(
                ["PACE envs/robot/robot.py:init_joints", "PACE envs/_base_task.py:_update_render",
                 "PACE envs/camera/camera.py:get_rgba,get_rgb", "scripts/g2_capture.py:capture_snapshot",
                 "Openpi-PACE src/openpi/policies/aloha_policy.py:AlohaInputs"],
                "All archive fields and hashes verified. Robot FK + fixed camera mounts project the red block in the "
                "correct wrist and image orientation. 17 requests x 3 camera arrays and RGB channel statistics below."),
            "resize_and_image_mask": verdict(
                ["Openpi-PACE src/openpi/training/config.py:ModelTransformFactory",
                 "src/openpi/transforms.py:ResizeImages", "packages/openpi-client/src/openpi_client/image_tools.py",
                 "src/openpi/models/model.py:Observation.from_dict"],
                dict(live_shape_chw=[3,240,320], resized_shape_hwc=[224,224,3], bilinear_content_hwc=[168,224,3],
                     zero_pad_top_bottom_px=[28,28], crop=False, masks=[True,True,True],
                     image_model_float_transform="uint8 / 255 * 2 - 1", exact_historical_model_image_tensor="UNKNOWN")),
            "state_measurement_and_kinematics": verdict(
                ["scripts/g2_capture.py:capture_arms", "scripts/g2_adapter.py:read_snapshot,Adapter.observe",
                 "scripts/g2_assets.py:load_arms"],
                dict(unit="radians for 12 arm joints, normalized physical aperture for 2 grippers",
                     order=["left_q1..q6","left_grip","right_q1..q6","right_grip"],
                     all_17_state_arrays_reconstructed_exactly=True,
                     target_finger_m_to_normalized="sum(q7,q8)/(2*0.044) clipped in unit interval")),
            "normalization": verdict(
                ["scripts/pace_model_smoke.py:load_model", "Openpi-PACE src/openpi/policies/policy_config.py",
                 "src/openpi/transforms.py:Normalize,Unnormalize,AbsoluteActions", "pace_download_manifest.json"],
                dict(asset="handover_block/assets/robotwin/norm_stats.json", asset_id="robotwin",
                     sha256=jobs[JOBS[0]]["identity"]["norm_stats"], state_and_action_dimensions=[14,14],
                     state_q01=bundle.stats["state"].q01.tolist(),state_q99=bundle.stats["state"].q99.tolist(),
                     action_q01=bundle.stats["actions"].q01.tolist(),action_q99=bundle.stats["actions"].q99.tolist(),
                     quantile_eps=1e-6, analytic_transform_roundtrip_only=True,
                     historical_unprocessed_sampler_tensor="UNKNOWN")),
            "language_and_serialization": verdict(
                ["scripts/g2_arx_direct.py:DirectController.cycle", "scripts/pace_runtime.py:RemotePolicy.infer_raw",
                 "scripts/g2e_model_entry.py:handler", "Openpi-PACE src/openpi/models/tokenizer.py:PaligemmaTokenizer"],
                dict(prompt=jobs[JOBS[0]]["requests"][0]["prompt"],
                     payload_keys=["images","prompt","state"], token_limit=200,
                     exact_historical_wire_and_token_tensor="UNKNOWN")),
            "absolute_action_and_window": verdict(
                ["Openpi-PACE src/openpi/policies/policy.py:Policy.infer",
                 "pace/src/openpi_pace/configs.py:make_configs",
                 "scripts/g2_arx_direct.py:DirectController.cycle,execute_prefix",
                 "scripts/g2_adapter.py:Adapter._actions"],
                dict(training_joint_delta_mask=JOINT_MASK.tolist(),adapt_to_pi=False,output_shape=[50,14],
                     client_adds_delta=False,max_prefix=50,approved_goals_reproduced=127,
                     first_prefix_127999421_consumed_discarded=[36,14],
                     second_prefix_127999421_consumed_discarded=[1,49])),
            "runtime_identity": verdict(
                ["release/SHA256SUMS", "effective-launch.json","model/startup.json","authorization-receipt.json"],
                "Per-job release scripts reproduce b7202f8, 28 checkpoint entries match the pinned manifest. "
                "Three current files differ in contact replanning or logging. See runtime_files."),
            "historical_server_tensors_and_training_distribution": verdict(
                ["scripts/g2_remaining_episode.py:run", "Openpi-PACE src/openpi/policies/policy.py:Policy.infer"],
                "Server did not archive wire payloads, post-transform tensors or sampler outputs. Exact checkpoint "
                "training LeRobot bytes, timing, camera distribution and prompt coverage are unavailable. "
                "CPU replays reconstruct configured transforms, not a historical server observation.", UNKNOWN)},
        evidence_semantics="Verified client-side request and reproducible pinned chain. CPU reconstructed normalized "
                           "state, tokens, resized images and tensor inverse checks are labeled cpu.")
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print({job:row["numeric_checks"] for job,row in jobs.items()})


if __name__=="__main__":
    main()
