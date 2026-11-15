"""Reconstruct archived input and output contracts from the fixed CPU evidence."""
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.spatial import ConvexHull
import yaml

from g2_adapter import Adapter, ArmSample, Snapshot, Stamp, deadline, read_snapshot, solve_pair
from g2_arx_direct import kinematic_pair
from g2_initial import matrix7
from g2_kinematics import pose_error
from input_audit import components, digest, read, camera_stats, MODEL_NAMES, JOINT_MASK
from robotwin_control_geometry import inverse_pose
from verify_robotwin_fk import Chain

from geometry import fk

ROOT = Path(__file__).resolve().parents[3]
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def policy_contract(image_source):
    prior = read(ROOT / "research/g2/arx-frozen-input-audit/input_adapter_audit.json")
    model = ROOT / "research/g2/arx-systematic-20261114/source-evidence/model"
    checkpoint = read(ROOT / "pace_download_manifest.json")
    stats_path = ROOT / "research/gate25/pace_handover_block_assets_robotwin_norm_stats.json"
    assert digest(stats_path.read_bytes()) == checkpoint["norm_stats_sha256"]
    for name in ("transforms.py", "aloha_policy.py", "policy_config.py", "config.py"):
        locked = prior["pinned_source_hashes"]["openpi-" + name]["sha256"]
        assert digest((model / name).read_bytes()) == locked, name
    transforms = components(model / "transforms.py", ("Normalize", "Unnormalize", "DeltaActions", "AbsoluteActions"),
                            dict(np=np, dataclasses=SimpleNamespace(dataclass=dataclass)))
    def rearrange(array, pattern):
        assert pattern == "c h w -> h w c"
        return array.transpose(1, 2, 0)
    aloha = components(model / "aloha_policy.py",
        ("AlohaInputs", "AlohaOutputs", "_decode_aloha", "_decode_state", "_encode_actions"),
        dict(np=np, dataclasses=SimpleNamespace(dataclass=dataclass), ClassVar=ClassVar,
             einops=SimpleNamespace(rearrange=rearrange)))
    stats = {k: SimpleNamespace(**{field: np.array(value) for field, value in block.items()})
             for k, block in read(stats_path)["norm_stats"].items()}
    image_sha = digest(image_source.read_bytes())
    assert image_sha == prior["pinned_source_hashes"]["openpi-image_tools.py"]["sha256"]
    image_tools = components(image_source, ("resize_with_pad", "_resize_with_pad_pil"),
                             dict(np=np, Image=Image))
    return SimpleNamespace(transforms=transforms, aloha=aloha, stats=stats, checkpoint=checkpoint,
                           image_tools=image_tools, image_source_sha256=image_sha)


def decode(bundle, state, chunk):
    normal = bundle.transforms.Normalize(None, use_quantiles=True)
    denormal = bundle.transforms.Unnormalize(None, use_quantiles=True)
    # Recover a normalized equivalent from each saved service output. The actual
    # latent denoiser output is absent and is not inferred from this equality.
    delta = bundle.transforms.DeltaActions(JOINT_MASK)(dict(state=state, actions=chunk.copy()))["actions"]
    np.testing.assert_array_equal(delta[:, ~JOINT_MASK], chunk[:, ~JOINT_MASK])
    encoded = normal._normalize_quantile(delta, bundle.stats["actions"])
    local = denormal._unnormalize_quantile(encoded, bundle.stats["actions"])
    recovered = bundle.transforms.AbsoluteActions(JOINT_MASK)(dict(state=state, actions=local))["actions"]
    recovered = bundle.aloha.AlohaOutputs(adapt_to_pi=False)(dict(actions=recovered))["actions"]
    np.testing.assert_allclose(recovered, chunk, atol=3e-15, rtol=0)
    normalized = normal._normalize_quantile(state, bundle.stats["state"])
    return dict(state_normalized=normalized,
                state_bins=np.digitize(normalized, np.linspace(-1, 1, 257)[:-1]) - 1,
                equivalent_normalized_actions=encoded, delta_joint_and_absolute_gripper=delta,
                denormalized_delta_max_error=float(np.max(np.abs(local - delta))),
                absolute_roundtrip_max_error=float(np.max(np.abs(recovered - chunk))),
                runtime_pre_unnormalize_tensor="UNAVAILABLE")


def rebuild_states(source, target, cfg, records, images_by_query):
    output = []
    for query, (request, images) in enumerate(zip(records, images_by_query), 1):
        stamp = Stamp(**request["stamp"])
        assert all(a["stamp"] == request["stamp"] for a in request["actual"])
        actual = tuple(ArmSample(stamp, a["instance"], tuple(a["names"]), np.asarray(a["qpos"]),
                                 np.asarray(a["world_base"])) for a in request["actual"])
        if query == 1:
            target = tuple(replace(arm, instance=a.instance) for arm, a in zip(target, actual))
            adapter = Adapter(source, target, [matrix7(p) for p in cfg["source_base_world"]],
                              matrix7(cfg["source_world_task"]), cfg["source_proxy_seed_rad"],
                              cfg["source_proxy_velocity_rad_s"], kinematic_pair, kinematic_pair, None,
                              cfg["grip_rule"], cfg["grip_max_clip"])
        snap = Snapshot(stamp, actual, {key: (stamp, array) for key, array in images.items()},
                        np.asarray(request["world_task"]))
        mapped = adapter.observe(snap, request["prompt"], 30)
        if not mapped.ok:
            raise ValueError(mapped.detail)
        q, fingers, frames, tools = read_snapshot(snap, target)
        state = np.asarray(request["state"], dtype=np.float32)
        np.testing.assert_allclose(mapped.payload["state"], state, rtol=0, atol=2e-6)
        proxy = fk(source, adapter.source_frames, state.reshape(2, 7)[:, :6])
        back, _ = solve_pair(target, frames, proxy, q, fingers, kinematic_pair, deadline(30))
        roundtrip = fk(target, frames, back)

        def error(a, b):
            delta = np.array([pose_error(x, y) for x, y in zip(a, b)])
            return dict(mm=1000 * np.linalg.norm(delta[:, :3], axis=1),
                        deg=np.rad2deg(np.linalg.norm(delta[:, 3:], axis=1)))
        output.append(dict(query=query, state_archived=state, state_rebuilt=mapped.payload["state"],
                           max_state_error=float(np.max(np.abs(state - mapped.payload["state"]))),
                           absolute_joint_names=[arm.chain.names for arm in source],
                           virtual_aloha_seed=mapped.diagnostic["ik_seed"],
                           virtual_aloha_ik=mapped.diagnostic["ik"], actual_joint_names=[a.names for a in actual],
                           actual_arms_rad=q, actual_fingers_m=fingers, gripper=mapped.diagnostic["grip_measurement"],
                           virtual_tcp_vs_actual=error(proxy, tools), reverse_tcp_vs_actual=error(roundtrip, tools),
                           reverse_branch_delta_rad=back - q, reverse_kinematic_only=True))
    return output


def clip_polygon(vertices, axis, edge, direction):
    """Keep a half plane while finding exact intersections with its boundary."""
    clipped = []
    for start, end in zip(np.roll(vertices, 1, axis=0), vertices):
        before = direction * (start[axis] - edge) >= 0
        after = direction * (end[axis] - edge) >= 0
        if before != after:
            weight = (edge - start[axis]) / (end[axis] - start[axis])
            clipped.append(start + weight * (end - start))
        if after:
            clipped.append(end)
    return np.array(clipped).reshape(-1, 2)


def visible_silhouette(projected, width, height):
    polygon = projected[ConvexHull(projected).vertices]
    for axis, edge, direction in ((0, 0, 1), (0, width-1, -1), (1, 0, 1), (1, height-1, -1)):
        polygon = clip_polygon(polygon, axis, edge, direction)
    return polygon


def substantial_red_regions(rgb):
    red, green, blue = rgb.astype(float)
    mask, count = label((red > 100) & (red > 1.7 * green) & (red > 1.7 * blue))
    if not count:
        return None, 0
    counts = np.bincount(mask.ravel())
    # A jaw can split the same object into distinct visible regions. Reject small
    # detached red speckles without discarding the upper portion of a split box.
    keep = 1 + np.flatnonzero(counts[1:] >= max(25, .05*counts[1:].max()))
    if not len(keep):
        return None, 0
    y, x = np.where(np.isin(mask, keep))
    return [int(x.min()), int(y.min()), int(x.max()), int(y.max())], len(x)


def image_semantics(bundle, request, images, physical_names, assets, member, is_arx, block_world, fov):
    encoded = bundle.aloha.AlohaInputs(adapt_to_pi=False)(
        dict(images=images, state=request["state"], prompt=request["prompt"]))
    np.testing.assert_array_equal(encoded["state"], request["state"])
    assert encoded["prompt"] == request["prompt"]
    cfg = yaml.safe_load((assets / member["archive_config_member"]).read_text())
    head, = [row for row in cfg["static_camera_list"] if row["name"] == "head_camera"]
    forward, left = (np.asarray(head[k]) / np.linalg.norm(head[k]) for k in ("forward", "left"))
    head_frame = np.eye(4)
    head_frame[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
    head_frame[:3, 3] = head["position"]
    wrist_frames = []
    for index, arm in enumerate(physical_names):
        endpoint = "camera_joint" if is_arx else ("left_camera_joint", "right_camera_joint")[index]
        chain = Chain(assets / member["urdf_member"], endpoint)
        wrist_frames.append(np.asarray(arm["world_base"]) @ chain.fk(dict(zip(arm["names"], arm["qpos"]))))
    local = np.array(list(product((-.03, .03), (-.03, .03), (-.10, .10))))
    world = local @ block_world[:3, :3].T + block_world[:3, 3]
    output = {}
    for key, target, camera in zip(CAMERAS, MODEL_NAMES, [head_frame, *wrist_frames]):
        rgb = images[key]
        assert rgb.shape == (3, 240, 320) and rgb.dtype == np.uint8
        np.testing.assert_array_equal(encoded["image"][target], rgb.transpose(1, 2, 0))
        assert encoded["image_mask"][target]
        _, height, width = rgb.shape
        focal = (height / 2) / np.tan(np.deg2rad(fov / 2))
        xyz = (world - camera[:3, 3]) @ camera[:3, :3]
        clipped = legacy = None
        if xyz[:,0].min() > 0:
            xy = np.column_stack([width/2 - focal * xyz[:,1]/xyz[:,0],
                                  height/2 - focal * xyz[:,2]/xyz[:,0]])
            silhouette = visible_silhouette(xy, width, height)
            if len(silhouette):
                clipped = np.r_[silhouette.min(axis=0), silhouette.max(axis=0)]
                legacy = np.clip(np.r_[xy.min(axis=0), xy.max(axis=0)], 0, [width-1,height-1]*2)
        classification = "VISIBLE" if clipped is not None else "OUTSIDE_OR_NEAR_PLANE"
        actual = camera_stats(rgb)
        red, connected_pixels = substantial_red_regions(rgb)
        pixel_error = float(np.max(np.abs(clipped - red))) if red is not None and clipped is not None else None
        legacy_error = float(np.max(np.abs(legacy - red))) if pixel_error is not None else None
        center = (block_world[:3, 3] - camera[:3, 3]) @ camera[:3, :3]
        cxy = [width/2 - focal * center[1]/center[0], height/2 - focal * center[2]/center[0]] if center[0] > 0 else None
        hwc = encoded["image"][target]
        resized = bundle.image_tools.resize_with_pad(hwc, 224, 224)
        assert resized.shape == (224, 224, 3) and resized.dtype == np.uint8
        assert not resized[:28].any() and not resized[196:].any()
        output[key] = dict(actual, model_field=target, camera_world=camera, fovy_deg=fov,
                           focal_px=focal, camera_center_offset_block_mm=1000*(inverse_pose(block_world) @ camera)[:3,3],
                           substantial_red_bounds_xyxy=red, substantial_red_pixels=connected_pixels,
                           block_center_pixel=cxy, box_expected_visibility=classification,
                           projected_bounds_xyxy=clipped,
                           independently_clamped_bounds_xyxy=legacy,
                           independently_clamped_max_error_px=legacy_error,
                           visible_red_bounds_max_error_px=pixel_error, rgb_resize_cpu_sha256=digest(resized.tobytes()),
                           optics_source="locked URDF mount and static camera configuration")
    return output
