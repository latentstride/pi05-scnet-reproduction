"""CPU-only inspection of sampled official RoboTwin HDF5 demonstrations.

Pass the official data/decode_image_bit.py from the pinned RoboTwin source.
This script does not execute a policy, IK, simulation, or pickle payloads.
"""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


def stats(values):
    a = np.asarray(values)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "finite": bool(np.isfinite(a).all()),
            "min": a.min(axis=0).tolist() if a.size else None,
            "max": a.max(axis=0).tolist() if a.size else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    code = args.decoder.read_bytes()
    decoder_blob = hashlib.sha1(("blob %d\0" % len(code)).encode() + code).hexdigest()
    if decoder_blob != "32ac399ef585a1ee6dea34eb35cb53c2814007b4":
        raise ValueError("Decoder differs from the audited official source")
    spec = importlib.util.spec_from_file_location("robotwin_official_codec", args.decoder)
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    manifest = json.loads((args.root / "sample_manifest.json").read_text())
    report = {"dataset_repository": manifest["repository"], "dataset_revision": manifest["revision"],
              "decoder_git_blob": decoder_blob, "episodes": [], "gpu_used": False,
              "simulation_executed": False, "policy_executed": False,
              "reader_versions": {"numpy": np.__version__, "h5py": h5py.__version__}}
    preview_rows = []
    for archive in manifest["archives"]:
        robot = archive["robot"]
        seeds = (args.root / robot / "metadata/seed.txt").read_text().split()
        for member in archive["selected"]:
            path = args.root / member["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
                raise ValueError("Sample hash mismatch: " + member["file"])
            ep = {"robot": robot, "episode": path.stem, "sha256": member["sha256"],
                  "seed_by_archive_order": int(seeds[int(re.search(r"\d+", path.stem).group())]),
                  "schema": {}, "images": {}, "quaternions": {}, "camera_frame_checks": {}}
            with h5py.File(path, "r") as f:
                datasets = {}
                f.visititems(lambda key, value: datasets.update({key: value}) if isinstance(value, h5py.Dataset) else None)
                ep["root_attributes"] = {str(k): str(v) for k, v in f.attrs.items()}
                ep["schema"] = {key: {"shape": list(v.shape), "dtype": str(v.dtype)} for key, v in datasets.items()}
                ep["time_fields"] = [k for k in datasets if re.search(r"(^|/)(timestamps?|times?|fps|dt|frame_index|step_index)$", k, re.I)]
                t = f["joint_action/vector"].shape[0]
                ep["frames"] = t
                ep["all_dataset_lengths_match"] = all(v.shape and v.shape[0] == t for v in datasets.values())
                ep["all_numeric_values_finite"] = all(np.isfinite(v[:]).all() for v in datasets.values() if v.dtype.kind in "fiu")
                joint = np.concatenate([f["joint_action/left_arm"][:], f["joint_action/left_gripper"][:][:, None],
                                        f["joint_action/right_arm"][:], f["joint_action/right_gripper"][:][:, None]], axis=1)
                ee = np.concatenate([f["endpose/left_endpose"][:], f["endpose/left_gripper"][:][:, None],
                                     f["endpose/right_endpose"][:], f["endpose/right_gripper"][:][:, None]], axis=1)
                ep["joint_state"] = stats(joint)
                ep["ee_state"] = stats(ee)
                ep["joint_vector_max_abs_difference"] = float(np.max(np.abs(joint - f["joint_action/vector"][:])))
                ep["first_joint_state"] = joint[0].tolist()
                ep["first_ee_state"] = ee[0].tolist()
                for arm in ["left", "right"]:
                    q = f["endpose/" + arm + "_endpose"][:, 3:7]
                    norm = np.linalg.norm(q, axis=1)
                    ep["quaternions"][arm] = {"shape": list(q.shape), "norm_min": float(norm.min()),
                        "norm_max": float(norm.max()), "max_abs_norm_error": float(np.max(np.abs(norm - 1)))}
                    a, b = f["endpose/" + arm + "_gripper"][:], f["joint_action/" + arm + "_gripper"][:]
                    ep[arm + "_gripper"] = {"min": float(a.min()), "max": float(a.max()),
                        "within_0_1": bool(((a >= 0) & (a <= 1)).all()),
                        "joint_vs_endpose_max_abs_difference": float(np.max(np.abs(a - b)))}
                row = []
                image_fields = [k for k in datasets if k.endswith("/rgb") or k == "third_view_rgb"]
                for key in image_fields:
                    marked, std_min, shapes = 0, float("inf"), set()
                    middle = None
                    for idx, bits in enumerate(f[key]):
                        image = codec.decode_image_bit(bits)
                        shapes.add(tuple(image.shape))
                        std_min = min(std_min, float(image.std()))
                        marked += int(codec._has_rgb_marker(bits))
                        if idx == t // 2:
                            middle = image
                    ep["images"][key] = {"frames_decoded": t, "decoded_shapes_hwc": [list(s) for s in sorted(shapes)],
                        "dtype": "uint8", "marked_standard_frames": marked, "legacy_frames": t - marked,
                        "minimum_pixel_std": std_min}
                    if key in ["observation/head_camera/rgb", "observation/left_camera/rgb", "observation/right_camera/rgb"]:
                        tile = Image.fromarray(middle).resize((320, 240))
                        draw = ImageDraw.Draw(tile)
                        draw.rectangle((0, 0, 320, 22), fill="black")
                        draw.text((5, 5), key.split("/")[1] + " frame=" + str(t // 2), fill="white")
                        row.append(tile)
                    prefix = key.rsplit("/", 1)[0]
                    if prefix + "/cam2world_gl" in f:
                        world_gl = f[prefix + "/cam2world_gl"][:].astype(float)
                        cv_world = np.tile(np.eye(4), (t, 1, 1))
                        cv_world[:, :3, :] = f[prefix + "/extrinsic_cv"][:]
                        gl_to_cv = np.diag([1., -1., -1., 1.])
                        residual = cv_world @ world_gl - gl_to_cv
                        ep["camera_frame_checks"][prefix] = {"max_abs_cv_world_times_world_gl_minus_axis_flip": float(np.abs(residual).max())}
                row_image = Image.new("RGB", (960, 270), "white")
                ImageDraw.Draw(row_image).text((5, 5), robot + " / " + path.stem, fill="black")
                for col, tile in enumerate(row):
                    row_image.paste(tile, (col * 320, 30))
                preview_rows.append(row_image)
            report["episodes"].append(ep)
            print("CHECKED", robot, path.stem, "frames", t, "views", len(ep["images"]), flush=True)
    canvas = Image.new("RGB", (960, 270 * len(preview_rows)), "white")
    for idx, row in enumerate(preview_rows):
        canvas.paste(row, (0, 270 * idx))
    canvas.save(args.out / "sample_observations.jpg", quality=90)
    report["all_static_numeric_checks_pass"] = all(
        ep["all_dataset_lengths_match"] and ep["all_numeric_values_finite"] and
        ep["joint_vector_max_abs_difference"] == 0 and
        all(q["max_abs_norm_error"] < 1e-5 for q in ep["quaternions"].values()) and
        all(ep[arm + "_gripper"]["within_0_1"] for arm in ["left", "right"])
        for ep in report["episodes"])
    (args.out / "probe_results.json").write_text(json.dumps(report, indent=2))
    print("STATIC_NUMERIC_CHECKS", report["all_static_numeric_checks_pass"], flush=True)


if __name__ == "__main__":
    main()
