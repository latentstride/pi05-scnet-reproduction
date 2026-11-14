"""Corroborate seed-derived object placement with original first-frame pixels."""
import argparse
from io import BytesIO
from itertools import product
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from robotwin_control_geometry import pose_from_wxyz


def verify(raw, prepared):
    output = []
    for path in sorted(prepared.glob("*/*/input.json")):
        item = json.loads(path.read_text())
        if not item["prepared"]:
            continue
        hdf5 = raw / item["robot"] / f"episode{item['episode']}.hdf5"
        with h5py.File(hdf5) as file:
            camera = file["observation/head_camera"]
            intrinsic, extrinsic = camera["intrinsic_cv"][0], camera["extrinsic_cv"][0][:3, :]
            with Image.open(BytesIO(bytes(camera["rgb"][0]))) as image:
                pixels = np.array(image.convert("RGB"), dtype=int)
                legacy = image.info.get("comment") != b"XPL-RGB1"
        if legacy:
            pixels = pixels[:, :, ::-1]
        red, green, blue = np.moveaxis(pixels, -1, 0)
        objects = {}
        for name, half, mask in (
            ("box", [.03, .03, .1], (red > 100) & (red > 1.7 * green) & (red > 1.7 * blue)),
            ("target_box", [.05, .05, .005], (blue > 70) & (blue > 1.7 * red) & (blue > 1.7 * green)),
        ):
            spawn = item["spawn_pose"][name]
            matrix = pose_from_wxyz(spawn["position_m"], spawn["quaternion_wxyz"])
            corners = np.array(list(product(*[(-v, v) for v in half])))
            xyz = corners @ matrix[:3, :3].T + matrix[:3, 3]
            homogeneous = (xyz @ extrinsic[:, :3].T + extrinsic[:, 3]) @ intrinsic.T
            uv = homogeneous[:, :2] / homogeneous[:, 2, None]
            height, width = mask.shape
            predicted = np.clip(np.r_[uv.min(0), uv.max(0)], 0, [width-1, height-1]*2)
            y, x = np.where(mask)
            observed = np.array([x.min(), y.min(), x.max(), y.max()]) if len(x) else None
            objects[name] = dict(predicted=predicted.tolist(), observed=None if observed is None else observed.tolist(),
                                 max_edge_error_px=None if observed is None else float(abs(predicted-observed).max()))
        output.append(dict(robot=item["robot"], episode=item["episode"], seed=item["seed"], objects=objects))
    return dict(scope="2D color bounds, not an original physical state measurement", episodes=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("raw", "prepared", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    result = verify(args.raw, args.prepared)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name in ("aloha-agilex", "arx-x5"):
        values = [r["objects"]["target_box"]["max_edge_error_px"] for r in result["episodes"] if r["robot"] == name]
        print(name, len(values), "target box max pixel residual", max(v for v in values if v is not None))
