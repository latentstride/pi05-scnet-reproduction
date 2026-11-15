"""Fit the known upright box to the archived head RGB silhouette on the table."""
import argparse
from hashlib import file_digest
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import yaml

from g2_initial import matrix7

CORNERS = np.array(list(product((-.03, .03), (-.03, .03), (-.10, .10))))


def fit(image, camera, plane_z, fovy):
    red, green, blue = image.astype(float)
    y, x = np.where((red > 100) & (red > 1.7 * green) & (red > 1.7 * blue))
    h, w = red.shape
    if not len(x) or x.min() == 0 or x.max() == w - 1 or y.min() == 0 or y.max() == h - 1:
        raise ValueError("no unclipped red box silhouette in the head image")
    u0, v0 = w / 2, h / 2
    focal = v0 / np.tan(np.deg2rad(fovy / 2))
    observed = np.array([x.min(), y.min(), x.max() + 1, y.max() + 1])
    forward, left = (np.array(camera[key]) / np.linalg.norm(camera[key]) for key in ("forward", "left"))
    rotation = np.stack((forward, left, np.cross(forward, left)), axis=1)
    origin = np.asarray(camera["position"])
    z = plane_z + .10

    def residual(p):
        px, py, yaw = p
        c, s = np.cos(yaw), np.sin(yaw)
        box_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        cam = (CORNERS @ box_rotation.T + [px, py, z] - origin) @ rotation
        xy = [u0 - focal * cam[:, 1] / cam[:, 0], v0 - focal * cam[:, 2] / cam[:, 0]]
        return np.array([min(xy[0]), min(xy[1]), max(xy[0]), max(xy[1])]) - observed

    u, v = (observed[:2] + observed[2:]) / 2
    ray = rotation @ [1, (u0-u)/focal, (v0-v)/focal]
    guess = (origin + ray * ((z - origin[2]) / ray[2]))[:2]
    candidates = [least_squares(residual, [*guess, yaw]) for yaw in np.linspace(-np.pi/4, np.pi/4, 5)]
    result = min(candidates, key=lambda r: r.cost)
    return np.r_[result.x[:2], z], result.x[2], observed, max(np.abs(result.fun))


def main(args):
    cfg = yaml.safe_load(args.camera.read_text())
    camera, = [c for c in cfg["static_camera_list"] if c["name"] == "head_camera"]
    fov = yaml.safe_load(args.intrinsics.read_text())["D435"]["fovy"]
    trial = json.loads(args.snapshot.read_text())
    table = matrix7(trial["initial"]["objects_task"]["table"])
    world_task = np.asarray(trial["request"]["world_task"])
    plane_z = (world_task @ table)[2, 3]
    with np.load(args.image, allow_pickle=False) as files:
        estimate, yaw, bounds, residual = fit(files["cam_high"], camera, plane_z, fov)
    # PhysX ground truth is held out until evaluation.
    oracle = (world_task @ np.asarray(trial["request"]["objects_task"]["block"]))[:3, 3]
    world_tcp = world_task @ np.asarray(trial["at"]["actual_tools_task"][0])
    with args.image.open("rb") as stream:
        image_digest = file_digest(stream, "sha256").hexdigest()
    report = dict(job=trial["job"], query=trial["query"], image_sha256=image_digest,
                  source="archived_head_rgb_and_known_upright_box_on_table",
                  estimate_xyz_world_m=estimate.tolist(), fitted_yaw_deg=float(np.rad2deg(yaw)),
                  comparison_oracle_xyz_world_m=oracle.tolist(), xy_error_mm=float(1000*np.linalg.norm((estimate-oracle)[:2])),
                  tcp_lateral_error_mm=float(1000*np.dot(world_tcp[:3, 1], estimate-oracle)),
                  red_bounds_xyxy=bounds.tolist(), maximum_fit_error_px=float(residual))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "image", "camera", "intrinsics", "output"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
