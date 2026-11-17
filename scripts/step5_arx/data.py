"""Convert the 50 RoboTwin ARX episodes and audit their recorded command semantics."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zlib

import h5py
import numpy as np
from PIL import Image

HORIZON = 50
DIM = 14
CAMERAS = ("head_camera", "left_camera", "right_camera")
ARM = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIP = np.array([6, 13])
PROMPT = "Pick up the red block with the left arm, hand it to the right arm, and place it on the target."
SEED = 42
SCHEMA = "robotwin-arx-x5-drive-target-v1"


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def partitions():
    # Hash sorting makes episode membership independent of NumPy's RNG version.
    ids = sorted(range(50), key=lambda ep: hashlib.sha256(f"step5:{SEED}:{ep}".encode()).digest())
    return dict(train=sorted(ids[:40]), val=sorted(ids[40:45]), test=sorted(ids[45:]))


def labels(q, t):
    if not 0 <= t < len(q) - HORIZON:
        raise IndexError(t)
    return q[t].copy(), q[t + 1:t + HORIZON + 1].copy()


def delta(state, targets):
    actions = targets.copy()
    actions[..., ARM] -= state[..., None, ARM]
    return actions


def absolute(state, actions):
    actions = actions.copy()
    actions[..., ARM] += state[..., None, ARM]
    return actions


def decode_rgb(blob):
    with Image.open(io.BytesIO(bytes(blob))) as frame:
        if frame.format != "JPEG":
            raise ValueError(f"Expected JPEG, found {frame.format}")
        rgb = np.asarray(frame.convert("RGB"))
        # RoboTwin's old cv2 writer took an RGB array as BGR. Marked XPolicyLab
        # JPEGs already have standard channel order; legacy archives need a swap.
        if frame.info.get("comment") == b"XPL-RGB1":
            return rgb
        return rgb[..., ::-1].copy()


def stats(values):
    values = values.astype(np.float64)
    low, high = np.quantile(values, [0.01, 0.99], axis=0)
    if np.any(high - low <= 1e-6):
        raise ValueError("Degenerate training quantile range")
    return dict(mean=values.mean(0).tolist(), std=values.std(0).tolist(),
                q01=low.tolist(), q99=high.tolist())


def inspect_episode(source, destination, ep, expected, summary, samples):
    with source.open("rb") as stream:
        raw = stream.read()
    if len(raw) != expected["bytes"] or f"{zlib.crc32(raw):08x}" != expected["crc32"]:
        raise ValueError(f"Episode {ep} disagrees with pinned ZIP member")
    evidence = dict(episode=ep, source_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    with h5py.File(io.BytesIO(raw), "r") as f:
        fields = {}
        def describe(name, node):
            if isinstance(node, h5py.Dataset):
                dtype = "fixed_bytes" if node.dtype.kind == "S" else str(node.dtype)
                fields[name] = [list(node.shape[1:]), dtype]
        f.visititems(describe)
        if summary.setdefault("fields", fields) != fields or dict(f.attrs):
            raise ValueError(f"Unexpected schema in episode {ep}")
        q = f["joint_action/vector"][:]
        composite = np.column_stack([f[f"joint_action/{name}"][:] for name in
                                    ("left_arm", "left_gripper", "right_arm", "right_gripper")])
        if q.ndim != 2 or q.shape != (len(q), DIM) or len(q) <= HORIZON:
            raise ValueError(f"Bad command dimensions in episode {ep}")
        if not np.isfinite(q).all() or not np.array_equal(q, composite):
            raise ValueError(f"Invalid or reordered commands in episode {ep}")
        g = q[:, GRIP]
        if np.any((g < 0) | (g > 1)) or np.any(g.min(0) != 0) or np.any(g.max(0) != 1):
            raise ValueError(f"Unexpected gripper direction/range in episode {ep}")
        drive_grip = np.column_stack([f[f"endpose/{side}_gripper"][:] for side in ("left", "right")])
        if drive_grip.shape != g.shape or not np.isfinite(drive_grip).all():
            raise ValueError(f"Invalid endpose gripper drive target in episode {ep}")
        evidence.update(
            frames=len(q), windows=len(q) - HORIZON,
            gripper_drive_minus_command_max=np.max(np.abs(drive_grip - g), axis=0).tolist(),
            gripper_drive_diff_frames=np.sum(np.abs(drive_grip - g) > 1e-6, axis=0).tolist(),
            gripper_first_closed=[int(np.flatnonzero(g[:, j] == 0)[0]) for j in range(2)],
            equal_adjacent_commands=int(np.all(q[1:] == q[:-1], axis=1).sum()),
            arm_max_next_saved_change_rad=float(np.abs(np.diff(q[:, ARM], axis=0)).max()),
        )
        for cam in CAMERAS:
            path = f"observation/{cam}"
            rgb = f[f"{path}/rgb"][:]
            if len(rgb) != len(q):
                raise ValueError(f"Misaligned {cam} in episode {ep}")
            for name in ("cam2world_gl", "extrinsic_cv", "intrinsic_cv"):
                matrix = f[f"{path}/{name}"][:]
                if len(matrix) != len(q) or not np.isfinite(matrix).all():
                    raise ValueError(f"Invalid {cam}/{name} in episode {ep}")
            for blob in rgb:
                shape = list(decode_rgb(blob).shape)
                if summary["rgb_shapes"].setdefault(cam, shape) != shape:
                    raise ValueError(f"Unexpected image shape for {cam} in episode {ep}")
            destination.create_dataset(cam, data=rgb)
    q = q.astype(np.float32)
    destination.create_dataset("drive_target", data=q)
    state = q[:-HORIZON]
    chunks = np.stack([labels(q, t)[1] for t in range(len(state))])
    round_trip = np.max(np.abs(absolute(state, delta(state, chunks)) - chunks))
    if round_trip > 5e-7:
        raise ValueError(f"Delta round trip failed in episode {ep}: {round_trip}")
    evidence["delta_roundtrip_max_abs"] = float(round_trip)
    if samples is not None:
        samples["state"].append(state)
        samples["actions"].append(delta(state, chunks).reshape(-1, DIM))
    return evidence


def convert(raw, index, output, report):
    unique = {Path(item["member"]).name: item for item in json.loads(index.read_text())}
    expected = {f"episode{ep}.hdf5" for ep in range(50)}
    if set(unique) != expected or {p.name for p in raw.glob("*.hdf5")} != expected:
        raise ValueError("Expected exactly the pinned episodes 0–49")
    if output.exists() or report.exists():
        raise FileExistsError("Use new output paths")
    samples = {"state": [], "actions": []}
    splits = partitions()
    info = dict(schema=SCHEMA, prompt=PROMPT, prompt_source="fixed task-level prompt",
                timing="next_saved_frame; physical time unknown", horizon=HORIZON,
                source_zip_index_sha256=digest(index), split=splits, rgb_shapes={}, episodes=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        partial = Path(work) / "converted.h5"
        with h5py.File(partial, "w") as dest:
            for name, ids in splits.items():
                for ep in ids:
                    member = f"episode{ep}.hdf5"
                    evidence = inspect_episode(raw / member, dest.create_group(f"{name}/{ep}"), ep,
                                               unique[member], info, samples if name == "train" else None)
                    info["episodes"].append(dict(split=name, **evidence))
            info["norm_stats"] = {k: stats(np.concatenate(parts)) for k, parts in samples.items()}
            dest.attrs["schema"] = SCHEMA
            dest.attrs["prompt"] = PROMPT
            dest.attrs["norm_stats"] = json.dumps(info["norm_stats"], sort_keys=True)
        info["converted_sha256"] = digest(partial)
        # Hard linking publishes the complete artifact without replacing an existing one.
        output.hardlink_to(partial)
    with report.open("x") as stream:
        json.dump(info, stream, indent=2, sort_keys=True)
        stream.write("\n")


class Dataset:
    """Index complete H50 windows by (episode, observation frame)."""
    def __init__(self, path):
        self.file = h5py.File(path, "r")
        if self.file.attrs.get("schema") != SCHEMA or self.file.attrs.get("prompt") != PROMPT:
            self.file.close()
            raise ValueError("Wrong ARX dataset/prompt schema")
        self.norm_stats = json.loads(self.file.attrs["norm_stats"])
        self.groups = {name: {int(ep): g for ep, g in self.file[name].items()}
                       for name in partitions()}
        if {name: sorted(groups) for name, groups in self.groups.items()} != partitions():
            self.file.close()
            raise ValueError("Episode membership differs from pinned splits")

    def indices(self, split, stride=1):
        for ep, group in sorted(self.groups[split].items()):
            last = len(group["drive_target"]) - HORIZON - 1
            for t in sorted({*range(0, last + 1, stride), last}):
                yield ep, t

    def read(self, split, ep, t):
        group = self.groups[split][ep]
        q = group["drive_target"]
        if not 0 <= t < len(q) - HORIZON:
            raise IndexError(t)
        return dict(state=q[t], actions=q[t + 1:t + HORIZON + 1], prompt=PROMPT,
                    images={cam: decode_rgb(group[cam][t]) for cam in CAMERAS})

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "zip-index", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    convert(args.raw, args.zip_index, args.output, args.report)


if __name__ == "__main__":
    main()
