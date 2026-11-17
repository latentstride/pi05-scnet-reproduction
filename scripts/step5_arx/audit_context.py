"""Compare archived native CPU replay traces and inspect available language sidecars."""
import argparse
import importlib.util
import json
from pathlib import Path
import re

import h5py
import numpy as np

from .config import write_json
from .data import ARM, CAMERAS, GRIP, PROMPT, decode_rgb, digest

TRACE_ARM = [*range(6), *range(8, 14)]
TRACE_NAMES = [f"{side}:joint{j}" for side in ("left", "right") for j in range(1, 9)]


def audit(raw, replay, metadata, decoder_path):
    spec = importlib.util.spec_from_file_location("official_decoder", decoder_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    images_checked = 0
    details = []
    arm_gaps = []
    grip_gaps = []
    for ep in range(50):
        trace = replay / f"episode{ep:02}/attempt/physics-trace.npz"
        with (np.load(trace) as z, h5py.File(raw / f"episode{ep}.hdf5", "r") as f):
            ticks = z["frame_ticks"]
            actual, drive = z["actual_qpos"], z["native_drive"]
            q = f["joint_action/vector"][:]
            if (ticks.shape != (len(q),) or not np.issubdtype(ticks.dtype, np.integer)
                    or ticks.min() < 0 or ticks.max() >= len(drive) or np.any(np.diff(ticks) < 0)
                    or actual.shape != drive.shape or drive.shape[1:] != (16,)
                    or z["joint_names"].tolist() != TRACE_NAMES or not np.isfinite([actual, drive]).all()):
                raise ValueError(f"Unexpected replay schema or frame mapping in episode {ep}")
            for camera in CAMERAS:
                for blob in f[f"observation/{camera}/rgb"][:]:
                    np.testing.assert_array_equal(decode_rgb(blob), official.decode_image_bit(blob))
                    images_checked += 1
            np.testing.assert_array_equal(drive[ticks][:, TRACE_ARM], q[:, ARM])
            np.testing.assert_array_equal(z["normalized_gripper_request"][ticks], q[:, GRIP])
            arms = np.abs(actual[ticks][:, TRACE_ARM] - drive[ticks][:, TRACE_ARM])
            grips = np.abs(actual[ticks][:, [6, 7, 14, 15]] - drive[ticks][:, [6, 7, 14, 15]])
            arm_gaps.append(arms)
            grip_gaps.append(grips)
            details.append(dict(episode=ep, matched_saved_rows=len(ticks), replay_sha256=digest(trace),
                                arm_gap_max_rad=float(arms.max()), arm_gap_rms_rad=float(np.sqrt((arms**2).mean())),
                                finger_gap_max_m=float(grips.max())))
    prompts = []
    for path in sorted(metadata.glob("episode*.json")):
        number = re.fullmatch(r"episode(\d+)\.json", path.name)
        if number is None or int(number[1]) not in range(50):
            raise ValueError(f"Unexpected language sidecar {path.name}")
        payload = json.loads(path.read_text())
        if not all(isinstance(payload[k], list) and payload[k]
                   and all(isinstance(v, str) and v.strip() for v in payload[k]) for k in ("seen", "unseen")):
            raise ValueError(f"Invalid task language in {path.name}")
        prompts.append(dict(episode=int(number[1]), sha256=digest(path), seen=len(payload["seen"]),
                            unseen=len(payload["unseen"]), seen_example=payload["seen"][0],
                            selected_prompt_is_exact_match=any(PROMPT in payload[k] for k in ("seen", "unseen"))))
    if sorted(row["episode"] for row in prompts) != list(range(50)):
        raise ValueError("Expected language sidecars for all 50 pinned episodes")
    gap = np.concatenate(arm_gaps)
    fingers = np.concatenate(grip_gaps)
    return dict(replay_provenance="native CPU expert replay, evidence commit b560854",
                replay_samples=details, matched_saved_rows=len(gap), arm_columns=TRACE_ARM,
                rgb_exact_comparisons=images_checked, reference_decoder_sha256=digest(decoder_path),
                arm_gap_rad={k: float(v) for k, v in zip(("median", "p95", "p99", "max"),
                             [*np.quantile(gap, [0.5, 0.95, 0.99]), gap.max()], strict=True)},
                arm_gap_rms_rad=float(np.sqrt((gap**2).mean())),
                finger_gap_m={k: float(v) for k, v in zip(("median", "p95", "max"),
                              [*np.quantile(fingers, [0.5, 0.95]), fingers.max()], strict=True)},
                language=dict(fixed_task_prompt=PROMPT, sidecar_count=len(prompts), sidecars=prompts,
                              absent_episodes=sorted(set(range(50)) - {r["episode"] for r in prompts})))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "replay", "metadata", "decoder", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    write_json(args.output, audit(args.raw, args.replay, args.metadata, args.decoder))


if __name__ == "__main__":
    main()
