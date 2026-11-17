"""ARX native drive-target policy I/O, without Aloha conversions."""
import numpy as np

from .data import ARM, CAMERAS, DIM, GRIP, HORIZON, PROMPT, absolute, delta

MODEL_DIM = 32
MODEL_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def normalization(x, stats, *, inverse=False):
    low = np.asarray(stats["q01"], dtype=np.float32)
    scale = np.asarray(stats["q99"], dtype=np.float32) - low + np.float32(1e-6)
    return ((x + 1) * 0.5 * scale + low) if inverse else ((x - low) / scale * 2 - 1)


def native_observation(obs):
    """Extract the same sampled command and RGB fields used in the archive."""
    return dict(state=obs["joint_action"]["vector"], prompt=PROMPT,
                images={cam: obs["observation"][cam]["rgb"] for cam in CAMERAS})


class NativeAdapter:
    def __init__(self, stats):
        for name in ("state", "actions"):
            low, high = (np.asarray(stats[name][key]) for key in ("q01", "q99"))
            if (low.shape != (DIM,) or high.shape != (DIM,) or not np.isfinite([low, high]).all()
                    or np.any(high - low <= 1e-6)):
                raise ValueError(f"Invalid {name} normalization")
        self.stats = stats

    def inputs(self, sample):
        state = np.asarray(sample["state"], dtype=np.float32)
        if state.shape != (DIM,) or not np.isfinite(state).all() or np.any((state[GRIP] < 0) | (state[GRIP] > 1)):
            raise ValueError("Expected finite command state [14] with grippers in [0, 1]")
        if sample["prompt"] != PROMPT or set(sample["images"]) != set(CAMERAS):
            raise ValueError("Expected the fixed task prompt and all three ARX cameras")
        images = {}
        for cam, name in zip(CAMERAS, MODEL_CAMERAS, strict=True):
            rgb = np.asarray(sample["images"][cam])
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or not rgb.size:
                raise ValueError(f"Expected RGB HWC uint8 in {cam}")
            images[name] = rgb
        result = dict(state=normalization(state, self.stats["state"]), image=images,
                      image_mask={name: np.bool_(True) for name in images}, prompt=PROMPT)
        if "actions" in sample:
            targets = np.asarray(sample["actions"], dtype=np.float32)
            if (targets.shape != (HORIZON, DIM) or not np.isfinite(targets).all()
                    or np.any((targets[:, GRIP] < 0) | (targets[:, GRIP] > 1))):
                raise ValueError("Expected finite native action targets [50,14] with grippers in [0, 1]")
            result["actions"] = normalization(delta(state, targets), self.stats["actions"])
        # Upstream model transforms resize, tokenize the unpadded state, then right-pad state/actions.
        return result

    def outputs(self, state, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (HORIZON, MODEL_DIM) or not np.isfinite(actions).all():
            raise ValueError("Expected finite model output [50,32]")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (DIM,) or not np.isfinite(state).all():
            raise ValueError("Expected finite raw state [14]")
        targets = absolute(state, normalization(actions[:, :DIM], self.stats["actions"], inverse=True))
        if not np.isfinite(targets).all():
            raise ValueError("Model output overflowed in native units")
        # Return raw commands for error measurement and the same native guard for both policy arms.
        return targets

    def control_outputs(self, state, actions):
        targets = self.outputs(state, actions)
        targets[:, GRIP] = np.clip(targets[:, GRIP], 0, 1)
        return targets
