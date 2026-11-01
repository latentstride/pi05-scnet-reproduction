"""CPU-only audit of released PACE stats against the six existing samples.

This checks scale coverage only; it does not run policy inference or establish
that target joint channels represent the same physical robot configuration.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import h5py

root = Path(__file__).resolve().parents[1]
audit = root / "research/gate25"
stats_path = audit / "pace_handover_block_assets_robotwin_norm_stats.json"
stats = json.loads(stats_path.read_text())["norm_stats"]
manifest = json.loads((audit / "pace_handover_manifest.json").read_text())
assert hashlib.sha256(stats_path.read_bytes()).hexdigest() == manifest["norm_stats_sha256"]
checks = {}
for name, fields in stats.items():
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in fields.items()}
    assert all(x.shape == (14,) and np.isfinite(x).all() for x in arrays.values())
    assert np.all(arrays["q99"] > arrays["q01"])
    checks[name] = {"dimensions": 14, "finite": True, "positive_quantile_spans": True}
lo, hi = np.asarray(stats["state"]["q01"]), np.asarray(stats["state"]["q99"])
rows = []
for robot in ["aloha-agilex", "arx-x5"]:
    samples = []
    for path in sorted((root / "research/robotwin/data_probe" / robot).glob("*.hdf5")):
        with h5py.File(path, "r") as handle:
            samples.append(handle["joint_action/vector"][:])
    assert len(samples) == 3
    states = np.concatenate(samples)
    outside = (states < lo) | (states > hi)
    rows.append({"robot": robot, "episodes": len(samples), "frames": len(states),
                 "state_shape": list(states.shape),
                 "fraction_scalar_values_outside_source_q01_q99": float(outside.mean()),
                 "fraction_frames_any_channel_outside": float(outside.any(axis=1).mean()),
                 "fraction_joint_scalar_values_outside": float(outside[:, [0,1,2,3,4,5,7,8,9,10,11,12]].mean()),
                 "outside_fraction_by_channel": outside.mean(axis=0).tolist()})
result = {"date": "2026-11-02", "stats_sha256_matches_release_manifest": True,
          "stats_checks": checks, "samples": rows, "gpu_used": False,
          "weights_loaded": False, "not_a_capability_or_safety_test": True,
          "note": "No clipping, target statistics substitution, or action relabeling was performed."}
(audit / "stats_probe.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
