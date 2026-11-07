"""Plot each recorded K26 hold step and all eight prediction summaries."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
with (root / "k26-slip.csv").open() as stream:
    samples = list(csv.DictReader(stream))
data = json.loads((root / "failure-evidence.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
ticks = [int(r["tick"]) for r in samples]
axes[0].plot(ticks, [float(r["drift_mm"]) for r in samples], color="#137a75", label="Measured drift")
axes[0].axhline(5, ls="--", color="#a84a35", label="Original limit")
axes[0].set(xlabel="Physics step", ylabel="Tool–block drift (mm)", title="Seed 1 · K26")
axes[0].legend(frameon=False)
labels, outside, inside = [], [], []
for seed in (0, 1):
    for k in (1, 4, 26, 31):
        trial = data["trials"][f"g2e-seed{seed}-native-w{k}"]
        for side in ("left", "right"):
            hand = trial["hands"][side]
            labels.append(f"{seed}/{k} {side[0].upper()}")
            inside.append(hand["in_prefix"])
            outside.append(hand["predictions"]-hand["in_prefix"])
positions = np.arange(len(labels))
axes[1].bar(positions, outside, color="#9da9b1", label="After prefix")
axes[1].bar(positions, inside, bottom=outside, color="#137a75", label="Within prefix")
axes[1].set(xlabel="Seed / K and hand", ylabel="Prediction blocks", title="First predicted closure")
axes[1].set_xticks(positions, labels, rotation=60, fontsize=7)
axes[1].legend(frameon=False)
for axis in axes:
    axis.spines[["right", "top"]].set_visible(False)
fig.savefig(root/"failure-evidence.pdf")
fig.savefig(root/"failure-evidence.png", dpi=220)
