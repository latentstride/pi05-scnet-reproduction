"""Render signed bias against the archived left-gripper requests."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(directory):
    arx = json.loads((directory / "first_query_analysis.json").read_text())["arx"]
    aloha = json.loads((directory / "aloha_arx_comparison.json").read_text())["aloha"]
    colors = ("#9e2433", "#cf5838", "#723a95", "#277c85")
    figure, (bias, grip) = plt.subplots(2, 1, figsize=(9.0, 6.1), sharex=True, layout="constrained",
                                         gridspec_kw={"height_ratios":[3,1]})
    for (name, run), color in zip(arx.items(), colors):
        rows = run["rows"]
        x = [r["row"] for r in rows]
        y = [r["tool_geometry"]["block_xyz_jaw_mm"][0][1] for r in rows]
        bias.plot(x, y, label=f"ARX {name}", color=color, linewidth=1.8)
        grip.plot(x, [r["gripper_request"][0] for r in rows], color=color, linewidth=1.8)
        event = run["physical"]["events"].get("first_single")
        if event:
            i = event["row"] - 1
            bias.scatter(x[i], y[i], color=color, edgecolor="white", s=55, zorder=3)
    native = np.array([[r["tool_geometry"]["block_xyz_jaw_mm"][0][1] for r in run["q1_rows"]]
                       for run in aloha.values()])
    bias.fill_between(x, native.min(axis=0), native.max(axis=0), color="#a2b3c3", alpha=.37,
                      label="Aloha success range (3)")
    for n, run in enumerate(aloha.values()):
        goal = run["first_contact_target"]
        if goal["query"] == 1:
            bias.scatter(goal["row"], goal["tool_geometry"]["block_xyz_jaw_mm"][0][1], color="#37495b",
                         marker="x", zorder=4, label="Aloha first single contact" if n == 0 else None)
    bias.axhline(0, color="#777", linewidth=.7)
    bias.set_title("First policy chunk before contact feedback", loc="left", weight="bold")
    bias.set_ylabel("Block in left jaw frame, signed Y (mm)")
    bias.legend(frameon=False, ncol=2, loc="upper right")
    grip.axhline(.5, color="#777", linewidth=.8, linestyle="--", label="Effective-close diagnostic cutoff")
    grip.axhline(.95, color="#bbb", linewidth=.7, linestyle=":")
    grip.set(xlabel="Archived row (1-based)", ylabel="Requested aperture", ylim=(.45,1.03), xlim=(1,50))
    grip.legend(frameon=False, loc="lower left")
    fig_note = "ARX dots mark the commanded row at first physical single-finger contact. G4 executed zero rows."
    figure.supxlabel(fig_note, fontsize=8, color="#45505d")
    for axis in (bias,grip):
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=.5)
    figure.savefig(directory / "first_query_bias.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot(parser.parse_args().directory)
