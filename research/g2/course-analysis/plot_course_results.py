"""Plot frozen Aloha and RoboTwin course results. Export vector PDF and 300 dpi PNG."""
# Panels: paired trials, synchronized time series, episode scatter, and empirical CDF.
# Layouts inherit BarComparison, LineTrend, PairedBoxScatter and ResidualDiagnostics.
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})


def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300,
                metadata={"CreationDate": None})
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


import argparse
from collections import Counter, defaultdict
import csv
import gzip
from hashlib import file_digest
import io
import json
from pathlib import Path
import warnings

from matplotlib import font_manager, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ALOHA = ROOT / "research/g2/aloha-six-20261111"
EXPERT = ROOT / "research/robotwin100_course/data"
BLUE, RED, GREEN = CATEGORICAL[:3]
ROBOTS = {"aloha-agilex": "Aloha-AgileX", "arx-x5": "ARX-X5"}
HANDOVER_FIELDS = ("trial", "tick", "action", "relative_s", "height_mm", "vertical_m_s",
                   "left_aperture_mm", "right_aperture_mm", "left_contact_fingers", "right_contact_fingers")


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def unpack(path):
    return json.loads(path.read_bytes())


def font_setup(path):
    font_manager.fontManager.addfont(str(path))
    font = font_manager.FontProperties(fname=path)
    mpl.rcParams["font.sans-serif"] = [font.get_name(), "DejaVu Sans"]
    smoke = plt.figure()
    smoke.text(.1, .5, r"实测夹爪开度 －1 mm，残差 0.05°，$\pi_{0.5}$")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing")
        smoke.canvas.draw()
    plt.close(smoke)


def archive_window(raw_root, summary, output):
    metadata = {}
    # Preserve every physics tick in the displayed interval. Verify each
    # original log in full before reading any samples for the figure.
    with output.open("xb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
            writer = csv.DictWriter(text, fieldnames=HANDOVER_FIELDS, lineterminator="\n")
            writer.writeheader()
            for g in ("G3", "G6"):
                item = next(row for row in summary["rows"] if row["experiment"] == g)
                relative, expected = next((path, h) for path, h in item["source_sha256"].items()
                                          if path.endswith("/physics.jsonl"))
                path = raw_root / relative
                if sha(path) != expected:
                    raise ValueError(f"{g} PhysX source hash mismatch")
                events = unpack(ALOHA / g / "stages.json")
                anchor = events["first_contact"]["both_two"]["tick"]
                transition = events["first_right_two_without_left_two_after_both"]["tick"]
                total = selected = 0
                with path.open() as source:
                    for line in source:
                        sample = json.loads(line)
                        tick = sample["tick"]
                        if total == 0:
                            baseline = sample["block_pose"][2][3]
                            dt = sample["step_period_s"]
                        if tick != total:
                            raise ValueError(f"{g} missing or disordered physics ticks")
                        total += 1
                        elapsed = (tick - anchor) * dt
                        if not -2 <= elapsed <= 13:
                            continue
                        contacts = {name for contact in sample["contacts"] if contact["points"]
                                    for name in contact["pair"]}
                        writer.writerow(dict(trial=g, tick=tick, action=sample["action"],
                                             relative_s=f"{elapsed:.8f}",
                                             height_mm=f'{(sample["block_pose"][2][3] - baseline) * 1000:.8f}',
                                             vertical_m_s=sample["block_linear_velocity"][2],
                                             left_aperture_mm=sample["aperture_m"][0] * 1000,
                                             right_aperture_mm=sample["aperture_m"][1] * 1000,
                                             left_contact_fingers=len(contacts & {"fl_link7", "fl_link8"}),
                                             right_contact_fingers=len(contacts & {"fr_link7", "fr_link8"})))
                        selected += 1
                if total != item["physics_steps"] + 1:
                    raise ValueError(f"{g} physics count differs from terminal receipt")
                metadata[g] = dict(source_path=relative, source_sha256=expected, samples=selected,
                                   original_samples=total, anchor_tick=anchor, transition_tick=transition,
                                   step_period_s=dt)
    return metadata


def heading(fig, title, subtitle):
    fig.text(.07, .97, title, fontsize=15, fontweight="bold", va="top", color=BLACK)
    fig.text(.07, .915, subtitle, fontsize=9, va="top", color="#444444")


def foot(fig, text):
    fig.text(.07, .029, text, fontsize=7.8, va="bottom", color="#444444")


def axes_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6E6E6", linewidth=.55)


def panels(fig, left=.10, right=.97, bottom=.20, top=.79):
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=.38)


def compare_aloha(summary):
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 4.5))
    heading(fig, "三个场景中的 Aloha 原生闭环", "同场景各运行一次  ·  Fixed H50 成功 3/3  ·  PACE 成功 2/3")
    rows = {(r["scene_seed"], r["selector"]["selector"]): r for r in summary["rows"]}
    scenes = sorted({seed for seed, _ in rows})
    assert len(rows) == 6 and len(scenes) == 3
    for ax, (metric, label) in zip(axes, (("completed_actions", "a  完成动作数"), ("queries", "b  模型查询次数"),
                                             ("physics_steps", "c  物理步数（千步）"))):
        divisor = 1000 if metric == "physics_steps" else 1
        for x, seed in enumerate(scenes):
            pair = [rows[seed, method] for method in ("fixed", "pace")]
            values = [r[metric] / divisor for r in pair]
            ax.plot([x-.14, x+.14], values, color="#B9B9B9", lw=1, zorder=1)
            for offset, value, row, color in zip((-.14, .14), values, pair, (BLUE, RED)):
                ax.scatter(x+offset, value, marker="o" if row["success"] else "X",
                           color=color, s=55, zorder=3, linewidths=.5, edgecolors="white")
                annotation = f"{value:.1f}" if divisor > 1 else str(int(value))
                ax.annotate(annotation, (x+offset, value), xytext=(0, 8 if offset > 0 else -11),
                            textcoords="offset points", ha="center", color=color, fontsize=7)
        ax.set(xticks=range(3), xticklabels=[str(s) for s in scenes], xlim=(-.43, 2.43),
               ylim=(0, ax.get_ylim()[1] * 1.13), xlabel="场景编号")
        ax.set_title(label, loc="left", fontweight="bold")
        axes_style(ax)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="Fixed H50"),
                        Line2D([], [], marker="o", ls="", color=RED, label="PACE"),
                        Line2D([], [], marker="X", ls="", color=BLACK, label="未完成")],
               loc="upper right", bbox_to_anchor=(.97, .90), ncols=3)
    foot(fig, "每点为一条实跑轨迹。三个配对的首块动作均不一致，结果用于描述性比较。")
    panels(fig, top=.74, bottom=.20)
    return fig


def handover(path, metadata):
    groups = defaultdict(list)
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            groups[row["trial"]].append(row)
    fig, axes = plt.subplots(4, 2, figsize=(9.2, 7.3), sharex=True, sharey="row")
    heading(fig, "场景 300000 的交接对照", "时间零点为各自首次双手四指同时接触；虚线为右双指仍接触、左侧失去双指接触")
    labels = ("物块高度变化（mm）", "物块垂直速度（m/s）", "实测指距（mm）", "指物接触指数")
    for col, (g, method) in enumerate((("G3", "Fixed H50"), ("G6", "PACE"))):
        seq = groups[g]
        meta = metadata[g]
        assert len(seq) == meta["samples"]
        assert np.all(np.diff([int(row["tick"]) for row in seq]) == 1)
        t = np.array([float(row["relative_s"]) for row in seq])
        value = lambda field: np.array([float(row[field]) for row in seq])
        threshold = (meta["transition_tick"] - meta["anchor_tick"]) * meta["step_period_s"]
        axes[0, col].set_title(f"{g}  {method}", loc="left", fontsize=11, fontweight="bold")
        for i, ax in enumerate(axes[:, col]):
            if col == 0:
                ax.set_ylabel(labels[i])
            ax.axvline(0, color=GREY, lw=.65)
            ax.axvline(threshold, color=BLACK, lw=.8, ls="--")
            axes_style(ax)
        axes[0, col].plot(t, value("height_mm"), color=BLUE, lw=1.2)
        axes[1, col].plot(t, value("vertical_m_s"), color=BLUE, lw=.75)
        for side, color, style in (("left", BLUE, "-"), ("right", RED, "--")):
            axes[2, col].plot(t, value(f"{side}_aperture_mm"), color=color, ls=style, lw=1)
            axes[3, col].step(t, value(f"{side}_contact_fingers"), color=color, ls=style, lw=.8, where="post")
        axes[3, col].set(yticks=[0, 1, 2], ylim=(-.15, 2.15), xlim=(-2, 13), xlabel="相对时间（s）")
    fig.legend(handles=[Line2D([], [], color=BLUE, label="左手"), Line2D([], [], color=RED, ls="--", label="右手")],
               loc="upper right", bbox_to_anchor=(.973, .89), ncols=2)
    foot(fig, "逐物理步记录；接触指数为与物块接触的手指数。G6 下落后仍有零星接触。")
    fig.subplots_adjust(left=.11, right=.97, top=.824, bottom=.115, wspace=.14, hspace=.23)
    return fig


def expert_statistics():
    episodes = defaultdict(list)
    with (EXPERT / "trajectory_quality.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "read_ok":
                raise ValueError(f"unexpected episode status: {row['status']}")
            episodes[row["robot"].casefold()].append(row)
    counts = defaultdict(Counter)
    residuals = defaultdict(list)
    with (EXPERT / "ik_results_full.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            robot = row["source_robot"].casefold()
            counts[robot][row["status"]] += 1
            if row["status"] == "kinematic_solution":
                residuals[robot].append((float(row["position_mm"]), float(row["angle_deg"])))
    stats = {}
    for robot in ROBOTS:
        records = episodes[robot]
        frames = np.array([int(r["frames"]) for r in records])
        duplicates = np.array([int(r["state_duplicate_rows"]) for r in records])
        values = np.array(residuals[robot])
        total = counts[robot].total()
        assert len(records) == 50 and len({r["episode"] for r in records}) == 50
        assert total == 2 * sum(frames) and np.all(np.isfinite(values))
        assert np.all(values[:, 0] <= .5) and np.all(values[:, 1] <= .05)
        stats[robot] = dict(episodes=len(records), frames=int(sum(frames)),
                            frames_min_median_max=[int(min(frames)), float(np.median(frames)), int(max(frames))],
                            images=sum(int(r["images"]) for r in records),
                            duplicate_states=int(sum(duplicates)),
                            all_ik_targets=total, solved=len(values), unsolved=total-len(values),
                            solve_rate_pct=len(values)/total*100,
                            position_mm_p50_p95_max=np.percentile(values[:, 0], [50, 95, 100]).tolist(),
                            angle_deg_p50_p95_max=np.percentile(values[:, 1], [50, 95, 100]).tolist())
    assert sum(s["frames"] for s in stats.values()) == 23814
    assert sum(s["solved"] for s in stats.values()) == 47541
    return episodes, residuals, stats


def plot_experts(episodes, stats):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.7))
    heading(fig, "两种本体的专家轨迹分布", "各 50 条轨迹  ·  合计 23,814 帧  ·  95,256 张图像")
    for j, (ax, label) in enumerate(zip(axes, ("a  记录长度（帧）", "b  重复状态占比（%）"))):
        for i, (robot, color) in enumerate(zip(ROBOTS, (BLUE, GREEN))):
            records = sorted(episodes[robot], key=lambda r: int(r["episode"]))
            frames = np.array([int(r["frames"]) for r in records])
            y = frames if j == 0 else 100 * np.array([int(r["state_duplicate_rows"]) for r in records]) / frames
            x = i + np.linspace(-.17, .17, len(records))
            ax.scatter(x, y, color=color, s=13, alpha=.67, linewidths=0)
            ax.hlines(np.median(y), i-.25, i+.25, colors=BLACK, linewidth=1.45, zorder=3)
            ax.annotate(f"{np.median(y):.0f}" if j == 0 else f"{np.median(y):.1f}%",
                        (i+.27, np.median(y)), va="center", ha="left", fontsize=8)
        ax.set(xlim=(-.43, 1.57), xticks=(0, 1), xticklabels=list(ROBOTS.values()))
        ax.set_title(label, loc="left", fontweight="bold")
        axes_style(ax)
    foot(fig, "每点一条轨迹，黑横线为轨迹中位数。帧数不换算为任务时长，重复状态全部保留。")
    panels(fig, left=.10, top=.80)
    return fig


def plot_ik(residuals, stats):
    fig, axes = plt.subplots(1, 3, figsize=(9.7, 4.6), gridspec_kw={"width_ratios":[1, 1.25, 1.25]})
    heading(fig, "跨本体几何筛查与残差分布", "47,628 个单臂位姿目标  ·  87 个有限搜索未解  ·  名义夹具")
    for index, (robot, color, ls, label) in enumerate(zip(
            ROBOTS, (BLUE, GREEN), ("-", "--"), ("Aloha → ARX", "ARX → Aloha"))):
        row = stats[robot]
        axes[0].scatter(index, row["solve_rate_pct"], s=48, color=color, zorder=3)
        axes[0].annotate(f'{row["solved"]:,}/{row["all_ik_targets"]:,}',
                         (index, row["solve_rate_pct"]), xytext=(0, -13),
                         textcoords="offset points", ha="center", fontsize=7.8, color=color)
        values = np.array(residuals[robot])
        for axis, column in zip(axes[1:], values.T):
            ordered = np.sort(column)
            axis.step(ordered, np.arange(1, len(ordered)+1) * 100 / len(ordered),
                      where="post", color=color, ls=ls, lw=1.4, label=label)
    axes[0].set(xticks=[0, 1], xticklabels=["正向", "反向"], xlim=(-.5, 1.5), ylim=(98.8, 100.15))
    for ax, title in zip(axes, ("a  有解率（%）", "b  位置残差（mm）", "c  姿态残差（°）")):
        ax.set_title(title, loc="left", fontweight="bold")
        axes_style(ax)
    for ax in axes[1:]:
        ax.set_ylim(0, 105)
    axes[1].set(ylabel="累计比例（%）", xlim=(0, .35))
    axes[2].set_xlim(0, .052)
    axes[2].axvline(.05, lw=.7, ls=":", color=BLACK)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.97, .91), ncols=2)
    foot(fig, "有解率包含全部尝试；残差累计曲线仅统计有解目标。该筛查未纳入碰撞、接触或时序。")
    panels(fig, left=.087, top=.75, bottom=.19)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for generated artifacts")
    parser.add_argument("--font", type=Path, required=True, help="installed Chinese font file")
    parser.add_argument("--raw-root", type=Path, help="verified G3/G6 PhysX log root to rebuild window")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    font_setup(args.font)
    summary = unpack(ALOHA / "summary.json")
    assert all(pair["first_chunk_exact_equal"] is False for pair in summary["pairs"])
    samples = HERE / "g3-g6-window.csv.gz"
    if args.raw_root:
        samples = args.output / samples.name
        metadata = archive_window(args.raw_root, summary, samples)
    else:
        source = unpack(HERE / "figure-data.json")
        metadata = source["handover_window"]
        if sha(samples) != source["input_sha256"][samples.name]:
            raise ValueError("handover window hash differs")
    episodes, residuals, stats = expert_statistics()
    figures = {
        "aloha-three-scenes": compare_aloha(summary),
        "aloha-g3-g6-handover": handover(samples, metadata),
        "expert-trajectory-statistics": plot_experts(episodes, stats),
        "expert-cross-embodiment-ik": plot_ik(residuals, stats),
    }
    for name, fig in figures.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Glyph .* missing")
            fig.canvas.draw()
            save_cns_figure(fig, args.output / name)
        plt.close(fig)
    data = dict(handover_window=metadata, expert=stats, input_sha256={
        samples.name: sha(samples), "aloha-summary.json": sha(ALOHA / "summary.json"),
        **{name: sha(EXPERT / name) for name in ("trajectory_quality.csv", "ik_results_full.csv")}})
    (args.output / "figure-data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
