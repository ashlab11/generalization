"""Build cleaner schoop plots from output folders or stats.json files."""

import argparse
import glob
import json
import os
from datetime import datetime

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import OmegaConf


COLORS = ["#d55e00", "#0072b2", "#cc79a7", "#56b4e9", "#009e73", "#e69f00", "#6a3d9a", "#1b9e77"]
LINESTYLES = ["-", "--", "-.", ":"]


def get_stats_files(input_path):
    input_path = os.path.expanduser(input_path)
    if os.path.isfile(input_path):
        if os.path.basename(input_path) != "stats.json":
            raise ValueError(f"Expected a stats.json file, got: {input_path}")
        return [input_path]
    if not os.path.isdir(input_path):
        raise ValueError(f"Path does not exist: {input_path}")

    files = []
    direct_stats = os.path.join(input_path, "stats.json")
    if os.path.isfile(direct_stats):
        files.append(direct_stats)

    patterns = [
        os.path.join(input_path, "**", "*training*", "stats.json"),
        os.path.join(input_path, "**", "*testing*", "stats.json"),
    ]
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    files = sorted(set(files))
    if not files:
        raise ValueError(f"No stats.json found under: {input_path}")
    return files


def get_cfg_value(cfg, key_path, default=None):
    cur = cfg
    for key in key_path:
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def style_for_label(label, idx):
    return {
        "color": COLORS[idx % len(COLORS)],
        "linestyle": LINESTYLES[(idx // len(COLORS)) % len(LINESTYLES)],
        "linewidth": 2.4,
    }


def rows_from_stats(stats_path):
    with open(stats_path, "r") as fp:
        stats = json.load(fp)

    run_dir = os.path.dirname(stats_path)
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path) if os.path.isfile(cfg_path) else {}

    problem = get_cfg_value(cfg, ["problem", "name"], "unknown")
    test_data = stats.get("test_data", get_cfg_value(cfg, ["problem", "test_data"], "unknown"))
    run_id = get_cfg_value(cfg, ["run_id"], stats.get("run_id", os.path.basename(run_dir)))

    rows = []
    for test_iter in stats.get("test_iters", []):
        key = str(test_iter)
        if key not in stats.get("test_acc", {}):
            continue
        rows.append({
            "problem": problem,
            "test_data": test_data,
            "run_id": str(run_id),
            "test_iter": int(test_iter),
            "test_acc": float(stats["test_acc"][key]),
            "max_iters": int(stats.get("max_iters", -1)),
            "run_dir": run_dir,
        })
    return rows


def build_table(stats_files):
    rows = []
    for stats_path in stats_files:
        rows.extend(rows_from_stats(stats_path))
    if not rows:
        raise ValueError("No usable rows found in stats files.")

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["problem", "test_data", "run_id", "test_iter"], as_index=False)
        .agg(test_acc_mean=("test_acc", "mean"), test_acc_sem=("test_acc", "sem"))
    )
    sem = agg["test_acc_sem"].fillna(0.0)
    agg["test_acc_low"] = agg["test_acc_mean"] - sem
    agg["test_acc_high"] = agg["test_acc_mean"] + sem
    return df, agg


def choose_single(value_name, series):
    values = [v for v in sorted(series.dropna().unique()) if str(v) != "unknown"]
    if len(values) == 1:
        return values[0]
    return None


def plot_schoop(raw_df, table, plot_name, plot_title, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    tr = int(raw_df["max_iters"].max())
    x_min = int(table["test_iter"].min()) if len(table) else 0
    ax.axvspan(x_min, tr, color="#bdbdbd", alpha=0.35, zorder=0)

    handles = []
    labels = []
    run_ids = sorted(table["run_id"].unique())
    for idx, run_id in enumerate(run_ids):
        method_df = table[table["run_id"] == run_id].sort_values("test_iter")
        style = style_for_label(run_id, idx)
        line, = ax.plot(
            method_df["test_iter"].to_numpy(),
            method_df["test_acc_mean"].to_numpy(),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=run_id,
            zorder=3,
        )
        ax.fill_between(
            method_df["test_iter"].to_numpy(),
            method_df["test_acc_low"].to_numpy(),
            method_df["test_acc_high"].to_numpy(),
            color=style["color"],
            alpha=0.20,
            linewidth=0,
            zorder=2,
        )
        handles.append(line)
        labels.append(run_id)

    handles.append(mpatches.Patch(facecolor="#bdbdbd", alpha=0.35, edgecolor="none"))
    labels.append("Train Regime")

    x_max = int(table["test_iter"].max())
    xticks = np.arange(25, x_max + 1, 25)
    if len(xticks) == 0:
        xticks = np.array(sorted(table["test_iter"].unique()))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=0, fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=14)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.set_xlim([max(1, x_min - 1), x_max + 1] if xlim is None else xlim)
    ax.set_ylim([0, 105] if ylim is None else ylim)
    ax.set_xlabel("Test-Time Iterations", fontsize=18, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=18, fontweight="bold")
    ax.set_title(plot_title, fontsize=24)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(False)

    ax.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        prop={"size": 14, "weight": "bold"},
    )
    plt.tight_layout()
    plt.savefig(plot_name, dpi=220)
    print(f"Saved {plot_name}")


def main():
    parser = argparse.ArgumentParser(description="Build schoop plot from outputs folder or stats.json")
    parser.add_argument("input_path", type=str,
                        help="Path to outputs folder, one run folder, or a stats.json file")
    parser.add_argument("--plot_name", type=str, default=None, help="Where to save the plot")
    parser.add_argument("--xlim", type=float, nargs="+", default=None, help="x limits for plotting")
    parser.add_argument("--ylim", type=float, nargs="+", default=None, help="y limits for plotting")
    parser.add_argument("--problem", type=str, default=None, help="Optional problem filter (e.g., sudoku)")
    parser.add_argument("--test_data", type=str, default=None, help="Optional test_data filter")
    args = parser.parse_args()

    if args.plot_name is None:
        now = datetime.now().strftime("%m%d-%H.%M")
        args.plot_name = f"schoop{now}.png"
        plot_title = "Schoopy Plot"
    else:
        plot_title = args.plot_name[:-4]

    stats_files = get_stats_files(args.input_path)
    raw_df, table = build_table(stats_files)

    if args.problem is not None:
        raw_df = raw_df[raw_df["problem"] == args.problem]
        table = table[table["problem"] == args.problem]
    if args.test_data is not None:
        raw_df = raw_df[raw_df["test_data"].astype(str) == args.test_data]
        table = table[table["test_data"].astype(str) == args.test_data]

    auto_problem = choose_single("problem", table["problem"]) if len(table) else None
    auto_test_data = choose_single("test_data", table["test_data"]) if len(table) else None
    if args.problem is None and auto_problem is not None:
        raw_df = raw_df[raw_df["problem"] == auto_problem]
        table = table[table["problem"] == auto_problem]
    if args.test_data is None and auto_test_data is not None:
        raw_df = raw_df[raw_df["test_data"] == auto_test_data]
        table = table[table["test_data"] == auto_test_data]

    if table.empty:
        raise ValueError("No rows left after filtering. Try removing --problem/--test_data.")

    print(table[["problem", "test_data", "run_id", "test_iter", "test_acc_mean", "test_acc_sem"]].round(3).to_markdown(index=False))
    plot_schoop(raw_df, table, args.plot_name, plot_title, xlim=args.xlim, ylim=args.ylim)


if __name__ == "__main__":
    main()
