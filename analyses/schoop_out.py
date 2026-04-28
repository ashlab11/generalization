# We copy the naming of Bansal et al. 2022 here -- they call it "Schoop" since that's what the graph
# looks like. We think it's cute!

#NOTE: wandb doesn't put out per-iter directly, so we grab it from the LOGS. DO NOT 
#DELETE the slurm logs if you wish to run this one in particular!!

from collections import defaultdict
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

VAL_MARKER = "Val accuracy:"
HARD_MARKER = "Test accuracy (hard data):"

OUT_DIR = Path("experiments/stability")

ENTITY, PROJECT = "asherlabovich-brown-university", "deep-thinking"
METHOD_LABELS = {("gru", "peri"): "gru", ("add", "peri"): "peri", ("add", "post"): "post", ("add", "pre"): "pre"}
PROBLEMS = ["chess", "sudoku", "prefix_sums"]
RESIDUALS = ["peri", "pre", "gru", "post"]
LRS = [0.0001, 0.0003, 0.001]
RECALLS = ["internal", "external", "none", "fixed"]
SEEDS = [0, 1, 2]
RECALL_TYPE = "none"
OUT_CSV = f"analyses/schoop_out_{RECALL_TYPE}.csv"
OUT_PLOT = f"analyses/schoop_out_{RECALL_TYPE}.png"

match RECALL_TYPE:
    case "internal":
        INJECTION_TYPE = "linear"
        RECALL_INNER = True
        NUM_BLOCKS = 1
    case "external":
        INJECTION_TYPE = "linear"
        RECALL_INNER = False
        NUM_BLOCKS = 1
    case "none":
        INJECTION_TYPE = "none"
        RECALL_INNER = False
        NUM_BLOCKS = 1
    case "fixed":
        INJECTION_TYPE = "none"
        RECALL_INNER = False
        NUM_BLOCKS = 15
    case _:
        raise ValueError(f"Unknown RECALL_TYPE: {RECALL_TYPE}")


#This api only gets runs for autonomous models
api = wandb.Api(timeout=60)
runs = api.runs(
    f"{ENTITY}/{PROJECT}",
    filters={
        "state": "finished",
        "config.sweep_name": "stability_final",
        "config.problem.model.injection_type": INJECTION_TYPE,
        "config.problem.model.recall_inner": {"$in": ["True", "true", True]} if RECALL_INNER else {"$in": ["False", "false", False]},
        "config.problem.hyp.lr": {"$in": LRS},
        "config.problem.model.num_blocks": NUM_BLOCKS,
    },
)

def run_id_from_run(run):
    p, h, m = run.config["problem"], run.config["problem"]["hyp"], run.config["problem"]["model"]
    residual = METHOD_LABELS[(m["residual_method"], m["norm_type"])]
    return f"{RECALL_TYPE}_{p['name']}_{residual}_lr{float(h['lr']):g}_seed{int(h['seed'])}"


def slurm_idx_from_run(run):
    p, h, m = run.config["problem"], run.config["problem"]["hyp"], run.config["problem"]["model"]
    lr = float(h["lr"])
    seed = int(h["seed"])
    residual = METHOD_LABELS[(m["residual_method"], m["norm_type"])]
    lr_idx = LRS.index(lr)
    prob_idx = PROBLEMS.index(p["name"])
    res_idx = RESIDUALS.index(residual)
    recall_idx = RECALLS.index(RECALL_TYPE)
    seed_idx = SEEDS.index(seed)
    return lr_idx + len(LRS) * (prob_idx + len(PROBLEMS) * (res_idx + len(RESIDUALS) * (recall_idx + len(RECALLS) * seed_idx)))


def dict_after_marker(text, marker):
    last = None
    prefix = f"INFO]: {marker}"
    for line in text.splitlines():
        if prefix not in line:
            continue
        payload = line.split(prefix, 1)[1].strip()
        if not payload.startswith("{"):
            continue
        pairs = re.findall(r"(\d+)\s*:\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", payload, flags=re.I)
        if not pairs:
            continue
        last = {int(k): float(v) for k, v in pairs}
    return last


def curves_from_out_file(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    val = dict_after_marker(text, VAL_MARKER)
    hard = dict_after_marker(text, HARD_MARKER)
    if val is None or hard is None:
        raise ValueError(f"{path} missing parseable {VAL_MARKER!r} / {HARD_MARKER!r} dicts")
    return val, hard


def out_file_for_run(run):
    idx = slurm_idx_from_run(run)
    run_id = run_id_from_run(run)
    candidates = sorted(OUT_DIR.glob(f"*_{idx}.out"))
    if not candidates:
        raise FileNotFoundError(f"No .out file found for slurm idx {idx} ({run_id})")
    if len(candidates) == 1:
        return candidates[0]

    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        if f"run_id: {run_id}" in text:
            return path
    raise FileNotFoundError(f"Found multiple .out files for slurm idx {idx}, but none matched run_id {run_id}")


def mean_curves_over_seeds(pairs):
    val = pd.DataFrame([pair[0] for pair in pairs]).mean(axis=0)
    hard = pd.DataFrame([pair[1] for pair in pairs]).mean(axis=0)
    return pd.DataFrame({"val_acc": val, "hard_acc": hard}).dropna().reset_index(names="iteration")


def plot_curves(curves):
    problems = sorted(curves["problem"].unique())
    colors = ["#3b6fb6", "#c65f46", "#4c9a5f"]
    fig, axes = plt.subplots(1, len(problems), figsize=(4.5 * len(problems), 4), sharey=True)
    if len(problems) == 1:
        axes = [axes]

    for ax, problem, color in zip(axes, problems, colors):
        data = curves[curves["problem"] == problem]
        ax.axvspan(0, 30, color="#e8e8e8", zorder=0)
        ax.plot(data["iteration"], data["val_acc"], color=color, alpha=0.45, linewidth=1.8)
        ax.plot(data["iteration"], data["hard_acc"], color=color, linewidth=3.0)
        ax.set_xlim(0, data["iteration"].max())
        ax.set_facecolor("white")
        ax.set_title(problem.replace("_", " "))
        ax.set_xlabel("Iteration")

    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


rows = []
runs = list(runs)
for run in runs:
    p, h, m = run.config["problem"], run.config["problem"]["hyp"], run.config["problem"]["model"]
    hist = run.history(keys=["_step", "val/val_acc", "val/hard_acc"], pandas=True)
    final_val = np.array(hist['val/val_acc'])[-1]
    final_hard = np.array(hist["val/hard_acc"])[-1]
    rows.append(
        {
            "problem": p["name"],
            "norm": METHOD_LABELS[(m["residual_method"], m["norm_type"])],
            "lr": float(h["lr"]),
            "seed": int(h["seed"]),
            "val_score": final_val,
            "hard_score": final_hard,
        }
    )

rows = pd.DataFrame(rows)
scores = rows.groupby(["problem", "norm", "lr"])["hard_score"].mean()
scores = scores[scores == scores.groupby("problem").transform("max")]
best_keys = set(scores.index)

valid_runs = []
for run in runs:
    p, h, m = run.config["problem"], run.config["problem"]["hyp"], run.config["problem"]["model"]
    key = (p["name"], METHOD_LABELS[(m["residual_method"], m["norm_type"])], float(h["lr"]))
    if key in best_keys:
        valid_runs.append(run)

by_key = defaultdict(list)
for run in valid_runs:
    p, h, m = run.config["problem"], run.config["problem"]["hyp"], run.config["problem"]["model"]
    key = (p["name"], METHOD_LABELS[(m["residual_method"], m["norm_type"])], float(h["lr"]))
    by_key[key].append(curves_from_out_file(out_file_for_run(run)))

curve_tables = []
for key, pairs in sorted(by_key.items(), key=lambda item: item[0]):
    table = mean_curves_over_seeds(pairs)
    problem, norm, lr = key
    table.insert(0, "lr", lr)
    table.insert(0, "norm", norm)
    table.insert(0, "problem", problem)
    curve_tables.append(table)

curves = pd.concat(curve_tables, ignore_index=True) if curve_tables else pd.DataFrame()
curves.to_csv(OUT_CSV, index=False)
if not curves.empty:
    plot_curves(curves)
print(curves)
