import os
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Summarize residual-path ablations across learning rates and seeds.

ENTITY = "asherlabovich-brown-university"
PROJECT = "deep-thinking"
OUT_DIR = "experiments/residual_path"

METHOD_ORDER = ["gru (peri)", "peri", "post", "pre"]
METHOD_LABELS = {
    ("gru", "peri"): "gru (peri)",
    ("add", "peri"): "peri",
    ("add", "post"): "post",
    ("add", "pre"): "pre",
}

os.makedirs(OUT_DIR, exist_ok=True)


def save_heatmap(summary_df, value_col, out_name, title, cbar_label, lr_order, cmap, fmt):
    pivot = summary_df.pivot(index="method", columns="lr", values=value_col)
    pivot = pivot.reindex(index=METHOD_ORDER, columns=lr_order)
    plt.figure(figsize=(8, 4.5))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, vmin=0, vmax=100, cbar_kws={"label": cbar_label})
    plt.title(title)
    plt.xlabel("Learning Rate")
    plt.ylabel("Residual Method / Norm")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{out_name}", dpi=200)
    plt.close()


runs = wandb.Api().runs(
    f"{ENTITY}/{PROJECT}",
    filters={
        "state": "finished",
        "config.sweep_name": {"$in": ["residual_path_ablation", "residual_path_ablation_2"]},
        "config.problem.hyp.seed": {"$exists": True},
    },
)

results = []
scatter_rows = []
for run in runs:
    history_df = run.history(keys=["_step", "val/hard_acc", "diagnostics/h_norm_ratio_mean"], pandas=True)
    history_df = history_df[history_df["_step"].between(10, 60) & history_df["val/hard_acc"].notna()]

    problem = run.config["problem"]
    hyp = problem["hyp"]
    model = problem["model"]
    method = METHOD_LABELS[(model["residual_method"], model["norm_type"])]
    lr = float(hyp["lr"])
    seed = int(hyp["seed"])

    hard_acc = history_df["val/hard_acc"]
    best_hard_acc = hard_acc.max()
    final_hard_acc = hard_acc.iloc[-1]
    results.append({
        "run_id": run.id,
        "method": method,
        "lr": lr,
        "seed": seed,
        "best_hard_acc": best_hard_acc,
        "reached_100": int(hard_acc.eq(100).any()),
        "stable": int(final_hard_acc >= best_hard_acc),
    })

    run_scatter = history_df[["_step", "val/hard_acc", "diagnostics/h_norm_ratio_mean"]].dropna()
    run_scatter = run_scatter.rename(columns={"val/hard_acc": "hard_acc", "diagnostics/h_norm_ratio_mean": "h_norm_ratio"})
    run_scatter = run_scatter.assign(run_id=run.id, method=method, lr=lr, seed=seed)
    scatter_rows.append(run_scatter)

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT_DIR}/results.csv", index=False)

seed_df = results_df.groupby(["method", "lr", "seed"], as_index=False).agg(
    best_hard_acc=("best_hard_acc", "max"),
    reached_100=("reached_100", "max"),
    stable=("stable", "max"),
)
summary_df = seed_df.groupby(["method", "lr"], as_index=False).agg(
    n_seeds=("seed", "nunique"),
    pct_runs_100=("reached_100", lambda x: 100 * x.mean()),
    avg_top_hard_acc=("best_hard_acc", "mean"),
    pct_stable=("stable", lambda x: 100 * x.mean()),
)
summary_df["method"] = pd.Categorical(summary_df["method"], categories=METHOD_ORDER, ordered=True)
summary_df = summary_df.sort_values(["method", "lr"]).reset_index(drop=True)
summary_df.to_csv(f"{OUT_DIR}/summary_by_method_lr.csv", index=False)
print(summary_df.to_string(index=False))

lr_order = sorted(summary_df["lr"].unique())
save_heatmap(
    summary_df,
    "pct_runs_100",
    "heatmap_pct_runs_100.png",
    "Residual Method Stability Across LR (% Runs Reaching 100)",
    "% runs reaching 100 hard acc",
    lr_order,
    "YlGnBu",
    ".1f",
)
save_heatmap(
    summary_df,
    "avg_top_hard_acc",
    "heatmap_avg_top_hard_acc.png",
    "Residual Method Stability Across LR (Avg Top Hard Acc)",
    "Average top hard accuracy",
    lr_order,
    "mako",
    ".2f",
)
save_heatmap(
    summary_df,
    "pct_stable",
    "heatmap_pct_stable.png",
    "Residual Method Stability Across LR (% Stable by Final >= Best)",
    "% runs where final hard acc >= best hard acc",
    lr_order,
    "crest",
    ".1f",
)

scatter_df = pd.concat(scatter_rows, ignore_index=True)
scatter_df.to_csv(f"{OUT_DIR}/validation_hardacc_normratio_points.csv", index=False)
high_acc_df = scatter_df[(scatter_df["hard_acc"] > 95) & (scatter_df['method'].isin(['pre', 'peri']))] #focus purely on residual
max_row = high_acc_df.loc[high_acc_df["h_norm_ratio"].idxmax()]
print(
    f"Highest h_norm_ratio with val/hard_acc > 95: {max_row['h_norm_ratio']:.6g} "
    f"(method={max_row['method']}, lr={max_row['lr']}, seed={int(max_row['seed'])}, step={int(max_row['_step'])}, hard_acc={max_row['hard_acc']:.2f})"
)
