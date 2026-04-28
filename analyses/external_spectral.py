import matplotlib.pyplot as plt
import pandas as pd
import wandb

ENTITY, PROJECT = "asherlabovich-brown-university", "deep-thinking"
PROBLEMS = ["chess", "sudoku", "prefix_sums"]
NORMS = ["pre", "peri"]
RECALLS = {"internal": True, "external": False}
COLORS = {"low": "#3b82f6", "medium": "#43a047", "high": "#ec6adf"}
LRS = {
    "chess": {"low": 1e-4, "medium": 3e-4, "high": 1e-3},
    "sudoku": {"low": 3e-4, "medium": 1e-3, "high": 3e-3},
    "prefix_sums": {"low": 1e-4, "medium": 3e-4, "high": 1e-3},
}
rows = []
api = wandb.Api(timeout=60)
for recall, recall_inner in RECALLS.items():
    for norm in NORMS:
        runs = api.runs(
            f"{ENTITY}/{PROJECT}",
            filters={
                "state": "finished",
                "config.sweep_name": "stability_final",
                "config.problem.model.norm_type": norm,
                "config.problem.model.recall_inner": {
                    "$in": ["True", "true", True]
                }
                if recall_inner
                else {"$in": ["False", "false", False]},
                "config.problem.model.injection_type": "linear",
                "config.problem.model.residual_method": "add",
            },
        )
        for run in runs:
            problem = run.config["problem"]["name"]
            lr = float(run.config["problem"]["hyp"]["lr"])
            seed = int(run.config["problem"]["hyp"]["seed"])
            lr_label = next(
                (label for label, value in LRS.get(problem, {}).items() if value == lr),
                None,
            )
            if lr_label is None:
                continue
            hist = run.history(
                keys=["_step", "diagnostics/Wx_radius"],
                pandas=True,
            ).dropna()
            rows += [
                {
                    "recall": recall,
                    "norm": norm,
                    "problem": problem,
                    "lr": lr,
                    "lr_label": lr_label,
                    "seed": seed,
                    "step": step,
                    "Wx_radius": radius,
                }
                for step, radius in zip(
                    hist["_step"], hist["diagnostics/Wx_radius"]
                )
            ]

df = pd.DataFrame(rows)
plot_df = (
    df.groupby(["recall", "problem", "lr_label", "step"], as_index=False)["Wx_radius"]
    .mean()
)

for recall in RECALLS:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, problem in zip(axes, PROBLEMS):
        for lr_label, color in COLORS.items():
            sub = plot_df[
                (plot_df.recall == recall)
                & (plot_df.problem == problem)
                & (plot_df.lr_label == lr_label)
            ]
            ax.plot(
                sub.step,
                sub.Wx_radius,
                label=lr_label,
                color=color,
                linewidth=2.2,
            )
        ax.set_title(problem.replace("_", " "))
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(r"Average $\rho(W_x)$")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"analyses/results/{recall}_spectral.png", dpi=300, bbox_inches="tight")
