import os
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#BEFORE RUNNING THIS, MAKE SURE YOU HAVE PERI, LR 0.0001, SEED 4 with LINEAR INJECTION RAN. If not, run it in run.bash.

ENTITY = "asherlabovich-brown-university"
PROJECT = "deep-thinking"
OUT_DIR = "experiments/residual_path"


runs = wandb.Api().runs(
    f"{ENTITY}/{PROJECT}",
    filters={
        "state": "finished",
        "config.run_id": "trans_peri_lr0.0001_seed4"
    },
)

run = next(runs)
history_df = run.history(keys=["_step", "train/val_acc", "diagnostics/Wx_radius"], pandas=True)

sns.set_style("whitegrid")
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(history_df["_step"], history_df["train/val_acc"], color="#2E86AB", linewidth=1.5, alpha=0.8)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Validation Accuracy", color="#2E86AB", fontsize=11)
ax1.set_ylim(0, 100)
ax1.set_yticks([0, 25, 50, 75, 100])
ax1.tick_params(axis="y", labelcolor="#2E86AB")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(history_df["_step"], history_df["diagnostics/Wx_radius"], color="#A23B72", linewidth=1.5, alpha=0.8)
ax2.set_ylabel("Spectral Radius", color="#A23B72", fontsize=11)
ax2.set_ylim(0.5, 1.3)
ax2.tick_params(axis="y", labelcolor="#A23B72")

plt.title("Validation Accuracy and Spectral Radius Over Training", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/radius_plot.png", dpi=150)
plt.show()
