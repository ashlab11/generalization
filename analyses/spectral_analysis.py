import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt

#Creates a scatter plot:
#x-axis is rho(W_x)
#y-axis is **normalized** accuracy. 
#focus entirely on pre-norm, across LR and across seeds
#each problem gets its own shape + color combo

ENTITY, PROJECT = "asherlabovich-brown-university", "deep-thinking"
OUT_DIR = "analyses/results"

runs = wandb.Api().runs(
        f"{ENTITY}/{PROJECT}",
        filters={"state": "finished", 
                 "config.sweep_name": "stability_final", 
                 "config.problem.model.norm_type": "peri", 
                 "config.problem.model.recall_inner": {"$in": ['False', 'false', False]}, 
                 "config.problem.model.injection_type": 'linear', 
                 "config.problem.model.residual_method": 'add'
                 })

for metric_key, name in zip(['val/hard_acc', 'train/val_acc'], ['hard', 'val']):
    rows = []
    for run in runs:
        p = run.config["problem"]['name']
        hist = run.history(keys=["_step", metric_key, "diagnostics/Wx_radius"], pandas=True)
        final_result = np.array(hist[metric_key])[-1]
        final_spectral = np.array(hist['diagnostics/Wx_radius'])[-1]
        rows.append({
            'problem': p, 
            'acc': final_result, 
            'spectral': final_spectral
        })
    rows = pd.DataFrame(rows)
    rows['acc'] = rows.groupby('problem')['acc'].transform(lambda x: x / x.max())
    problems = sorted(rows['problem'].unique())
    cmap = plt.colormaps['tab20']
    markers = 'ov^<>s8P*pHDX'
    fig, ax = plt.subplots()
    for i, prob in enumerate(problems):
        sub = rows[rows['problem'] == prob]
        ax.scatter(
            sub['spectral'], sub['acc'],
            color=cmap(i / max(len(problems) - 1, 1)),
            marker=markers[i % len(markers)],
            label=prob,
            s=36,
            edgecolors='k',
            linewidths=0.25,
            alpha=0.9,
        )
    ax.axvline(1, color='0.35', linestyle='--', linewidth=1, zorder=0)
    ax.set_xlabel(r'$\rho(W_x)$')
    ax.set_ylabel('Normalized accuracy')
    ax.set_title(f"Spectral norm and {name} accuracy")
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/spectral_{name}")
    plt.close(fig)
