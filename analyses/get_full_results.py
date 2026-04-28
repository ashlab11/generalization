import os
from pandas.core.indexes.base import final
import wandb
import pandas as pd
import numpy as np

# Build val/hard LaTeX tables for the stability sweep.
ENTITY, PROJECT = "asherlabovich-brown-university", "deep-thinking"
OUT_DIR = "analyses/results"
GET_RUNS = True
WANDB_TIMEOUT = 180
WANDB_METRICS = {"val": "val/val_acc", "hard": "val/hard_acc"}
METHOD_LABELS = {("gru", "peri"): "gru", ("add", "peri"): "peri", ("add", "post"): "post", ("add", "pre"): "pre"}
PROBLEM_ORDER = ["chess", "sudoku", "prefix_sums"]
RECALL_ORDER = ["internal", "external", "none", "fixed"]
METHOD_ORDER = ["gru", "peri", "post", "pre"]
PROBLEM_TEX = {"chess": "Chess", "sudoku": "Sudoku", "prefix_sums": r"Prefix\_sums"}

# Three LRs shown per problem in the main val/hard tables (edit per problem).
PROBLEM_LRS = {
    "chess": (0.0001, 0.0003, 0.001),
    "sudoku": (0.0003, 0.001, 0.003),
    "prefix_sums": (0.0001, 0.0003, 0.001),
}

# Sudoku-only transparency table: all four training LRs.
SUDOKU_LRS_4 = (0.0001, 0.0003, 0.001, 0.003)
TABLE_SPECS = [
    ("table_val.tex", "val", PROBLEM_LRS, PROBLEM_ORDER),
    ("table_hard.tex", "hard", PROBLEM_LRS, PROBLEM_ORDER),
    ("table_sudoku_val_4lr.tex", "val", {"sudoku": SUDOKU_LRS_4}, ["sudoku"]),
    ("table_sudoku_hard_4lr.tex", "hard", {"sudoku": SUDOKU_LRS_4}, ["sudoku"]),
]


def lr_label(lr):
    m = {0.0001: "1e-4", 0.0003: "3e-4", 0.001: "1e-3", 0.003: "3e-3", 0.01: "1e-2"}
    k = min(m.keys(), key=lambda t: abs(float(lr) - t))
    return m[k] if abs(float(lr) - k) < max(1e-12, abs(k) * 1e-6) else f"{lr:g}"


def recall_kind(m):
    if int(m.get("num_blocks", 1)) >= 15 and int(m.get("max_iters", 30)) == 1:
        return "fixed"
    if m.get("injection_type") == "none":
        return "none"
    ri = m.get("recall_inner")
    return "internal" if ri is True or str(ri).lower() == "true" else "external"


def rows_from_wandb():
    # Pull each finished run and keep the best metric value between steps 10 and 60.
    runs = wandb.Api(timeout=WANDB_TIMEOUT).runs(
        f"{ENTITY}/{PROJECT}",
        filters={"state": "finished", "config.sweep_name": "stability_final"},
    )
    rows = []
    for run in runs:
        p, h, m = run.config["problem"], run.config["problem"]["hyp"], run.config["problem"]["model"]
        hist = run.history(keys=["_step", *WANDB_METRICS.values()], pandas=True)
        final_val = np.array(hist[WANDB_METRICS["val"]])[-1]
        final_hard =  np.array(hist[WANDB_METRICS["hard"]])[-1]
        
        rows.append({
            "problem": p["name"],
            "recall": recall_kind(m),
            "norm": METHOD_LABELS[(m["residual_method"], m["norm_type"])],
            "lr": float(h["lr"]),
            "seed": int(h["seed"]),
            "val": final_val,
            "hard": final_hard,
        })
    return pd.DataFrame(rows)


def mean_snapped_to_lrs(df, problem_lrs, metric):
    def snap(p, lr):
        return min(problem_lrs[p], key=lambda t: abs(float(lr) - t))

    x = df.copy()
    x["lr_k"] = [snap(p, l) for p, l in zip(x["problem"], x["lr"])]
    return x.groupby(["problem", "recall", "norm", "lr_k"], as_index=False)[metric].mean()


def full_table_tex(mean, metric, problem_lrs, problem_order):
    kv = {
        (r.problem, r.recall, r.norm, float(r.lr_k)): getattr(r, metric)
        for r in mean.itertuples(index=False)
    }

    def cell(prob, rec, nor, lr):
        v = kv.get((prob, rec, nor, float(lr)))
        return r"\multicolumn{1}{c}{---}" if v is None or pd.isna(v) else f"{v:.2f}"

    col = 3
    cmid_parts = []
    heads1_parts = []
    heads2_parts = []
    for p in problem_order:
        lrs = problem_lrs[p]
        n_lr = len(lrs)
        cmid_parts.append(rf"\cmidrule(lr){{{col}-{col + n_lr - 1}}}")
        col += n_lr
        heads1_parts.append(rf"\multicolumn{{{n_lr}}}{{c}}{{{PROBLEM_TEX[p]}}}")
        lr_cell = lambda e: rf"\multicolumn{{1}}{{c}}{{\footnotesize\num{{{lr_label(e)}}}}}"
        heads2_parts.append(" & ".join(lr_cell(e) for e in lrs))

    n_data = sum(len(problem_lrs[p]) for p in problem_order)
    s_cols = "S[table-format=2.2]" * n_data

    body = []
    norm_sep = r"\specialrule{0.2pt}{0.15em}{0.15em}"
    for ni, nor in enumerate(METHOD_ORDER):
        if ni:
            body.append(norm_sep)
        for ri, rec in enumerate(RECALL_ORDER):
            row_cells = " & ".join(
                cell(p, rec, nor, lr) for p in problem_order for lr in problem_lrs[p]
            )
            ncol = rf"\multirow{{{len(RECALL_ORDER)}}}{{*}}{{{nor}}}" if ri == 0 else ""
            body.append(f"{ncol} & {rec} & {row_cells} \\\\")

    return "\n".join(
        [
            rf"\begin{{tabular}}{{@{{}}ll {s_cols}@{{}}}}",
            r"\toprule",
            rf"\multirow{{2}}{{*}}{{Norm}} & \multirow{{2}}{{*}}{{Recall}} & {' & '.join(heads1_parts)} \\",
            "".join(cmid_parts),
            rf" &  & {' & '.join(heads2_parts)} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def write_table(df, out_name, metric, problem_lrs, problem_order):
    mean = mean_snapped_to_lrs(df[df["problem"].isin(problem_order)], problem_lrs, metric)
    with open(f"{OUT_DIR}/{out_name}", "w") as f:
        f.write(full_table_tex(mean, metric, problem_lrs, problem_order) + "\n")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    runs_path = f"{OUT_DIR}/runs_val_hard.csv"
    df = rows_from_wandb() if GET_RUNS else pd.read_csv(runs_path)
    if GET_RUNS:
        df.to_csv(runs_path, index=False)

    mean = df.groupby(["problem", "recall", "norm", "lr"], as_index=False)[["val", "hard"]].mean()
    mean.to_csv(f"{OUT_DIR}/mean_val_hard.csv", index=False)

    for out_name, metric, problem_lrs, problem_order in TABLE_SPECS:
        write_table(df, out_name, metric, problem_lrs, problem_order)
