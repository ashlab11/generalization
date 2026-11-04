import os
import wandb
import pandas as pd
import numpy as np

# Build val/hard LaTeX tables (mean ± SD over seeds) for the stability sweep.
ENTITY, PROJECT = "asherlabovich-brown-university", "deep-thinking"
OUT_DIR = "analyses/results"
GET_RUNS = True
WANDB_TIMEOUT = 300
FINAL_STEP = 99  # last epoch (epochs=100); val/* metrics are logged there
WANDB_METRICS = {"val": "val/val_acc", "hard": "val/hard_acc"}
METHOD_LABELS = {("gru", "peri"): "gru", ("add", "peri"): "peri", ("add", "post"): "post", ("add", "pre"): "pre"}
PROBLEM_ORDER = ["chess", "sudoku", "prefix_sums"]
RECALL_ORDER = ["internal", "external", "none", "fixed"]
METHOD_ORDER = ["gru", "peri", "post", "pre"]
PROBLEM_TEX = {"chess": "Chess", "sudoku": "Sudoku", "prefix_sums": r"Prefix\_sums"}
DECIMALS = 1

# Three LRs shown per problem in the main val/hard tables (edit per problem).
PROBLEM_LRS = {
    "chess": (0.0001, 0.0003, 0.001),
    "sudoku": (0.0003, 0.001, 0.003),
    "prefix_sums": (0.0001, 0.0003, 0.001),
}

# Fixed-depth prefix-sums can't see the whole 512-bit input with local attention, so it is left out of hard data.
EXCLUDED = {("hard", "prefix_sums", "fixed")}

# Each table is a list of column groups: (header, metric, problem, lrs).
SUDOKU_LRS_4 = (0.0001, 0.0003, 0.001, 0.003)
TABLE_SPECS = [
    ("table_val.tex", [(PROBLEM_TEX[p], "val", p, PROBLEM_LRS[p]) for p in PROBLEM_ORDER]),
    ("table_hard.tex", [(PROBLEM_TEX[p], "hard", p, PROBLEM_LRS[p]) for p in PROBLEM_ORDER]),
    # Sudoku transparency table: validation and hard accuracy across all four training LRs.
    ("table_sudoku_4lr.tex", [("Validation Accuracy", "val", "sudoku", SUDOKU_LRS_4),
                              ("Hard Accuracy", "hard", "sudoku", SUDOKU_LRS_4)]),
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
    # Final val/hard accuracy (at the last test iteration) of each finished run, read from the run summary.
    runs = wandb.Api(timeout=WANDB_TIMEOUT).runs(
        f"{ENTITY}/{PROJECT}",
        filters={"state": "finished", "config.sweep_name": "stability_final"},
        per_page=500,
    )
    rows = []
    for run in runs:
        p, h, m = run.config["problem"], run.config["problem"]["hyp"], run.config["problem"]["model"]
        s = run.summary
        assert s.get("_step") == FINAL_STEP, f"{run.name}: summary is from step {s.get('_step')}, not {FINAL_STEP}"
        rows.append({
            "problem": p["name"],
            "recall": recall_kind(m),
            "norm": METHOD_LABELS[(m["residual_method"], m["norm_type"])],
            "lr": float(h["lr"]),
            "seed": int(h["seed"]),
            "created": run.created_at,
            "val": s[WANDB_METRICS["val"]],
            "hard": s[WANDB_METRICS["hard"]],
        })
    df = pd.DataFrame(rows)
    # A config/seed that finished twice keeps its latest run.
    return df.sort_values("created").drop_duplicates(["problem", "recall", "norm", "lr", "seed"], keep="last")


def cell_stats(df):
    """(problem, recall, norm, lr, metric) -> (mean, sd) over seeds."""
    g = df.assign(lr=df["lr"].round(6)).groupby(["problem", "recall", "norm", "lr"])
    out = {}
    for metric in WANDB_METRICS:
        for (p, rec, nor, lr), r in g[metric].agg(["mean", "std"]).iterrows():
            if (metric, p, rec) not in EXCLUDED:
                out[(p, rec, nor, lr, metric)] = (r["mean"], r["std"])
    return out


def rank_marks(values):
    """Map each value to 'best' / 'second' by its rounded mean (ties share a mark)."""
    distinct = sorted({round(v, DECIMALS) for v in values}, reverse=True)
    marks = {}
    if distinct:
        marks[distinct[0]] = "best"
    if len(distinct) > 1:
        marks[distinct[1]] = "second"
    return marks


def full_table_tex(stats, groups):
    columns = [(gi, p, lr, metric) for gi, (_, metric, p, lrs) in enumerate(groups) for lr in lrs]

    # Best and second-best mean in each column group (e.g. all of Chess), across every row and LR.
    group_marks = [
        rank_marks([stats[(p, rec, nor, lr, metric)][0] for lr in lrs for nor in METHOD_ORDER for rec in RECALL_ORDER
                    if (p, rec, nor, lr, metric) in stats])
        for _, metric, p, lrs in groups
    ]

    def cell(gi, prob, lr, metric, rec, nor):
        v = stats.get((prob, rec, nor, lr, metric))
        if v is None or pd.isna(v[0]):
            return "---"
        mean, sd = v
        text = f"{mean:.{DECIMALS}f}{{\\scriptsize$\\pm${sd:.{DECIMALS}f}}}"
        mark = group_marks[gi].get(round(mean, DECIMALS))
        if mark == "best":
            return rf"\textbf{{{text}}}"
        if mark == "second":
            return rf"\underline{{{text}}}"
        return text

    col = 3
    cmid_parts = []
    heads1_parts = []
    heads2_parts = []
    for header, _, _, lrs in groups:
        n_lr = len(lrs)
        cmid_parts.append(rf"\cmidrule(lr){{{col}-{col + n_lr - 1}}}")
        col += n_lr
        heads1_parts.append(rf"\multicolumn{{{n_lr}}}{{c}}{{{header}}}")
        lr_cell = lambda e: rf"{{\footnotesize\num{{{lr_label(e)}}}}}"
        heads2_parts.append(" & ".join(lr_cell(e) for e in lrs))

    c_cols = "c" * len(columns)

    body = []
    norm_sep = r"\specialrule{0.2pt}{0.15em}{0.15em}"
    for ni, nor in enumerate(METHOD_ORDER):
        if ni:
            body.append(norm_sep)
        for ri, rec in enumerate(RECALL_ORDER):
            row_cells = " & ".join(cell(*c, rec, nor) for c in columns)
            ncol = rf"\multirow{{{len(RECALL_ORDER)}}}{{*}}{{{nor}}}" if ri == 0 else ""
            body.append(f"{ncol} & {rec} & {row_cells} \\\\")

    return "\n".join(
        [
            rf"\begin{{tabular}}{{@{{}}ll {c_cols}@{{}}}}",
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


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    runs_path = f"{OUT_DIR}/runs_val_hard.csv"
    df = rows_from_wandb() if GET_RUNS else pd.read_csv(runs_path)
    if GET_RUNS:
        df.to_csv(runs_path, index=False)

    summary = df.groupby(["problem", "recall", "norm", "lr"])[["val", "hard"]].agg(["mean", "std", "count"])
    summary.to_csv(f"{OUT_DIR}/mean_sd_val_hard.csv")
    print(f"{len(df)} runs; seeds per cell: {sorted(set(summary[('val', 'count')]))}")

    stats = cell_stats(df)
    for out_name, groups in TABLE_SPECS:
        with open(f"{OUT_DIR}/{out_name}", "w") as f:
            f.write(full_table_tex(stats, groups) + "\n")
