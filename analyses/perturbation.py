"""Measure how random parameter perturbations change hard-set accuracy."""

import argparse
import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf

import deepthinking as dt

ENTITY = "asherlabovich-brown-university"
PROJECT = "deep-thinking"


@contextmanager
def working_directory(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def recall_label(recall_inner):
    return "internal" if str(recall_inner).lower() == "true" else "external"


def model_label(cfg):
    model = cfg.problem.model
    hyp = cfg.problem.hyp
    return f"{model.residual_method}/{model.norm_type} lr={float(hyp.lr):g} seed={int(hyp.seed)}"

def dataset_key(cfg):
    hyp = cfg.problem.hyp
    return (
        cfg.problem.name,
        cfg.problem.test_data,
        int(hyp.test_batch_size),
        int(getattr(cfg.problem, "max_test_samples", -1)),
    )


def metric_value(run, metric):
    value = run.summary.get(metric)
    if value is not None:
        return float(value)
    history = run.history(keys=[metric], pandas=True).dropna()
    return float(history[metric].iloc[-1])

def find_best_runs(problem, sweep_name, metric):
    filters = {
        "state": "finished",
        "config.problem.name": problem,
        "config.problem.model.injection_type": "linear",
        "config.problem.model.num_blocks": 1,
    }
    if sweep_name:
        filters["config.sweep_name"] = sweep_name
    best = {}
    for run in wandb.Api(timeout=180).runs(f"{ENTITY}/{PROJECT}", filters=filters):
        recall = recall_label(run.config["problem"]["model"]["recall_inner"])
        score = metric_value(run, metric)
        run_id = run.config.get("run_id") or run.name or run.id
        print(f"{recall:>8} {score:6.2f} {run_id}")
        if recall not in best or score > best[recall]["score"]:
            best[recall] = {"run": run, "run_id": run_id, "score": score}
    return best


def get_local_candidate(run_id, checkpoint_root):
    for cfg_path in sorted(Path(checkpoint_root).glob("**/.hydra/config.yaml")):
        cfg = OmegaConf.load(cfg_path)
        if str(getattr(cfg, "run_id", "")) != run_id:
            continue
        run_dir = cfg_path.parent.parent
        checkpoint = run_dir / "model_hardbest.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for run_id={run_id}: {checkpoint}")
        return {
            "cfg": cfg,
            "checkpoint": checkpoint,
            "recall": recall_label(cfg.problem.model.recall_inner),
            "run_dir": run_dir,
            "label": model_label(cfg),
        }
    raise FileNotFoundError(f"Could not find a local run directory for run_id={run_id}")


def get_test_loader(candidate):
    with working_directory(candidate["run_dir"]):
        return dt.utils.get_dataloaders(candidate["cfg"].problem)["test"]


def load_model(candidate, device):
    cfg = OmegaConf.create(OmegaConf.to_container(candidate["cfg"], resolve=True))
    cfg.problem.model.model_path = str(candidate["checkpoint"])
    return dt.utils.load_model_from_checkpoint(cfg.problem.name, cfg.problem.model, device)[0], cfg


def accuracy_at_iter(net, loader, cfg, device, eval_iter):
    use_amp = bool(getattr(cfg.problem.hyp, "use_amp", False)) and device == "cuda"
    return dt.test(net, [loader], cfg.problem.hyp.test_mode, [eval_iter], cfg.problem.name, device, use_amp)[0][0][eval_iter]


def get_direction(params, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for param in params:
        noise = torch.randn(param.shape, generator=generator, device=device, dtype=param.dtype)
        yield param.norm() * noise / (noise.norm() + 1e-12)


def apply_perturbation(params, rho, device, seed):
    for param, direction in zip(params, get_direction(params, device, seed)):
        param.add_(rho * direction)


def restore_params(params, base_params):
    for param, base in zip(params, base_params):
        param.copy_(base)


def perturb_model(candidate, loader, device, eval_iter, percents, n_perturbations):
    net, cfg = load_model(candidate, device)
    params = [param for param in net.parameters()]
    base_params = [param.detach().clone() for param in params]
    clean_acc = accuracy_at_iter(net, loader, cfg, device, eval_iter)
    rows = []
    for percent in percents:
        rho = percent / 100.0
        for seed in range(n_perturbations):
            apply_perturbation(params, rho, device, seed)
            perturbed_acc = accuracy_at_iter(net, loader, cfg, device, eval_iter)
            restore_params(params, base_params)
            rows.append(
                {
                    "problem": cfg.problem.name,
                    "recall": candidate["recall"],
                    "model": candidate["label"],
                    "run_dir": str(candidate["run_dir"]),
                    "checkpoint": str(candidate["checkpoint"]),
                    "eval_iter": eval_iter,
                    "perturbation_seed": seed,
                    "perturbation_percent": percent,
                    "clean_acc": clean_acc,
                    "perturbed_acc": perturbed_acc,
                    "accuracy_drop": clean_acc - perturbed_acc,
                }
            )
    del net
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def plot_results(summary, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"internal": "#1f77b4", "external": "#d62728"}
    for recall in ["internal", "external"]:
        data = summary[summary["recall"] == recall]
        if data.empty:
            continue
        ax.plot(data["perturbation_percent"], data["mean_drop"], label=recall, color=colors[recall], linewidth=2.2)
        ax.fill_between(
            data["perturbation_percent"],
            data["mean_drop"] - data["std_drop"].fillna(0),
            data["mean_drop"] + data["std_drop"].fillna(0),
            color=colors[recall],
            alpha=0.18,
        )
    ax.set_xlabel("% norm of perturbation")
    ax.set_ylabel(f"Accuracy drop at {int(summary['eval_iter'].iloc[0])} iterations")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Perturb the best internal and external recall models and measure hard-set accuracy drop.")
    parser.add_argument("--problem", default="chess")
    parser.add_argument("--checkpoint-root", default="outputs")
    parser.add_argument("--out-dir", default="analyses/results")
    parser.add_argument("--metric", default="val/hard_acc")
    parser.add_argument("--sweep-name", default="stability_final")
    parser.add_argument("--eval-iter", type=int, default=500)
    parser.add_argument("--n-values", type=int, default=10)
    parser.add_argument("--n-perturbations", type=int, default=5)
    parser.add_argument("--max-percent", type=float, default=100.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    best_runs = find_best_runs(args.problem, args.sweep_name, args.metric)
    best = {
        recall: get_local_candidate(best_runs[recall]["run_id"], args.checkpoint_root)
        for recall in ["internal", "external"]
    }
    keys = {dataset_key(best[recall]["cfg"]) for recall in best}
    assert len(keys) == 1, keys

    loader = get_test_loader(best["internal"])

    percents = np.linspace(0.0, args.max_percent, args.n_values)
    rows = []
    for recall in ["internal", "external"]:
        print(f"Selected {recall}: {best_runs[recall]['score']:.2f} {best[recall]['label']} {best[recall]['run_dir']}")
        rows.extend(perturb_model(best[recall], loader, device, args.eval_iter, percents, args.n_perturbations))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["problem", "recall", "eval_iter", "perturbation_percent"], as_index=False).agg(
        mean_drop=("accuracy_drop", "mean"),
        std_drop=("accuracy_drop", "std"),
    )
    raw.to_csv(out_dir / f"{args.problem}_perturbation_raw.csv", index=False)
    summary.to_csv(out_dir / f"{args.problem}_perturbation_summary.csv", index=False)
    plot_results(summary, out_dir / f"{args.problem}_perturbation.png")


if __name__ == "__main__":
    main()
