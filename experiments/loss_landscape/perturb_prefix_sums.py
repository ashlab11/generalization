"""Perturbation-based loss landscape comparison for prefix sums runs."""

import argparse
import copy
import csv
import os
from contextlib import nullcontext

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import deepthinking as dt
from deepthinking.utils.testing import get_predicted


def parse_csv_list(value):
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError(f"Invalid comma-separated list: {value}")
    return parts


def resolve_model_labels(run_dirs, model_labels):
    if len(model_labels) != len(run_dirs):
        raise ValueError("--model-label must be passed once per --run-dir (same count and order)")
    labels = []
    for label in model_labels:
        if label not in {"fixed15", "looped30"}:
            raise ValueError(f"Invalid model label: {label}. Expected fixed15 or looped30.")
        labels.append(label)
    return labels


def load_cfg(run_dir):
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing run config: {cfg_path}")
    return OmegaConf.load(cfg_path)


def load_net_from_checkpoint(cfg, checkpoint_path, device):
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.problem.model.model_path = checkpoint_path
    net, _, _ = dt.utils.load_model_from_checkpoint(cfg_copy.problem.name, cfg_copy.problem.model, device)
    return net


def get_split_loaders(cfg):
    loaders = dt.utils.get_dataloaders(cfg.problem)
    return {"val": loaders["val"], "test": loaders["test"]}


def maybe_subsample_loader(loader, sample_size, sample_seed):
    if sample_size is None:
        return loader
    dataset_len = len(loader.dataset)
    if sample_size >= dataset_len:
        return loader

    rng = np.random.default_rng(sample_seed)
    indices = rng.choice(dataset_len, size=sample_size, replace=False).tolist()
    subset = Subset(loader.dataset, indices)
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=False,
        persistent_workers=getattr(loader, "persistent_workers", False),
    )


def evaluate_loss_acc(net, loader, cfg, device, batch_limit=None):
    criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    use_amp = bool(getattr(cfg.problem.hyp, "use_amp", False)) and device == "cuda"

    net.eval()
    total_loss = 0.0
    total_loss_count = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_limit is not None and batch_idx >= batch_limit:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            targets = targets.view(targets.size(0), -1)

            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True) if use_amp else nullcontext()
            with autocast_ctx:
                all_outputs = net(inputs, iters_to_do=cfg.problem.model.max_iters)

            outputs = all_outputs[:, -1]
            logits = outputs.view(outputs.size(0), outputs.size(1), -1)
            losses = criterion(logits, targets)

            loss_sum = losses.sum().item()
            loss_count = int(losses.numel())
            predicted = get_predicted(inputs, outputs, cfg.problem.name)
            batch_correct = int(torch.amin(predicted == targets, dim=[1]).sum().item())

            total_loss += loss_sum
            total_loss_count += loss_count
            total_correct += batch_correct
            total_samples += int(targets.size(0))

    if total_loss_count == 0 or total_samples == 0:
        raise ValueError("No data was evaluated. Check dataloader and --batch-limit.")

    mean_loss = total_loss / total_loss_count
    acc = 100.0 * total_correct / total_samples
    return mean_loss, acc


def perturb_model_in_place(param_items, rho, device):
    deltas = []
    for _, param in param_items:
        weight_norm = param.detach().norm()
        if weight_norm.item() == 0.0:
            delta = torch.zeros_like(param)
        else:
            u = torch.randn_like(param, device=device)
            u_norm = u.norm()
            delta = rho * weight_norm * (u / (u_norm + 1e-12))
        param.add_(delta)
        deltas.append(delta)
    return deltas


def restore_model_in_place(param_items, base_values):
    for (name, param), base in zip(param_items, base_values):
        _ = name
        param.copy_(base)


def write_raw_csv(rows, out_path):
    fields = [
        "model_label", "run_id", "run_dir", "milestone", "eval_split", "rho", "direction_idx",
        "loss", "acc", "delta_loss", "delta_acc"
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["model_label"], row["milestone"], row["eval_split"], row["rho"])
        grouped.setdefault(key, []).append(row["delta_loss"])

    summary = []
    for key, values in sorted(grouped.items()):
        model_label, milestone, eval_split, rho = key
        arr = np.asarray(values, dtype=np.float64)
        summary.append({
            "model_label": model_label,
            "milestone": milestone,
            "eval_split": eval_split,
            "rho": rho,
            "n": int(arr.size),
            "delta_loss_median": float(np.median(arr)),
            "delta_loss_p10": float(np.percentile(arr, 10)),
            "delta_loss_p90": float(np.percentile(arr, 90)),
        })
    return summary


def write_agg_csv(rows, out_path):
    fields = [
        "model_label", "milestone", "eval_split", "rho", "n",
        "delta_loss_median", "delta_loss_p10", "delta_loss_p90"
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(agg_rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    milestones = ["valbest", "hardbest"]
    eval_splits = ["val", "test"]
    model_order = ["fixed15", "looped30"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for i, milestone in enumerate(milestones):
        for j, eval_split in enumerate(eval_splits):
            ax = axes[i][j]
            for model_label in model_order:
                rows = [
                    r for r in agg_rows
                    if r["model_label"] == model_label and r["milestone"] == milestone and r["eval_split"] == eval_split
                ]
                if not rows:
                    continue
                rows = sorted(rows, key=lambda x: x["rho"])
                x = [r["rho"] for r in rows]
                y = [r["delta_loss_median"] for r in rows]
                y_lo = [r["delta_loss_p10"] for r in rows]
                y_hi = [r["delta_loss_p90"] for r in rows]
                ax.plot(x, y, linewidth=2, label=model_label)
                ax.fill_between(x, y_lo, y_hi, alpha=0.2)

            ax.set_xscale("log")
            ax.set_title(f"{milestone} / {eval_split}")
            ax.set_ylabel("delta loss")
            ax.grid(True, alpha=0.3)
            if i == 1:
                ax.set_xlabel("rho")
            if i == 0 and j == 1:
                ax.legend(loc="best")

    fig.suptitle("Prefix-Sums Perturbation Robustness (Layerwise Relative Sphere)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def infer_run_id(cfg, run_dir):
    cfg_run_id = getattr(cfg, "run_id", None)
    if cfg_run_id:
        return str(cfg_run_id)
    return os.path.basename(os.path.abspath(run_dir))


def main():
    parser = argparse.ArgumentParser(description="Perturb prefix-sums checkpoints and compare robustness.")
    parser.add_argument("--run-dir", action="append", required=True, help="Training run directory (repeatable).")
    parser.add_argument("--model-label", action="append", required=True, help="Model label per run-dir: fixed15 or looped30.")
    parser.add_argument("--milestones", default="valbest,hardbest", help="Comma list from {valbest,hardbest}.")
    parser.add_argument("--eval-splits", default="val,test", help="Comma list from {val,test}.")
    parser.add_argument("--scope", default="global", choices=["global"], help="Perturbation scope.")
    parser.add_argument("--rho-min", type=float, default=1e-4)
    parser.add_argument("--rho-max", type=float, default=3e-1)
    parser.add_argument("--rho-count", type=int, default=10)
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--batch-limit", type=int, default=None, help="Optional max batches per split evaluation.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional random subset size per eval split.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for random eval subset sampling.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for perturbation directions.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    labels = resolve_model_labels(args.run_dir, args.model_label)
    milestones = parse_csv_list(args.milestones)
    eval_splits = parse_csv_list(args.eval_splits)

    for m in milestones:
        if m not in {"valbest", "hardbest"}:
            raise ValueError(f"Invalid milestone: {m}")
    for s in eval_splits:
        if s not in {"val", "test"}:
            raise ValueError(f"Invalid eval split: {s}")

    rho_values = np.geomspace(args.rho_min, args.rho_max, num=args.rho_count).astype(np.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_name = {
        "valbest": "model_valbest.pth",
        "hardbest": "model_hardbest.pth",
    }

    rows = []

    for run_dir, model_label in zip(args.run_dir, labels):
        cfg = load_cfg(run_dir)
        if cfg.problem.name != "prefix_sums":
            raise ValueError(f"Run is not prefix_sums: {run_dir}")

        run_id = infer_run_id(cfg, run_dir)
        split_loaders = get_split_loaders(cfg)
        split_loaders = {
            split_name: maybe_subsample_loader(split_loader, args.sample_size, args.sample_seed)
            for split_name, split_loader in split_loaders.items()
        }

        for milestone in milestones:
            ckpt_path = os.path.join(run_dir, checkpoint_name[milestone])
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing checkpoint for {milestone}: {ckpt_path}")

            net = load_net_from_checkpoint(cfg, ckpt_path, device)
            param_items = [(name, p) for name, p in net.named_parameters() if p.requires_grad]
            base_values = [p.detach().clone() for _, p in param_items]

            baseline_by_split = {}
            for eval_split in eval_splits:
                base_loss, base_acc = evaluate_loss_acc(net, split_loaders[eval_split], cfg, device, batch_limit=args.batch_limit)
                baseline_by_split[eval_split] = (base_loss, base_acc)
                rows.append({
                    "model_label": model_label,
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "milestone": milestone,
                    "eval_split": eval_split,
                    "rho": 0.0,
                    "direction_idx": -1,
                    "loss": base_loss,
                    "acc": base_acc,
                    "delta_loss": 0.0,
                    "delta_acc": 0.0,
                })

            for rho in rho_values:
                for direction_idx in range(args.directions):
                    torch.manual_seed(args.seed + direction_idx)
                    if device == "cuda":
                        torch.cuda.manual_seed_all(args.seed + direction_idx)
                    with torch.no_grad():
                        perturb_model_in_place(param_items, float(rho), device)

                    for eval_split in eval_splits:
                        loss, acc = evaluate_loss_acc(net, split_loaders[eval_split], cfg, device, batch_limit=args.batch_limit)
                        base_loss, base_acc = baseline_by_split[eval_split]
                        rows.append({
                            "model_label": model_label,
                            "run_id": run_id,
                            "run_dir": run_dir,
                            "milestone": milestone,
                            "eval_split": eval_split,
                            "rho": float(rho),
                            "direction_idx": direction_idx,
                            "loss": float(loss),
                            "acc": float(acc),
                            "delta_loss": float(loss - base_loss),
                            "delta_acc": float(acc - base_acc),
                        })

                    with torch.no_grad():
                        restore_model_in_place(param_items, base_values)

    raw_csv = os.path.join(args.out_dir, "perturb_raw.csv")
    agg_csv = os.path.join(args.out_dir, "perturb_agg.csv")
    plot_png = os.path.join(args.out_dir, "robustness_curves.png")

    write_raw_csv(rows, raw_csv)
    agg_rows = aggregate_rows(rows)
    write_agg_csv(agg_rows, agg_csv)
    make_plot(agg_rows, plot_png)

    print(f"Wrote raw rows: {raw_csv}")
    print(f"Wrote aggregate rows: {agg_csv}")
    print(f"Wrote plot: {plot_png}")


if __name__ == "__main__":
    main()
