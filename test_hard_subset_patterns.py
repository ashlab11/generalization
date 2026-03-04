"""Run a saved maze model on a hard-data subset and summarize failure patterns."""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from easy_to_hard_data import MazeDataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path when script is run from nested directories.
for parent in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
    if (parent / "deepthinking").is_dir():
        parent_str = str(parent)
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)
        break

import deepthinking as dt
from deepthinking.utils.tools import get_model
from deepthinking.utils.testing import get_predicted


def count_components(binary_map):
    h, w = binary_map.shape
    seen = torch.zeros((h, w), dtype=torch.bool)
    components = 0
    for r in range(h):
        for c in range(w):
            if not binary_map[r, c] or seen[r, c]:
                continue
            components += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if seen[nr, nc] or not binary_map[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return components


def component_sizes(binary_map):
    h, w = binary_map.shape
    seen = torch.zeros((h, w), dtype=torch.bool)
    sizes = []
    for r in range(h):
        for c in range(w):
            if not binary_map[r, c] or seen[r, c]:
                continue
            size = 0
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if seen[nr, nc] or not binary_map[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            sizes.append(size)
    return sizes


def find_endpoints(binary_map):
    h, w = binary_map.shape
    endpoints = []
    for r in range(h):
        for c in range(w):
            if not binary_map[r, c]:
                continue
            degree = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and binary_map[nr, nc]:
                    degree += 1
            if degree == 1:
                endpoints.append((int(r), int(c)))
    return endpoints


def endpoint_distance(endpoints):
    if len(endpoints) != 2:
        return -1
    (r1, c1), (r2, c2) = endpoints
    return abs(r1 - r2) + abs(c1 - c2)


def path_degree_stats(binary_map):
    h, w = binary_map.shape
    degree_counts = {0: 0, 1: 0, 2: 0, "3+": 0}
    for r in range(h):
        for c in range(w):
            if not binary_map[r, c]:
                continue
            degree = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and binary_map[nr, nc]:
                    degree += 1
            if degree in (0, 1, 2):
                degree_counts[degree] += 1
            else:
                degree_counts["3+"] += 1
    return degree_counts


def path_centroid(binary_map):
    pts = torch.nonzero(binary_map, as_tuple=False)
    if pts.numel() == 0:
        return None
    center = pts.float().mean(dim=0)
    return float(center[0].item()), float(center[1].item())


def bbox_stats(binary_map):
    pts = torch.nonzero(binary_map, as_tuple=False)
    if pts.numel() == 0:
        return {"height": 0, "width": 0, "area": 0, "aspect": 0.0}
    rmin = int(pts[:, 0].min().item())
    rmax = int(pts[:, 0].max().item())
    cmin = int(pts[:, 1].min().item())
    cmax = int(pts[:, 1].max().item())
    height = rmax - rmin + 1
    width = cmax - cmin + 1
    area = height * width
    aspect = float(max(height, width) / max(1, min(height, width)))
    return {"height": height, "width": width, "area": area, "aspect": aspect}


def overlap_stats(target_path, pred_path, mask_map):
    tp = int((target_path & pred_path).sum().item())
    fp = int((~target_path & pred_path & mask_map).sum().item())
    fn = int((target_path & ~pred_path).sum().item())
    tn = int((~target_path & ~pred_path & mask_map).sum().item())
    union = tp + fp + fn
    jaccard = float(tp / union) if union > 0 else 1.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "jaccard": jaccard, "precision": precision, "recall": recall}


def summarize_group(records, group_name):
    if not records:
        return {
            group_name: "",
            "count": 0,
            "exact_acc": 0.0,
            "mean_bit_acc": 0.0,
            "mean_jaccard": 0.0,
            "mean_len_diff": 0.0,
            "mean_pred_components": 0.0,
        }
    count = len(records)
    return {
        "count": count,
        "exact_acc": round(100.0 * sum(r["exact"] for r in records) / count, 3),
        "mean_bit_acc": round(100.0 * sum(r["bit_acc"] for r in records) / count, 3),
        "mean_jaccard": round(100.0 * sum(r["jaccard"] for r in records) / count, 3),
        "mean_len_diff": round(sum(r["len_diff"] for r in records) / count, 3),
        "mean_pred_components": round(sum(r["pred_components"] for r in records) / count, 3),
    }


def make_subset_loader(dataset, subset_size, subset_mode, seed, batch_size):
    total_size = len(dataset)
    indices = list(range(total_size))
    if subset_size > 0 and subset_size < total_size:
        if subset_mode == "random":
            rng = random.Random(seed)
            rng.shuffle(indices)
        indices = indices[:subset_size]
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, indices, total_size


def maze_split_exists(root, split_name):
    return os.path.exists(os.path.join(root, split_name, "inputs.npy")) and os.path.exists(
        os.path.join(root, split_name, "solutions.npy")
    )


def resolve_maze_data_root(test_size, data_root=None):
    split_name = f"maze_data_test_{test_size}"
    candidates = []
    if data_root:
        candidates.append(data_root)
    env_root = os.environ.get("MAZE_DATA_ROOT")
    if env_root:
        candidates.append(env_root)

    repo_root = str(Path(__file__).resolve().parent)
    candidates.extend(
        [
            os.path.join(repo_root, "data"),
            os.path.abspath(os.path.join(repo_root, "..", "data")),
            "/users/alabovic/deep-thinking/data",
            "/oscar/home/alabovic/deep-thinking/data",
            "/users/alabovic/generalization/data",
            "/oscar/home/alabovic/generalization/data",
        ]
    )

    deduped = []
    for c in candidates:
        c = os.path.abspath(os.path.expanduser(c))
        if c not in deduped:
            deduped.append(c)

    for root in deduped:
        if maze_split_exists(root, split_name):
            return root

    tried = "\n".join(deduped)
    raise FileNotFoundError(
        f"Could not find existing {split_name} in any candidate root.\n"
        f"Tried:\n{tried}\n"
        "Pass --data-root /path/to/data or set MAZE_DATA_ROOT."
    )


def load_model_with_compile_key_fallback(problem_name, model_args, device):
    try:
        return dt.utils.load_model_from_checkpoint(problem_name, model_args, device)
    except RuntimeError as exc:
        err = str(exc)
        if "._orig_mod." not in err:
            raise

        hidden_dim = getattr(model_args, "hidden_dim", None) or getattr(model_args, "width", None)
        if hidden_dim is None:
            raise ValueError("Must provide either 'hidden_dim' or 'width' in model config")
        in_channels = getattr(model_args, "in_channels", None)
        if in_channels is None:
            if problem_name == "chess":
                in_channels = 12
            elif problem_name == "prefix_sums":
                in_channels = 1
            elif problem_name in {"rule110", "cellular"}:
                in_channels = 1
            else:
                in_channels = 3
        extra_args = {
            k: v
            for k, v in dict(model_args).items()
            if k
            not in {
                "model",
                "model_path",
                "width",
                "hidden_dim",
                "max_iters",
                "test_iterations",
                "in_channels",
                "init_method",
            }
        }
        net = get_model(model_args.model, hidden_dim, in_channels=in_channels, max_iters=model_args.max_iters, **extra_args)
        net = net.to(device)

        state = torch.load(model_args.model_path, map_location=device)
        state_dict = state["net"]
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_state_dict[key.replace("._orig_mod.", ".")] = value
        model_keys = set(net.state_dict().keys())
        cleaned_state_dict = {
            k: v
            for k, v in cleaned_state_dict.items()
            if not (k.startswith("init_norm.") and k not in model_keys)
        }
        net.load_state_dict(cleaned_state_dict, strict=True)
        epoch = state.get("epoch", -1) + 1
        optimizer_state_dict = state.get("optimizer")
        return net, epoch, optimizer_state_dict


def summarize_bins(records, field, bins):
    rows = []
    eps = 1e-9
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if i == len(bins) - 2:
            in_bin = [r for r in records if (r[field] >= lo - eps and r[field] <= hi + eps)]
        else:
            in_bin = [r for r in records if (r[field] >= lo - eps and r[field] < hi - eps)]
        if not in_bin:
            rows.append({"range": f"[{lo},{hi})", "count": 0, "exact_acc": 0.0, "mean_bit_acc": 0.0})
            continue
        count = len(in_bin)
        exact_acc = 100.0 * sum(r["exact"] for r in in_bin) / count
        mean_bit_acc = 100.0 * sum(r["bit_acc"] for r in in_bin) / count
        rows.append(
            {
                "range": f"[{lo},{hi})",
                "count": count,
                "exact_acc": round(exact_acc, 3),
                "mean_bit_acc": round(mean_bit_acc, 3),
            }
        )
    return rows


def make_bins(values, quantiles, pad):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [0.0, 1.0]
    lo = float(values.min())
    hi = float(values.max())
    if abs(hi - lo) < 1e-12:
        return [lo - pad, hi + pad]
    q = np.quantile(values, quantiles).tolist()
    bins = [float(x) for x in q]
    bins[0] = lo
    bins[-1] = hi
    deduped = [bins[0]]
    for x in bins[1:]:
        if x > deduped[-1] + 1e-12:
            deduped.append(x)
    if len(deduped) < 2:
        deduped = [lo - pad, hi + pad]
    else:
        deduped[-1] = deduped[-1] + pad
    return deduped


def build_error_tags(exact, target_len, pred_len, target_components, pred_components, target_endpoints, pred_endpoints):
    if exact:
        return ""

    tags = []
    if pred_len == 0:
        tags.append("empty_pred")
    if pred_components > 1:
        tags.append("pred_disconnected")
    if target_len > 0 and pred_len < max(1, int(0.7 * target_len)):
        tags.append("too_short")
    if target_len > 0 and pred_len > int(1.3 * target_len):
        tags.append("too_long")
    if len(target_endpoints) == 2:
        if len(pred_endpoints) != 2:
            tags.append("endpoint_count_wrong")
        elif set(pred_endpoints) != set(target_endpoints):
            tags.append("endpoint_mismatch")
    if target_components > 1:
        tags.append("target_multicomponent")
    if not tags:
        tags.append("other")
    return "|".join(tags)


def build_cumulative_threshold_rows(records, value_fn):
    values = []
    exact = []
    for r in records:
        value = value_fn(r)
        if value is None:
            continue
        values.append(float(value))
        exact.append(int(r["exact"]))
    if not values:
        return []

    vals = np.asarray(values, dtype=np.float64)
    exact_arr = np.asarray(exact, dtype=np.int64)
    order = np.argsort(vals, kind="mergesort")
    vals_sorted = vals[order]
    exact_sorted = exact_arr[order]
    n = vals_sorted.size
    cum_correct = np.cumsum(exact_sorted)
    total_correct = int(cum_correct[-1])
    uniq_vals, first_idx, counts = np.unique(vals_sorted, return_index=True, return_counts=True)

    rows = []
    for threshold, start, count in zip(uniq_vals, first_idx, counts):
        end = int(start + count - 1)
        le_count = end + 1
        le_correct = int(cum_correct[end])
        ge_count = n - int(start)
        before_start_correct = int(cum_correct[start - 1]) if start > 0 else 0
        ge_correct = total_correct - before_start_correct
        rows.append(
            {
                "threshold": float(threshold),
                "count_le": int(le_count),
                "exact_acc_le": 100.0 * le_correct / max(1, le_count),
                "count_ge": int(ge_count),
                "exact_acc_ge": 100.0 * ge_correct / max(1, ge_count),
            }
        )
    return rows


def save_cumulative_plot(rows, metric_name, metric_label, output_dir):
    if not rows:
        return None, None
    thresholds = [r["threshold"] for r in rows]
    acc_le = [r["exact_acc_le"] for r in rows]
    acc_ge = [r["exact_acc_ge"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(thresholds, acc_le, label="Exact acc for value <= threshold", linewidth=2.0)
    ax.plot(thresholds, acc_ge, label="Exact acc for value >= threshold", linewidth=2.0)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_ylim(0.0, 101.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(f"Cumulative Exact Accuracy vs {metric_label}")
    fig.tight_layout()
    png_path = os.path.join(output_dir, f"cumulative_{metric_name}.png")
    fig.savefig(png_path, dpi=170)
    plt.close(fig)

    csv_path = os.path.join(output_dir, f"cumulative_{metric_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "count_le", "exact_acc_le", "count_ge", "exact_acc_ge"])
        writer.writeheader()
        writer.writerows(rows)
    return png_path, csv_path


def get_ge_100_threshold(rows, min_count):
    for r in rows:
        if r["count_ge"] >= min_count and r["exact_acc_ge"] >= 100.0 - 1e-12:
            return float(r["threshold"]), int(r["count_ge"])
    return None, 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate a hard-data subset and find error patterns.")
    parser.add_argument(
        "--run-dir",
        default="outputs/length_generalization/training-concat_post_recurrent_local_r3_mazes_lr0.0001",
        help="Training output directory containing model_best.pth and .hydra/config.yaml",
    )
    parser.add_argument("--subset-size", type=int, default=500, help="Number of hard test samples to analyze")
    parser.add_argument("--subset-mode", choices=["first", "random"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-iter", type=int, default=None, help="Iteration to evaluate; default is max_iters")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for analysis")
    parser.add_argument("--test-data", type=int, default=None, help="Override hard test size")
    parser.add_argument("--data-root", type=str, default=None, help="Root containing maze_data_test_<size>")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", default=None, help="Default: <run-dir>/hard_subset_patterns")
    parser.add_argument(
        "--cumulative-min-count",
        type=int,
        default=5,
        help="Minimum sample count for reporting >=threshold 100%% exact-accuracy cutoffs",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    run_cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    model_path = os.path.join(run_dir, "model_best.pth")
    if not os.path.exists(run_cfg_path):
        raise FileNotFoundError(f"Missing run config: {run_cfg_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    cfg = OmegaConf.load(run_cfg_path)
    if cfg.problem.name != "mazes":
        raise ValueError(f"This script is maze-specific, got problem={cfg.problem.name}")

    cfg.problem.model.model_path = model_path
    if args.test_data is not None:
        cfg.problem.test_data = int(args.test_data)
    cfg.problem.hyp.test_batch_size = max(1, int(args.batch_size))

    output_dir = args.output_dir or os.path.join(run_dir, "hard_subset_patterns")
    os.makedirs(output_dir, exist_ok=True)

    match args.device:
        case "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        case "cpu":
            device = "cpu"
        case "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            device = "cuda"

    use_amp = bool(getattr(cfg.problem.hyp, "use_amp", False)) and device == "cuda" and not args.no_amp

    data_root = resolve_maze_data_root(int(cfg.problem.test_data), args.data_root)
    test_dataset = MazeDataset(data_root, train=False, size=int(cfg.problem.test_data), download=False)
    subset_loader, subset_indices, total_test_size = make_subset_loader(
        test_dataset, args.subset_size, args.subset_mode, args.seed, cfg.problem.hyp.test_batch_size
    )

    net, _, _ = load_model_with_compile_key_fallback(cfg.problem.name, cfg.problem.model, device)
    net.eval()

    eval_iter = int(cfg.problem.model.max_iters) if args.eval_iter is None else int(args.eval_iter)
    if eval_iter < 1:
        raise ValueError("--eval-iter must be >= 1")

    records = []
    pointer = 0
    with torch.no_grad():
        for inputs, targets in subset_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long().view(targets.size(0), -1)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                all_outputs = net(inputs, iters_to_do=eval_iter)
            outputs = all_outputs[:, eval_iter - 1]
            predicted = get_predicted(inputs, outputs, cfg.problem.name).view(targets.size(0), -1)
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

            eq = predicted == targets
            exact_per_sample = torch.amin(eq, dim=1)
            bit_per_sample = (eq & mask).sum(dim=1).float() / mask.sum(dim=1).clamp(min=1).float()

            h, w = inputs.shape[-2], inputs.shape[-1]
            for i in range(inputs.size(0)):
                global_idx = subset_indices[pointer + i]
                target_map = targets[i].view(h, w).detach().cpu()
                pred_map = predicted[i].view(h, w).detach().cpu()
                mask_map = mask[i].view(h, w).detach().cpu()
                target_path = (target_map > 0) & mask_map
                pred_path = (pred_map > 0) & mask_map

                target_len = int(target_path.sum().item())
                pred_len = int(pred_path.sum().item())
                open_cells = int(mask_map.sum().item())
                open_ratio = float(open_cells / max(1, h * w))

                target_components = count_components(target_path)
                pred_components = count_components(pred_path)
                target_comp_sizes = component_sizes(target_path)
                pred_comp_sizes = component_sizes(pred_path)
                target_endpoints = find_endpoints(target_path)
                pred_endpoints = find_endpoints(pred_path)
                target_ep_dist = endpoint_distance(target_endpoints)
                pred_ep_dist = endpoint_distance(pred_endpoints)
                target_degree = path_degree_stats(target_path)
                pred_degree = path_degree_stats(pred_path)
                target_centroid = path_centroid(target_path)
                pred_centroid = path_centroid(pred_path)
                target_bbox = bbox_stats(target_path)
                pred_bbox = bbox_stats(pred_path)
                overlap = overlap_stats(target_path, pred_path, mask_map)

                exact = bool(exact_per_sample[i].item())
                bit_acc = float(bit_per_sample[i].item())
                error_tags = build_error_tags(
                    exact,
                    target_len,
                    pred_len,
                    target_components,
                    pred_components,
                    target_endpoints,
                    pred_endpoints,
                )

                records.append(
                    {
                        "sample_idx": int(global_idx),
                        "exact": int(exact),
                        "bit_acc": bit_acc,
                        "open_cells": open_cells,
                        "open_ratio": open_ratio,
                        "target_len": target_len,
                        "pred_len": pred_len,
                        "len_diff": pred_len - target_len,
                        "target_components": target_components,
                        "pred_components": pred_components,
                        "target_largest_component": max(target_comp_sizes) if target_comp_sizes else 0,
                        "pred_largest_component": max(pred_comp_sizes) if pred_comp_sizes else 0,
                        "target_endpoints": len(target_endpoints),
                        "pred_endpoints": len(pred_endpoints),
                        "target_endpoint_dist": target_ep_dist,
                        "pred_endpoint_dist": pred_ep_dist,
                        "target_degree_0": target_degree[0],
                        "target_degree_1": target_degree[1],
                        "target_degree_2": target_degree[2],
                        "target_degree_3p": target_degree["3+"],
                        "pred_degree_0": pred_degree[0],
                        "pred_degree_1": pred_degree[1],
                        "pred_degree_2": pred_degree[2],
                        "pred_degree_3p": pred_degree["3+"],
                        "target_bbox_area": target_bbox["area"],
                        "pred_bbox_area": pred_bbox["area"],
                        "target_bbox_aspect": target_bbox["aspect"],
                        "pred_bbox_aspect": pred_bbox["aspect"],
                        "target_centroid_r": -1.0 if target_centroid is None else target_centroid[0],
                        "target_centroid_c": -1.0 if target_centroid is None else target_centroid[1],
                        "pred_centroid_r": -1.0 if pred_centroid is None else pred_centroid[0],
                        "pred_centroid_c": -1.0 if pred_centroid is None else pred_centroid[1],
                        "centroid_l1": -1.0
                        if (target_centroid is None or pred_centroid is None)
                        else abs(target_centroid[0] - pred_centroid[0]) + abs(target_centroid[1] - pred_centroid[1]),
                        "tp": overlap["tp"],
                        "fp": overlap["fp"],
                        "fn": overlap["fn"],
                        "tn": overlap["tn"],
                        "jaccard": overlap["jaccard"],
                        "precision": overlap["precision"],
                        "recall": overlap["recall"],
                        "error_tags": error_tags,
                    }
                )
            pointer += inputs.size(0)

    if not records:
        raise RuntimeError("No samples were analyzed.")

    exact_acc = 100.0 * sum(r["exact"] for r in records) / len(records)
    mean_bit_acc = 100.0 * sum(r["bit_acc"] for r in records) / len(records)

    tag_counts = Counter()
    for r in records:
        if r["exact"] == 1 or not r["error_tags"]:
            continue
        for tag in r["error_tags"].split("|"):
            tag_counts[tag] += 1

    len_values = np.array([r["target_len"] for r in records], dtype=np.float64)
    open_values = np.array([r["open_ratio"] for r in records], dtype=np.float64)
    jaccard_values = np.array([r["jaccard"] for r in records], dtype=np.float64)
    pred_comp_values = np.array([r["pred_components"] for r in records], dtype=np.float64)
    len_bins = make_bins(len_values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], pad=1.0)
    open_bins = make_bins(open_values, [0.0, 0.25, 0.5, 0.75, 1.0], pad=1e-6)
    jaccard_bins = make_bins(jaccard_values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], pad=1e-6)
    pred_comp_bins = make_bins(pred_comp_values, [0.0, 0.5, 0.8, 0.95, 1.0], pad=1.0)
    valid_ep_records = [r for r in records if r["target_endpoint_dist"] >= 0]
    ep_bins = [0.0, 16.0, 32.0, 48.0, 64.0, 1e9]

    failed_records = [r for r in records if r["exact"] == 0]
    solved_records = [r for r in records if r["exact"] == 1]
    bit_acc_bands = {
        "<70": [r for r in records if r["bit_acc"] < 0.70],
        "70-80": [r for r in records if 0.70 <= r["bit_acc"] < 0.80],
        "80-90": [r for r in records if 0.80 <= r["bit_acc"] < 0.90],
        "90-95": [r for r in records if 0.90 <= r["bit_acc"] < 0.95],
        "95-99": [r for r in records if 0.95 <= r["bit_acc"] < 0.99],
        "99-100": [r for r in records if 0.99 <= r["bit_acc"] <= 1.0],
    }

    summary = {
        "run_dir": run_dir,
        "checkpoint": model_path,
        "test_data": int(cfg.problem.test_data),
        "subset_mode": args.subset_mode,
        "subset_size_requested": int(args.subset_size),
        "subset_size_used": len(records),
        "total_hard_test_size": int(total_test_size),
        "eval_iter": int(eval_iter),
        "data_root": data_root,
        "device": device,
        "use_amp": bool(use_amp),
        "exact_acc_percent": round(exact_acc, 4),
        "mean_bit_acc_percent": round(mean_bit_acc, 4),
        "num_exact": int(len(solved_records)),
        "num_failed": int(len(failed_records)),
        "open_cells_minmax": [int(min(r["open_cells"] for r in records)), int(max(r["open_cells"] for r in records))],
        "target_len_minmax": [int(min(r["target_len"] for r in records)), int(max(r["target_len"] for r in records))],
        "pred_len_minmax": [int(min(r["pred_len"] for r in records)), int(max(r["pred_len"] for r in records))],
        "len_diff_mean": round(float(np.mean([r["len_diff"] for r in records])), 4),
        "len_diff_abs_mean": round(float(np.mean([abs(r["len_diff"]) for r in records])), 4),
        "jaccard_mean_percent": round(100.0 * float(np.mean([r["jaccard"] for r in records])), 4),
        "precision_mean_percent": round(100.0 * float(np.mean([r["precision"] for r in records])), 4),
        "recall_mean_percent": round(100.0 * float(np.mean([r["recall"] for r in records])), 4),
        "solved_mean_target_len": round(float(np.mean([r["target_len"] for r in solved_records])) if solved_records else 0.0, 4),
        "failed_mean_target_len": round(float(np.mean([r["target_len"] for r in failed_records])) if failed_records else 0.0, 4),
        "solved_mean_pred_components": round(float(np.mean([r["pred_components"] for r in solved_records])) if solved_records else 0.0, 4),
        "failed_mean_pred_components": round(float(np.mean([r["pred_components"] for r in failed_records])) if failed_records else 0.0, 4),
        "error_tag_counts": dict(tag_counts),
        "by_target_len_quantile": summarize_bins(records, "target_len", len_bins),
        "by_open_ratio_quantile": summarize_bins(records, "open_ratio", open_bins),
        "by_jaccard_quantile": summarize_bins(records, "jaccard", jaccard_bins),
        "by_pred_components_quantile": summarize_bins(records, "pred_components", pred_comp_bins),
        "by_target_endpoint_distance_valid_only": summarize_bins(valid_ep_records, "target_endpoint_dist", ep_bins),
        "target_endpoint_invalid_count": int(sum(1 for r in records if r["target_endpoint_dist"] < 0)),
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(output_dir, "samples.csv")
    fieldnames = [
        "sample_idx",
        "exact",
        "bit_acc",
        "open_cells",
        "open_ratio",
        "target_len",
        "pred_len",
        "len_diff",
        "target_components",
        "pred_components",
        "target_largest_component",
        "pred_largest_component",
        "target_endpoints",
        "pred_endpoints",
        "target_endpoint_dist",
        "pred_endpoint_dist",
        "target_degree_0",
        "target_degree_1",
        "target_degree_2",
        "target_degree_3p",
        "pred_degree_0",
        "pred_degree_1",
        "pred_degree_2",
        "pred_degree_3p",
        "target_bbox_area",
        "pred_bbox_area",
        "target_bbox_aspect",
        "pred_bbox_aspect",
        "target_centroid_r",
        "target_centroid_c",
        "pred_centroid_r",
        "pred_centroid_c",
        "centroid_l1",
        "tp",
        "fp",
        "fn",
        "tn",
        "jaccard",
        "precision",
        "recall",
        "error_tags",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    failures_path = os.path.join(output_dir, "failures.csv")
    with open(failures_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed_records)

    solved_path = os.path.join(output_dir, "solved.csv")
    with open(solved_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(solved_records)

    error_rows = []
    for tag in sorted(tag_counts):
        group = [r for r in failed_records if tag in (r["error_tags"].split("|") if r["error_tags"] else [])]
        row = {"error_tag": tag}
        row.update(summarize_group(group, "error_tag"))
        error_rows.append(row)
    error_summary_path = os.path.join(output_dir, "error_tag_summary.csv")
    with open(error_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["error_tag", "count", "exact_acc", "mean_bit_acc", "mean_jaccard", "mean_len_diff", "mean_pred_components"],
        )
        writer.writeheader()
        writer.writerows(error_rows)

    comp_rows = []
    for comp in sorted(set(r["pred_components"] for r in records)):
        group = [r for r in records if r["pred_components"] == comp]
        row = {"pred_components": comp}
        row.update(summarize_group(group, "pred_components"))
        comp_rows.append(row)
    comp_summary_path = os.path.join(output_dir, "pred_components_summary.csv")
    with open(comp_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pred_components", "count", "exact_acc", "mean_bit_acc", "mean_jaccard", "mean_len_diff", "mean_pred_components"],
        )
        writer.writeheader()
        writer.writerows(comp_rows)

    bit_rows = []
    for band in ["<70", "70-80", "80-90", "90-95", "95-99", "99-100"]:
        row = {"bit_acc_band": band}
        row.update(summarize_group(bit_acc_bands[band], "bit_acc_band"))
        bit_rows.append(row)
    bit_summary_path = os.path.join(output_dir, "bit_acc_band_summary.csv")
    with open(bit_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bit_acc_band", "count", "exact_acc", "mean_bit_acc", "mean_jaccard", "mean_len_diff", "mean_pred_components"],
        )
        writer.writeheader()
        writer.writerows(bit_rows)

    cumulative_specs = [
        ("jaccard", "Jaccard", lambda r: r["jaccard"]),
        ("precision", "Precision", lambda r: r["precision"]),
        ("recall", "Recall", lambda r: r["recall"]),
        ("bit_acc", "Bit Accuracy", lambda r: r["bit_acc"]),
        ("target_len", "Target Path Length", lambda r: r["target_len"]),
        ("pred_len", "Predicted Path Length", lambda r: r["pred_len"]),
        ("abs_len_diff", "Absolute Length Difference", lambda r: abs(r["len_diff"])),
        ("pred_components", "Predicted Components", lambda r: r["pred_components"]),
        ("centroid_l1", "Centroid L1 Distance", lambda r: None if r["centroid_l1"] < 0 else r["centroid_l1"]),
        ("target_bbox_aspect", "Target BBox Aspect", lambda r: r["target_bbox_aspect"]),
    ]
    cumulative_thresholds = {}
    cumulative_files = []
    for metric_name, metric_label, metric_fn in cumulative_specs:
        rows = build_cumulative_threshold_rows(records, metric_fn)
        png_path, csv_curve_path = save_cumulative_plot(rows, metric_name, metric_label, output_dir)
        if png_path is None:
            continue
        ge_threshold, ge_count = get_ge_100_threshold(rows, args.cumulative_min_count)
        cumulative_thresholds[metric_name] = {
            "label": metric_label,
            "ge_100_threshold": ge_threshold,
            "ge_100_support": ge_count,
            "min_count_used": int(args.cumulative_min_count),
        }
        cumulative_files.append({"metric": metric_name, "png": png_path, "csv": csv_curve_path})

    summary["cumulative_thresholds"] = cumulative_thresholds
    summary["cumulative_files"] = cumulative_files
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {failures_path}")
    print(f"Wrote {solved_path}")
    print(f"Wrote {error_summary_path}")
    print(f"Wrote {comp_summary_path}")
    print(f"Wrote {bit_summary_path}")
    for item in cumulative_files:
        print(f"Wrote {item['png']}")
        print(f"Wrote {item['csv']}")


if __name__ == "__main__":
    main()
