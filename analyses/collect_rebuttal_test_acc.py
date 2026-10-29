#!/usr/bin/env python3
"""Print test_acc at iter 500 from every stats.json under rebuttal_runs."""

import argparse
import json
from pathlib import Path

ITER = "500"


def acc_at(stats: dict, key: str, iter_: str) -> float | None:
    d = stats.get(key, {})
    return d.get(iter_, d.get(int(iter_)) if iter_.isdigit() else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("rebuttal_runs"))
    p.add_argument("--iter", default=ITER, help="test iteration (default: 500)")
    args = p.parse_args()

    paths = sorted(args.root.glob("*/stats.json"))
    if not paths:
        print(f"No stats.json under {args.root.resolve()}")
        return

    rows = []
    for path in paths:
        stats = json.loads(path.read_text())
        run = path.parent.name
        acc = acc_at(stats, "test_acc", args.iter)
        rows.append((run, acc))

    w = max(len(r[0]) for r in rows)
    print(f"{'run':<{w}}  test_acc@{args.iter}")
    print("-" * (w + 16))
    for run, acc in rows:
        val = "-" if acc is None else f"{acc:.4f}"
        print(f"{run:<{w}}  {val}")


if __name__ == "__main__":
    main()
