# Stability and Generalization in Looped Transformers

This repository contains the code for **"Stability and Generalization in Looped Transformers"**.

[![arXiv](https://img.shields.io/badge/arXiv-2604.15259-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.15259)

This repository began from Avi Schwarzschild et. al's work on deep thinking, but has since been substantially updated.

## Repository Layout

- [`deepthinking/`](deepthinking/) contains the model definitions and shared utilities used by the training and evaluation scripts.
- [`experiments/`](experiments/) contains shell scripts for launching the main experiment sweeps, including the Slurm-oriented norm/learning-rate runs.
- [`config/`](config/) contains Hydra configuration files for training, testing, model settings, and problem-specific hyperparameters.
- [`analyses/`](analyses/) contains scripts for turning completed experiment logs into tables and plots used in the paper and appendix.

The main entry points are [`train_model.py`](train_model.py) and [`test_model.py`](test_model.py).

## Setup With uv

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, reproducible dependency installation from [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock). The project currently requires Python 3.11.

Install `uv` if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On macOS with Homebrew, this also works:

```bash
brew install uv
```

From the repository root, install Python 3.11 and create the local environment:

```bash
uv python install 3.11
```

Then sync dependencies. Choose exactly one CUDA extra for the machine you are using:

```bash
# CUDA 12.8
uv sync --frozen --extra cu128

# CUDA 12.6
uv sync --frozen --extra cu126

# CUDA 11.8
uv sync --frozen --extra cu118
```

`uv sync` creates a `.venv/` in the repository if one does not already exist and installs the locked package versions. The `--frozen` flag tells `uv` to use the committed lockfile without updating it, which is usually what you want when reproducing experiments.

Run commands through the environment with `uv run`, e.g.:

```bash
uv run python train_model.py problem=chess
uv run python test_model.py problem=chess
```

You can also activate the environment directly:

```bash
source .venv/bin/activate
```

## Experiments and Analyses

The scripts in [`experiments/`](experiments/) launch the main training sweeps. These are intended for a Slurm cluster and can take substantial GPU time.

After runs finish, use the scripts in [`analyses/`](analyses/) to generate paper tables and plots from the saved logs and outputs.
