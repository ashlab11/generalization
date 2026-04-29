#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-he
#SBATCH --constraint=b200

set -euo pipefail
source .venv/bin/activate

python analyses/perturbation.py \
  --problem sudoku \
  --checkpoint-root outputs \
  --out-dir analyses/results \
  --metric val/hard_acc \
  --sweep-name stability_final \
  --eval-iter 500 \
  --n-values 10 \
  --n-perturbations 5 \
  --max-percent 100.0 \
  --device auto