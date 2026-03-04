#!/bin/bash
#SBATCH --job-name=prefix_perturb
#SBATCH -o experiments/loss_landscape/%A.out
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-he

set -euo pipefail
source .venv/bin/activate

# Example usage:
# sbatch experiments/loss_landscape/run_perturb.bash \
#   outputs/length_generalization/training-fixed_run \
#   outputs/length_generalization/training-looped_run

FIXED_RUN_DIR="$1"
LOOPED_RUN_DIR="$2"
OUT_DIR="experiments/loss_landscape/out_${SLURM_JOB_ID}"

python experiments/loss_landscape/perturb_prefix_sums.py \
  --run-dir "$FIXED_RUN_DIR" \
  --model-label fixed15 \
  --run-dir "$LOOPED_RUN_DIR" \
  --model-label looped30 \
  --milestones valbest,hardbest \
  --eval-splits val,test \
  --scope global \
  --rho-min 1e-4 \
  --rho-max 3e-1 \
  --rho-count 6 \
  --directions 8 \
  --sample-size 1000 \
  --sample-seed 0 \
  --out-dir "$OUT_DIR"
