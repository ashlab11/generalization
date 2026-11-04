#!/bin/bash
#SBATCH --job-name=stab_jacobian
#SBATCH -o experiments/stability/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --gres=gpu:1
# Jacobian runs (Appendix B.3, Tables 4-5), sudoku seed 0:
# 2 norm × 2 recall × 2 LR × 2 blocks = 16. Measure afterwards with measure_all.bash.
#SBATCH --array=0-15
#SBATCH --partition=gpu-he
#SBATCH --constraint=b200

set -euo pipefail
source .venv/bin/activate

NORMS=("post" "pre")
RECALL_TYPES=("internal" "external")
LRS=("0.001" "0.003")
BLOCKS=(1 2)

IDX=$SLURM_ARRAY_TASK_ID
LR=${LRS[$((IDX % 2))]}
RECALL_TYPE=${RECALL_TYPES[$((IDX / 2 % 2))]}
NORM_TYPE=${NORMS[$((IDX / 4 % 2))]}
NUM_BLOCKS=${BLOCKS[$((IDX / 8))]}

RECALL_INNER=false
[[ "$RECALL_TYPE" == internal ]] && RECALL_INNER=true

EXP_NAME="${RECALL_TYPE}_sudoku_${NORM_TYPE}_lr${LR}_seed0"
[[ "$NUM_BLOCKS" == 2 ]] && EXP_NAME="${EXP_NAME}_2blocks"

python train_model.py \
    name=stability_jacobian \
    +run_id=$EXP_NAME \
    problem=sudoku \
    problem/model=transformer \
    problem.hyp.alpha=1 \
    problem.hyp.epochs=100 \
    problem.hyp.optimizer=adamw \
    problem.hyp.weight_decay=0.01 \
    problem.hyp.lr=$LR \
    problem.hyp.seed=0 \
    problem.hyp.use_amp=true \
    problem.hyp.train_mode=progressive \
    problem.hyp.rand_method=basic \
    problem.model.test_iterations.low=1 \
    problem.model.test_iterations.high=500 \
    problem.hyp.train_batch_size=500 \
    problem.model.kernel_size=5 \
    problem.model.hidden_dim=256 \
    problem.model.norm_type="$NORM_TYPE" \
    problem.model.attn_type=full \
    problem.model.num_sinks=0 \
    problem.model.num_blocks=$NUM_BLOCKS \
    problem.model.max_iters=30 \
    problem.model.injection_type=linear \
    problem.model.recall_inner=$RECALL_INNER \
    problem.model.residual_method=add \
    +problem.model.qk_normalization=true \
    problem.model.init_method=default \
    problem.hyp.full_only_hard=false \
    problem.model.ccot=none \
    hydra.run.dir=jacobian_runs/$EXP_NAME \
    compile=true \
    +sweep_name=stability_jacobian
