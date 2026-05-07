#!/bin/bash
#SBATCH --job-name=stab_sudoku_lr003
#SBATCH -o experiments/stability/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --gres=gpu:1
# sudoku only, LR=0.003: 4 recall × 4 residual × 3 seed = 48
#SBATCH --array=0-47
#SBATCH --partition=gpu-he
#SBATCH --constraint=b200

set -euo pipefail
source .venv/bin/activate

RESIDUAL_NAMES=("peri" "pre" "gru" "post")
RESIDUAL_METHODS=("add" "add" "gru" "add")
NORM_TYPES=("peri" "pre" "peri" "post")
LR="0.003"
SEEDS=(0 1 2)
RECALL_TYPES=("internal" "external" "none" "fixed")

IDX=$SLURM_ARRAY_TASK_ID
NUM_SEED=${#SEEDS[@]}
NUM_RES=${#RESIDUAL_NAMES[@]}
NUM_RECALL=${#RECALL_TYPES[@]}

SEED_IDX=$((IDX % NUM_SEED))
Q=$((IDX / NUM_SEED))
RES_IDX=$((Q % NUM_RES))
Q=$((Q / NUM_RES))
RECALL_IDX=$((Q % NUM_RECALL))

PROBLEM=sudoku
SEED=${SEEDS[$SEED_IDX]}
RESIDUAL_NAME=${RESIDUAL_NAMES[$RES_IDX]}
RESIDUAL_METHOD=${RESIDUAL_METHODS[$RES_IDX]}
NORM_TYPE=${NORM_TYPES[$RES_IDX]}
RECALL_TYPE=${RECALL_TYPES[$RECALL_IDX]}

ATTN_TYPE=full
INJECTION=linear
RECALL_INNER=false
NUM_BLOCKS=1
TEST_ITERS_LO=1
TEST_ITERS_HI=500
MAX_ITERS=30

case "$RECALL_TYPE" in
  internal)
    INJECTION=linear
    RECALL_INNER=true
    ;;
  external)
    INJECTION=linear
    RECALL_INNER=false
    ;;
  none)
    INJECTION=none
    RECALL_INNER=false
    ;;
  fixed)
    INJECTION=none
    RECALL_INNER=false
    NUM_BLOCKS=15
    MAX_ITERS=1
    TEST_ITERS_LO=1
    TEST_ITERS_HI=1
    ;;
  *)
    echo "Unknown recall type: $RECALL_TYPE" >&2
    exit 1
    ;;
esac

EXP_NAME="${RECALL_TYPE}_${PROBLEM}_${RESIDUAL_NAME}_lr${LR}_seed${SEED}"

python train_model.py \
    name=stability_all \
    +run_id=$EXP_NAME \
    problem=$PROBLEM \
    problem/model=transformer \
    problem.hyp.epochs=100 \
    problem.hyp.optimizer=adamw \
    problem.hyp.weight_decay=0.01 \
    problem.hyp.lr=$LR \
    problem.hyp.seed=$SEED \
    problem.hyp.use_amp=true \
    problem.hyp.train_mode=progressive \
    problem.hyp.rand_method=basic \
    problem.model.test_iterations.low=$TEST_ITERS_LO \
    problem.model.test_iterations.high=$TEST_ITERS_HI \
    problem.hyp.train_batch_size=500 \
    problem.model.kernel_size=5 \
    problem.model.hidden_dim=256 \
    problem.model.norm_type="$NORM_TYPE" \
    problem.model.attn_type="$ATTN_TYPE" \
    problem.model.num_sinks=0 \
    problem.model.num_blocks=$NUM_BLOCKS \
    problem.model.max_iters=$MAX_ITERS \
    problem.model.injection_type="$INJECTION" \
    problem.model.recall_inner=$RECALL_INNER \
    problem.model.residual_method="$RESIDUAL_METHOD" \
    +problem.model.qk_normalization=true \
    problem.model.init_method=default \
    problem.hyp.full_only_hard=false \
    problem.model.ccot=none \
    compile=true \
    +sweep_name=stability_final
