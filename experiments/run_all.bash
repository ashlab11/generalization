#!/bin/bash
#SBATCH --job-name=stability_all
#SBATCH -o experiments/stability/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --gres=gpu:1
# 4 recall × 4 residual × 3 prob × 3 LR × 3 seed = 432 (LR fastest, seed slowest)
#SBATCH --array=0-431
#SBATCH --partition=gpu-he
#SBATCH --constraint=b200

set -euo pipefail
source .venv/bin/activate

PROBLEMS=("chess" "sudoku" "prefix_sums")
RESIDUAL_NAMES=("peri" "pre" "gru" "post")
RESIDUAL_METHODS=("add" "add" "gru" "add")
NORM_TYPES=("peri" "pre" "peri" "post")
LRS=("0.0001" "0.0003" "0.001")
SEEDS=(0 1 2)
RECALL_TYPES=("internal" "external" "none" "fixed")

IDX=$SLURM_ARRAY_TASK_ID
NUM_LR=${#LRS[@]}
NUM_SEED=${#SEEDS[@]}
NUM_PROB=${#PROBLEMS[@]}
NUM_RES=${#RESIDUAL_NAMES[@]}
NUM_RECALL=${#RECALL_TYPES[@]}

LR_IDX=$((IDX % NUM_LR))
Q=$((IDX / NUM_LR))
PROB_IDX=$((Q % NUM_PROB))
Q=$((Q / NUM_PROB))
RES_IDX=$((Q % NUM_RES))
Q=$((Q / NUM_RES))
RECALL_IDX=$((Q % NUM_RECALL))
SEED_IDX=$((Q / NUM_RECALL))

PROBLEM=${PROBLEMS[$PROB_IDX]}
SEED=${SEEDS[$SEED_IDX]}
RESIDUAL_NAME=${RESIDUAL_NAMES[$RES_IDX]}
RESIDUAL_METHOD=${RESIDUAL_METHODS[$RES_IDX]}
NORM_TYPE=${NORM_TYPES[$RES_IDX]}
LR=${LRS[$LR_IDX]}
RECALL_TYPE=${RECALL_TYPES[$RECALL_IDX]}

ATTN_TYPE=full
[[ "$PROBLEM" == prefix_sums ]] && ATTN_TYPE=local

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
    problem.hyp.alpha=1 \
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
    hydra.run.dir=../scratch/outputs/$EXP_NAME \
    compile=true \
    +sweep_name=stability_final \