#!/bin/bash
#SBATCH --job-name=length_gen
#SBATCH -o experiments/length_generalization/%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -n 6
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --array=13,15,17,19
#SBATCH --partition=gpu-he
#SBATCH --constraint=b200
#SBATCH --exclude=gpu4002

source .venv/bin/activate

IDX=$SLURM_ARRAY_TASK_ID

PROBLEMS=("prefix_sums" "mazes")
LRS=("0.0001" "0.0003" "0.001")
MODEL_NAMES=("fixed15_full" "recurrent_full" "recurrent_local_r5" "recurrent_local_rbig")

PROBLEM_IDX=$((IDX % 2))
IDX=$((IDX / 2))
LR_IDX=$((IDX % 3))
IDX=$((IDX / 3))
MODEL_IDX=$((IDX % 4))

PROBLEM=${PROBLEMS[$PROBLEM_IDX]}
LR=${LRS[$LR_IDX]}
MODEL_NAME=${MODEL_NAMES[$MODEL_IDX]}

case "$MODEL_IDX" in
  0)
    ATTN_TYPE="full"
    NUM_BLOCKS=15
    MAX_ITERS=1
    TEST_LOW=1
    TEST_HIGH=1
    INJECTION_TYPE="none"
    KERNEL_SIZE=11
    ;;
  1)
    ATTN_TYPE="full"
    NUM_BLOCKS=1
    MAX_ITERS=30
    TEST_LOW=1
    TEST_HIGH=250
    INJECTION_TYPE="linear"
    KERNEL_SIZE=11
    ;;
  2)
    ATTN_TYPE="local"
    NUM_BLOCKS=1
    MAX_ITERS=30
    TEST_LOW=1
    TEST_HIGH=250
    INJECTION_TYPE="linear"
    KERNEL_SIZE=11
    ;;
  3)
    ATTN_TYPE="local"
    NUM_BLOCKS=1
    MAX_ITERS=30
    TEST_LOW=1
    TEST_HIGH=250
    INJECTION_TYPE="linear"
    if [ "$PROBLEM" = "prefix_sums" ]; then
      KERNEL_SIZE=31
    else
      KERNEL_SIZE=9
    fi
    ;;
esac

if [ "$PROBLEM" = "prefix_sums" ]; then
  TRAIN_BATCH_SIZE=500
else
  TRAIN_BATCH_SIZE=100
fi

EXP_NAME="${MODEL_NAME}_${PROBLEM}_lr${LR}_prog"

python train_model.py \
    name=length_generalization \
    +run_id=$EXP_NAME \
    problem=$PROBLEM \
    problem/model=transformer \
    problem.hyp.epochs=100 \
    problem.hyp.optimizer=adamw \
    problem.hyp.weight_decay=0.01 \
    problem.hyp.lr=$LR \
    problem.hyp.use_amp=true \
    problem.hyp.train_mode=progressive \
    problem.hyp.rand_method=basic \
    problem.model.max_iters=$MAX_ITERS \
    problem.model.num_blocks=$NUM_BLOCKS \
    problem.model.test_iterations.low=$TEST_LOW \
    problem.hyp.train_batch_size=$TRAIN_BATCH_SIZE \
    problem.model.test_iterations.high=$TEST_HIGH \
    problem.model.hidden_dim=256 \
    problem.model.norm_type=post \
    problem.model.attn_type=$ATTN_TYPE \
    problem.model.num_sinks=0 \
    problem.model.kernel_size=$KERNEL_SIZE \
    problem.model.injection_type=$INJECTION_TYPE \
    problem.model.recall_inner=false \
    problem.model.residual_method=add \
    +problem.model.qk_normalization=true \
    problem.model.init_method=default \
    problem.model.ccot=none \
    profile=false \
    compile=false \
    +sweep_name=length_generalization_2
