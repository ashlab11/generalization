#!/bin/bash
#SBATCH --job-name=length_gen
#SBATCH -o experiments/length_generalization/%j.out
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -n 6
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-he
#SBATCH --constraint=b200
#SBATCH --exclude=gpu4002

source .venv/bin/activate

PROBLEM="mazes"
LR="0.0003"
ATTN_TYPE="conv"
MAX_ITERS=30
TEST_LOW=1
TEST_HIGH=250

if [ "$PROBLEM" = "prefix_sums" ]; then
  TRAIN_BATCH_SIZE=500
else
  TRAIN_BATCH_SIZE=100
fi

EXP_NAME="cnn_${PROBLEM}_lr${LR}_mazes"

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
    problem.model.num_blocks=1 \
    problem.model.test_iterations.low=$TEST_LOW \
    problem.hyp.train_batch_size=$TRAIN_BATCH_SIZE \
    problem.model.test_iterations.high=$TEST_HIGH \
    problem.model.hidden_dim=256 \
    problem.model.norm_type=post \
    problem.model.attn_type=$ATTN_TYPE \
    problem.model.num_sinks=0 \
    problem.model.kernel_size=3 \
    problem.model.injection_type=linear \
    problem.model.recall_inner=false \
    problem.model.residual_method=add \
    +problem.model.qk_normalization=true \
    problem.model.init_method=default \
    problem.model.ccot=none \
    profile=false \
    compile=false \
    +sweep_name=length_generalization_2
