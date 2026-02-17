#!/bin/bash
#SBATCH --job-name=difficulty_gen_fixed15
#SBATCH -o experiments/difficulty_generalization/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -n 6
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --partition=gpu-he
#SBATCH --constraint=b200
#SBATCH --exclude=gpu4002

source .venv/bin/activate

IDX=$SLURM_ARRAY_TASK_ID

PROBLEMS=("chess" "sudoku")
LRS=("0.0001" "0.0003" "0.001")

PROBLEM_IDX=$((IDX % 2))
LR_IDX=$((IDX / 2))

PROBLEM=${PROBLEMS[$PROBLEM_IDX]}
LR=${LRS[$LR_IDX]}

EXP_NAME="fixed1_${PROBLEM}_${LR}"

python train_model.py \
    name=difficulty_generalization \
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
    problem.hyp.train_batch_size=500 \
    problem.model.num_blocks=1 \
    problem.model.max_iters=1 \
    problem.model.test_iterations.low=1 \
    problem.model.test_iterations.high=1 \
    problem.model.hidden_dim=256 \
    problem.model.norm_type=pre \
    problem.model.attn_type=full \
    problem.model.num_sinks=1 \
    problem.model.injection_type=none \
    problem.model.recall_inner=false \
    problem.model.residual_method=add \
    +problem.model.qk_normalization=true \
    problem.model.init_method=default \
    problem.model.ccot=none \
    profile=false \
    compile=false \
    +sweep_name=fixed15_no_recall_pre
