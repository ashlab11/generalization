#!/bin/bash
#SBATCH --job-name=difficulty_gen
#SBATCH -o experiments/difficulty_generalization/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -n 6
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --array=2,5
#SBATCH --partition=gpu-he
#SBATCH --constraint=b200

source .venv/bin/activate

IDX=$SLURM_ARRAY_TASK_ID

PROBLEMS=("chess" "sudoku")
MODES=("softmin" "progressive" "upweight")

PROBLEM_IDX=$((IDX / 3))
MODE_IDX=$((IDX % 3))

PROBLEM=${PROBLEMS[$PROBLEM_IDX]}
MODE=${MODES[$MODE_IDX]}

EXP_NAME="${PROBLEM}_mode${MODE}_upweight"

python train_model.py \
    name=difficulty_generalization \
    +run_id=$EXP_NAME \
    problem=$PROBLEM \
    problem/model=transformer \
    problem.hyp.epochs=100 \
    problem.hyp.optimizer=adamw \
    problem.hyp.weight_decay=0.01 \
    problem.hyp.lr=0.0003 \
    problem.hyp.use_amp=true \
    problem.hyp.train_mode=$MODE \
    problem.hyp.rand_method=basic \
    problem.model.test_iterations.low=1 \
    problem.hyp.train_batch_size=500 \
    problem.model.test_iterations.high=500 \
    problem.model.hidden_dim=256 \
    problem.model.norm_type=peri \
    problem.model.attn_type=full \
    problem.model.num_sinks=1 \
    problem.model.injection_type=concat \
    problem.model.recall_inner=true \
    problem.model.residual_method=gru \
    +problem.model.qk_normalization=true \
    problem.model.init_method=default \
    profile=false \
    compile=false \
    +sweep_name=difficulty_generalization
