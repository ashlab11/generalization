#!/bin/bash
#SBATCH --job-name=transformer
#SBATCH -o ablations/transformer/%A.out
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --partition=gpu-he
#SBATCH --constraint=mig

source .venv/bin/activate

# Run training
python train_model.py \
    name=rule110 \
    +run_id=rule110_full \
    problem=rule110 \
    problem/model=dt_transformer \
    problem.hyp.optimizer=adamw \
    problem.hyp.weight_decay=0.01 \
    problem.model.test_iterations.high=500 \
    problem.model.width=256 \
    problem.model.norm_type=peri \
    problem.model.attn_type=full \
    +problem.model.qk_normalization=true \
    problem.model.injection_type=none \
    problem.model.residual_method=gru \
    +problem.model.full_concat=true \
    problem.hyp.lr=0.0001 \
    problem.hyp.use_amp=true \
    +sweep_name=rule110

