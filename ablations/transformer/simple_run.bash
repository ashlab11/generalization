#!/bin/bash
#SBATCH --job-name=transformer
#SBATCH -o ablations/transformer/%A.out
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --partition=gpu-he
#SBATCH --constraint=nomig

source .venv/bin/activate

# Run training
python train_model.py \
    name=conv_low_softmin \
    +run_id=conv_low_softmin \
    problem=prefix_sums \
    problem/model=transformer \
    problem.hyp.optimizer=adamw \
    problem.hyp.weight_decay=0.01 \
    problem.model.num_sinks=1 \
    problem.hyp.train_mode=softmin \
    problem.model.test_iterations.high=500 \
    problem.model.hidden_dim=256 \
    problem.model.norm_type=peri \
    problem.model.attn_type=conv \
    +problem.model.qk_normalization=true \
    problem.model.injection_type=concat \
    problem.model.residual_method=gru \
    problem.model.recall_inner=true \
    problem.hyp.lr=0.001 \
    problem.hyp.use_amp=true \
    +sweep_name=generalization

