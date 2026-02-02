#!/bin/bash
#SBATCH --job-name=transformer_abl
#SBATCH -o ablations/transformer/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --array=1-6
#SBATCH --partition=gpu-he
#SBATCH --constraint=nomig

source .venv/bin/activate
# Comparing attention mechanisms!
IDX=$((SLURM_ARRAY_TASK_ID - 1))

case $IDX in
    0)
        RULE=184
        ;;
    1)
        RULE=108
        ;;
    2)
        RULE=30
        ;;
    3)
        RULE=90
        ;;
    4) 
        RULE=110
        ;;
    5) 
        RULE=54
        ;;
esac

# Run training
python train_model.py \
    name=transformer_ablation \
    problem.hyp.train_mode=softmin \
    problem=cellular \
    problem/model=dt_transformer \
    problem.hyp.optimizer=adamw \
    problem.hyp.epochs=500 \
    problem.hyp.lr_schedule="[400,450]" \
    problem.train_data=16 \
    problem.test_t_min=17 \
    problem.test_data=32 \
    problem.model.num_blocks=1 \
    problem.hyp.weight_decay=0.01 \
    +run_id=cellular_${RULE} \
    problem.model.test_iterations.low=1 \
    problem.model.test_iterations.high=500 \
    problem.hyp.rand_method=basic \
    problem.model.hidden_dim=256 \
    problem.model.norm_type=peri \
    problem.model.attn_type=local \
    problem.model.num_sinks=1 \
    problem.model.residual_method=gru \
    +problem.model.qk_normalization=true \
    +problem.model.full_concat=true \
    problem.model.monotone_lambda=0 \
    problem.hyp.lr=0.001 \
    problem.hyp.use_amp=true \
    +sweep_name=cellular
