#!/bin/bash
#SBATCH --job-name=init_abl
#SBATCH -o ablations/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --partition=gpu-he
#SBATCH --constraint=nomig

source .venv/bin/activate

IDX=$SLURM_ARRAY_TASK_ID

case $IDX in
    0)
        EXP_NAME="init_default"
        INIT_METHOD="default"
        ;;
    1)
        EXP_NAME="init_xavier"
        INIT_METHOD="xavier"
        ;;
    2)
        EXP_NAME="init_xavier_small"
        INIT_METHOD="xavier_small"
        ;;
    3)
        EXP_NAME="init_orthogonal"
        INIT_METHOD="orthogonal"
        ;;
esac

# Optional: export SEED=0 for deterministic comparisons

python train_model.py \
    name=init_ablation \
    +run_id=$EXP_NAME\_softmin \
    problem=prefix_sums \
    problem/model=transformer \
    problem.hyp.epochs=100 \
    problem.hyp.optimizer=adamw \
    problem.hyp.weight_decay=0.01 \
    problem.hyp.lr=0.001 \
    problem.hyp.use_amp=true \
    problem.hyp.train_mode=softmin \
    problem.hyp.rand_method=basic \
    problem.model.test_iterations.low=1 \
    problem.model.test_iterations.high=500 \
    problem.model.hidden_dim=256 \
    problem.model.norm_type=peri \
    problem.model.attn_type=conv \
    problem.model.num_sinks=1 \
    problem.model.injection_type=concat \
    problem.model.recall_inner=true \
    problem.model.residual_method=gru \
    +problem.model.qk_normalization=true \
    problem.model.init_method=$INIT_METHOD \
    +sweep_name=init_ablation

echo "Training complete for $EXP_NAME"
