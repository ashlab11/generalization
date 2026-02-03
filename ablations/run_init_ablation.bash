#!/bin/bash
#SBATCH --job-name=init_abl
#SBATCH -o ablations/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --array=4,6
#SBATCH --partition=gpu-he
#SBATCH --constraint=nomig

source .venv/bin/activate

IDX=$SLURM_ARRAY_TASK_ID

case $IDX in
    0)
        EXP_NAME="default_amp_prog"
        INIT_METHOD="default"
        USE_AMP=true
        TRAIN_MODE="progressive"
        ;;
    1)
        EXP_NAME="xavier_amp_prog"
        INIT_METHOD="xavier"
        USE_AMP=true
        TRAIN_MODE="progressive"
        ;;
    2)
        EXP_NAME="default_fp32_prog"
        INIT_METHOD="default"
        USE_AMP=false
        TRAIN_MODE="progressive"
        ;;
    3)
        EXP_NAME="xavier_fp32_prog"
        INIT_METHOD="xavier"
        USE_AMP=false
        TRAIN_MODE="progressive"
        ;;
    4)
        EXP_NAME="default_amp_new_softmin"
        INIT_METHOD="default"
        USE_AMP=true
        TRAIN_MODE="softmin"
        ;;
    5)
        EXP_NAME="xavier_amp_new_softmin"
        INIT_METHOD="xavier"
        USE_AMP=true
        TRAIN_MODE="softmin"
        ;;
    6)
        EXP_NAME="default_fp32_new_softmin"
        INIT_METHOD="default"
        USE_AMP=false
        TRAIN_MODE="softmin"
        ;;
    7)
        EXP_NAME="xavier_fp32_softmin"
        INIT_METHOD="xavier"
        USE_AMP=false
        TRAIN_MODE="softmin"
        ;;
esac

# Optional: export SEED=0 for deterministic comparisons

python train_model.py \
    name=init_ablation \
    +run_id=$EXP_NAME\_eps \
    problem=prefix_sums \
    problem/model=transformer \
    problem.hyp.epochs=100 \
    problem.hyp.optimizer=adamw \
    problem.hyp.weight_decay=0.01 \
    problem.hyp.lr=0.001 \
    problem.hyp.use_amp=$USE_AMP \
    problem.hyp.train_mode=$TRAIN_MODE \
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
