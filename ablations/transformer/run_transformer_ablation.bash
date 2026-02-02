#!/bin/bash
#SBATCH --job-name=transformer_abl
#SBATCH -o ablations/transformer/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --array=2-2
#SBATCH --partition=gpu-he
#SBATCH --constraint=nomig

source .venv/bin/activate
# Comparing attention mechanisms!
IDX=$((SLURM_ARRAY_TASK_ID - 1))

case $IDX in 
    0)
        EXP_NAME="geiping_qk_full_softmin"
        ATTN_TYPE=full
        NORM_TYPE=post
        RESIDUAL_METHOD=add
        QK_NORMALIZATION=true
        INJECTION_TYPE=concat
        FULL_CONCAT=false
        SINK=0
        MONOTONE_LAMBDA=0
        ;;
    1)
        EXP_NAME="gru_full_softmin_noise"
        ATTN_TYPE=full
        NORM_TYPE=peri
        RESIDUAL_METHOD=gru
        QK_NORMALIZATION=true
        INJECTION_TYPE=none
        FULL_CONCAT=true
        NOISE_PROB=0.05
        SINK=1
        MONOTONE_LAMBDA=0
        ;;
    2)
        EXP_NAME="gru_local_sink_softmin"
        ATTN_TYPE=local
        NORM_TYPE=peri
        RESIDUAL_METHOD=gru
        QK_NORMALIZATION=true
        INJECTION_TYPE=none
        FULL_CONCAT=true
        NOISE_PROB=0
        SINK=4
        MONOTONE_LAMBDA=0
        ;;
    3) 
        EXP_NAME="gru_full_sink_monotone_softmin"
        ATTN_TYPE=full
        NORM_TYPE=peri
        RESIDUAL_METHOD=gru
        QK_NORMALIZATION=true
        INJECTION_TYPE=none
        FULL_CONCAT=true
        SINK=1
        MONOTONE_LAMBDA=0.01
        ;;
    4) 
        EXP_NAME="cnn"
        ATTN_TYPE=conv
        NORM_TYPE=peri
        RESIDUAL_METHOD=gru
        QK_NORMALIZATION=true
        INJECTION_TYPE=none
        FULL_CONCAT=true
        SINK=1
        MONOTONE_LAMBDA=0
        ;;
esac

# Run training
python train_model.py \
    name=transformer_ablation \
    problem.hyp.train_mode=softmin \
    problem=rule110 \
    problem/model=dt_transformer \
    problem.hyp.optimizer=adamw \
    problem.hyp.epochs=500 \
    problem.train_data=32 \
    problem.test_t_min=16 \
    problem.test_data=16 \
    +problem.exclude_t=[16] \
    problem.model.num_blocks=1 \
    problem.hyp.weight_decay=0.01 \
    +run_id=$EXP_NAME \
    problem.model.test_iterations.low=1 \
    problem.model.test_iterations.high=500 \
    problem.hyp.rand_method=basic \
    problem.model.hidden_dim=256 \
    problem.model.norm_type=$NORM_TYPE \
    problem.model.attn_type=$ATTN_TYPE \
    problem.model.num_sinks=$SINK \
    problem.model.residual_method=$RESIDUAL_METHOD \
    +problem.model.qk_normalization=$QK_NORMALIZATION \
    +problem.model.full_concat=$FULL_CONCAT \
    problem.model.monotone_lambda=$MONOTONE_LAMBDA \
    problem.hyp.lr=0.001 \
    problem.hyp.use_amp=true \
    problem.model.noise_prob=$NOISE_PROB \
    +sweep_name=rule110_correct

echo "Training complete for $EXP_NAME"
