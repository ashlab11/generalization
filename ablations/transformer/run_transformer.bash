#!/bin/bash
#SBATCH --job-name=transformer
#SBATCH -o ablations/transformer/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array 1-18
#SBATCH -p gpu-b200

source .venv/bin/activate
# Generate all combinations: injection_type (3) × norm_type (3) × recall_inner (2) = 18
# Order: for each injection_type, for each norm_type, for each recall_inner
IDX=$((SLURM_ARRAY_TASK_ID - 1))

INJECTION_IDX=$((IDX / 6))
NORM_IDX=$(((IDX % 6) / 2))
RECALL_IDX=$((IDX % 2))

case $INJECTION_IDX in
    0) INJECTION_TYPE=add ;;
    1) INJECTION_TYPE=concat ;;
    2) INJECTION_TYPE=none ;;
esac

case $NORM_IDX in
    0) NORM_TYPE=pre ;;
    1) NORM_TYPE=peri ;;
    2) NORM_TYPE=post ;;
esac

case $RECALL_IDX in
    0) RECALL_INNER=false ;;
    1) RECALL_INNER=true ;;
esac

EXP_NAME="spectral_low_${RECALL_INNER}_${NORM_TYPE}_${INJECTION_TYPE}"

# Run training
python train_model.py \
    name=transformer_ablation \
    problem/model=dt_transformer \
    +run_id=$EXP_NAME \
    problem.model.test_iterations.high=500 \
    problem.model.num_blocks=1 \
    problem.model.width=256 \
    problem.model.injection_type=$INJECTION_TYPE \
    problem.model.norm_type=$NORM_TYPE \
    problem.model.recall_inner=$RECALL_INNER \
    problem.model.attn_type=local \
    problem.hyp.lr=0.001 \
    problem.hyp.use_amp=true \
    +problem.model.spectral=true \
    +sweep_num=2

echo "Training complete for $EXP_NAME"