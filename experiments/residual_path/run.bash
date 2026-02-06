#!/bin/bash
#SBATCH --job-name=residual_path_abl
#SBATCH -o experiments/residual_path/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-47
#SBATCH --partition=gpu-he
#SBATCH --constraint=nomig

source .venv/bin/activate

IDX=$SLURM_ARRAY_TASK_ID

RESIDUAL_NAMES=("peri" "pre" "gru" "post")
RESIDUAL_METHODS=("add" "add" "gru" "add")
NORM_TYPES=("peri" "pre" "peri" "post")
LRS=("0.0001" "0.0003" "0.001")
SEEDS=("0" "1" "2" "3")

SEED_IDX=$((IDX / 12))
RESIDUAL_IDX=$(((IDX % 12) / 3))
LR_IDX=$((IDX % 3))

SEED=${SEEDS[$SEED_IDX]}
RESIDUAL_NAME=${RESIDUAL_NAMES[$RESIDUAL_IDX]}
RESIDUAL_METHOD=${RESIDUAL_METHODS[$RESIDUAL_IDX]}
NORM_TYPE=${NORM_TYPES[$RESIDUAL_IDX]}
LR=${LRS[$LR_IDX]}

EXP_NAME="${RESIDUAL_NAME}_lr${LR}_seed${SEED}"

python train_model.py \
    name=residual_path_ablation \
    +run_id=$EXP_NAME \
    problem=prefix_sums \
    problem/model=transformer \
    problem.hyp.epochs=100 \
    problem.hyp.optimizer=adamw \
    problem.hyp.weight_decay=0.01 \
    problem.hyp.lr=$LR \
    +problem.hyp.seed=$SEED \
    problem.hyp.use_amp=false \
    problem.hyp.train_mode=progressive \
    problem.hyp.rand_method=basic \
    problem.model.test_iterations.low=1 \
    problem.model.test_iterations.high=500 \
    problem.model.hidden_dim=256 \
    problem.model.norm_type="$NORM_TYPE" \
    problem.model.attn_type=conv \
    problem.model.num_sinks=1 \
    problem.model.injection_type=linear \
    problem.model.recall_inner=false \
    problem.model.residual_method="$RESIDUAL_METHOD" \
    +problem.model.qk_normalization=true \
    problem.model.init_method=default \
    +sweep_name=residual_path_ablation