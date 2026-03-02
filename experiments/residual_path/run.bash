#!/bin/bash
#SBATCH --job-name=residual_path_abl
#SBATCH -o experiments/residual_path/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-47
#SBATCH --partition=gpu-he
#SBATCH --constraint=b200

set -euo pipefail
source .venv/bin/activate

RESIDUAL_NAMES=("peri" "pre" "gru" "post")
RESIDUAL_METHODS=("add" "add" "gru" "add")
NORM_TYPES=("peri" "pre" "peri" "post")
LRS=("0.0001" "0.0003" "0.001")

# Example: N_SEEDS=8 sbatch --array=0-$((8*4*3-1)) experiments/residual_path/run.bash
N_SEEDS="${N_SEEDS:-4}"
SEEDS=($(seq 0 $((N_SEEDS - 1))))

IDX=$SLURM_ARRAY_TASK_ID
NUM_RESIDUALS=${#RESIDUAL_NAMES[@]}
NUM_LRS=${#LRS[@]}
JOBS_PER_SEED=$((NUM_RESIDUALS * NUM_LRS))

SEED_IDX=$((IDX / JOBS_PER_SEED))
RESIDUAL_IDX=$(((IDX % JOBS_PER_SEED) / NUM_LRS))
LR_IDX=$((IDX % NUM_LRS))

SEED=${SEEDS[$SEED_IDX]}
RESIDUAL_NAME=${RESIDUAL_NAMES[$RESIDUAL_IDX]}
RESIDUAL_METHOD=${RESIDUAL_METHODS[$RESIDUAL_IDX]}
NORM_TYPE=${NORM_TYPES[$RESIDUAL_IDX]}
LR=${LRS[$LR_IDX]}


EXP_NAME="concat_${RESIDUAL_NAME}_lr${LR}_seed${SEED}"

python train_model.py \
    name=residual_path_ablation \
    +run_id=$EXP_NAME \
    problem=prefix_sums \
    problem/model=transformer \
    problem.hyp.epochs=100 \
    problem.hyp.optimizer=adamw \
    problem.hyp.weight_decay=0.01 \
    problem.hyp.lr=$LR \
    problem.hyp.seed=$SEED \
    problem.hyp.use_amp=true \
    problem.hyp.train_mode=progressive \
    problem.hyp.rand_method=basic \
    problem.model.test_iterations.low=1 \
    problem.model.test_iterations.high=500 \
    problem.model.kernel_size=5 \
    problem.model.hidden_dim=256 \
    problem.model.norm_type="$NORM_TYPE" \
    problem.model.attn_type=local \
    problem.model.num_sinks=0 \
    problem.model.injection_type=concat \
    problem.model.recall_inner=true \
    problem.model.residual_method="$RESIDUAL_METHOD" \
    +problem.model.qk_normalization=true \
    problem.model.init_method=default \
    problem.hyp.full_only_hard=true \
    problem.model.ccot=none \
    compile=true \
    +sweep_name=residual_path_ablation_transformer
