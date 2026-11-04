#!/bin/bash
# measure_all.sh — run measure_jacobian.py over all 16 Jacobian checkpoints (1 and 2 blocks).
# Run from the repo root after training finishes. OUT_ROOT picks the run directory (default jacobian_runs).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
conda deactivate 2>/dev/null || true
export PYTHONPATH="$REPO_ROOT"

OUT_ROOT="$(realpath "${OUT_ROOT:-$REPO_ROOT/jacobian_runs}")"
export OUT_ROOT

NORMS=(post post post post pre pre pre pre)
RECALLS=(internal internal external external internal internal external external)
LRS=(0.001 0.003 0.001 0.003 0.001 0.003 0.001 0.003)

for NUM_BLOCKS in 1 2; do
for i in 0 1 2 3 4 5 6 7; do
  NORM=${NORMS[$i]}
  RECALL=${RECALLS[$i]}
  LR=${LRS[$i]}
  RECALL_INNER=false
  [[ "$RECALL" == internal ]] && RECALL_INNER=true

  RUN_DIR="$OUT_ROOT/${RECALL}_sudoku_${NORM}_lr${LR}_seed0"
  [[ "$NUM_BLOCKS" == 2 ]] && RUN_DIR="${RUN_DIR}_2blocks"
  CKPT="$RUN_DIR/model_hardbest.pth"
  [[ -f "$CKPT" ]] || CKPT="$RUN_DIR/model_best.pth"
  [[ -f "$CKPT" ]] || { echo "SKIP (no checkpoint): $RUN_DIR"; continue; }

  echo "=== Measuring $RUN_DIR ==="
  python "$REPO_ROOT/analyses/measure_jacobian.py" \
      problem=sudoku \
      problem/model=transformer \
      problem.model.hidden_dim=256 \
      problem.model.norm_type="$NORM" \
      problem.model.injection_type=linear \
      problem.model.recall_inner=$RECALL_INNER \
      problem.model.residual_method=add \
      problem.model.num_blocks=$NUM_BLOCKS \
      problem.model.max_iters=30 \
      problem.model.attn_type=full \
      problem.model.num_sinks=0 \
      problem.model.kernel_size=5 \
      problem.model.ccot=none \
      +problem.model.qk_normalization=true \
      problem.model.model_path="$CKPT" \
      hydra.run.dir="$RUN_DIR/jacobian_measurement"
done
done

echo
echo "=== Combined table ==="
python - << 'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob(os.path.join(os.environ["OUT_ROOT"], "*/jacobian_measurements.json"))):
    s = json.load(open(p))
    f = lambda z: "-" if z is None else f"{z:.3f}"
    rows.append(f"| {os.path.basename(os.path.dirname(p))} "
                f"| {f(s['rho_Wx_proxy'])} | {f(s['rho_full_mean_converged'])} "
                f"| {f(s['rho_full_no_norm_mean_converged'])} "
                f"| {s['n_converged']}/{s['n_inputs']} "
                f"| {f(s.get('cos_sim_init_mean_converged'))} "
                f"| {f(s.get('cos_sim_fixed_point_mean_converged'))} |")
print("| config | rho(Wx) | rho(M_full) | rho(M_full, no outer norm) | converged | cos sim init | cos sim fixed pt |")
print("|---|---|---|---|---|---|---|")
print("\n".join(rows))
PY