#!/usr/bin/env bash
# Publishable ablation orchestrator.
#
# Runs (sequentially, to avoid CPU oversubscription on 16 cores):
#   1) Minimal Stackelberg (no CoPO underneath) -- 1M env steps x N seeds
#   2) CoPO baseline (original fork, LCF coordination) -- 1M env steps x N seeds
#
# Both runs write to ./<exp_name>/ in the current working directory and log
# success / crash / out / reward via MultiAgentDrivingCallbacks.
#
# Usage:
#   bash run_publishable_ablation.sh                  # default: 2 seeds each
#   NUM_SEEDS=3 bash run_publishable_ablation.sh      # 3 seeds each
#   NUM_SEEDS=1 START_SEED=200 \                      # add a 3rd seed later
#       bash run_publishable_ablation.sh
#
# Logs are tee'd into ./logs/.

set -e -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- conda env ---
# shellcheck disable=SC1091
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stackelberg

# --- knobs ---
NUM_SEEDS="${NUM_SEEDS:-2}"
START_SEED="${START_SEED:-0}"
SUFFIX=""
if [ "$START_SEED" -ne 0 ]; then
    # Append start_seed to the exp_name so adding more seeds later does not
    # collide with previous seed dirs in Tune's bookkeeping.
    SUFFIX="_s${START_SEED}"
fi

STACK_EXP="minimal_stackelberg_1M${SUFFIX}"
COPO_EXP="copo_baseline_1M${SUFFIX}"

mkdir -p logs

export PYTHONPATH="$REPO_ROOT/CoPO/copo_code"

echo "================================================================"
echo "Publishable ablation"
echo "  num_seeds   : $NUM_SEEDS  (start_seed=$START_SEED)"
echo "  Stackelberg : $STACK_EXP   (1,000,000 env steps per seed)"
echo "  CoPO base   : $COPO_EXP    (1,000,000 env steps per seed)"
echo "  CPU only (RTX 5070 sm_120 unsupported by torch 2.4.1+cu121)"
echo "================================================================"

# 1) Stackelberg first (lighter; ~1.6h per seed).
echo ""
echo "[1/2] Launching Minimal Stackelberg ..."
python CoPO/copo_code/copo/torch_stackelberg/train_minimal_stackelberg_1M.py \
    --exp-name "$STACK_EXP" \
    --num-seeds "$NUM_SEEDS" \
    --start-seed "$START_SEED" \
    --num-gpus 0 \
    2>&1 | tee "logs/${STACK_EXP}.log"

# 2) CoPO baseline (heavier; ~3-4h per seed).
echo ""
echo "[2/2] Launching CoPO baseline ..."
python CoPO/copo_code/copo/torch_stackelberg/train_copo_baseline_1M.py \
    --exp-name "$COPO_EXP" \
    --num-seeds "$NUM_SEEDS" \
    --start-seed "$START_SEED" \
    --num-gpus 0 \
    2>&1 | tee "logs/${COPO_EXP}.log"

echo ""
echo "Done. Plot with:"
echo "  python plot_publishable_ablation.py --stack-dir $STACK_EXP --copo-dir $COPO_EXP"
