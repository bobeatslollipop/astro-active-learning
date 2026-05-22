#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

# ── Shared hyperparameters (same as run_experiments.sh) ──
WARM_START=bp_rp_lamost_normalized_low_teff.h5
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=100
EVAL_EVERY=10
LAMBDA_MP=0.01
WASS_POOL=45000
C=10000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=16
N_SNAPSHOTS=10
SOFT_TOPK=20
SOFTMAX_POOL=100000

# ── Lambda geometrically sweep: 0.1, 1, 10, 100 ──
LAMBDAS=(300)

for lambda in "${LAMBDAS[@]}"; do
  out_dir="l2_sweep_results/al_wasserstein_l2_${TOTAL_QUERIES}_lambda_${lambda}"
  echo ""
  echo "============================================================"
  echo "  Strategy: wasserstein  |  Reweighting: l2"
  echo "  Reweight Lambda: ${lambda}"
  echo "  Output:   ${out_dir}"
  echo "============================================================"
  
  python active_learning.py \
    --warm-start-file "$WARM_START" \
    --full-data-file  "$FULL_DATA" \
    --feh-threshold   "$FEH_THRESHOLD" \
    --strategy        "wasserstein" \
    --reweighting     "l2" \
    --soft-topk       "$SOFT_TOPK" \
    --softmax-pool-size "$SOFTMAX_POOL" \
    --reweight-lambda "$lambda" \
    --total-queries   "$TOTAL_QUERIES" \
    --eval-every      "$EVAL_EVERY" \
    --lambda-MP       "$LAMBDA_MP" \
    --wass-pool-size  "$WASS_POOL" \
    --C               "$C" \
    --eval-size       "$EVAL_SIZE" \
    --seed            "$SEED" \
    --n-trials        "$N_TRIALS" \
    --n-snapshots     "$N_SNAPSHOTS" \
    --out-dir         "$out_dir"
done

echo ""
echo "All sweep experiments completed."
