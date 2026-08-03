#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

RESULTS_ROOT="${RESULTS_ROOT:-results/active_learning}"

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
REWEIGHT_POOL=100000

# ── Wasserstein-L2 query + Voronoi-L2 reweighting lambda sweep ──
LAMBDAS=(300 1000 3000 10000 30000)

for lambda in "${LAMBDAS[@]}"; do
  out_dir="${RESULTS_ROOT}/truelygreedy_l2_sweep/al_wasserstein_l2_${TOTAL_QUERIES}_lambda_${lambda}"
  echo ""
  echo "============================================================"
  echo "  Strategy: wasserstein_l2  |  Reweighting: voronoi_l2"
  echo "  Reweight Lambda: ${lambda}"
  echo "  Output:   ${out_dir}"
  echo "============================================================"
  
  python active_learning.py \
    --warm-start-file "$WARM_START" \
    --full-data-file  "$FULL_DATA" \
    --feh-threshold   "$FEH_THRESHOLD" \
    --strategy        "wasserstein_l2" \
    --reweighting     "voronoi_l2" \
    --soft-topk       "$SOFT_TOPK" \
    --reweight-pool-size "$REWEIGHT_POOL" \
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
