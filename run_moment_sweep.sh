#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

# Shared hyperparameters for the linear moment-matching path.
WARM_START=bp_rp_lamost_normalized_low_teff.h5
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=100
EVAL_EVERY=10
LAMBDA_MP=0.01
WASS_POOL=45000
C=10000.0
RIDGE_ALPHA=1000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=16
N_SNAPSHOTS=10
SOFT_TOPK=20
SOFTMAX_POOL=100000
REWEIGHTING=moment_l2
REWEIGHT_LAMBDA=1.0
MOMENT_WEIGHT_ITERS=200
MODEL=ridge

# Ridge values for the linear moment/design objective.
MOMENT_RIDGES=(1 10 100 1000 3000 10000)

for moment_ridge in "${MOMENT_RIDGES[@]}"; do
  out_dir="moment_l2_sweep_results/al_moment_matching_${MODEL}_${TOTAL_QUERIES}_ridge_${moment_ridge}"
  echo ""
  echo "============================================================"
  echo "  Strategy: moment_matching  |  Model: ${MODEL}"
  echo "  Reweighting: ${REWEIGHTING}  |  Reweight Lambda: ${REWEIGHT_LAMBDA}"
  echo "  Moment Ridge: ${moment_ridge}  |  Ridge Alpha: ${RIDGE_ALPHA}"
  echo "  Output: ${out_dir}"
  echo "============================================================"

  python active_learning.py \
    --warm-start-file "$WARM_START" \
    --full-data-file  "$FULL_DATA" \
    --feh-threshold   "$FEH_THRESHOLD" \
    --strategy        "moment_matching" \
    --model           "$MODEL" \
    --ridge-alpha     "$RIDGE_ALPHA" \
    --moment-ridge    "$moment_ridge" \
    --reweighting     "$REWEIGHTING" \
    --soft-topk       "$SOFT_TOPK" \
    --softmax-pool-size "$SOFTMAX_POOL" \
    --reweight-lambda "$REWEIGHT_LAMBDA" \
    --moment-weight-iters "$MOMENT_WEIGHT_ITERS" \
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
echo "All moment matching sweep experiments completed."
