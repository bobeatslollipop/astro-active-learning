#!/usr/bin/env bash
set -euo pipefail

# ── Shared hyperparameters (edit once, applies to all experiments) ──
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
SOFTMAX_POOL=500000

# ── Experiments: "strategy  reweighting  out_dir" ──
EXPERIMENTS=(
  # "random        none  al_random_${TOTAL_QUERIES}"
  # "uncertainty   none  al_uncertainty_${TOTAL_QUERIES}"
  # "uncertainty   hard  al_uncertainty_hard_${TOTAL_QUERIES}"
  # "wasserstein   none  al_wasserstein_${TOTAL_QUERIES}"
  # "wasserstein   hard  al_wasserstein_hard_${TOTAL_QUERIES}"
  # "kmedianpp     none  al_kmedianpp_${TOTAL_QUERIES}"
  # "random        hard  al_random_hard_${TOTAL_QUERIES}"
  # "kmedianpp     hard  al_kmedianpp_hard_${TOTAL_QUERIES}"
  "wasserstein   soft  al_wasserstein_soft_${TOTAL_QUERIES}"
  "kmedianpp     soft  al_kmedianpp_soft_${TOTAL_QUERIES}"
  "random        soft  al_random_soft_${TOTAL_QUERIES}"
  "uncertainty   soft  al_uncertainty_soft_${TOTAL_QUERIES}"
)

# ── Run each experiment ──
for exp in "${EXPERIMENTS[@]}"; do
  read -r strategy reweighting out_dir <<< "$exp"
  echo ""
  echo "============================================================"
  echo "  Strategy: ${strategy}  |  Reweighting: ${reweighting}"
  echo "  Output:   ${out_dir}"
  echo "============================================================"
  python active_learning.py \
    --warm-start-file "$WARM_START" \
    --full-data-file  "$FULL_DATA" \
    --feh-threshold   "$FEH_THRESHOLD" \
    --strategy        "$strategy" \
    --reweighting     "$reweighting" \
    --soft-topk       "$SOFT_TOPK" \
    --softmax-pool-size "$SOFTMAX_POOL" \
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
echo "All experiments completed."
