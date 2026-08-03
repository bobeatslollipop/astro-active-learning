#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

# Shared setup for the XGBoost + Voronoi-L2 sampling-method comparison.
WARM_START=bp_rp_lamost_normalized_low_teff.h5
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=150
EVAL_EVERY=10
LAMBDA_MP=0.01
WASS_POOL=45000
WASS_PLAN_SIZE="$EVAL_EVERY"
C=10000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=5
N_SNAPSHOTS=15
SOFT_TOPK=20
REWEIGHT_POOL=100000
REWEIGHT_LAMBDA=1000.0
VORONOI_L2_MAX_ITER=8
VORONOI_L2_INITIAL_MAX_ITER=16

# XGBoost configuration from the full-training xgb_deeper benchmark.
XGB_N_ESTIMATORS=700
XGB_MAX_DEPTH=6
XGB_LEARNING_RATE=0.03
XGB_SUBSAMPLE=0.8
XGB_COLSAMPLE_BYTREE=0.8
XGB_MIN_CHILD_WEIGHT=10
XGB_GAMMA=0.0
XGB_REG_LAMBDA=2
XGB_TREE_METHOD=hist
XGB_DEVICE=cuda
XGB_N_JOBS=-1

RESULTS_ROOT="${RESULTS_ROOT:-results/active_learning}"
RESULT_ROOT="${RESULTS_ROOT}/Jul23-xgb-sampling-150q"
mkdir -p "${RESULT_ROOT}/figures/final"

# This is a replacement run; clear any previous partial output for this root.
rm -rf "$RESULT_ROOT"

# Experiments: "strategy  reweighting  out_dir"
EXPERIMENTS=(
  "kmedianpp       voronoi_l2    ${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}"
  "random          voronoi_l2    ${RESULT_ROOT}/al_xgb_random_l2_${TOTAL_QUERIES}"
  "wasserstein_l2  voronoi_l2    ${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}"
)

for exp in "${EXPERIMENTS[@]}"; do
  read -r strategy reweighting out_dir <<< "$exp"
  echo ""
  echo "============================================================"
  echo "  Strategy: ${strategy}  |  Reweighting: ${reweighting}"
  echo "  Model:    xgboost"
  echo "  Output:   ${out_dir}"
  echo "============================================================"

  python active_learning.py \
    --warm-start-file "$WARM_START" \
    --full-data-file  "$FULL_DATA" \
    --feh-threshold   "$FEH_THRESHOLD" \
    --strategy        "$strategy" \
    --reweighting     "$reweighting" \
    --soft-topk       "$SOFT_TOPK" \
    --reweight-pool-size "$REWEIGHT_POOL" \
    --reweight-lambda "$REWEIGHT_LAMBDA" \
    --voronoi-l2-max-iter "$VORONOI_L2_MAX_ITER" \
    --voronoi-l2-initial-max-iter "$VORONOI_L2_INITIAL_MAX_ITER" \
    --total-queries   "$TOTAL_QUERIES" \
    --eval-every      "$EVAL_EVERY" \
    --lambda-MP       "$LAMBDA_MP" \
    --wass-pool-size  "$WASS_POOL" \
    --wass-plan-size  "$WASS_PLAN_SIZE" \
    --C               "$C" \
    --eval-size       "$EVAL_SIZE" \
    --seed            "$SEED" \
    --n-trials        "$N_TRIALS" \
    --n-snapshots     "$N_SNAPSHOTS" \
    --model           xgboost \
    --xgb-n-estimators "$XGB_N_ESTIMATORS" \
    --xgb-max-depth "$XGB_MAX_DEPTH" \
    --xgb-learning-rate "$XGB_LEARNING_RATE" \
    --xgb-subsample "$XGB_SUBSAMPLE" \
    --xgb-colsample-bytree "$XGB_COLSAMPLE_BYTREE" \
    --xgb-min-child-weight "$XGB_MIN_CHILD_WEIGHT" \
    --xgb-gamma "$XGB_GAMMA" \
    --xgb-reg-lambda "$XGB_REG_LAMBDA" \
    --xgb-tree-method "$XGB_TREE_METHOD" \
    --xgb-device "$XGB_DEVICE" \
    --xgb-n-jobs "$XGB_N_JOBS" \
    --out-dir         "$out_dir"
done

echo ""
echo "============================================================"
echo "  Generating sampling-method PR-AUC comparison"
echo "============================================================"

python compare_auc_trials.py \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}" \
  "${RESULT_ROOT}/al_xgb_random_l2_${TOTAL_QUERIES}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}" \
  --out "${RESULT_ROOT}/figures/final/xgb_voronoi_l2_sampling_${TOTAL_QUERIES}q_auc.png" \
  --labels "XGB kmedian++ + Voronoi-L2" "XGB random + Voronoi-L2" "XGB Wasserstein-L2 + Voronoi-L2" \
  --cmap-runs none

echo ""
echo "============================================================"
echo "  Generating reweighting ESS comparison"
echo "============================================================"

python compare_weight_l2_trials.py \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}" \
  "${RESULT_ROOT}/al_xgb_random_l2_${TOTAL_QUERIES}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}" \
  --metric effective_sample_size \
  --out "${RESULT_ROOT}/figures/final/xgb_voronoi_l2_sampling_${TOTAL_QUERIES}q_effective_sample_size.png" \
  --labels "XGB kmedian++ + Voronoi-L2" "XGB random + Voronoi-L2" "XGB Wasserstein-L2 + Voronoi-L2"

echo ""
echo "All XGBoost Voronoi-L2 150-query sampling experiments completed."
