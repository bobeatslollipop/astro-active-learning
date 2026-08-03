#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

# Shared data/model setup matching the 200-query XGBoost sampling comparison.
WARM_START=bp_rp_lamost_normalized_low_teff.h5
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=200
EVAL_EVERY=10
LAMBDA_MP=0.01
WASS_POOL=45000
WASS_PLAN_SIZE=10
C=10000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=10
N_SNAPSHOTS=20
SOFT_TOPK=20
REWEIGHT_POOL=100000
REWEIGHT_LAMBDA=1000.0
VORONOI_L2_MAX_ITER=5

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
RESULT_ROOT="${RESULTS_ROOT}/xgboost_voronoi_l2_sampling_${TOTAL_QUERIES}q_10seeds"
mkdir -p "${RESULT_ROOT}/figures/final"
OLD_WASS_DIR="${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}"
NEW_WASS_DIR="${RESULT_ROOT}/al_xgb_wasserstein_l2_replan10_${TOTAL_QUERIES}"

echo ""
echo "============================================================"
echo "  Strategy: wasserstein_l2  |  Reweighting: voronoi_l2"
echo "  Model:    xgboost"
echo "  Plan:     rebuild every ${WASS_PLAN_SIZE} queries"
echo "  Output:   ${NEW_WASS_DIR}"
echo "============================================================"

python active_learning.py \
  --warm-start-file "$WARM_START" \
  --full-data-file  "$FULL_DATA" \
  --feh-threshold   "$FEH_THRESHOLD" \
  --strategy        wasserstein_l2 \
  --reweighting     voronoi_l2 \
  --soft-topk       "$SOFT_TOPK" \
  --reweight-pool-size "$REWEIGHT_POOL" \
  --reweight-lambda "$REWEIGHT_LAMBDA" \
  --voronoi-l2-max-iter "$VORONOI_L2_MAX_ITER" \
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
  --out-dir "$NEW_WASS_DIR"

echo ""
echo "============================================================"
echo "  Generating cached-plan vs replan10 PR-AUC comparison"
echo "============================================================"

python compare_auc_trials.py \
  "$OLD_WASS_DIR" \
  "$NEW_WASS_DIR" \
  --out "${RESULT_ROOT}/figures/final/xgb_wasserstein_l2_cached_vs_replan10_auc.png" \
  --labels "Wasserstein-L2 cached 200-query plan" "Wasserstein-L2 replan every 10 queries" \
  --cmap-runs none

echo ""
echo "XGBoost Wasserstein-L2 replan10 experiment completed."
