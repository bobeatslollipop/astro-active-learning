#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

WARM_START=bp_rp_lamost_normalized_low_teff.h5
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=200
EVAL_EVERY=10
LAMBDA_MP=0.01
WASS_POOL=45000
C=10000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=10
N_SNAPSHOTS=20

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

RESULT_ROOT="xgboost_voronoi_l2_sampling_${TOTAL_QUERIES}q_10seeds"
OUT_DIR="${RESULT_ROOT}/al_xgb_random_none_${TOTAL_QUERIES}"

echo ""
echo "============================================================"
echo "  Strategy: random  |  Reweighting: none"
echo "  Model:    xgboost"
echo "  Output:   ${OUT_DIR}"
echo "============================================================"

python active_learning.py \
  --warm-start-file "$WARM_START" \
  --full-data-file  "$FULL_DATA" \
  --feh-threshold   "$FEH_THRESHOLD" \
  --strategy        random \
  --reweighting     none \
  --total-queries   "$TOTAL_QUERIES" \
  --eval-every      "$EVAL_EVERY" \
  --lambda-MP       "$LAMBDA_MP" \
  --wass-pool-size  "$WASS_POOL" \
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
  --out-dir         "$OUT_DIR"

echo ""
echo "============================================================"
echo "  Generating four-way sampling/reweighting PR-AUC comparison"
echo "============================================================"

python compare_auc_trials.py \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}" \
  "${RESULT_ROOT}/al_xgb_random_l2_${TOTAL_QUERIES}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}" \
  "${RESULT_ROOT}/al_xgb_random_none_${TOTAL_QUERIES}" \
  --out "${RESULT_ROOT}/xgb_voronoi_l2_sampling_${TOTAL_QUERIES}q_with_random_no_reweight_auc.png" \
  --labels "XGB kmedian++ + Voronoi-L2" "XGB random + Voronoi-L2" "XGB Wasserstein-L2 + Voronoi-L2" "XGB random + no reweighting" \
  --cmap-runs none

echo ""
echo "Random no-reweighting baseline and four-way comparison completed."
