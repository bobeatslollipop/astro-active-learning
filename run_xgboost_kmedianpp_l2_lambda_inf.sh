#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

# Same setup as run_xgboost_kmedianpp_l2_lambda_sweep.sh, but with uniform
# covariate weights: lambda = infinity corresponds to --reweighting none.
WARM_START=bp_rp_lamost_normalized_low_teff.h5
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=150
EVAL_EVERY=10
LAMBDA_MP=0.01
WASS_POOL=45000
C=10000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=5
N_SNAPSHOTS=15

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
RESULT_ROOT="${RESULTS_ROOT}/xgboost_kmedianpp_l2_lambda_sweep_${TOTAL_QUERIES}q_5seeds"
mkdir -p "${RESULT_ROOT}/figures/final"
OUT_DIR="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_inf"

echo ""
echo "============================================================"
echo "  Strategy: kmedianpp  |  Reweighting: none"
echo "  Model:    xgboost"
echo "  Lambda:   infinity (uniform covariate weights)"
echo "  Output:   ${OUT_DIR}"
echo "============================================================"

python active_learning.py \
  --warm-start-file "$WARM_START" \
  --full-data-file  "$FULL_DATA" \
  --feh-threshold   "$FEH_THRESHOLD" \
  --strategy        kmedianpp \
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
  --out-dir "$OUT_DIR"

echo ""
echo "============================================================"
echo "  Generating finite-lambda + uniform PR-AUC comparison"
echo "============================================================"

python compare_auc_trials.py \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_100" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_1000" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_10000" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_100000" \
  "$OUT_DIR" \
  --out "${RESULT_ROOT}/figures/final/xgb_kmedianpp_l2_lambda_sweep_auc_with_uniform.png" \
  --labels "lambda=100" "lambda=1e3" "lambda=1e4" "lambda=1e5" "lambda=inf (uniform)" \
  --cmap-runs viridis

echo ""
echo "============================================================"
echo "  Generating finite-lambda + uniform weight-concentration comparisons"
echo "============================================================"

python compare_weight_l2_trials.py \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_100" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_1000" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_10000" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_100000" \
  "$OUT_DIR" \
  --metric objective_l2_norm \
  --out "${RESULT_ROOT}/figures/final/xgb_kmedianpp_l2_lambda_sweep_weight_l2_norm_with_uniform.png" \
  --labels "lambda=100" "lambda=1e3" "lambda=1e4" "lambda=1e5" "lambda=inf (uniform)"

python compare_weight_l2_trials.py \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_100" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_1000" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_10000" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_100000" \
  "$OUT_DIR" \
  --metric effective_sample_size \
  --out "${RESULT_ROOT}/figures/final/xgb_kmedianpp_l2_lambda_sweep_effective_sample_size_with_uniform.png" \
  --labels "lambda=100" "lambda=1e3" "lambda=1e4" "lambda=1e5" "lambda=inf (uniform)"

echo ""
echo "XGBoost kmedian++ lambda=infinity uniform baseline completed."
