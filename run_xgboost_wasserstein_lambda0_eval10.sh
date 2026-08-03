#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

# Unregularized lambda=0 counterpart to the archived Jul24 Wasserstein-L2 sweep.
# The lambda=0 limit is pure Wasserstein greedy querying with hard Voronoi
# reweighting.  Keep lambda_MP=0.01 from the previous Jul24 sweep.
WARM_START=bp_rp_lamost_normalized_low_teff.h5
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=150
EVAL_EVERY=10
LAMBDA_MP=0.01
REWEIGHT_LAMBDA=0.0
WASS_POOL=45000
C=10000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=5
N_SNAPSHOTS=15
SOFT_TOPK=20
WASS_PLAN_SIZE="$EVAL_EVERY"

# XGBoost configuration from the prior Jul24 Wasserstein-L2 sweep.
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
RESULT_ROOT="${RESULTS_ROOT}/Jul24-wass-lambda"
mkdir -p "${RESULT_ROOT}/figures/final"
OUT_DIR="${RESULT_ROOT}/al_xgb_wasserstein_hard_150_lambda_0_eval10"

if [[ -e "$OUT_DIR" ]] && [[ -n "$(find "$OUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: $OUT_DIR" >&2
  exit 1
fi

echo ""
echo "============================================================"
echo "  Strategy:       wasserstein"
echo "  Reweighting:    hard"
echo "  lambda_MP:      ${LAMBDA_MP}"
echo "  Lambda:         ${REWEIGHT_LAMBDA} (unregularized)"
echo "  Eval every:     ${EVAL_EVERY}"
echo "  Output:         ${OUT_DIR}"
echo "============================================================"

python active_learning.py \
  --warm-start-file "$WARM_START" \
  --full-data-file  "$FULL_DATA" \
  --feh-threshold   "$FEH_THRESHOLD" \
  --strategy        wasserstein \
  --reweighting     hard \
  --soft-topk       "$SOFT_TOPK" \
  --reweight-lambda "$REWEIGHT_LAMBDA" \
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
  --out-dir "$OUT_DIR"

echo ""
echo "============================================================"
echo "  Generating lambda=0 vs regularized Wasserstein comparisons"
echo "============================================================"

python compare_auc_trials.py \
  "$OUT_DIR" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_150_lambda_300" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_150_lambda_1000" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_150_lambda_3000" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_150_lambda_10000" \
  --out "${RESULT_ROOT}/figures/final/xgb_wasserstein_lambda0_vs_l2_pr_auc_trapz.png" \
  --metric pr_auc \
  --labels "lambda=0 hard" "lambda=300" "lambda=1e3" "lambda=3e3" "lambda=1e4" \
  --cmap-runs viridis

python compare_weight_l2_trials.py \
  "$OUT_DIR" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_150_lambda_300" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_150_lambda_1000" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_150_lambda_3000" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_150_lambda_10000" \
  --out "${RESULT_ROOT}/figures/final/xgb_wasserstein_lambda0_vs_l2_effective_sample_size.png" \
  --metric effective_sample_size \
  --labels "lambda=0 hard" "lambda=300" "lambda=1e3" "lambda=3e3" "lambda=1e4"

echo ""
echo "All lambda=0 Wasserstein eval10 experiments completed."
