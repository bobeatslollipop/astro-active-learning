#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

# Final 0.01 MP-weight baseline:
# query with unregularized Wasserstein greedy sampling, then train XGBoost
# without covariate/Voronoi reweighting.  Keep the historical 1% MP class-weight
# setup and XGBoost hyperparameters used by the Jul23/Jul24 comparison runs.
WARM_START=bp_rp_lamost_normalized_low_teff.h5
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=150
EVAL_EVERY=10
LAMBDA_MP=0.01
REWEIGHT_LAMBDA=0.0
WASS_POOL=45000
WASS_PLAN_SIZE="$EVAL_EVERY"
C=10000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=5
N_SNAPSHOTS=15
SOFT_TOPK=20

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

RESULT_ROOT="0.01MP-weight"
OUT_DIR="${RESULT_ROOT}/al_xgb_wasserstein_none_${TOTAL_QUERIES}_lambda_0_queryonly"

if [[ -e "$OUT_DIR" ]] && [[ -n "$(find "$OUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: $OUT_DIR" >&2
  exit 1
fi

echo ""
echo "============================================================"
echo "  Strategy:       wasserstein"
echo "  Reweighting:    none"
echo "  lambda_MP:      ${LAMBDA_MP}"
echo "  Query lambda:   ${REWEIGHT_LAMBDA} (unregularized Wasserstein)"
echo "  Eval every:     ${EVAL_EVERY}"
echo "  Output:         ${OUT_DIR}"
echo "============================================================"

python active_learning.py \
  --warm-start-file "$WARM_START" \
  --full-data-file  "$FULL_DATA" \
  --feh-threshold   "$FEH_THRESHOLD" \
  --strategy        wasserstein \
  --reweighting     none \
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
echo "  Generating 0.01 MP-weight baseline comparison"
echo "============================================================"

python compare_auc_trials.py \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_10000" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_10000" \
  "${RESULT_ROOT}/al_xgb_wasserstein_hard_${TOTAL_QUERIES}_lambda_0_eval10" \
  "${RESULT_ROOT}/al_xgb_random_none_${TOTAL_QUERIES}" \
  "$OUT_DIR" \
  --metric pr_auc \
  --out "${RESULT_ROOT}/xgb_0p01mp_weight_baselines_pr_auc_trapz.png" \
  --labels \
    "kmedian++ + L2 lambda=1e4" \
    "Wasserstein-L2 greedy lambda=1e4" \
    "Wasserstein hard lambda=0" \
    "random + no reweight" \
    "Wasserstein query lambda=0 + no reweight" \
  --cmap-runs none

echo ""
echo "Wasserstein query-only no-reweight baseline completed."
