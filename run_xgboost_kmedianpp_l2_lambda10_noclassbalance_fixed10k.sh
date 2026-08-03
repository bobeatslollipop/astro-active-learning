#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

WARM_START=bp_rp_lamost_normalized_low_teff.h5
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=150
EVAL_EVERY=10
LAMBDA_MP=1
CLASS_BALANCE_MODE=none
TRAIN_WEIGHT_SUM_MODE=fixed
TRAIN_WEIGHT_SUM=10000.0
EVAL_SOURCE=full_heldout
WASS_POOL=45000
C=10000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=5
N_SNAPSHOTS=15
SOFT_TOPK=20
REWEIGHT_POOL=100000
VORONOI_L2_MAX_ITER=8
VORONOI_L2_INITIAL_MAX_ITER=16

LAMBDA=10

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
RESULT_ROOT="${RESULTS_ROOT}/xgb_kmedianpp_noclassbalance_fixed10k_fullheldout_${TOTAL_QUERIES}q_${N_TRIALS}seeds"
mkdir -p "${RESULT_ROOT}/figures/final"
OUT_DIR="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDA}"
UNIFORM_OUT_DIR="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_inf"

for required in \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_100" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_1000" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_10000" \
  "$UNIFORM_OUT_DIR"; do
  if [[ ! -d "$required" ]]; then
    echo "Missing required existing result directory: $required" >&2
    exit 1
  fi
done

if [[ -e "$OUT_DIR" ]] && [[ -n "$(find "$OUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty lambda=10 directory: $OUT_DIR" >&2
  exit 1
fi

echo ""
echo "============================================================"
echo "  Adding lambda=10 to no-class-balance fixed-10k kmedian++ run"
echo "  Output: ${OUT_DIR}"
echo "============================================================"

python active_learning.py \
  --warm-start-file "$WARM_START" \
  --full-data-file "$FULL_DATA" \
  --feh-threshold "$FEH_THRESHOLD" \
  --strategy kmedianpp \
  --reweighting voronoi_l2 \
  --soft-topk "$SOFT_TOPK" \
  --reweight-pool-size "$REWEIGHT_POOL" \
  --reweight-lambda "$LAMBDA" \
  --voronoi-l2-max-iter "$VORONOI_L2_MAX_ITER" \
  --voronoi-l2-initial-max-iter "$VORONOI_L2_INITIAL_MAX_ITER" \
  --total-queries "$TOTAL_QUERIES" \
  --eval-every "$EVAL_EVERY" \
  --lambda-MP "$LAMBDA_MP" \
  --class-balance-mode "$CLASS_BALANCE_MODE" \
  --train-weight-sum-mode "$TRAIN_WEIGHT_SUM_MODE" \
  --train-weight-sum "$TRAIN_WEIGHT_SUM" \
  --eval-source "$EVAL_SOURCE" \
  --wass-pool-size "$WASS_POOL" \
  --C "$C" \
  --eval-size "$EVAL_SIZE" \
  --seed "$SEED" \
  --n-trials "$N_TRIALS" \
  --n-snapshots "$N_SNAPSHOTS" \
  --model xgboost \
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
echo "  Plotting lambda=1e1, 1e2, 1e3, 1e4, inf AP curves"
echo "============================================================"

python compare_auc_trials.py \
  "$OUT_DIR" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_100" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_1000" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_10000" \
  "$UNIFORM_OUT_DIR" \
  --out "${RESULT_ROOT}/figures/final/xgb_kmedianpp_noclassbalance_fixed10k_ap_lambda1e1_1e2_1e3_1e4_inf.png" \
  --metric average_precision \
  --labels "lambda=1e1" "lambda=1e2" "lambda=1e3" "lambda=1e4" "lambda=inf (uniform)" \
  --cmap-runs viridis

echo ""
echo "Added lambda=10 and saved the five-curve AP plot."
