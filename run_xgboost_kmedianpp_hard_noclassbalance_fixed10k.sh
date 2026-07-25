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

RESULT_ROOT="xgb_kmedianpp_noclassbalance_fixed10k_fullheldout_${TOTAL_QUERIES}q_${N_TRIALS}seeds"
HARD_OUT_DIR="${RESULT_ROOT}/al_xgb_kmedianpp_hard_${TOTAL_QUERIES}_lambda_0"

DIR_L100="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_100"
DIR_L1K="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_1000"
DIR_L10K="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_10000"
DIR_L100K="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_100000"
DIR_L1M="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_1000000"
DIR_INF="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_inf"

for required in "$DIR_L100" "$DIR_L1K" "$DIR_L10K" "$DIR_L100K" "$DIR_L1M" "$DIR_INF"; do
  if [[ ! -d "$required" ]]; then
    echo "Missing required existing result directory: $required" >&2
    exit 1
  fi
done

if [[ -e "$HARD_OUT_DIR" ]] && [[ -n "$(find "$HARD_OUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: $HARD_OUT_DIR" >&2
  exit 1
fi

echo ""
echo "============================================================"
echo "  Adding hard Voronoi no-class-balance fixed-10k kmedian++ run"
echo "  Strategy:       kmedianpp with dedicated query RNG"
echo "  Reweighting:    hard Voronoi"
echo "  Class balance:  none"
echo "  Weight sum:     fixed ${TRAIN_WEIGHT_SUM}"
echo "  Output:         ${HARD_OUT_DIR}"
echo "============================================================"

python active_learning.py \
  --warm-start-file "$WARM_START" \
  --full-data-file "$FULL_DATA" \
  --feh-threshold "$FEH_THRESHOLD" \
  --strategy kmedianpp \
  --reweighting hard \
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
  --out-dir "$HARD_OUT_DIR"

echo ""
echo "============================================================"
echo "  Plotting no-class-balance curves with hard Voronoi"
echo "============================================================"

python compare_auc_trials.py \
  "$HARD_OUT_DIR" "$DIR_L100" "$DIR_L1K" "$DIR_L10K" "$DIR_L100K" "$DIR_L1M" "$DIR_INF" \
  --out "${RESULT_ROOT}/xgb_kmedianpp_noclassbalance_fixed10k_average_precision_with_hard.png" \
  --metric average_precision \
  --labels "lambda=0 hard" "lambda=1e2" "lambda=1e3" "lambda=1e4" "lambda=1e5" "lambda=1e6" "lambda=inf (uniform)" \
  --cmap-runs viridis

python compare_auc_trials.py \
  "$HARD_OUT_DIR" "$DIR_L100" "$DIR_L1K" "$DIR_L10K" "$DIR_L100K" "$DIR_L1M" "$DIR_INF" \
  --out "${RESULT_ROOT}/xgb_kmedianpp_noclassbalance_fixed10k_pr_auc_trapz_with_hard.png" \
  --metric pr_auc \
  --labels "lambda=0 hard" "lambda=1e2" "lambda=1e3" "lambda=1e4" "lambda=1e5" "lambda=1e6" "lambda=inf (uniform)" \
  --cmap-runs viridis

python compare_weight_l2_trials.py \
  "$HARD_OUT_DIR" "$DIR_L100" "$DIR_L1K" "$DIR_L10K" "$DIR_L100K" "$DIR_L1M" "$DIR_INF" \
  --metric objective_l2_norm \
  --out "${RESULT_ROOT}/xgb_kmedianpp_noclassbalance_fixed10k_weight_l2_norm_with_hard.png" \
  --labels "lambda=0 hard" "lambda=1e2" "lambda=1e3" "lambda=1e4" "lambda=1e5" "lambda=1e6" "lambda=inf (uniform)"

python compare_weight_l2_trials.py \
  "$HARD_OUT_DIR" "$DIR_L100" "$DIR_L1K" "$DIR_L10K" "$DIR_L100K" "$DIR_L1M" "$DIR_INF" \
  --metric effective_sample_size \
  --out "${RESULT_ROOT}/xgb_kmedianpp_noclassbalance_fixed10k_effective_sample_size_with_hard.png" \
  --labels "lambda=0 hard" "lambda=1e2" "lambda=1e3" "lambda=1e4" "lambda=1e5" "lambda=1e6" "lambda=inf (uniform)"

echo ""
echo "Hard Voronoi no-class-balance kmedian++ run completed."
