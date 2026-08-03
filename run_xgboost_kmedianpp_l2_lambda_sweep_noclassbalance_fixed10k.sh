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

LAMBDAS=(100 1000 10000 100000 1000000)

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
UNIFORM_OUT_DIR="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_inf"

if [[ -e "$RESULT_ROOT" ]]; then
  echo "Refusing to overwrite existing result root: $RESULT_ROOT" >&2
  exit 1
fi

COMMON_ARGS=(
  --warm-start-file "$WARM_START"
  --full-data-file "$FULL_DATA"
  --feh-threshold "$FEH_THRESHOLD"
  --strategy kmedianpp
  --total-queries "$TOTAL_QUERIES"
  --eval-every "$EVAL_EVERY"
  --lambda-MP "$LAMBDA_MP"
  --class-balance-mode "$CLASS_BALANCE_MODE"
  --train-weight-sum-mode "$TRAIN_WEIGHT_SUM_MODE"
  --train-weight-sum "$TRAIN_WEIGHT_SUM"
  --eval-source "$EVAL_SOURCE"
  --wass-pool-size "$WASS_POOL"
  --C "$C"
  --eval-size "$EVAL_SIZE"
  --seed "$SEED"
  --n-trials "$N_TRIALS"
  --n-snapshots "$N_SNAPSHOTS"
  --model xgboost
  --xgb-n-estimators "$XGB_N_ESTIMATORS"
  --xgb-max-depth "$XGB_MAX_DEPTH"
  --xgb-learning-rate "$XGB_LEARNING_RATE"
  --xgb-subsample "$XGB_SUBSAMPLE"
  --xgb-colsample-bytree "$XGB_COLSAMPLE_BYTREE"
  --xgb-min-child-weight "$XGB_MIN_CHILD_WEIGHT"
  --xgb-gamma "$XGB_GAMMA"
  --xgb-reg-lambda "$XGB_REG_LAMBDA"
  --xgb-tree-method "$XGB_TREE_METHOD"
  --xgb-device "$XGB_DEVICE"
  --xgb-n-jobs "$XGB_N_JOBS"
)

for lambda in "${LAMBDAS[@]}"; do
  out_dir="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${lambda}"
  echo ""
  echo "============================================================"
  echo "  Strategy:       kmedianpp with dedicated query RNG"
  echo "  Reweighting:    voronoi_l2"
  echo "  Class balance:  none"
  echo "  Weight sum:     fixed ${TRAIN_WEIGHT_SUM}"
  echo "  Lambda:         ${lambda}"
  echo "  Output:         ${out_dir}"
  echo "============================================================"

  python active_learning.py \
    "${COMMON_ARGS[@]}" \
    --reweighting voronoi_l2 \
    --soft-topk "$SOFT_TOPK" \
    --reweight-pool-size "$REWEIGHT_POOL" \
    --reweight-lambda "$lambda" \
    --voronoi-l2-max-iter "$VORONOI_L2_MAX_ITER" \
    --voronoi-l2-initial-max-iter "$VORONOI_L2_INITIAL_MAX_ITER" \
    --out-dir "$out_dir"
done

echo ""
echo "============================================================"
echo "  Strategy:       kmedianpp with dedicated query RNG"
echo "  Reweighting:    none"
echo "  Class balance:  none"
echo "  Weight sum:     fixed ${TRAIN_WEIGHT_SUM}"
echo "  Lambda:         infinity (uniform covariate weights)"
echo "  Output:         ${UNIFORM_OUT_DIR}"
echo "============================================================"

python active_learning.py \
  "${COMMON_ARGS[@]}" \
  --reweighting none \
  --out-dir "$UNIFORM_OUT_DIR"

echo ""
echo "============================================================"
echo "  Generating no-class-balance AP and PR-AUC comparisons"
echo "============================================================"

python compare_auc_trials.py \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[0]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[1]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[2]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[3]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[4]}" \
  "$UNIFORM_OUT_DIR" \
  --out "${RESULT_ROOT}/figures/final/xgb_kmedianpp_noclassbalance_fixed10k_average_precision.png" \
  --metric average_precision \
  --labels "lambda=1e2" "lambda=1e3" "lambda=1e4" "lambda=1e5" "lambda=1e6" "lambda=inf (uniform)" \
  --cmap-runs viridis

python compare_auc_trials.py \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[0]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[1]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[2]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[3]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[4]}" \
  "$UNIFORM_OUT_DIR" \
  --out "${RESULT_ROOT}/figures/final/xgb_kmedianpp_noclassbalance_fixed10k_pr_auc_trapz.png" \
  --metric pr_auc \
  --labels "lambda=1e2" "lambda=1e3" "lambda=1e4" "lambda=1e5" "lambda=1e6" "lambda=inf (uniform)" \
  --cmap-runs viridis

echo ""
echo "============================================================"
echo "  Generating no-class-balance weight-concentration comparisons"
echo "============================================================"

python compare_weight_l2_trials.py \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[0]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[1]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[2]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[3]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[4]}" \
  "$UNIFORM_OUT_DIR" \
  --metric objective_l2_norm \
  --out "${RESULT_ROOT}/figures/final/xgb_kmedianpp_noclassbalance_fixed10k_weight_l2_norm.png" \
  --labels "lambda=1e2" "lambda=1e3" "lambda=1e4" "lambda=1e5" "lambda=1e6" "lambda=inf (uniform)"

python compare_weight_l2_trials.py \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[0]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[1]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[2]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[3]}" \
  "${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[4]}" \
  "$UNIFORM_OUT_DIR" \
  --metric effective_sample_size \
  --out "${RESULT_ROOT}/figures/final/xgb_kmedianpp_noclassbalance_fixed10k_effective_sample_size.png" \
  --labels "lambda=1e2" "lambda=1e3" "lambda=1e4" "lambda=1e5" "lambda=1e6" "lambda=inf (uniform)"

echo ""
echo "All no-class-balance kmedian++ Voronoi-L2 lambda sweep experiments completed."
