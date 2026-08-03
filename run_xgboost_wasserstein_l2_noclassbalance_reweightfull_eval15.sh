#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

WARM_START=bp_rp_lamost_normalized_low_teff.h5
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=150
EVAL_EVERY=15
LAMBDA_MP=1
CLASS_BALANCE_MODE=none
TRAIN_WEIGHT_SUM_MODE=fixed
TRAIN_WEIGHT_SUM=10000.0
EVAL_SOURCE=full_heldout
REWEIGHT_SOURCE=full_non_eval
WASS_POOL=45000
WASS_PLAN_SIZE="$EVAL_EVERY"
C=10000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=5
N_SNAPSHOTS=10
SOFT_TOPK=20
REWEIGHT_POOL=100000
VORONOI_L2_MAX_ITER=8
VORONOI_L2_INITIAL_MAX_ITER=16

LAMBDAS=(10 100 1000 10000)

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
RESULT_ROOT="${RESULTS_ROOT}/xgb_wasserstein_l2_noclassbalance_fixed10k_fullheldout_reweightfull_${TOTAL_QUERIES}q_${N_TRIALS}seeds_eval${EVAL_EVERY}"
mkdir -p "${RESULT_ROOT}/figures/final"
HARD_OUT_DIR="${RESULT_ROOT}/al_xgb_wasserstein_hard_${TOTAL_QUERIES}_lambda_0"

if [[ -e "$RESULT_ROOT" ]]; then
  echo "Refusing to overwrite existing result root: $RESULT_ROOT" >&2
  exit 1
fi

COMMON_ARGS=(
  --warm-start-file "$WARM_START"
  --full-data-file "$FULL_DATA"
  --feh-threshold "$FEH_THRESHOLD"
  --total-queries "$TOTAL_QUERIES"
  --eval-every "$EVAL_EVERY"
  --lambda-MP "$LAMBDA_MP"
  --class-balance-mode "$CLASS_BALANCE_MODE"
  --train-weight-sum-mode "$TRAIN_WEIGHT_SUM_MODE"
  --train-weight-sum "$TRAIN_WEIGHT_SUM"
  --eval-source "$EVAL_SOURCE"
  --reweight-source "$REWEIGHT_SOURCE"
  --include-zero-snapshot
  --wass-pool-size "$WASS_POOL"
  --wass-plan-size "$WASS_PLAN_SIZE"
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

echo ""
echo "============================================================"
echo "  Strategy:       wasserstein"
echo "  Reweighting:    hard Voronoi"
echo "  Lambda:         0"
echo "  Reweight src:   ${REWEIGHT_SOURCE}"
echo "  Eval every:     ${EVAL_EVERY} queries, including 0-query snapshot"
echo "  Output:         ${HARD_OUT_DIR}"
echo "============================================================"

python active_learning.py \
  "${COMMON_ARGS[@]}" \
  --strategy wasserstein \
  --reweighting hard \
  --reweight-lambda 0.0 \
  --out-dir "$HARD_OUT_DIR"

for lambda in "${LAMBDAS[@]}"; do
  out_dir="${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${lambda}"
  echo ""
  echo "============================================================"
  echo "  Strategy:       wasserstein_l2"
  echo "  Reweighting:    voronoi_l2"
  echo "  Lambda:         ${lambda}"
  echo "  Reweight src:   ${REWEIGHT_SOURCE}"
  echo "  Eval every:     ${EVAL_EVERY} queries, including 0-query snapshot"
  echo "  Output:         ${out_dir}"
  echo "============================================================"

  python active_learning.py \
    "${COMMON_ARGS[@]}" \
    --strategy wasserstein_l2 \
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
echo "  Generating Wasserstein lambda sweep plots"
echo "============================================================"

python compare_auc_trials.py \
  "$HARD_OUT_DIR" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[0]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[1]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[2]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[3]}" \
  --out "${RESULT_ROOT}/figures/final/xgb_wasserstein_l2_noclassbalance_reweightfull_average_precision.png" \
  --metric average_precision \
  --labels "lambda=0 hard" "lambda=1e1" "lambda=1e2" "lambda=1e3" "lambda=1e4" \
  --cmap-runs viridis

python compare_auc_trials.py \
  "$HARD_OUT_DIR" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[0]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[1]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[2]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[3]}" \
  --out "${RESULT_ROOT}/figures/final/xgb_wasserstein_l2_noclassbalance_reweightfull_pr_auc_trapz.png" \
  --metric pr_auc \
  --labels "lambda=0 hard" "lambda=1e1" "lambda=1e2" "lambda=1e3" "lambda=1e4" \
  --cmap-runs viridis

python compare_weight_l2_trials.py \
  "$HARD_OUT_DIR" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[0]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[1]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[2]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[3]}" \
  --metric objective_l2_norm \
  --out "${RESULT_ROOT}/figures/final/xgb_wasserstein_l2_noclassbalance_reweightfull_weight_l2_norm.png" \
  --labels "lambda=0 hard" "lambda=1e1" "lambda=1e2" "lambda=1e3" "lambda=1e4"

python compare_weight_l2_trials.py \
  "$HARD_OUT_DIR" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[0]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[1]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[2]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[3]}" \
  --metric effective_sample_size \
  --out "${RESULT_ROOT}/figures/final/xgb_wasserstein_l2_noclassbalance_reweightfull_effective_sample_size.png" \
  --labels "lambda=0 hard" "lambda=1e1" "lambda=1e2" "lambda=1e3" "lambda=1e4"

echo ""
echo "Wasserstein lambda sweep with full-non-eval reweight target completed."
