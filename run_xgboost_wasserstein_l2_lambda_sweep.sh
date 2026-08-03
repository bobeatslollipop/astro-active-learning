#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

# Shared data/model setup.
WARM_START=bp_rp_lamost_normalized_low_teff.h5
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=150
EVAL_EVERY=10
LAMBDA_MP=1
TRAIN_WEIGHT_SUM_MODE=initial_labeled
EVAL_SOURCE=full_heldout
WASS_POOL=45000
C=10000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=5
N_SNAPSHOTS=15
SOFT_TOPK=20
REWEIGHT_POOL=100000
WASS_PLAN_SIZE="$EVAL_EVERY"
VORONOI_L2_MAX_ITER=128

LAMBDAS=(100 1000 10000 100000)

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
RESULT_ROOT="${RESULTS_ROOT}/xgb_wasserstein_l2_lambdaMP1_fullheldout_initialweightsum_${TOTAL_QUERIES}q_${N_TRIALS}seeds"
mkdir -p "${RESULT_ROOT}/figures/final"

for lambda in "${LAMBDAS[@]}"; do
  out_dir="${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${lambda}"
  echo ""
  echo "============================================================"
  echo "  Strategy: wasserstein_l2  |  Reweighting: voronoi_l2"
  echo "  Model:    xgboost"
  echo "  Lambda:   ${lambda}"
  echo "  Output:   ${out_dir}"
  echo "============================================================"

  python active_learning.py \
    --warm-start-file "$WARM_START" \
    --full-data-file  "$FULL_DATA" \
    --feh-threshold   "$FEH_THRESHOLD" \
    --strategy        wasserstein_l2 \
    --reweighting     voronoi_l2 \
    --soft-topk       "$SOFT_TOPK" \
    --reweight-pool-size "$REWEIGHT_POOL" \
    --reweight-lambda "$lambda" \
    --voronoi-l2-max-iter "$VORONOI_L2_MAX_ITER" \
    --total-queries   "$TOTAL_QUERIES" \
    --eval-every      "$EVAL_EVERY" \
    --lambda-MP       "$LAMBDA_MP" \
    --train-weight-sum-mode "$TRAIN_WEIGHT_SUM_MODE" \
    --eval-source     "$EVAL_SOURCE" \
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
    --out-dir "$out_dir"
done

echo ""
echo "============================================================"
echo "  Generating Wasserstein-L2 lambda-sweep AP comparison"
echo "============================================================"

python compare_auc_trials.py \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[0]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[1]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[2]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[3]}" \
  --out "${RESULT_ROOT}/figures/final/xgb_wasserstein_l2_lambda_sweep_average_precision.png" \
  --metric average_precision \
  --labels "lambda=100" "lambda=1e3" "lambda=1e4" "lambda=1e5" \
  --cmap-runs viridis

python compare_auc_trials.py \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[0]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[1]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[2]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[3]}" \
  --out "${RESULT_ROOT}/figures/final/xgb_wasserstein_l2_lambda_sweep_pr_auc_trapz.png" \
  --metric pr_auc \
  --labels "lambda=100" "lambda=1e3" "lambda=1e4" "lambda=1e5" \
  --cmap-runs viridis

python compare_weight_l2_trials.py \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[0]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[1]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[2]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${LAMBDAS[3]}" \
  --out "${RESULT_ROOT}/figures/final/xgb_wasserstein_l2_lambda_sweep_effective_sample_size.png" \
  --metric effective_sample_size \
  --labels "lambda=100" "lambda=1e3" "lambda=1e4" "lambda=1e5"

echo ""
echo "All XGBoost Wasserstein-L2 lambda sweep experiments completed."
