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
LAMBDA_MP=0.01
WASS_POOL=45000
C=10000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=10
N_SNAPSHOTS=15
SOFT_TOPK=20
SOFTMAX_POOL=100000
VORONOI_L2_MAX_ITER=5

# Only run missing low-lambda points; existing 1000/3000/10000 outputs are reused.
LAMBDAS=(100 300)
COMPARE_LAMBDAS=(100 300 1000 3000 10000)

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

RESULT_ROOT="xgboost_wasserstein_l2_lambda_sweep_${TOTAL_QUERIES}q_10seeds"

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
    --softmax-pool-size "$SOFTMAX_POOL" \
    --reweight-lambda "$lambda" \
    --voronoi-l2-max-iter "$VORONOI_L2_MAX_ITER" \
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
    --out-dir "$out_dir"
done

echo ""
echo "============================================================"
echo "  Generating Wasserstein-L2 lambda-sweep PR-AUC comparison"
echo "============================================================"

python compare_auc_trials.py \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${COMPARE_LAMBDAS[0]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${COMPARE_LAMBDAS[1]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${COMPARE_LAMBDAS[2]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${COMPARE_LAMBDAS[3]}" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_${COMPARE_LAMBDAS[4]}" \
  --out "${RESULT_ROOT}/xgb_wasserstein_l2_lambda_sweep_auc.png" \
  --labels "lambda=100" "lambda=300" "lambda=1e3" "lambda=3e3" "lambda=1e4" \
  --cmap-runs viridis

echo ""
echo "All XGBoost Wasserstein-L2 lambda sweep experiments completed."
