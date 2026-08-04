#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

# Matched baselines for the no-class-balancing Wasserstein-L2 sweep.
# These runs intentionally reuse the Wasserstein no-class-balancing settings:
# fixed total train weight 10k, lambda_MP=1, full-heldout eval, full-non-eval
# reweight target, 150 queries, eval every 15, and 5 seeds.
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
VORONOI_L2_MAX_ITER=512
VORONOI_L2_RELATIVE_GAP_TOL=1e-2
VORONOI_L2_GRADIENT_TOL=1e-4
VORONOI_L2_STABILITY_WINDOW=10
VORONOI_L2_DUAL_RELATIVE_TOL=1e-4
VORONOI_L2_WEIGHT_L1_TOL=5e-3
VORONOI_L2_STABILITY_PATIENCE=2

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
WASS_NONE_OUT="${RESULT_ROOT}/al_xgb_wasserstein_none_${TOTAL_QUERIES}_lambda_0_queryonly"
KMED_L2_OUT="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_100"
RANDOM_NONE_OUT="${RESULT_ROOT}/al_xgb_random_none_${TOTAL_QUERIES}_lambda_inf"

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

run_if_missing() {
  local out_dir="$1"
  shift
  if [[ -e "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -print -quit)" ]]; then
    echo "Output already exists; skipping: $out_dir"
    return
  fi
  "$@"
}

echo ""
echo "============================================================"
echo "  Matched baseline 1: pure Wasserstein query + no reweight"
echo "  Output: ${WASS_NONE_OUT}"
echo "============================================================"
run_if_missing "$WASS_NONE_OUT" \
  python active_learning.py \
    "${COMMON_ARGS[@]}" \
    --strategy wasserstein \
    --reweighting none \
    --soft-topk "$SOFT_TOPK" \
    --reweight-lambda 0.0 \
    --out-dir "$WASS_NONE_OUT"

echo ""
echo "============================================================"
echo "  Matched baseline 2: kmedian++ + Voronoi-L2 lambda=100"
echo "  Output: ${KMED_L2_OUT}"
echo "============================================================"
run_if_missing "$KMED_L2_OUT" \
  python active_learning.py \
    "${COMMON_ARGS[@]}" \
    --strategy kmedianpp \
    --reweighting voronoi_l2 \
    --soft-topk "$SOFT_TOPK" \
    --reweight-pool-size "$REWEIGHT_POOL" \
    --reweight-lambda 100 \
    --voronoi-l2-max-iter "$VORONOI_L2_MAX_ITER" \
    --voronoi-l2-relative-gap-tol "$VORONOI_L2_RELATIVE_GAP_TOL" \
    --voronoi-l2-gradient-tol "$VORONOI_L2_GRADIENT_TOL" \
    --voronoi-l2-stability-window "$VORONOI_L2_STABILITY_WINDOW" \
    --voronoi-l2-dual-relative-tol "$VORONOI_L2_DUAL_RELATIVE_TOL" \
    --voronoi-l2-weight-l1-tol "$VORONOI_L2_WEIGHT_L1_TOL" \
    --voronoi-l2-stability-patience "$VORONOI_L2_STABILITY_PATIENCE" \
    --out-dir "$KMED_L2_OUT"

echo ""
echo "============================================================"
echo "  Matched baseline 3: random sampling + no reweight"
echo "  Output: ${RANDOM_NONE_OUT}"
echo "============================================================"
run_if_missing "$RANDOM_NONE_OUT" \
  python active_learning.py \
    "${COMMON_ARGS[@]}" \
    --strategy random \
    --reweighting none \
    --soft-topk "$SOFT_TOPK" \
    --reweight-lambda 1.0 \
    --out-dir "$RANDOM_NONE_OUT"

echo ""
echo "============================================================"
echo "  Generating matched no-class-balancing comparison plots"
echo "============================================================"

python compare_auc_trials.py \
  "${RESULT_ROOT}/al_xgb_wasserstein_hard_${TOTAL_QUERIES}_lambda_0" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_10" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_100" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_1000" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_10000" \
  "$WASS_NONE_OUT" \
  "$KMED_L2_OUT" \
  "$RANDOM_NONE_OUT" \
  --out "${RESULT_ROOT}/figures/final/xgb_noclassbalance_matched_baselines_average_precision.png" \
  --metric average_precision \
  --labels \
    "Wass hard lambda=0" \
    "Wass-L2 lambda=10" \
    "Wass-L2 lambda=100" \
    "Wass-L2 lambda=1000" \
    "Wass-L2 lambda=10000" \
    "Wass query lambda=0 + no reweight" \
    "kmedian++ L2 lambda=100" \
    "random + no reweight" \
  --cmap-runs none

python compare_auc_trials.py \
  "${RESULT_ROOT}/al_xgb_wasserstein_hard_${TOTAL_QUERIES}_lambda_0" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_10" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_100" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_1000" \
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_${TOTAL_QUERIES}_lambda_10000" \
  "$WASS_NONE_OUT" \
  "$KMED_L2_OUT" \
  "$RANDOM_NONE_OUT" \
  --out "${RESULT_ROOT}/figures/final/xgb_noclassbalance_matched_baselines_pr_auc_trapz.png" \
  --metric pr_auc \
  --labels \
    "Wass hard lambda=0" \
    "Wass-L2 lambda=10" \
    "Wass-L2 lambda=100" \
    "Wass-L2 lambda=1000" \
    "Wass-L2 lambda=10000" \
    "Wass query lambda=0 + no reweight" \
    "kmedian++ L2 lambda=100" \
    "random + no reweight" \
  --cmap-runs none

echo ""
echo "Matched no-class-balancing baselines completed."
