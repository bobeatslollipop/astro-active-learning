#!/usr/bin/env bash
set -euo pipefail

# Physical GPU 1 only. Within this process it is exposed to CUDA as device 0.
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

WARM_START=bp_rp_lamost_normalized_low_teff_100k_seed42.h5
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=150
EVAL_EVERY=30
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
N_SNAPSHOTS=5
SOFT_TOPK=20
REWEIGHT_POOL=100000
VORONOI_L2_MAX_ITER=128
VORONOI_L2_OBJECTIVE_TOL=1e-4
VORONOI_L2_OBJECTIVE_PATIENCE=2

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
FAMILY="xgb_noclassbalance_100k_warm_fixed10k_fullheldout_reweightfull_${TOTAL_QUERIES}q_${N_TRIALS}seeds_eval${EVAL_EVERY}_v2"
RESULT_ROOT="${RESULTS_ROOT}/${FAMILY}"

WASS_HARD_OUT="${RESULT_ROOT}/al_xgb_wasserstein_hard_${TOTAL_QUERIES}_lambda_0"
WASS_NONE_OUT="${RESULT_ROOT}/al_xgb_wasserstein_none_${TOTAL_QUERIES}_lambda_0_queryonly"
KMED_L2_OUT="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_100"
RANDOM_NONE_OUT="${RESULT_ROOT}/al_xgb_random_none_${TOTAL_QUERIES}_lambda_inf"
LAMBDAS=(10 100 1000 10000)

for required_file in "$WARM_START" "$FULL_DATA"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required data file is missing: $required_file" >&2
    exit 1
  fi
done

mkdir -p "${RESULT_ROOT}/figures/final"

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

L2_ARGS=(
  --soft-topk "$SOFT_TOPK"
  --reweight-pool-size "$REWEIGHT_POOL"
  --voronoi-l2-max-iter "$VORONOI_L2_MAX_ITER"
  --voronoi-l2-objective-tol "$VORONOI_L2_OBJECTIVE_TOL"
  --voronoi-l2-objective-patience "$VORONOI_L2_OBJECTIVE_PATIENCE"
)

is_completed_run() {
  local out_dir="$1"
  [[ -f "${out_dir}/params.json" ]] || return 1
  [[ -f "${out_dir}/results.json" ]] || return 1
  [[ -f "${out_dir}/auc_trials.json" ]] || return 1
  [[ -f "${out_dir}/query_plan_trials.json" ]] || return 1
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("run", {}).get("status") == "completed" else 1)' \
    "${out_dir}/params.json"
}

run_one() {
  local label="$1"
  local out_dir="$2"
  shift 2

  if [[ -d "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -print -quit)" ]]; then
    if is_completed_run "$out_dir"; then
      echo "Completed output exists; skipping: $out_dir"
      return 0
    fi
    echo "Refusing to overwrite incomplete output for ${label}: $out_dir" >&2
    exit 1
  fi

  echo ""
  echo "============================================================"
  echo "  ${label}"
  echo "  Physical GPU: 1"
  echo "  Warm input:   ${WARM_START}"
  echo "  Snapshots:    0,30,60,90,120,150 queries"
  echo "  Output:       ${out_dir}"
  echo "============================================================"
  "$@"
}

for lambda in "${LAMBDAS[@]}"; do
  out_dir="${RESULT_ROOT}/al_xgb_wasserstein_l2_v2_${TOTAL_QUERIES}_lambda_${lambda}"
  run_one "Full-Voronoi Wasserstein-L2 v2 lambda=${lambda}" "$out_dir" \
    python active_learning.py \
      "${COMMON_ARGS[@]}" \
      "${L2_ARGS[@]}" \
      --strategy wasserstein_l2 \
      --reweighting voronoi_l2 \
      --reweight-lambda "$lambda" \
      --out-dir "$out_dir"
done

run_one "Wasserstein hard lambda=0" "$WASS_HARD_OUT" \
  python active_learning.py \
    "${COMMON_ARGS[@]}" \
    --strategy wasserstein \
    --reweighting hard \
    --reweight-lambda 0.0 \
    --out-dir "$WASS_HARD_OUT"

run_one "Pure Wasserstein query plus no reweight" "$WASS_NONE_OUT" \
  python active_learning.py \
    "${COMMON_ARGS[@]}" \
    --strategy wasserstein \
    --reweighting none \
    --soft-topk "$SOFT_TOPK" \
    --reweight-lambda 0.0 \
    --out-dir "$WASS_NONE_OUT"

run_one "kmedian++ plus Voronoi-L2 lambda=100" "$KMED_L2_OUT" \
  python active_learning.py \
    "${COMMON_ARGS[@]}" \
    "${L2_ARGS[@]}" \
    --strategy kmedianpp \
    --reweighting voronoi_l2 \
    --reweight-lambda 100 \
    --out-dir "$KMED_L2_OUT"

run_one "Random query plus no reweight" "$RANDOM_NONE_OUT" \
  python active_learning.py \
    "${COMMON_ARGS[@]}" \
    --strategy random \
    --reweighting none \
    --soft-topk "$SOFT_TOPK" \
    --reweight-lambda 1.0 \
    --out-dir "$RANDOM_NONE_OUT"

WASS_L2_DIRS=(
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_v2_${TOTAL_QUERIES}_lambda_${LAMBDAS[0]}"
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_v2_${TOTAL_QUERIES}_lambda_${LAMBDAS[1]}"
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_v2_${TOTAL_QUERIES}_lambda_${LAMBDAS[2]}"
  "${RESULT_ROOT}/al_xgb_wasserstein_l2_v2_${TOTAL_QUERIES}_lambda_${LAMBDAS[3]}"
)

echo ""
echo "============================================================"
echo "  Generating 100K-warm, eval30 comparison plots"
echo "============================================================"

python compare_auc_trials.py \
  "$WASS_HARD_OUT" "${WASS_L2_DIRS[@]}" \
  --out "${RESULT_ROOT}/figures/final/wasserstein_l2_v2_lambda_sweep_average_precision.png" \
  --metric average_precision \
  --labels "hard lambda=0" "v2 lambda=10" "v2 lambda=100" "v2 lambda=1000" "v2 lambda=10000" \
  --cmap-runs viridis

python compare_auc_trials.py \
  "$WASS_HARD_OUT" "${WASS_L2_DIRS[@]}" \
  --out "${RESULT_ROOT}/figures/final/wasserstein_l2_v2_lambda_sweep_pr_auc_trapz.png" \
  --metric pr_auc \
  --labels "hard lambda=0" "v2 lambda=10" "v2 lambda=100" "v2 lambda=1000" "v2 lambda=10000" \
  --cmap-runs viridis

python compare_weight_l2_trials.py \
  "$WASS_HARD_OUT" "${WASS_L2_DIRS[@]}" \
  --metric objective_l2_norm \
  --out "${RESULT_ROOT}/figures/final/wasserstein_l2_v2_lambda_sweep_weight_l2_norm.png" \
  --labels "hard lambda=0" "v2 lambda=10" "v2 lambda=100" "v2 lambda=1000" "v2 lambda=10000"

python compare_weight_l2_trials.py \
  "$WASS_HARD_OUT" "${WASS_L2_DIRS[@]}" \
  --metric effective_sample_size \
  --out "${RESULT_ROOT}/figures/final/wasserstein_l2_v2_lambda_sweep_effective_sample_size.png" \
  --labels "hard lambda=0" "v2 lambda=10" "v2 lambda=100" "v2 lambda=1000" "v2 lambda=10000"

ALL_RUNS=(
  "$WASS_HARD_OUT"
  "${WASS_L2_DIRS[@]}"
  "$WASS_NONE_OUT"
  "$KMED_L2_OUT"
  "$RANDOM_NONE_OUT"
)
ALL_LABELS=(
  "Wass hard lambda=0"
  "Wass-L2 v2 lambda=10"
  "Wass-L2 v2 lambda=100"
  "Wass-L2 v2 lambda=1000"
  "Wass-L2 v2 lambda=10000"
  "Wass query lambda=0 + no reweight"
  "kmedian++ L2 lambda=100"
  "random + no reweight"
)

python compare_auc_trials.py \
  "${ALL_RUNS[@]}" \
  --out "${RESULT_ROOT}/figures/final/noclassbalance_8run_average_precision.png" \
  --metric average_precision \
  --labels "${ALL_LABELS[@]}" \
  --cmap-runs none

python compare_auc_trials.py \
  "${ALL_RUNS[@]}" \
  --out "${RESULT_ROOT}/figures/final/noclassbalance_8run_pr_auc_trapz.png" \
  --metric pr_auc \
  --labels "${ALL_LABELS[@]}" \
  --cmap-runs none

echo ""
echo "All eight 100K-warm no-class-rebalance runs completed."
