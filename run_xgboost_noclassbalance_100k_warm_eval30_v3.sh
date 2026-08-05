#!/usr/bin/env bash
set -euo pipefail

# The selected physical GPU is exposed to this process as CUDA device 0.
PHYSICAL_GPU="${PHYSICAL_GPU:-1}"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

WARM_START="${WARM_START:-bp_rp_lamost_normalized_low_teff_100k_seed42.h5}"
FULL_DATA=bp_rp_lamost_normalized.h5
FEH_THRESHOLD=-2.0
TOTAL_QUERIES=150
EVAL_EVERY=30
LAMBDA_MP="${LAMBDA_MP:-1}"
CLASS_BALANCE_MODE="${CLASS_BALANCE_MODE:-none}"
TRAIN_WEIGHT_SUM_MODE=fixed
TRAIN_WEIGHT_SUM=10000.0
EVAL_SOURCE=full_heldout
REWEIGHT_SOURCE=full_non_eval
WASS_POOL="${WASS_POOL:-45000}"
WASS_PLAN_SIZE="$EVAL_EVERY"
C=10000.0
EVAL_SIZE=500000
SEED=42
N_TRIALS=5
N_SNAPSHOTS=5
SOFT_TOPK=20
REWEIGHT_POOL="${REWEIGHT_POOL:-100000}"
VORONOI_L2_MAX_ITER=1024
VORONOI_L2_RELATIVE_GAP_TOL=1e-2
VORONOI_L2_GRADIENT_TOL=1e-4
VORONOI_L2_STABILITY_WINDOW=10
VORONOI_L2_DUAL_RELATIVE_TOL=1e-4
VORONOI_L2_WEIGHT_L1_TOL=5e-3
VORONOI_L2_STABILITY_PATIENCE=2
WASSERSTEIN_L2_COORDINATE_STEPS=32
WASSERSTEIN_L2_CORRECTIVE_MAX_SWEEPS=128
WASSERSTEIN_L2_CORRECTIVE_DUAL_RELATIVE_TOL=1e-8
WASSERSTEIN_L2_CORRECTIVE_Z_RELATIVE_TOL=1e-6
WASSERSTEIN_L2_CORRECTIVE_PATIENCE=2

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
FAMILY="${EXPERIMENT_FAMILY:-xgb_noclassbalance_100k_warm_fixed10k_fullheldout_reweightfull_${TOTAL_QUERIES}q_${N_TRIALS}seeds_eval${EVAL_EVERY}_v3}"
RESULT_ROOT="${RESULTS_ROOT}/${FAMILY}"
RUN_LAMBDA_10000="${RUN_LAMBDA_10000:-1}"
RUN_WASSERSTEIN_L2="${RUN_WASSERSTEIN_L2:-1}"
RUN_KMEDIAN="${RUN_KMEDIAN:-1}"
KMEDIAN_LAMBDAS_STRING="${KMEDIAN_LAMBDAS:-100}"
RUN_LABEL="${RUN_LABEL:-100K-warm, eval30}"
PLOT_PREFIX="${PLOT_PREFIX:-noclassbalance}"

WASS_HARD_OUT="${RESULT_ROOT}/al_xgb_wasserstein_hard_${TOTAL_QUERIES}_lambda_0"
WASS_NONE_OUT="${RESULT_ROOT}/al_xgb_wasserstein_none_${TOTAL_QUERIES}_lambda_0_queryonly"
RANDOM_NONE_OUT="${RESULT_ROOT}/al_xgb_random_none_${TOTAL_QUERIES}_lambda_inf"
LAMBDAS=(10 100 1000)
EARLY_LAMBDAS=(10 100)
LATE_LAMBDAS=(1000)
if [[ "$RUN_LAMBDA_10000" == "1" ]]; then
  LAMBDAS+=(10000)
  LATE_LAMBDAS+=(10000)
elif [[ "$RUN_LAMBDA_10000" != "0" ]]; then
  echo "RUN_LAMBDA_10000 must be 0 or 1; got: $RUN_LAMBDA_10000" >&2
  exit 1
fi
if [[ "$RUN_KMEDIAN" != "0" && "$RUN_KMEDIAN" != "1" ]]; then
  echo "RUN_KMEDIAN must be 0 or 1; got: $RUN_KMEDIAN" >&2
  exit 1
fi
if [[ "$RUN_WASSERSTEIN_L2" != "0" && "$RUN_WASSERSTEIN_L2" != "1" ]]; then
  echo "RUN_WASSERSTEIN_L2 must be 0 or 1; got: $RUN_WASSERSTEIN_L2" >&2
  exit 1
fi
read -r -a KMEDIAN_LAMBDAS_ARRAY <<< "$KMEDIAN_LAMBDAS_STRING"
if [[ "$RUN_KMEDIAN" == "1" && "${#KMEDIAN_LAMBDAS_ARRAY[@]}" -eq 0 ]]; then
  echo "KMEDIAN_LAMBDAS must contain at least one lambda when RUN_KMEDIAN=1" >&2
  exit 1
fi

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
  --reweight-pool-size "$REWEIGHT_POOL"
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
  --voronoi-l2-max-iter "$VORONOI_L2_MAX_ITER"
  --voronoi-l2-relative-gap-tol "$VORONOI_L2_RELATIVE_GAP_TOL"
  --voronoi-l2-gradient-tol "$VORONOI_L2_GRADIENT_TOL"
  --voronoi-l2-stability-window "$VORONOI_L2_STABILITY_WINDOW"
  --voronoi-l2-dual-relative-tol "$VORONOI_L2_DUAL_RELATIVE_TOL"
  --voronoi-l2-weight-l1-tol "$VORONOI_L2_WEIGHT_L1_TOL"
  --voronoi-l2-stability-patience "$VORONOI_L2_STABILITY_PATIENCE"
  --wasserstein-l2-coordinate-steps "$WASSERSTEIN_L2_COORDINATE_STEPS"
  --wasserstein-l2-corrective-max-sweeps "$WASSERSTEIN_L2_CORRECTIVE_MAX_SWEEPS"
  --wasserstein-l2-corrective-dual-relative-tol "$WASSERSTEIN_L2_CORRECTIVE_DUAL_RELATIVE_TOL"
  --wasserstein-l2-corrective-z-relative-tol "$WASSERSTEIN_L2_CORRECTIVE_Z_RELATIVE_TOL"
  --wasserstein-l2-corrective-patience "$WASSERSTEIN_L2_CORRECTIVE_PATIENCE"
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
  echo "  Physical GPU: ${PHYSICAL_GPU}"
  echo "  Class balance: ${CLASS_BALANCE_MODE} (lambda_MP=${LAMBDA_MP})"
  echo "  Warm input:   ${WARM_START}"
  echo "  Snapshots:    0,30,60,90,120,150 queries"
  echo "  Output:       ${out_dir}"
  echo "============================================================"
  "$@"
}

run_wasserstein_l2() {
  local lambda="$1"
  local out_dir="${RESULT_ROOT}/al_xgb_wasserstein_l2_v3_${TOTAL_QUERIES}_lambda_${lambda}"
  run_one "Power-cell Wasserstein-L2 v3 lambda=${lambda}" "$out_dir" \
    python active_learning.py \
      "${COMMON_ARGS[@]}" \
      "${L2_ARGS[@]}" \
      --strategy wasserstein_l2 \
      --reweighting voronoi_l2 \
      --reweight-lambda "$lambda" \
      --out-dir "$out_dir"
}

run_kmedian_l2() {
  local lambda="$1"
  local out_dir="${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${lambda}"
  run_one "kmedian++ plus Voronoi-L2 lambda=${lambda}" "$out_dir" \
    python active_learning.py \
      "${COMMON_ARGS[@]}" \
      "${L2_ARGS[@]}" \
      --strategy kmedianpp \
      --reweighting voronoi_l2 \
      --reweight-lambda "$lambda" \
      --out-dir "$out_dir"
}

# Put kmedian++ first when requested so the non-Wasserstein regularized
# baseline completes before the slower high-lambda Wasserstein solves.
if [[ "$RUN_KMEDIAN" == "1" ]]; then
  for lambda in "${KMEDIAN_LAMBDAS_ARRAY[@]}"; do
    run_kmedian_l2 "$lambda"
  done
fi

run_one "Wasserstein hard lambda=0" "$WASS_HARD_OUT" \
  python active_learning.py \
    "${COMMON_ARGS[@]}" \
    --strategy wasserstein \
    --reweighting hard \
    --reweight-lambda 0.0 \
    --out-dir "$WASS_HARD_OUT"

if [[ "$RUN_WASSERSTEIN_L2" == "1" ]]; then
  for lambda in "${EARLY_LAMBDAS[@]}"; do
    run_wasserstein_l2 "$lambda"
  done
fi

run_one "Pure Wasserstein query plus no reweight" "$WASS_NONE_OUT" \
  python active_learning.py \
    "${COMMON_ARGS[@]}" \
    --strategy wasserstein \
    --reweighting none \
    --soft-topk "$SOFT_TOPK" \
    --reweight-lambda 0.0 \
    --out-dir "$WASS_NONE_OUT"

run_one "Random query plus no reweight" "$RANDOM_NONE_OUT" \
  python active_learning.py \
    "${COMMON_ARGS[@]}" \
    --strategy random \
    --reweighting none \
    --soft-topk "$SOFT_TOPK" \
    --reweight-lambda 1.0 \
    --out-dir "$RANDOM_NONE_OUT"

# The highest-lambda Wasserstein solves deliberately run last.
if [[ "$RUN_WASSERSTEIN_L2" == "1" ]]; then
  for lambda in "${LATE_LAMBDAS[@]}"; do
    run_wasserstein_l2 "$lambda"
  done
fi

WASS_L2_DIRS=()
WASS_SWEEP_LABELS=("hard lambda=0")
if [[ "$RUN_WASSERSTEIN_L2" == "1" ]]; then
  for lambda in "${LAMBDAS[@]}"; do
    WASS_L2_DIRS+=("${RESULT_ROOT}/al_xgb_wasserstein_l2_v3_${TOTAL_QUERIES}_lambda_${lambda}")
    WASS_SWEEP_LABELS+=("v3 lambda=${lambda}")
  done
fi

KMED_L2_DIRS=()
KMED_SWEEP_LABELS=()
if [[ "$RUN_KMEDIAN" == "1" ]]; then
  for lambda in "${KMEDIAN_LAMBDAS_ARRAY[@]}"; do
    KMED_L2_DIRS+=("${RESULT_ROOT}/al_xgb_kmedianpp_l2_${TOTAL_QUERIES}_lambda_${lambda}")
    KMED_SWEEP_LABELS+=("kmedian++ L2 lambda=${lambda}")
  done
fi

echo ""
echo "============================================================"
echo "  Generating ${RUN_LABEL} comparison plots"
echo "============================================================"

if [[ "$RUN_WASSERSTEIN_L2" == "1" ]]; then
  python compare_auc_trials.py \
    "$WASS_HARD_OUT" "${WASS_L2_DIRS[@]}" \
    --out "${RESULT_ROOT}/figures/final/wasserstein_l2_v3_lambda_sweep_average_precision.png" \
    --metric average_precision \
    --labels "${WASS_SWEEP_LABELS[@]}" \
    --cmap-runs viridis

  python compare_auc_trials.py \
    "$WASS_HARD_OUT" "${WASS_L2_DIRS[@]}" \
    --out "${RESULT_ROOT}/figures/final/wasserstein_l2_v3_lambda_sweep_pr_auc_trapz.png" \
    --metric pr_auc \
    --labels "${WASS_SWEEP_LABELS[@]}" \
    --cmap-runs viridis

  python compare_weight_l2_trials.py \
    "$WASS_HARD_OUT" "${WASS_L2_DIRS[@]}" \
    --metric objective_l2_norm \
    --out "${RESULT_ROOT}/figures/final/wasserstein_l2_v3_lambda_sweep_weight_l2_norm.png" \
    --labels "${WASS_SWEEP_LABELS[@]}"

  python compare_weight_l2_trials.py \
    "$WASS_HARD_OUT" "${WASS_L2_DIRS[@]}" \
    --metric effective_sample_size \
    --out "${RESULT_ROOT}/figures/final/wasserstein_l2_v3_lambda_sweep_effective_sample_size.png" \
    --labels "${WASS_SWEEP_LABELS[@]}"
fi

if [[ "$RUN_KMEDIAN" == "1" ]]; then
  python compare_auc_trials.py \
    "${KMED_L2_DIRS[@]}" \
    --out "${RESULT_ROOT}/figures/final/kmedianpp_l2_lambda_sweep_average_precision.png" \
    --metric average_precision \
    --labels "${KMED_SWEEP_LABELS[@]}" \
    --cmap-runs viridis

  python compare_auc_trials.py \
    "${KMED_L2_DIRS[@]}" \
    --out "${RESULT_ROOT}/figures/final/kmedianpp_l2_lambda_sweep_pr_auc_trapz.png" \
    --metric pr_auc \
    --labels "${KMED_SWEEP_LABELS[@]}" \
    --cmap-runs viridis

  python compare_weight_l2_trials.py \
    "${KMED_L2_DIRS[@]}" \
    --metric objective_l2_norm \
    --out "${RESULT_ROOT}/figures/final/kmedianpp_l2_lambda_sweep_weight_l2_norm.png" \
    --labels "${KMED_SWEEP_LABELS[@]}"

  python compare_weight_l2_trials.py \
    "${KMED_L2_DIRS[@]}" \
    --metric effective_sample_size \
    --out "${RESULT_ROOT}/figures/final/kmedianpp_l2_lambda_sweep_effective_sample_size.png" \
    --labels "${KMED_SWEEP_LABELS[@]}"
fi

ALL_RUNS=(
  "$WASS_HARD_OUT"
)
ALL_LABELS=("Wass hard lambda=0")
if [[ "$RUN_WASSERSTEIN_L2" == "1" ]]; then
  ALL_RUNS+=("${WASS_L2_DIRS[@]}")
  for lambda in "${LAMBDAS[@]}"; do
    ALL_LABELS+=("Wass-L2 v3 lambda=${lambda}")
  done
fi
ALL_RUNS+=("$WASS_NONE_OUT")
ALL_LABELS+=("Wass query lambda=0 + no reweight")
if [[ "$RUN_KMEDIAN" == "1" ]]; then
  ALL_RUNS+=("${KMED_L2_DIRS[@]}")
  ALL_LABELS+=("${KMED_SWEEP_LABELS[@]}")
fi
ALL_RUNS+=("$RANDOM_NONE_OUT")
ALL_LABELS+=("random + no reweight")
RUN_COUNT="${#ALL_RUNS[@]}"

python compare_auc_trials.py \
  "${ALL_RUNS[@]}" \
  --out "${RESULT_ROOT}/figures/final/${PLOT_PREFIX}_${RUN_COUNT}run_average_precision.png" \
  --metric average_precision \
  --labels "${ALL_LABELS[@]}" \
  --cmap-runs none

python compare_auc_trials.py \
  "${ALL_RUNS[@]}" \
  --out "${RESULT_ROOT}/figures/final/${PLOT_PREFIX}_${RUN_COUNT}run_pr_auc_trapz.png" \
  --metric pr_auc \
  --labels "${ALL_LABELS[@]}" \
  --cmap-runs none

echo ""
echo "All ${RUN_COUNT} ${RUN_LABEL} runs completed."
