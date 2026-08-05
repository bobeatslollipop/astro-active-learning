#!/usr/bin/env bash
set -euo pipefail

# Reproduce the historical 1:100 MP:MR total training-weight ratio using the
# current 50K warm-start, 100K target/candidate, and eval30 protocol. The
# regularized sweep uses kmedian++ rather than Wasserstein-L2 v3.
export PHYSICAL_GPU="${PHYSICAL_GPU:-0}"
export WARM_START=bp_rp_lamost_normalized_low_teff_50k_seed42.h5
export WASS_POOL=100000
export REWEIGHT_POOL=100000
export LAMBDA_MP=0.01
export CLASS_BALANCE_MODE=ratio
export RUN_WASSERSTEIN_L2=0
export RUN_KMEDIAN=1
export KMEDIAN_LAMBDAS="10 100 1000"
export RUN_LAMBDA_10000=0
export RUN_LABEL="50K warm, MP:MR weight ratio 0.01, 100K reweight target, 100K query candidates, eval30"
export PLOT_PREFIX="classbalance_mp0p01"
export EXPERIMENT_FAMILY="xgb_classbalance_mp0p01_50k_warm_fixed10k_fullheldout_reweight100k_candidate100k_150q_5seeds_eval30_kmedian"

exec bash "$(dirname "$0")/run_xgboost_noclassbalance_100k_warm_eval30_v3.sh"
