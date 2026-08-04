#!/usr/bin/env bash
set -euo pipefail

export WARM_START=bp_rp_lamost_normalized_low_teff_50k_seed42.h5
export WASS_POOL=50000
export REWEIGHT_POOL=50000
export RUN_LAMBDA_10000=0
export RUN_LABEL="50K-warm, 50K reweight target, 50K query candidates, eval30"
export EXPERIMENT_FAMILY="xgb_noclassbalance_50k_warm_fixed10k_fullheldout_reweight50k_candidate50k_150q_5seeds_eval30_v3"

exec bash "$(dirname "$0")/run_xgboost_noclassbalance_100k_warm_eval30_v3.sh"
