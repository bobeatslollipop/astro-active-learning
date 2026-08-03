# XGBoost Wasserstein-L2 Lambda Sweep, 150 Queries, 10 Seeds

This archive captures the XGBoost active-learning lambda sweep for
`wasserstein_l2` sampling with `voronoi_l2` reweighting.

Setup:
- Dataset: `bp_rp_lamost_normalized.h5`
- Warm start: `bp_rp_lamost_normalized_low_teff.h5`
- Model: XGBoost, deeper config from the full-eval benchmark
- Sampling: `wasserstein_l2`
- Reweighting: `voronoi_l2`
- Extra queries: 150
- Eval interval: 10 queries
- Seeds/trials: 10
- Lambda values in final comparison: 100, 300, 1000, 3000, 10000
- Wasserstein pool size: 45000
- Reweighting pool size: 100000
- Voronoi-L2 max iterations: 5
- GPU: CUDA device 1

Contents:
- `results/`: copied experiment outputs and comparison plots.
- `run_xgboost_wasserstein_l2_lambda_sweep.sh`: runner used for the final low-lambda completion and comparison plot.
- `logs/`: logs from the original partial run and the final 100/300 completion run.

The original result directory was copied, not moved, so existing result links still work.
