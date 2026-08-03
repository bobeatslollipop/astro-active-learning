# XGBoost Voronoi-L2 Sampling Comparison, 100 Queries

This folder archives the completed GPU run comparing three sampling strategies
under Voronoi-L2 reweighting:

- `kmedianpp`
- `random`
- `wasserstein_l2`

Setup:

- Data: `bp_rp_lamost_normalized.h5`
- Warm start: `bp_rp_lamost_normalized_low_teff.h5`
- Model: XGBoost binary classifier
- XGBoost device: CUDA via `tree_method=hist, device=cuda`
- Reweighting: `voronoi_l2`
- Reweight lambda: `1000`
- Voronoi-L2 optimizer max iterations: `5`
- Total queried samples: `100`
- Evaluation interval: every `10` queries
- Trials/seeds: `16`, starting from seed `42`
- Eval subsample size: `500000`
- Wasserstein query pool size: `45000`
- Voronoi-L2 softmax/reweighting pool size: `100000`

Contents:

- `results/`: all experiment outputs and comparison plots.
- `logs/`: tmux run log from the completed GPU run.
- `run_xgboost_voronoi_l2_experiments.sh`: runner used for this archive.
