# Astro Active Learning

## Project Structure

This project implements an active learning framework for stellar classification, focusing on identifying metal-poor stars from imbalanced and biased datasets.

*   **`active_learning.py`**: The core active learning loop. It supports query strategies (`random`, `purely_random`, `uncertainty`, `entropy`, `margin`, `wasserstein`, `entropicOT`, `kmedianpp`, `moment_matching`), final models (`logistic`, `ridge`), and dataset-shift reweighting (`none`, `hard`, `soft`, `voronoi_l2`, `kl`, `moment_l2`).
*   **`run_experiments.sh`, `run_voronoi_l2_sweep.sh`, `run_moment_sweep.sh`**: Shell scripts for batch active-learning runs, Voronoi-L2 reweighting sweeps, and linear moment-L2 sweeps.
*   **`linear_classifier.py`, `linear_regression.py`, `two_layers.py`**: Baseline model training scripts (logistic regression, linear regression, and a 2-layer neural network) for training on fixed datasets.
*   **`compare_auc_trials.py`**: A plotting utility to compare PR-AUC learning curves across multiple active learning runs.
*   **`visualize_embedding.py`, `visualize_feh_dist.py`**: Utilities for UMAP/t-SNE embedding visualizations and plotting metallicity ([Fe/H]) distributions.
*   **`normalize_h5.py`**: Script to L2-normalize the stellar spectra features in the HDF5 datasets.
*   **`experiment_results_100/`, `experiment_results_6k/`**: Output directories containing logs, weights, and plots from active learning experiments.
*   **`random_training/`, `low_temp_training/`**: Scripts and data splits for baseline model training.

## Dataset Preparation

Active-learning runs use `bp_rp_lamost_normalized.h5` and the biased warm-start file `bp_rp_lamost_normalized_low_teff.h5`. For fixed-dataset baseline scripts, generate a random train/test split with:

```bash
cd random_training
python generate_dataset.py --seed 42 --file-path ../bp_rp_lamost_normalized.h5 --feh-threshold -2.0 --train-frac 0.8 --mr-ratio 1
cd ..
```

This generates `random_train_set.h5` and `random_test_set.h5` inside the `random_training` folder.

## Optional Fixed-Dataset Baselines

These scripts are secondary baselines. The main experiment interface is `active_learning.py`.

```bash
python linear_classifier.py --run-name default_run --data-split random --optimizer irls --lambda-MP 0.1 --weight-decay 0.0 --feh-threshold -2.0
python3 linear_regression.py --run-name ridge_baseline --data-split random --optimizer exact --weight-decay 1.0 --low-feh-weight 0.3 --feh-threshold -2.0
```

`linear_classifier.py` supports `--optimizer adam|irls`; `linear_regression.py` supports `--optimizer adam|exact`, where `exact + --weight-decay` is ridge regression. `two_layers.py` remains an Adam-only neural baseline.

## Embedding Visualization

Visualize high-dimensional BP/RP embeddings using UMAP, t-SNE, or PCA, colored by metallicity ([Fe/H]).

```bash
python visualize_embedding.py --method umap
python visualize_embedding.py --method umap --threshold -2.0
python visualize_embedding.py --method umap --threshold -2.0 --continuous
python visualize_embedding.py --method umap --threshold -2.0 --eval_weights linear_0.1/linear_model_weights.csv
```

## Active Learning (Warm Start)

Trains a classifier via active learning, starting from a biased initial set (e.g. low-$T_{\rm eff}$ stars) and iteratively querying the full population. The default final model is logistic regression; `--model ridge` uses regularized linear regression as a classifier.

```bash
CUDA_VISIBLE_DEVICES=1

python active_learning.py \
  --warm-start-file bp_rp_lamost_normalized_low_teff.h5 \
  --full-data-file  bp_rp_lamost_normalized.h5 \
  --feh-threshold   -2.0 \
  --strategy        wasserstein \
  --model           logistic \
  --reweighting     voronoi_l2 \
  --reweight-lambda 3000 \
  --softmax-pool-size 100000 \
  --total-queries   100 \
  --eval-every      10 \
  --lambda-MP       0.01 \
  --wass-pool-size  50000 \
  --C               10000.0 \
  --eval-size       500000 \
  --seed            42 \
  --n-trials        16 \
  --n-snapshots     10 \
  --out-dir         al_wasserstein_voronoi_l2_100_lambda_3000
```

Outputs (in `--out-dir`): `results.json`, `params.json`, `final_weights.csv`, PR curves, weight-distribution plots for reweighted runs, and multi-trial summaries such as `auc_trials.json`/`auc_trials.png` when `--n-trials > 1`.

### Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--warm-start-file` | `bp_rp_lamost_normalized_low_teff.h5` | H5 file for the biased warm-start set. |
| `--full-data-file` | `bp_rp_lamost_normalized.h5` | H5 file for the full population (pool + eval). |
| `--feh-threshold` | `-2.0` | Fe/H cut: < threshold → MP (0), ≥ threshold → MR (1). |
| `--strategy` | `uncertainty` | Query strategy: `random`, `purely_random`, `uncertainty`, `entropy`, `margin`, `wasserstein`, `entropicOT`, `kmedianpp`, `moment_matching`. |
| `--total-queries` | `3000` | Total points to query from the pool. |
| `--eval-every` | `200` | Retrain and evaluate every k queries. |
| `--model` | `logistic` | Final classifier: `logistic` or `ridge`. |
| `--lambda-MP` | `1.0` | Desired total-weight ratio MP/MR. Per-sample weights auto-scale: $w_{MP} = \lambda \cdot n_{MR}/n_{MP}$. |
| `--C` | `1.0` | Inverse L2 regularisation strength for the logistic regression classifier. Larger values mean weaker classifier regularization. |
| `--ridge-alpha` | `1.0` | L2 regularization strength for `--model ridge`. Larger values mean stronger ridge regularization. |
| `--reweighting` | `none` | Covariate-shift correction: `none`=uniform weights, `hard`=hard Voronoi weights, `soft`=temperature softmin weights, `voronoi_l2`=L2-regularized Wasserstein/Voronoi final weights, `kl`=KL-regularized Wasserstein/Voronoi final weights, `moment_l2`=linear second-moment weights with L2 weight regularization. |
| `--reweight-lambda` | `1.0` | Regularization strength for `voronoi_l2`, `kl`, and `moment_l2` reweighting. Larger values produce less concentrated final weights. |
| `--voronoi-l2-max-iter` | `15` | Maximum LBFGS iterations for `voronoi_l2`/`kl` reweighting at each snapshot. |
| `--temperature` | `1.0` | Temperature for `soft` reweighting. Smaller values approach hard Voronoi weights. |
| `--soft-topk` | `0` | Top-K nearest labeled points for `soft` reweighting. `0` auto-calibrates per snapshot. |
| `--softmax-pool-size` | `None` | Subsample pool to this size for `soft`, `voronoi_l2`, `kl`, and `moment_l2` reweighting. `None` uses the full pool. Hard reweighting is unaffected. |
| `--eval-size` | `100000` | Size of random eval subsample drawn from the full population. |
| `--warm-start-max` | `None` | Cap warm-start size (subsampled if exceeded). |
| `--pool-max` | `None` | Cap full-population size (subsampled if exceeded). |
| `--wass-pool-size` | `50000` | Subpool size for `wasserstein` and `entropicOT` query planning. |
| `--eot-temperature` | `1.0` | Temperature for `entropicOT` query planning. Smaller values approach hard Wasserstein selection. |
| `--moment-ridge` | `1.0` | Ridge regularization used inside the `moment_matching` query objective. |
| `--moment-weight-iters` | `200` | Projected subgradient iterations for `--reweighting moment_l2`. |
| `--n-trials` | `1` | Number of independent trials. Multi-trial runs save mean/std PR-AUC summaries. |
| `--n-snapshots` | `3` | Number of evenly spaced PR-AUC snapshot points. |
| `--seed` | `42` | Random seed. |
| `--out-dir` | `al_{strategy}` | Output directory. |

### Query Strategies

| Strategy | Description |
| :--- | :--- |
| `random` | Uniform random sampling (baseline). |
| `purely_random` | Uniform random sampling without the biased warm-start set. |
| `uncertainty` | Sample points near predicted probability 0.5. |
| `entropy` | Sample points by predictive entropy. |
| `margin` | Pick points with smallest \|decision function\| (closest to boundary). |
| `wasserstein` | Greedy core-set: maximise coverage of the full population. |
| `entropicOT` | Entropic optimal-transport variant of Wasserstein-style coverage. |
| `kmedianpp` | k-median++ style geometric coverage baseline. |
| `moment_matching` | Greedy ridge linear-design selection that reduces target second-moment prediction discrepancy. |

### Reweighting Methods

| Method | Description |
| :--- | :--- |
| `none` | Train with class-ratio weights only. |
| `hard` | Hard Voronoi assignment from target-pool points to labeled points. |
| `soft` | Temperature-softened Voronoi assignment, optionally truncated to top-K neighbors. |
| `voronoi_l2` | Wasserstein/Voronoi final weights with an L2 concentration penalty controlled by `--reweight-lambda`. |
| `kl` | Wasserstein final weights with a KL/entropy-style concentration penalty controlled by `--reweight-lambda`. |
| `moment_l2` | Linear-regression moment weights: minimize target second-moment mismatch plus an L2 weight-spreading penalty. |

## Comparing AUC across Experiments

`compare_auc_trials.py` reads the `auc_trials.json` files produced by `active_learning.py` (the raw data behind each `auc_trials.png`) and overlays the PR-AUC learning curves from multiple experiments on one figure. Each curve shows the **mean ± 1σ** band across all trials.

```bash
# Auto-discover all experiment subdirectories that contain auc_trials.json
python compare_auc_trials.py --base-dir presentation_runs/ --cmap-runs viridis


# Compare a specific subset, with custom legend labels and output file
python compare_auc_trials.py \
  al_random_100 al_uncertainty_100 al_kmedianpp_100 al_wasserstein_hard_100 \
  --labels "Random" "Uncertainty" "K-Median++" "Wasserstein (hard)" \
  --out comparison_100.png

# Also overlay the individual trial lines (lighter, thinner)
python compare_auc_trials.py \
  al_random_100 al_uncertainty_100 \
  --labels "Random" "Uncertainty" \
  --show-trials \
  --out comparison_with_trials.png

# Custom figure size and title
python compare_auc_trials.py \
  al_random_100 al_uncertainty_100 al_kmedianpp_100 \
  --figsize 14 7 \
  --title "100-Query Active Learning: PR-AUC Comparison" \
  --out comparison_100_wide.png
```

### Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `dirs` | *(auto)* | Positional list of experiment directories. Auto-discovers all subdirs with `auc_trials.json` if omitted. |
| `--labels` / `-l` | *(from dir name)* | Legend labels (must match the number of directories). |
| `--out` / `-o` | `auc_comparison.png` | Output image path. |
| `--show-trials` | off | Overlay individual trial lines on top of the mean curve. |
| `--figsize W H` | `12 7` | Figure width and height in inches. |
| `--title` | *(default string)* | Plot title. |
| `--base-dir` | `.` | Root directory for auto-discovery mode. |
| `--cmap-runs` | `coolwarm` | Continuous Matplotlib colormap for parameter sweeps. Use `none` for the discrete palette. |

## Experiment Conclusions: 100 Queries (`experiment_results_100`)

The `experiment_results_100` directory contains the results of an active learning evaluation where a logistic regression model is initialized on a heavily biased sample (low-temperature stars only) and updated using a very small budget of 100 newly queried stars from the full population.

In this "biased initial sample + few queries" regime, the **Wasserstein hard** strategy (core-set Wasserstein querying combined with hard Voronoi reweighting) emerges as the top-performing method. It effectively corrects the covariate shift of the initial biased sample while efficiently querying the most informative points to construct a representative target distribution. We strongly recommend `Wasserstein hard` as the primary active learning strategy for scenarios with severe initial bias and extremely limited query budgets.
