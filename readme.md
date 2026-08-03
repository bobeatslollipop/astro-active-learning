# Astro Active Learning

Active-learning experiments for identifying metal-poor stars from Gaia BP/RP
features under severe class imbalance and a biased low-temperature warm start.

The repository is organized as a research workflow rather than an installable
Python package: one command-line entry point, several focused implementation
modules, shell runners for reproducible experiment families, and checked-in
summaries of completed runs.

## Scientific setup

The default task is binary stellar metallicity classification:

- metal poor (MP): Fe/H < -2
- metal rich (MR): Fe/H >= -2
- active-learning convention: MP = 0 and MR = 1
- features: 55 BP coefficients, 55 RP coefficients, and E(B-V)

The local full dataset contains 5,283,562 valid rows, including 10,201 MP
stars. The default warm-start file contains 450,000 low-temperature rows,
including 1,580 MP stars. This creates both extreme class imbalance and a
covariate shift between the initial labeled set and the full population.

All current code uses the same canonical encoding: MP = 0 and MR = 1. Metric
helpers explicitly convert MP into the positive metric target and locate its
probability through `model.classes_`, so the scientific positive class does not
depend on a hard-coded probability column. The one preserved historical
full-dataset run used MP = 1 at training time; its params file records both that
legacy provenance and the current project convention.

## Repository layout

| Path | Purpose |
| --- | --- |
| active_learning.py | Stable CLI and backward-compatible import surface. |
| al_data.py | HDF5 loading, feature ordering, normalization, and runtime helpers. |
| al_queries.py | Random, uncertainty, Wasserstein-family, k-median++, and moment-matching query strategies. |
| al_reweighting.py | Hard/soft Voronoi, Voronoi-L2, KL, and moment-L2 sample weights. |
| al_models.py | Training-weight normalization and Logistic, Ridge, and XGBoost classifiers. |
| al_reporting.py | Metrics, trial summaries, plots, and final model summaries. |
| al_runner.py | Data splitting, multi-trial orchestration, snapshot training, and artifact writing. |
| al_metadata.py | Single-file params schema, fingerprints, hashes, Git state, and environment metadata. |
| compare_auc_trials.py | Compare PR-AUC or average-precision curves across runs. |
| compare_weight_l2_trials.py | Compare weight concentration and effective sample size. |
| run_xgboost_full_eval.py | Supervised full-dataset model-family benchmark. |
| run_*.sh | Reproducible experiment families and comparison-plot generation. |
| normalize_h5.py | Create coefficient-normalized HDF5 data. |
| low_temp_training/ | Build the biased low-temperature warm-start data. |
| random_training/ | Build fixed random train/test baseline datasets. |
| visualize_embedding.py | PCA, t-SNE, or UMAP views of the feature space. |
| visualize_feh_dist.py | Fe/H distribution visualization. |
| results/active_learning/ | Active-learning families and their individual runs. |
| results/full_data/ | Supervised full-dataset benchmarks. |
| results/diagnostics/ | Optimizer and implementation diagnostics. |
| results/archive/ | Self-contained archives of older completed runs. |
| results/logs/ | Local-only long-running experiment logs. |

Result directories no longer live at repository root. Their names are useful
hints, but `params.json` is the authoritative description of a run. See
`results/README.md` for the artifact tracking policy.

## Environment

Python 3.10 or 3.11 is recommended.

Create an isolated environment and install the declared dependencies:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

The numerical requirements intentionally require SciPy 1.11 or newer so it is
compatible with NumPy 1.26 or newer.

### CUDA notes

PyTorch and XGBoost GPU support depend on the local CUDA driver and wheel/build.
If the generic pip installation does not match the machine, install the
appropriate PyTorch build first and then install the remaining requirements.

The GPU runners normally use:

~~~bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-cache
~~~

For XGBoost 2 or newer, the active-learning runners use tree_method=hist with
device=cuda. Use nvidia-smi and the run log to verify that a nominal GPU run is
actually using a GPU.

UMAP visualization uses umap-learn on CPU. visualize_embedding.py can also use
RAPIDS cuML when it is installed separately in a compatible CUDA environment.

## Data preparation

The main commands expect these local files in the repository root:

- bp_rp_lamost_normalized.h5
- bp_rp_lamost_normalized_low_teff.h5

They are intentionally not tracked by Git.

To normalize a raw columnar HDF5 file:

~~~bash
python normalize_h5.py
~~~

To rebuild the low-temperature warm-start file, adjust paths in
low_temp_training/extract_low_teff.py and run:

~~~bash
python low_temp_training/extract_low_teff.py
~~~

For the secondary fixed-dataset baselines:

~~~bash
cd random_training
python generate_dataset.py \
  --seed 42 \
  --file-path ../bp_rp_lamost_normalized.h5 \
  --feh-threshold -2.0 \
  --train-frac 0.8 \
  --mr-ratio 1
cd ..
~~~

## Active-learning CLI

All existing commands continue to use active_learning.py.

A small CPU-friendly smoke run is:

~~~bash
python active_learning.py \
  --warm-start-max 5000 \
  --pool-max 20000 \
  --eval-size 5000 \
  --strategy random \
  --model logistic \
  --reweighting none \
  --total-queries 20 \
  --eval-every 10 \
  --n-snapshots 2 \
  --n-trials 1 \
  --out-dir /tmp/astro-al-smoke
~~~

A representative XGBoost run is:

~~~bash
python active_learning.py \
  --warm-start-file bp_rp_lamost_normalized_low_teff.h5 \
  --full-data-file bp_rp_lamost_normalized.h5 \
  --feh-threshold -2.0 \
  --strategy kmedianpp \
  --reweighting voronoi_l2 \
  --reweight-lambda 100 \
  --model xgboost \
  --total-queries 150 \
  --eval-every 15 \
  --n-snapshots 10 \
  --n-trials 5 \
  --eval-source full_heldout \
  --reweight-source full_non_eval \
  --class-balance-mode none \
  --train-weight-sum-mode fixed \
  --train-weight-sum 10000 \
  --include-zero-snapshot \
  --out-dir my_experiment
~~~

Use python active_learning.py --help for the complete option list.

## Query strategies

| Strategy | Description |
| --- | --- |
| random | Uniform random query baseline. |
| purely_random | Random querying without the biased warm-start set. |
| uncertainty | Stochastic sampling weighted toward probability 0.5. |
| entropy | Predictive-entropy sampling. |
| margin | Smallest decision-function margin. |
| wasserstein | Greedy geometric coverage on a random target/candidate subpool. |
| wasserstein_l2 | Wasserstein coverage plus the complete updated Voronoi cell-mass L2 penalty. |
| entropicOT | Temperature-controlled entropic transport variant. |
| kmedianpp | k-median++ geometric coverage baseline. |
| moment_matching | Ridge-design selection that reduces feature-moment mismatch. |

Wasserstein-family planning is approximate even before regularization because it
operates on random subpools controlled by wass_pool_size and periodically
rebuilds plans according to wass_plan_size.

### Wasserstein-L2 interpretation

Current implementation version 2 scores every candidate using nearest-neighbour
Wasserstein coverage plus the squared masses of all Voronoi cells after adding
that candidate:

~~~text
WWD(S union {u}, T) + lambda * [m_u^2 + sum_i (w_i - c_i,u)^2]
~~~

The candidate-target distance matrix is reused, while candidate-by-cell capture
counts are reduced in bounded chunks. Historical runs without
`query_implementation_version` used version 1, which penalized only the new
candidate's captured mass. Those existing artifacts are not rewritten or
reinterpreted as version 2.

Version 2 is the complete L2 penalty for the nearest-neighbour Voronoi plan. It
still does not re-solve the transport plan and all support weights for every
candidate. The later Voronoi-L2 reweighting solve therefore does not make the
preceding ranking exact. Describe it as full-Voronoi-L2 greedy, not exact
regularized-OT greedy.

The CLI enforces that strategy=wasserstein_l2 is paired with
reweighting=voronoi_l2 and a positive shared regularization value.

For small-scale auditing, `diagnose_wasserstein_l2_objectives.py` compares
historical captured-mass v1, full-Voronoi v2, and candidate-wise solves of the
complete regularized objective:

~~~bash
python diagnose_wasserstein_l2_objectives.py \
  --support-size 6 \
  --pool-size 12 \
  --n-pick 3 \
  --reweight-lambda 100 \
  --out-dir results/diagnostics/wasserstein_l2_exact_oracle
~~~

The diagnostic writes JSON, CSV, and text summaries. It calls a winner exact
only when one candidate's feasible primal upper bound is strictly below every
competitor's dual lower bound; overlapping intervals are reported as
`unresolved`.

## Reweighting methods

| Method | Description |
| --- | --- |
| none | No covariate-shift correction; only final weight normalization/class handling is applied. |
| hard | Hard nearest-neighbor Voronoi mass assignment. |
| soft | Temperature-softened Voronoi mass assignment. |
| voronoi_l2 | Wasserstein/Voronoi weights with an L2 concentration penalty. |
| kl | Wasserstein/Voronoi weights with KL-style regularization. |
| moment_l2 | Feature second-moment matching with L2 weight spreading. |

The raw reweighting distribution and the final training weights are distinct.
After reweighting, the training layer can optionally enforce an MP/MR total
weight ratio and always normalizes to the selected training-weight total.

Voronoi-L2 uses one convergence policy for the initial and warm-started solves:
at most 128 accepted L-BFGS updates, with early stopping after two consecutive
absolute dual-objective improvements below `1e-4`. A gradient infinity norm at
or below `1e-5` also stops the solve. Override the budget or objective rule with
`--voronoi-l2-max-iter`, `--voronoi-l2-objective-tol`, and
`--voronoi-l2-objective-patience`.

Every accepted iterate is printed and appended to the run-local
`optimizer.log`. The same records are saved by trial and query snapshot in
`optimizer_trace_trials.json`, including the minimized dual objective, a
feasible primal upper bound, dual lower bound, primal-dual gap, gradient
infinity norm, raw weight sum, stopping reason, and elapsed time.

## Evaluation and reproducibility

Two evaluation modes exist:

- pool: legacy behavior; evaluation rows are sampled from the query pool and
  remain eligible for querying.
- full_heldout: evaluation rows are sampled first and removed from both the
  warm-start training set and the query pool using source_id.

Use full_heldout for new matched comparisons.

For each trial the runner records the exact queried pool indices, source IDs,
labels, and batches. k-median++ uses a dedicated query RNG stream so
reweighting/training draws do not change its query plan.

## Output schema

A multi-trial run normally writes:

| Artifact | Contents |
| --- | --- |
| params.json | Schema-v2 inputs, derived data/split statistics, Git/environment metadata, hashes, status, and timings. |
| results.json | Detailed evaluation snapshots for the first trial. |
| auc_trials.json | PR-AUC, average precision, queried MP counts, losses, and weight statistics for all trials. |
| query_plan_trials.json | Exact query plan for every trial. |
| weight_stats_trials.json | Weight L2 norm, ESS, top-mass, and related concentration statistics. |
| final_weights.csv | Linear coefficients or XGBoost feature importances. |
| figures/generated/auc_trials.png | Mean and one-standard-deviation PR-AUC curves. |
| figures/generated/average_precision_trials.png | Mean and one-standard-deviation AP curves. |
| figures/generated/pr_curve_mp.png | MP precision-recall curves from first-trial snapshots. |
| figures/generated/test_loss_*_trials.png | Per-class test-loss trajectories. |
| figures/generated/weight_*_trials.png | Weight concentration and ESS trajectories. |

Do not infer a run configuration from its folder name alone. Read params.json
before combining or comparing results.

### Params schema and run status

New active-learning and full-data runs write exactly one metadata/configuration
file, `params.json`. It groups data, split, query, reweighting, training, trial,
environment, and timing information. The run section contains:

- `status`: `running`, `completed`, or `failed`
- `config_hash`: hash of all scientific inputs
- `protocol_id`: hash of settings that must match across a comparison family
- Git commit/branch/dirty state and the original command line

The file is written atomically after data preparation and updated when the run
finishes. A process killed without a Python exception remains marked `running`,
which is intentionally distinguishable from a completed experiment.

Historical active-learning files remain flat schema v1. They have only been
annotated with the verified label convention, artifact-layout version, and old
path; unavailable historical environment or Git facts were not invented.

### Git policy for results

The complete local result tree is retained, but Git uses a narrow allowlist.
Compact JSON/CSV records, archived launchers and README files, and curated
family plots under `figures/final/` are shareable. Per-run plots under
`figures/generated/`, logs, top-candidate tables, checkpoints, and large raw
arrays remain local. Use `git check-ignore -v PATH` when adding a new artifact
type.

## Experiment families

### Full-dataset model benchmark

`results/full_data/natural_seed42` compares Logistic regression with shallow,
medium, and deeper XGBoost configurations on a natural stratified split.

The recorded eval PR-AUC values are approximately:

| Model | Eval PR-AUC |
| --- | ---: |
| Logistic | 0.0594 |
| XGBoost shallow | 0.5398 |
| XGBoost medium | 0.5925 |
| XGBoost deeper | 0.6467 |

The deeper configuration (700 trees, depth 6, learning rate 0.03) became the
default model family for the later active-learning runners.

### Legacy 1 percent MP-weight families

These include `results/active_learning/0.01MP-weight`,
`results/active_learning/Jul23-kmedpp-lambda`,
`results/active_learning/Jul24-wass-lambda`, and the earlier 100/150/200-query
XGBoost families.

They generally use lambda_MP=0.01 and legacy pool evaluation. They remain useful
historical baselines but should not be mixed with later full-heldout runs.

### Full-heldout and no-class-balance families

The later experiments progressively introduced:

- lambda_MP=1
- full-heldout evaluation
- explicit training-weight-sum semantics
- class_balance_mode=none
- full_non_eval as the reweighting target
- the zero-query warm-start snapshot

The canonical matched comparison root is:

`results/active_learning/xgb_wasserstein_l2_noclassbalance_fixed10k_fullheldout_reweightfull_150q_5seeds_eval15`

It uses 150 queries, evaluation every 15 queries, five trials, fixed total
training weight 10,000, no forced class balancing, and a 100,000-row reweighting
subsample from the full non-evaluation population.

Final mean average precision at 150 queries:

| Run | Mean AP |
| --- | ---: |
| kmedian++ + Voronoi-L2, lambda=100 | 0.2968 |
| Wasserstein-L2, lambda=100 | 0.2823 |
| Wasserstein-L2, lambda=1000 | 0.2811 |
| Wasserstein-L2, lambda=10000 | 0.2800 |
| Wasserstein-L2, lambda=10 | 0.2561 |
| random + no reweight | 0.2486 |
| Wasserstein query + no reweight | 0.2464 |
| hard Wasserstein | 0.2233 |

Within this matched protocol, k-median++ plus Voronoi-L2 at lambda=100 is the
strongest tested method. Query-only Wasserstein does not outperform random
sampling, and hard Voronoi reweighting performs poorly because its weights are
extremely concentrated.

## Batch runners

Important current runners include:

- run_xgboost_noclassbalance_100k_warm_eval30_v2.sh
- run_xgboost_kmedianpp_l2_lambda_sweep_noclassbalance_fixed10k.sh
- run_xgboost_wasserstein_l2_noclassbalance_reweightfull_eval15.sh
- run_xgboost_noclassbalance_matched_baselines.sh
- run_xgboost_wasserstein_lambda0_eval10.sh
- run_xgboost_voronoi_l2_sampling_200q_10seeds.sh
- run_xgboost_full_eval.py

Runners default to `results/active_learning/` and accept a `RESULTS_ROOT`
environment override. Most refuse to overwrite completed outputs or skip
already populated subdirectories. Always inspect the selected result root,
`params.json`, and log before relaunching.

## Comparing completed runs

Average precision:

~~~bash
python compare_auc_trials.py \
  results/active_learning/family/run_a \
  results/active_learning/family/run_b \
  results/active_learning/family/run_c \
  --metric average_precision \
  --labels A B C \
  --out comparison_average_precision.png
~~~

Trapezoidal PR-AUC:

~~~bash
python compare_auc_trials.py \
  results/active_learning/family/run_a \
  results/active_learning/family/run_b \
  results/active_learning/family/run_c \
  --metric pr_auc \
  --labels A B C \
  --out comparison_pr_auc.png
~~~

Weight concentration:

~~~bash
python compare_weight_l2_trials.py \
  results/active_learning/family/run_a \
  results/active_learning/family/run_b \
  results/active_learning/family/run_c \
  --metric effective_sample_size \
  --labels A B C \
  --out comparison_ess.png
~~~

Only combine runs whose params.json files agree on the data split, evaluation
source, reweighting source, class-balance behavior, training-weight total,
query budget, evaluation schedule, model hyperparameters, and trial count.

## Known limitations

- Wasserstein-L2 v2 uses the full updated nearest-neighbour Voronoi mass
  penalty, but is not a candidate-wise solution of the complete regularized
  transport objective. Historical unversioned runs used captured-mass v1.
- Voronoi-L2 stops using explicit objective/gradient criteria or the 128-update
  ceiling; inspect `optimizer_trace_trials.json` before treating a ceiling hit
  as convergence.
- Historical schema-v1 params files predate environment, Git-state, and hash
  metadata; only facts recoverable from the artifacts are annotated.
- The preserved full-dataset baseline was originally trained with MP = 1 even
  though all current code uses MP = 0. Its params file records this explicitly.
- Generated figures are intentionally local; retain the JSON/CSV inputs needed
  to reconstruct any plot selected for publication.
