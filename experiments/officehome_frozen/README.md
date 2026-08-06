# Office-Home frozen ResNet-50 features

This directory is an isolated domain-adaptation experiment. It extracts a
deterministic, frozen `ResNet50_Weights.IMAGENET1K_V1` representation and fits
weighted 65-class multinomial logistic regression. It also provides reusable
random/full-pool Wasserstein selection and hard/regularized-Wasserstein
reweighting adapters. It does not fine-tune the backbone or fit PCA.

The primary protocol is a class-stratified 80/20 target heldout split. The
source domain is fully labeled, the target-pool manifest exposed to selection
contains no labels, and target private labels are read only by the trainer
after query IDs are fixed or by the evaluator.

## Environment

From the repository root:

```bash
python3 -m venv --system-site-packages experiments/officehome_frozen/.venv
experiments/officehome_frozen/.venv/bin/python -m pip install \
  -r experiments/officehome_frozen/requirements.txt
```

The optional Hugging Face fallback has a separate dependency file:

```bash
experiments/officehome_frozen/.venv/bin/python -m pip install \
  -r experiments/officehome_frozen/requirements-hf-fallback.txt
```

## 1. Acquire the original dataset

The default downloader uses the archive linked from the
[Office-Home official page](https://www.hemanthdv.org/officeHomeDataset.html).
It does not use Kaggle and does not automatically switch providers.

```bash
PY=experiments/officehome_frozen/.venv/bin/python
EXP=experiments/officehome_frozen
OUT=results/domain_adaptation/officehome_frozen
DATASET_SOURCE=official

$PY $EXP/scripts/download_officehome.py --source official
```

`--data-root` overrides the data directory. Otherwise `OFFICEHOME_ROOT` is
used, followed by `$EXP/.data`. Existing validated domain directories are
reused without downloading again.

If and only if the official download is unavailable, explicitly use the
allowed fallback. Its provider and dataset fingerprint are recorded in every
subsequent dataset artifact:

```bash
$PY -m pip install -r $EXP/requirements-hf-fallback.txt
$PY $EXP/scripts/download_officehome.py --source huggingface
DATASET_SOURCE=huggingface
```

## 2. Manifest and frozen features

```bash
$PY $EXP/scripts/build_manifest.py \
  --output-dir $OUT/dataset \
  --dataset-source $DATASET_SOURCE

$PY $EXP/scripts/extract_resnet50_features.py \
  --data-root $EXP/.data \
  --manifest $OUT/dataset/manifest.csv \
  --output-dir $OUT/features \
  --device auto --batch-size 64 --workers 4 --seed 0

$PY $EXP/scripts/normalize_features.py \
  --raw-features $OUT/features/resnet50_imagenet1k_v1_raw.npy \
  --output-dir $OUT/features
```

Extraction uses RGB conversion, shorter-side resize to 256, a deterministic
224 center crop, and ImageNet V1 channel normalization. The model is frozen,
in eval mode, and called inside `torch.inference_mode()`. Both raw and row-wise
L2-normalized float32 arrays are retained.

## 3. Primary Art to Clipart heldout baseline

```bash
TASK=$OUT/tasks/heldout/art_to_clipart_seed0
RUN=$OUT/runs/heldout/art_to_clipart_seed0/source_only

$PY $EXP/scripts/make_task_split.py \
  --manifest-private $OUT/dataset/manifest_private.csv \
  --source-domain art --target-domain clipart \
  --protocol heldout --seed 0 --output-dir $TASK

$PY $EXP/scripts/select_l2.py \
  --features $OUT/features/resnet50_imagenet1k_v1_l2.npy \
  --feature-manifest $OUT/features/manifest.csv \
  --source-manifest $TASK/source_labeled.csv \
  --grid 1e-5 1e-4 1e-3 1e-2 1e-1 \
  --folds 3 --device auto --seed 0 \
  --output-dir $RUN/l2_cv

RHO=$($PY -c "import json; print(json.load(open('$RUN/l2_cv/l2_selection.json'))['selected_rho'])")

$PY $EXP/scripts/train_weighted_logreg.py \
  --features $OUT/features/resnet50_imagenet1k_v1_l2.npy \
  --feature-manifest $OUT/features/manifest.csv \
  --task-dir $TASK --l2 "$RHO" --device auto --seed 0 \
  --max-iter 500 --tolerance 1e-6 --output-dir $RUN

$PY $EXP/scripts/evaluate_logreg.py \
  --model $RUN/model.pt \
  --features $OUT/features/resnet50_imagenet1k_v1_l2.npy \
  --feature-manifest $OUT/features/manifest.csv \
  --task-dir $TASK --output-dir $RUN
```

The CV command reads only `source_labeled.csv`. Target-test labels are loaded
only by the last command. For every heldout run, that evaluator reports both
the heldout 20% metrics and transductive metrics on the complete target domain
(plus the complete target domain with queried rows removed).

## 4. Query-ID and sample-weight interface

Query files contain exactly one `row_id` column. Weight files contain exactly
`row_id,weight` and must cover every source plus queried-target training row
once. Weights must be finite, nonnegative, and are normalized to sum to one.
No class balancing is added.

The following creates a mechanical interface fixture; it is not a selection
algorithm:

```bash
$PY $EXP/scripts/make_example_query_inputs.py \
  --task-dir $TASK --num-queries 5 --output-dir $RUN/example_query

$PY $EXP/scripts/train_weighted_logreg.py \
  --features $OUT/features/resnet50_imagenet1k_v1_l2.npy \
  --feature-manifest $OUT/features/manifest.csv \
  --task-dir $TASK \
  --query-ids $RUN/example_query/query_ids.csv \
  --sample-weights $RUN/example_query/sample_weights.csv \
  --l2 "$RHO" --device auto --seed 0 \
  --output-dir $OUT/runs/heldout/art_to_clipart_seed0/example_query
```

Campaign selectors consume `target_pool_public.csv` plus label-free cached
features and output only query IDs and geometry metadata. They never receive
either private target manifest.

## 5. Transductive and all directed pairs

For the transductive protocol, the target pool is the full target domain. The
evaluator reports both full-target metrics and metrics on target rows not in
the query-ID file.

```bash
$PY $EXP/scripts/run_all_domain_pairs.py \
  --manifest-private $OUT/dataset/manifest_private.csv \
  --protocol heldout --seed 0 --output-root $OUT/tasks
```

This produces all 12 task directories. Reuse the same frozen feature cache and
run `select_l2.py`, `train_weighted_logreg.py`, and `evaluate_logreg.py` for
each pair.

## 6. Round-1 baseline and global-lambda campaign

The recoverable campaign runner covers all 12 directed pairs, seeds 0--4, a
150-query budget, complete public target geometry, four no-lambda baselines,
and random-query regularized-Wasserstein calibration on lambdas
`10,100,1000,10000`. Seed 0 selects one global lambda by equal-pair mean
heldout cross-entropy; seeds 1--4 reuse it without recalibration.

Run the required Art-to-Clipart seed-0 gate first:

```bash
env CUDA_VISIBLE_DEVICES=1 $PY $EXP/scripts/run_officehome_campaign.py \
  --stage smoke --device auto
```

After it passes, launch the resumable sequential full campaign in tmux:

```bash
PHYSICAL_GPU=1 STAGE=full \
  bash $EXP/scripts/launch_round1_campaign.sh
```

The launcher prints the tmux session, PID file, log, and follow commands. A
second invocation does not overwrite a live session. Completed runs validate
their compact record and are skipped; safe partial training/evaluation is
resumed, while inconsistent partial artifacts and failed regularized solves
remain explicit. The regularized solver doubles its accepted-iteration ceiling
once after a failed convergence check and never silently uses a failed solve.

Aggregation can be refreshed without retraining:

```bash
$PY $EXP/scripts/run_officehome_campaign.py --stage aggregate --device auto
```

Important reusable artifacts include shared `queries/.../query_ids.csv`, each
run's `weights/sample_weights.csv`, `run_record.json`,
`aggregates/per_run_results.csv`, `lambda_calibration_seed0.csv`, and
`selected_lambda.json`. Large models, features, predictions, full optimizer
traces, and logs remain local.

## Objective and theory record

Training minimizes

```text
sum_i v_i CrossEntropy(W z_i + b, y_i)
  + rho/2 * (||W||_F^2 + ||b||_2^2),  sum_i v_i = 1.
```

The bias is deliberately regularized. Since the fitted objective should not
exceed the zero classifier's `log(65)`, the run records the check

```text
||W||_F^2 + ||b||_2^2 <= 2 log(65) / rho.
```

For bounded operator norm, softmax cross-entropy is Lipschitz in the feature
vector; hard argmax/0-1 loss is not. Accuracy is reported for practical
comparison, while cross-entropy is the theory-aligned loss.

## Tests

```bash
$PY -m pytest -q $EXP/tests
```

After the real dataset and pretrained weights are available, run the opt-in
integration smoke test with the same device policy as feature extraction:

```bash
OFFICEHOME_REAL_SMOKE=1 OFFICEHOME_SMOKE_DEVICE=auto \
  $PY -m pytest -q $EXP/tests/test_real_smoke.py
```

Generated images, feature arrays, models, predictions, and logs remain local.
Compact metadata, metrics, CV summaries, and optimization histories are the
auditable records.
