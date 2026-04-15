## Dataset Generation
Before training, you need to generate the split training and testing datasets.
```bash
cd random_training
python generate_dataset.py --seed 42 --file-path ../bp_rp_lamost_normalized.h5 --feh-threshold -2.0 --train-frac 0.8 --mr-ratio 1
cd ..
```
This generates `random_train_set.h5` and `random_test_set.h5` inside the `random_training` folder.

## Training Linear Classifier
To train the model:
```bash
python linear_classifier.py --run-name default_run --seed 42 --feh-threshold -2.0 --optimizer irls --lr 1.0 --epochs 500 --batch-size 30000 --lr-end-factor 1.0 --lambda-MP 0.1 --weight-decay 0.0 --momentum 0.0 --data-split random

python3 linear_regression.py --run-name weight_0.3 --seed 42 --optimizer exact --lr 1.0 --epochs 500 --batch-size 30000 --lr-end-factor 1.0 --weight-decay 0.0 --momentum 0.0 --data-split random --low-feh-weight 0.3 --cutoff 10 --feh-threshold -2.0
```

All outputs (weights, loss plots, and evaluation confusion matrices) will be saved in the `linear_{run_name}` directory.

### Available Training Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--run-name` | `str` | `None` | Name of the run. Outputs will be saved to `linear_{run_name}/`. |

| `--data-split` | `str` | `random` | Semantic name for the dataset split. Triggers auto-discovery of H5 files. (See below) |
| `--train-file` | `str` | `None` | Path to custom train H5 file. Overrides `--data-split`. |
| `--test-file`| `str` | `None` | Path to custom test H5 file. Overrides `--data-split`. |
| `--lambda-MP` | `float` | `2.0` | Reweight factor for Metal-Poor (MP) class. MP weight = $\lambda_{MP} / (1+\lambda_{MP})$, MR weight = $1/(1+\lambda_{MP})$. |
| `--feh-threshold` | `float` | `-2.0` | [Fe/H] threshold defining the boundary between MP and MR classes. |
| `--seed` | `int` | `42` | Random seed for reproducibility. |
| `--optimizer` | `str` | `adam` | Optimizer to use: `adam`, `sgd`, or `irls` (Iteratively Reweighted Least Squares). |
| `--lr` | `float` | `1.0` | Initial learning rate. |
| `--lr-end-factor`| `float` | `1.0` | Final learning rate multiplier (linear scheduler). |
| `--epochs` | `int` | `500` | Number of training epochs. |
| `--batch-size` | `int` | `30000` | Batch size for training. |
| `--weight-decay` | `float` | `0.0` | Weight decay factor for L2 regularization. |
| `--momentum` | `float` | `0.0` | Momentum factor for SGD optimizer. Ignored if `--optimizer=adam`. |
| `--low-feh-weight` | `float` | `1.0` | Regression only: Weight multiplier for samples with true Fe/H < -2.0. |
| `--cutoff` | `float` | `None` | Regression only: Exclude star properties whose target Fe/H is strictly greater than this value during both training and evaluation. |

### Embedding Visualization

Visualize high-dimensional BP/RP embeddings using UMAP, t-SNE, or PCA, colored by metallicity ([Fe/H]).

- **Standard Heatmap**: Randomly samples data and displays a continuous color gradient.
  ```bash
  python visualize_embedding.py --method umap
  ```
- **Balanced Classification**: Use `--threshold` to enable balanced sampling (ensuring rare metal-poor stars are well-represented) and display binary Red/Blue classes.
  ```bash
  python visualize_embedding.py --method umap --threshold -2.0
  ```
- **Balanced Heatmap**: Use both `--threshold` and `--continuous` to combine balanced sampling with a continuous color gradient.
  ```bash
  python visualize_embedding.py --method umap --threshold -2.0 --continuous
  ```
- **Error Diagnosis and Decision Boundary**: Use `--eval_weights <path_to_csv>` to overlay classification errors (false positives/negatives) as large triangles on the plot, and draw the linear classifier's decision boundary (`Logit = 0`).
  ```bash
  python visualize_embedding.py --method umap --threshold -2.0 --eval_weights linear_0.1/linear_model_weights.csv
  ```

## Active Learning (Warm Start)

Trains a logistic regression classifier via active learning, starting from a biased initial set (e.g. low-$T_{\rm eff}$ stars) and iteratively querying the full population.

```bash
python active_learning.py \
  --warm-start-file bp_rp_lamost_normalized_low_teff.h5 \
  --full-data-file  bp_rp_lamost_normalized.h5 \
  --feh-threshold   -2.0 \
  --strategy        uncertainty \
  --total-queries   500 \
  --eval-every      50 \
  --lambda-MP       1.0 \
  --C               1.0 \
  --eval-size       50000 \
  --seed            42 \
  --out-dir         al_uncertainty
```

```powershell
python active_learning.py `
  --warm-start-file bp_rp_lamost_normalized_low_teff.h5 `
  --full-data-file  bp_rp_lamost_normalized.h5 `
  --feh-threshold   -2.0 `
  --strategy        random `
  --total-queries   500 `
  --eval-every      50 `
  --lambda-MP       1.0 `
  --C               5.0 `
  --eval-size       50000 `
  --seed            42 `
  --out-dir         al_random
```

Outputs (in `--out-dir`): `results.json`, `final_weights.csv`, `params.json`, `learning_curve.png`, `class_distribution.png`.

### Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--warm-start-file` | `bp_rp_lamost_normalized_low_teff.h5` | H5 file for the biased warm-start set. |
| `--full-data-file` | `bp_rp_lamost_normalized.h5` | H5 file for the full population (pool + eval). |
| `--feh-threshold` | `-2.0` | Fe/H cut: < threshold → MP (0), ≥ threshold → MR (1). |
| `--strategy` | `uncertainty` | Query strategy: `random`, `uncertainty`, `margin`, `wasserstein`. |
| `--total-queries` | `500` | Total points to query from the pool. |
| `--eval-every` | `50` | Retrain and evaluate every k queries. |
| `--lambda-MP` | `1.0` | Desired total-weight ratio MP/MR. Per-sample weights auto-scale: $w_{MP} = \lambda \cdot n_{MR}/n_{MP}$. |
| `--C` | `1.0` | Inverse regularisation strength. |
| `--eval-size` | `100000` | Size of random eval subsample drawn from the full population. |
| `--warm-start-max` | `None` | Cap warm-start size (subsampled if exceeded). |
| `--pool-max` | `None` | Cap full-population size (subsampled if exceeded). |
| `--seed` | `42` | Random seed. |
| `--out-dir` | `al_{strategy}` | Output directory. |

### Query Strategies

| Strategy | Description |
| :--- | :--- |
| `random` | Uniform random sampling (baseline). |
| `uncertainty` | Pick points with predicted probability closest to 0.5. |
| `margin` | Pick points with smallest \|decision function\| (closest to boundary). |
| `wasserstein` | Greedy core-set: maximise coverage of the full population. |