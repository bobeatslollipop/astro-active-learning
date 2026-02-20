## Training Linear Classifier
To train the model with default settings:
```bash
python linear_classifier.py --run-name default_run --seed 42 --file-path ./bp_rp_lamost_normalized.h5 --feh-threshold -2.0 --train-frac 0.8 --mr-ratio 1 --optimizer adam --lr 1.0 --epochs 500 --batch-size 30000 --lr-end-factor 1.0 --lambda-MP 1.0 --weight-decay 0.0 --momentum 0.0 --use-ebv
```

All outputs (weights, loss plots, and evaluation confusion matrices) will be saved in the `linear_{run_name}` directory.

### Available Training Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--run-name` | `str` | `None` | Name of the run. Outputs will be saved to `linear_{run_name}/`. |
| `--use-ebv` | `flag` | `False` | Include `ebv` as a training feature. |
| `--lambda-MP` | `float` | `2.0` | Reweight factor for Metal-Poor (MP) class. MP weight = $\lambda_{MP} / (1+\lambda_{MP})$, MR weight = $1/(1+\lambda_{MP})$. |
| `--feh-threshold` | `float` | `-2.0` | [Fe/H] threshold defining the boundary between MP and MR classes. |
| `--seed` | `int` | `42` | Random seed for reproducibility. |
| `--train-frac` | `float` | `0.8` | Fraction of metal-poor stars used for training (remaining used for test). |
| `--mr-ratio` | `int` | `1` | Ratio of Metal-Rich to Metal-Poor stars in the training/test sets. |
| `--optimizer` | `str` | `adam` | Optimizer to use: `adam` or `sgd`. |
| `--lr` | `float` | `1.0` | Initial learning rate. |
| `--lr-end-factor`| `float` | `1.0` | Final learning rate multiplier (linear scheduler). |
| `--epochs` | `int` | `500` | Number of training epochs. |
| `--batch-size` | `int` | `30000` | Batch size for training. |
| `--weight-decay` | `float` | `0.0` | Weight decay factor for L2 regularization. |
| `--momentum` | `float` | `0.0` | Momentum factor for SGD optimizer. Ignored if `--optimizer=adam`. |
| `--file-path` | `str` | `./bp_rp_lamost_normalized.h5` | Path to the input H5 file. |

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
- **Error Diagnosis**: Use `--eval_weights` to overlay classification errors (false positives/negatives) as large triangles on the plot.
  ```bash
  python visualize_embedding.py --method umap --threshold -2.0 --continuous --eval_weights
  ```