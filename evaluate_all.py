import h5py
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use-ebv', action='store_true', help="Evaluate model with 'ebv' feature.")
args = parser.parse_args()

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

print("Loading weights...")
weights_file = 'linear_model_weights_ebv.csv' if args.use_ebv else 'linear_model_weights.csv'
with open(weights_file, 'r') as f:
    rows = list(csv.DictReader(f))
bias = float(next(r['weight'] for r in rows if r['feature'] == 'BIAS'))
features_in_weights = [r['feature'] for r in rows if r['feature'] != 'BIAS']
weights = np.array([float(r['weight']) for r in rows if r['feature'] != 'BIAS'])

print("Loading dataset...")
file_path = 'bp_rp_lamost_normalized.h5'
with h5py.File(file_path, 'r') as f:
    keys = list(f.keys())
    bp_cols = sorted([k for k in keys if k.startswith('bp_')], key=numerical_sort_key)
    rp_cols = sorted([k for k in keys if k.startswith('rp_')], key=numerical_sort_key)
    feature_cols = bp_cols + rp_cols
    norm_cols = set(feature_cols) # Only these columns will be normalized
    if args.use_ebv:
        feature_cols.append('ebv')
    
    # Check that the columns match exactly
    assert features_in_weights == feature_cols, f"Feature mismatch! Weights features: {len(features_in_weights)}, Dataset features: {len(feature_cols)}"
    
    full_feh = f['feh'][:].astype(np.float64)
    valid_mask = np.isfinite(full_feh)
    
    feh_valid = full_feh[valid_mask]
    
    # Class 0: MP (Fe/H < -2.0)
    # Class 1: MR (Fe/H >= -2.0)
    y_true = (feh_valid >= -2.0).astype(int) 
    
    print(f"Total valid samples with Fe/H labels: {len(y_true)}")
    
    # We will compute properties column by column to be extremely memory-efficient
    print("Computing L2 norms per sample (excluding ebv)...")
    sum_sq = np.zeros(len(y_true), dtype=np.float64)
    for i, col_name in enumerate(feature_cols):
        if col_name not in norm_cols: continue
        if i % 20 == 0: print(f" Computing norm: {i}/{len(feature_cols)} features...")
        sum_sq += np.nan_to_num(f[col_name][:][valid_mask]) ** 2
    norms = np.sqrt(sum_sq) + 1e-8
    
    # Now compute the linear prediction outputs (logits)
    print("Computing predictions using learned weights...")
    logits = np.zeros(len(y_true), dtype=np.float32) + bias
    
    for i, col_name in enumerate(feature_cols):
        if i % 20 == 0: print(f" Applying weights: {i}/{len(feature_cols)} features...")
        col_valid = np.nan_to_num(f[col_name][:][valid_mask])
        if col_name in norm_cols:
            logits += (col_valid / norms) * weights[i]
        else:
            logits += col_valid * weights[i]

    y_pred = (logits > 0.0).astype(int)
    
print("Calculating metrics...")
acc = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {acc:.4%}")

cm = confusion_matrix(y_true, y_pred)

print(f"\n--- Results ---")
print(f"True MP (< -2): {np.sum(y_true == 0):,} samples")
print(f"True MR (>= -2): {np.sum(y_true == 1):,} samples")
print('-'*30)
print(f"Predicted MP: {np.sum(y_pred == 0):,} samples")
print(f"Predicted MR: {np.sum(y_pred == 1):,} samples")
print(f"Overall Accuracy: {acc:.4%}\n")

print("Confusion Matrix:")
print("                 | Pred MP (0) | Pred MR (1)")
print("--------------------------------------------")
print(f"True MP (0)     | {cm[0, 0]:11d} | {cm[0, 1]:11d}")
print(f"True MR (1)     | {cm[1, 0]:11d} | {cm[1, 1]:11d}")

# Plotting Confusion Matrix
print("\nPlotting Confusion Matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['MP (Fe/H < -2)', 'MR (Fe/H >= -2)'])
# Format with comma separators thousands
disp.plot(cmap='Blues', ax=ax, values_format=',') 

# Apply logarithmic color scale due to massive class imbalance
from matplotlib.colors import LogNorm
disp.im_.set_norm(LogNorm(vmin=max(cm.min(), 1), vmax=cm.max()))

plt.title(f'Overall Evaluation (Accuracy: {acc:.2%})')
plt.tight_layout()
out_file = 'confusion_matrix_all_data_ebv.png' if args.use_ebv else 'confusion_matrix_all_data.png'
plt.savefig(out_file, dpi=300)
print(f"Saved confusion matrix plot to {out_file}.")
