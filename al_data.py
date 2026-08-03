"""Data loading and shared runtime helpers for active-learning experiments."""

import re
import time

import h5py
import numpy as np


MP_LABEL = 0
MR_LABEL = 1
CANONICAL_LABEL_ENCODING = {
    "MP": MP_LABEL,
    "MR": MR_LABEL,
    "positive_metric_class": "MP",
}


def labels_from_feh(feh, feh_threshold=-2.0):
    """Return canonical binary labels: MP=0 for Fe/H<threshold, MR=1 otherwise."""
    feh = np.asarray(feh)
    return np.where(feh < feh_threshold, MP_LABEL, MR_LABEL).astype(np.int32)


def mp_target(y):
    """Return a 0/1 metric target whose positive class is metal poor."""
    return (np.asarray(y) == MP_LABEL).astype(np.int32)


def probability_for_label(clf, X, label):
    """Return a class probability by label instead of assuming a column index."""
    probabilities = np.asarray(clf.predict_proba(X), dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(
            f"predict_proba must return a 2-D array, got shape {probabilities.shape}."
        )

    classes = getattr(clf, "classes_", None)
    if classes is None:
        if probabilities.shape[1] != 2:
            raise ValueError(
                "A classifier without classes_ must expose exactly two canonical "
                "probability columns [MP, MR]."
            )
        classes = np.array([MP_LABEL, MR_LABEL], dtype=np.int32)
    classes = np.asarray(classes)
    matches = np.flatnonzero(classes == label)
    if len(matches) != 1:
        raise ValueError(f"Classifier classes_={classes.tolist()} has no unique label {label!r}.")
    return probabilities[:, int(matches[0])]


def mp_probability(clf, X):
    """Return P(MP) under the canonical MP=0 convention."""
    return probability_for_label(clf, X, MP_LABEL)


def _configure_torch_runtime():
    """Enable fast CUDA matmul settings when PyTorch/CUDA are available."""
    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        print("  [Torch] CUDA detected; TF32 matmul enabled where supported.")
    except Exception as exc:
        print(f"  [Torch] Warning: CUDA runtime tuning skipped: {exc}")


def _timing(label, start_time):
    """Print elapsed wall time for a pipeline stage."""
    print(f"  [Timing] {label}: {time.perf_counter() - start_time:.2f}s")


def _json_scalar(value):
    """Convert numpy/HDF5 scalars to plain JSON-safe Python values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def _nsort(s):
    """Natural sort key for strings like 'bp_2', 'bp_10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]


def _feature_cols(h5_keys):
    """Return ordered feature column names from an h5 key list."""
    bp = sorted([k for k in h5_keys if k.startswith("bp_")], key=_nsort)
    rp = sorted([k for k in h5_keys if k.startswith("rp_")], key=_nsort)
    cols = bp + rp
    if "ebv" in h5_keys:
        cols.append("ebv")
    return cols


def load_features_and_labels(h5_path, feh_threshold=-2.0, max_samples=None, seed=42):
    """Load BP/RP + ebv features and binary Fe/H label from an h5 file.

    Returns (X, y, source_ids) where X is L2-normalised, y ∈ {0=MP, 1=MR}.

    Optimisations over the original:
      - Uses float32 instead of float64 (halves memory, speeds up compute).
      - Reads HDF5 slices directly via sorted fancy indexing instead of
        loading entire columns then sub-indexing.
    """
    with h5py.File(h5_path, "r") as f:
        cols = _feature_cols(list(f.keys()))
        n = f[cols[0]].shape[0]

        # Optional subsample
        if max_samples is not None and max_samples < n:
            idx = np.sort(np.random.RandomState(seed).choice(n, max_samples, replace=False))
        else:
            idx = None  # read everything – use slice for speed

        # --- Fast column read ---------------------------------------------------
        # When idx is None we read the whole dataset at once (contiguous I/O).
        # When idx is a sorted int array HDF5 can still do a relatively fast
        # fancy-index read without materialising the full column first.
        if idx is None:
            parts = [np.nan_to_num(f[c][()], nan=0.0).astype(np.float32) for c in cols]
            feh = f["feh"][()].astype(np.float32)
            sids = f["source_id"][()] if "source_id" in f else None
        else:
            parts = [np.nan_to_num(f[c][idx], nan=0.0).astype(np.float32) for c in cols]
            feh = f["feh"][idx].astype(np.float32)
            sids = f["source_id"][idx] if "source_id" in f else None

        X = np.column_stack(parts)
        del parts  # free intermediate list

    valid = np.isfinite(feh)
    if not valid.all():
        X, feh = X[valid], feh[valid]
        if sids is not None:
            sids = sids[valid]

    # L2-normalise spectral coefficients (everything except ebv)
    end = -1 if cols[-1] == "ebv" else X.shape[1]
    norms = np.linalg.norm(X[:, :end], axis=1, keepdims=True) + 1e-8
    X[:, :end] /= norms

    y = labels_from_feh(feh, feh_threshold)
    return X, y, sids
