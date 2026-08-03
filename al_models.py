"""Classifier training, sample-weight normalization, and evaluation."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from al_data import MP_LABEL, MR_LABEL

DEFAULT_TRAIN_WEIGHT_SUM = 10_000.0


def _class_ratio_sample_weights(y, lambda_MP=1.0, sample_weight=None,
                                target_sum=None, class_balance_mode="ratio"):
    """Build final training weights, optionally locking MP/MR total ratio."""
    if target_sum is None:
        raise ValueError("target_sum must be resolved explicitly before training.")
    target_sum = float(target_sum)
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError(f"target_sum must be positive and finite, got {target_sum!r}.")

    n_MP, n_MR = int(np.sum(y == MP_LABEL)), int(np.sum(y == MR_LABEL))

    if sample_weight is not None:
        sw = np.array(sample_weight, dtype=np.float64)
    else:
        sw = np.ones(len(y), dtype=np.float64)

    if class_balance_mode == "none":
        final_w = np.array(sw, dtype=np.float64, copy=True)
        final_w[~np.isfinite(final_w)] = 0.0
        final_w = np.maximum(final_w, 0.0)
        total_w = final_w.sum()
        if total_w > 0:
            final_w *= (target_sum / total_w)
        return final_w
    if class_balance_mode != "ratio":
        raise ValueError(f"Unknown class_balance_mode: {class_balance_mode!r}.")

    final_w = np.zeros_like(sw)
    mp_mask = (y == MP_LABEL)
    mr_mask = (y == MR_LABEL)

    if n_MP > 0:
        sum_mp = sw[mp_mask].sum()
        final_w[mp_mask] = sw[mp_mask] * (lambda_MP / sum_mp) if sum_mp > 0 else lambda_MP / n_MP

    if n_MR > 0:
        sum_mr = sw[mr_mask].sum()
        final_w[mr_mask] = sw[mr_mask] * (1.0 / sum_mr) if sum_mr > 0 else 1.0 / n_MR

    total_w = final_w.sum()
    if total_w > 0:
        final_w *= (target_sum / total_w)
    return final_w


def _resolve_train_weight_target_sum(mode, fixed_sum, initial_labeled_count, current_labeled_count):
    """Resolve the training sample-weight total for the current snapshot."""
    if mode == "fixed":
        return float(fixed_sum)
    if mode == "initial_labeled":
        return float(initial_labeled_count)
    if mode == "current_labeled":
        return float(current_labeled_count)
    raise ValueError(f"Unknown train weight sum mode: {mode!r}")


def _final_weight_summary(y, final_w, target_sum, lambda_MP, *,
                          class_balance_mode="ratio", rtol=1e-6):
    """Validate and summarize final class-balanced training weights."""
    y = np.asarray(y)
    final_w = np.asarray(final_w, dtype=np.float64)
    mp_sum = float(final_w[y == MP_LABEL].sum())
    mr_sum = float(final_w[y == MR_LABEL].sum())
    total = float(final_w.sum())
    target_sum = float(target_sum)
    atol = max(1e-6, abs(target_sum) * 1e-8)
    if not np.isclose(total, target_sum, rtol=rtol, atol=atol):
        raise RuntimeError(
            f"Final training weight total {total:.12g} does not match target "
            f"{target_sum:.12g}."
        )
    if class_balance_mode == "ratio" and np.isclose(float(lambda_MP), 1.0, rtol=rtol, atol=1e-12):
        half = target_sum / 2.0
        if not np.isclose(mp_sum, half, rtol=rtol, atol=atol):
            raise RuntimeError(
                f"MP final training weight total {mp_sum:.12g} does not match "
                f"target/2 {half:.12g}."
            )
        if not np.isclose(mr_sum, half, rtol=rtol, atol=atol):
            raise RuntimeError(
                f"MR final training weight total {mr_sum:.12g} does not match "
                f"target/2 {half:.12g}."
            )
    return {
        "train_weight_target_sum": target_sum,
        "train_weight_actual_sum": total,
        "train_weight_MP_sum": mp_sum,
        "train_weight_MR_sum": mr_sum,
    }


def train_logistic(X, y, lambda_MP=1.0, C=1.0, prev_clf=None, sample_weight=None,
                   target_sum=None, class_balance_mode="ratio"):
    """Train logistic regression with guaranteed class weight totals.

    Regardless of the per-sample weights provided (e.g. Voronoi weights),
    the final training weights are rescaled in two steps:

    1. **Class-ratio lock**: MP weights are scaled to sum to lambda_MP,
       MR weights to 1.0.  Within each class, relative Voronoi weights
       are preserved.
    2. **Global normalisation**: all weights are uniformly rescaled to a
       fixed total. This makes the data-fit term comparable across snapshots,
       so C has a stable, consistent
       meaning throughout the active-learning loop (n_labeled grows
       from warm-start size to warm-start + all queries).

    If prev_clf is given, its coefficients are used to warm-start LBFGS
    so that convergence takes only a few iterations.
    """
    # Step 2: Normalise to a fixed total.
    # The sklearn objective is: sum_i(w_i * loss_i) + (1/2C)*||coef||^2
    # sklearn does NOT normalise sample_weight internally, so sum(w_i) sets
    # the scale of the data-fit term.  In active learning n_labeled grows
    # over time; if we normalised to n_labeled the fit term would grow with
    # every snapshot, making C effectively weaker and weaker.  Normalising
    # to a fixed total keeps the data-fit term comparable throughout — C then has a fixed,
    # dataset-size-independent meaning.  The class ratio and within-class
    # Voronoi corrections are unaffected (we only multiply by a scalar).
    final_w = _class_ratio_sample_weights(
        y, lambda_MP, sample_weight,
        target_sum=target_sum,
        class_balance_mode=class_balance_mode,
    )

    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                             warm_start=True)
    # Seed from previous solution so LBFGS starts near the optimum
    if prev_clf is not None:
        clf.coef_ = prev_clf.coef_.copy()
        clf.intercept_ = prev_clf.intercept_.copy()
        clf.classes_ = prev_clf.classes_.copy()
    clf.fit(X, y, sample_weight=final_w)
    return clf


class RidgeRegressionClassifier:
    """Ridge regression used as a binary classifier with MP as the low score."""

    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef, dtype=np.float64).reshape(1, -1)
        self.intercept_ = np.asarray([intercept], dtype=np.float64)
        self.classes_ = np.array([MP_LABEL, MR_LABEL], dtype=np.int32)

    def decision_function(self, X):
        return np.asarray(X) @ self.coef_.ravel() + self.intercept_[0]

    def predict(self, X):
        # y=0 is MP, y=1 is MR. The ridge target is -1 for MP and +1 for MR.
        return np.where(
            self.decision_function(X) >= 0.0, MR_LABEL, MP_LABEL
        ).astype(np.int32)

    def predict_proba(self, X):
        scores = np.clip(self.decision_function(X), -50.0, 50.0)
        p_mr = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - p_mr, p_mr])


def train_ridge_classifier(X, y, lambda_MP=1.0, alpha=1.0, sample_weight=None,
                           target_sum=None, class_balance_mode="ratio"):
    """Train weighted ridge regression on targets MP=-1, MR=+1."""
    final_w = _class_ratio_sample_weights(
        y, lambda_MP, sample_weight,
        target_sum=target_sum,
        class_balance_mode=class_balance_mode,
    )
    X = np.asarray(X, dtype=np.float64)
    target = np.where(y == MP_LABEL, -1.0, 1.0).astype(np.float64)

    X_pad = np.column_stack([X, np.ones(len(X), dtype=np.float64)])
    sqrt_w = np.sqrt(final_w)[:, None]
    Xw = X_pad * sqrt_w
    yw = target[:, None] * sqrt_w

    gram = Xw.T @ Xw
    rhs = Xw.T @ yw
    reg = max(float(alpha), 0.0) * np.eye(X_pad.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    gram += reg

    try:
        sol = np.linalg.solve(gram, rhs).ravel()
    except np.linalg.LinAlgError:
        sol = (np.linalg.pinv(gram) @ rhs).ravel()

    return RidgeRegressionClassifier(sol[:-1], sol[-1])


class XGBoostBinaryClassifier:
    """Thin adapter exposing the classifier interface used by this script."""

    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([MP_LABEL, MR_LABEL], dtype=np.int32)

    @property
    def feature_importances_(self):
        return getattr(self.model, "feature_importances_", None)

    def predict_proba(self, X):
        proba = np.asarray(self.model.predict_proba(X), dtype=np.float64)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    def decision_function(self, X):
        p_mr = np.clip(
            self.predict_proba(X)[:, MR_LABEL], 1e-12, 1.0 - 1e-12
        )
        return np.log(p_mr / (1.0 - p_mr))

    def predict(self, X):
        return np.where(
            self.predict_proba(X)[:, MR_LABEL] >= 0.5, MR_LABEL, MP_LABEL
        ).astype(np.int32)


def train_xgboost_classifier(X, y, lambda_MP=1.0, sample_weight=None, *,
                             n_estimators=400, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=1.0, gamma=0.0,
                             reg_lambda=1.0, tree_method="hist",
                             device="auto", n_jobs=-1, random_state=42,
                             target_sum=None, class_balance_mode="ratio"):
    """Train an XGBoost boosted-tree classifier, following Yao et al.'s model family.

    The paper uses XGBoost multiclass classifiers over metallicity bins.  The
    active-learning loop here has only binary MP/MR labels, so this implements
    the same boosted decision-tree family as a binary classifier while preserving
    the existing class-ratio and reweighting semantics.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "--model xgboost requires the optional 'xgboost' package. "
            "Install it in this environment, e.g. `pip install xgboost`, "
            "then rerun with --model xgboost."
        ) from exc

    final_w = _class_ratio_sample_weights(
        y, lambda_MP, sample_weight,
        target_sum=target_sum,
        class_balance_mode=class_balance_mode,
    )
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        min_child_weight=float(min_child_weight),
        gamma=float(gamma),
        reg_lambda=float(reg_lambda),
        tree_method=str(tree_method),
        n_jobs=int(n_jobs),
        random_state=int(random_state),
    )
    if device is not None and str(device).lower() != "auto":
        params["device"] = str(device)

    try:
        model = XGBClassifier(use_label_encoder=False, **params)
    except TypeError:
        model = XGBClassifier(**params)

    model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32),
              sample_weight=final_w)
    return XGBoostBinaryClassifier(model)


def evaluate(clf, X, y):
    """Return a flat dict of metrics including per-class average log-loss."""
    yp = clf.predict(X)
    labels = [MP_LABEL, MR_LABEL]
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, yp, labels=labels, zero_division=0
    )

    # Per-class average log-loss:  -mean[ y*log(p) + (1-y)*log(1-p) ] for each class
    probs = clf.predict_proba(X)  # columns: [P(class=0), P(class=1)]
    eps = 1e-15
    # For each sample: log-loss = -[y==0]*log(P(0)) - [y==1]*log(P(1))
    log_loss_per_sample = -np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0))
    mp_mask = (y == MP_LABEL)
    mr_mask = (y == MR_LABEL)
    loss_MP = float(log_loss_per_sample[mp_mask].mean()) if mp_mask.any() else 0.0
    loss_MR = float(log_loss_per_sample[mr_mask].mean()) if mr_mask.any() else 0.0

    return {
        "accuracy": float(accuracy_score(y, yp)),
        "precision_MP": float(prec[0]), "recall_MP": float(rec[0]), "f1_MP": float(f1[0]),
        "precision_MR": float(prec[1]), "recall_MR": float(rec[1]), "f1_MR": float(f1[1]),
        "loss_MP": loss_MP, "loss_MR": loss_MR,
        "confusion_matrix": confusion_matrix(y, yp, labels=labels).tolist(),
    }
