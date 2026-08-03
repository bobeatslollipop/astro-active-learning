"""Metrics, plots, and model-summary outputs for active-learning runs."""

import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from al_data import MP_LABEL, MR_LABEL, _feature_cols, mp_probability, mp_target


def _generated_figure_path(out_dir, filename):
    """Return a local-only per-run figure path, creating its directory."""
    figure_dir = os.path.join(out_dir, "figures", "generated")
    os.makedirs(figure_dir, exist_ok=True)
    return os.path.join(figure_dir, filename)


def _record(metrics, n_queries, y_labeled):
    """Augment a metrics dict with bookkeeping fields."""
    metrics["n_queries"] = n_queries
    metrics["n_labeled"] = len(y_labeled)
    metrics["n_labeled_MP"] = int(np.sum(y_labeled == MP_LABEL))
    metrics["n_labeled_MR"] = int(np.sum(y_labeled == MR_LABEL))
    return metrics


def _log(m):
    """One-line summary of a metrics snapshot."""
    print(f"[Query {m['n_queries']:4d}] Acc={m['accuracy']:.4f}  "
          f"Loss(test MP={m['loss_MP']:.4f} MR={m['loss_MR']:.4f} "
          f"avg={m['avg_test_loss']:.4f})  "
          f"labeled={m['n_labeled']} (MP={m['n_labeled_MP']}, MR={m['n_labeled_MR']})")


# ── Plotting ─────────────────────────────────────────────


def compute_pr_auc(clf, X_eval, y_eval):
    """Compute Precision-Recall AUC for the MP class."""
    from sklearn.metrics import precision_recall_curve, auc
    y_true_mp = mp_target(y_eval)
    y_scores = mp_probability(clf, X_eval)
    precision, recall, _ = precision_recall_curve(y_true_mp, y_scores)
    precision, recall = precision[:-1], recall[:-1]
    if len(recall) < 2:
        return 0.0
    return auc(recall, precision)


def compute_average_precision(clf, X_eval, y_eval):
    """Compute sklearn average precision for the MP class."""
    from sklearn.metrics import average_precision_score
    y_true_mp = mp_target(y_eval)
    y_scores = mp_probability(clf, X_eval)
    return float(average_precision_score(y_true_mp, y_scores))


def _save_auc_trials_plot(auc_query_points, all_trial_aucs, out_dir, n_trials):
    """Plot PR-AUC across trials with confidence region (mean ± std)."""
    # Pad any short trial lists with NaN (e.g. if clf was None at a snapshot)
    max_len = max(len(t) for t in all_trial_aucs)
    padded = [t + [float('nan')] * (max_len - len(t)) for t in all_trial_aucs]
    aucs = np.array(padded, dtype=float)  # (n_trials, n_snapshots)
    mean_auc = np.nanmean(aucs, axis=0)
    std_auc  = np.nanstd(aucs, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(auc_query_points, mean_auc, 'o-', color='#4A90D9', lw=2, markersize=6,
            label='Mean PR-AUC')
    ax.fill_between(auc_query_points, mean_auc - std_auc, mean_auc + std_auc,
                    alpha=0.25, color='#4A90D9', label='±1 std')

    # Overlay individual trial lines faintly
    for t in range(n_trials):
        ax.plot(auc_query_points, aucs[t], '-', color='#999999', alpha=0.3, lw=0.8)

    ax.set_xlabel('Number of Queries', fontsize=12)
    ax.set_ylabel('PR-AUC (MP Class)', fontsize=12)
    ax.set_title(f'PR-AUC across {n_trials} Trials', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_file = _generated_figure_path(out_dir, 'auc_trials.png')
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved AUC trials plot to {out_file}")


def _save_average_precision_trials_plot(query_points, all_trial_aps, out_dir, n_trials):
    """Plot average precision across trials with confidence region (mean ± std)."""
    max_len = max(len(t) for t in all_trial_aps)
    padded = [t + [float('nan')] * (max_len - len(t)) for t in all_trial_aps]
    aps = np.array(padded, dtype=float)
    mean_ap = np.nanmean(aps, axis=0)
    std_ap = np.nanstd(aps, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(query_points, mean_ap, 'o-', color='#5A9E7A', lw=2, markersize=6,
            label='Mean AP')
    ax.fill_between(query_points, mean_ap - std_ap, mean_ap + std_ap,
                    alpha=0.25, color='#5A9E7A', label='±1 std')

    for t in range(n_trials):
        ax.plot(query_points, aps[t], '-', color='#999999', alpha=0.3, lw=0.8)

    ax.set_xlabel('Number of Queries', fontsize=12)
    ax.set_ylabel('Average Precision (MP Class)', fontsize=12)
    ax.set_title(f'Average Precision across {n_trials} Trials', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_file = _generated_figure_path(out_dir, 'average_precision_trials.png')
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved average-precision trials plot to {out_file}")


def _save_test_loss_trials_plot(eval_query_points, all_trial_test_losses, out_dir, n_trials):
    """Plot per-class test loss across trials with mean ± std (log-scale y-axis).

    Generates two plots: test_loss_MP_trials.png and test_loss_MR_trials.png.
    Each plot shows the average test loss evolution over training with variance bands.
    """
    max_len = max(len(t) for t in all_trial_test_losses)
    # Each element in all_trial_test_losses is a list of dicts
    # with keys "loss_MP", "loss_MR", "avg_test_loss"

    for class_key, class_label, color, filename in [
        ("loss_MP", "MP", "#E07070", "test_loss_MP_trials.png"),
        ("loss_MR", "MR", "#4A90D9", "test_loss_MR_trials.png"),
    ]:
        # Extract per-trial loss arrays
        trial_losses = []
        for trial_data in all_trial_test_losses:
            losses = [d[class_key] for d in trial_data]
            # Pad with NaN if needed
            losses += [float('nan')] * (max_len - len(losses))
            trial_losses.append(losses)

        arr = np.array(trial_losses, dtype=float)  # (n_trials, n_evals)
        mean_loss = np.nanmean(arr, axis=0)
        std_loss  = np.nanstd(arr, axis=0)

        # Trim eval_query_points to match
        qp = eval_query_points[:max_len]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(qp, mean_loss, 'o-', color=color, lw=2, markersize=5,
                label=f'Mean Test Loss ({class_label})')
        ax.fill_between(qp, mean_loss - std_loss, mean_loss + std_loss,
                        alpha=0.25, color=color, label='±1 std')

        # Overlay individual trial lines faintly
        for t in range(n_trials):
            ax.plot(qp, arr[t], '-', color='#999999', alpha=0.3, lw=0.8)

        ax.set_xlabel('Number of Queries', fontsize=12)
        ax.set_ylabel(f'Test Log-Loss ({class_label})', fontsize=12)
        ax.set_yscale('log')
        ax.set_title(f'{class_label} Test Loss across {n_trials} Trials',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_file = _generated_figure_path(out_dir, filename)
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        print(f"Saved {class_label} test loss trials plot to {out_file}")


def _compute_reweight_stats(sample_weight, y_labeled, n_queries):
    """Summarize concentration of reweighting weights before class-ratio scaling."""
    if sample_weight is None:
        return None

    w = np.asarray(sample_weight, dtype=np.float64).ravel()
    if len(w) == 0:
        return None

    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        return None

    p = w / total
    l2_sq = float(np.dot(p, p))
    l2_norm = float(np.sqrt(l2_sq))
    ess = float(1.0 / l2_sq) if l2_sq > 0.0 else float("inf")
    nonzero = p > 0

    def top_mass(k):
        k = min(int(k), len(p))
        if k <= 0:
            return 0.0
        if k == len(p):
            return 1.0
        return float(np.partition(p, -k)[-k:].sum())

    y_arr = np.asarray(y_labeled)
    mp_mask = y_arr == MP_LABEL
    mr_mask = y_arr == MR_LABEL

    return {
        "n_queries": int(n_queries),
        "n_labeled": int(len(w)),
        "weight_sum": total,
        "objective_l2_norm": l2_norm,
        "objective_l2_sq": l2_sq,
        "effective_sample_size": ess,
        "effective_sample_fraction": float(ess / len(w)) if len(w) else float("nan"),
        "max_mass": float(np.max(p)),
        "top10_mass": top_mass(10),
        "top100_mass": top_mass(100),
        "nonzero_count": int(np.count_nonzero(nonzero)),
        "nonzero_fraction": float(np.mean(nonzero)),
        "mp_mass": float(p[mp_mask].sum()) if len(mp_mask) == len(p) else float("nan"),
        "mr_mass": float(p[mr_mask].sum()) if len(mr_mask) == len(p) else float("nan"),
        "returned_weight_l2_norm": float(np.linalg.norm(w)),
        "returned_weight_l2_sq": float(np.dot(w, w)),
    }


def _save_weight_stats_trials_plot(all_trial_weight_stats, out_dir, n_trials):
    if not all_trial_weight_stats or not any(all_trial_weight_stats):
        return

    query_points = sorted({
        int(d["n_queries"])
        for trial_data in all_trial_weight_stats
        for d in trial_data
        if "n_queries" in d
    })
    if not query_points:
        return

    q_to_col = {q: i for i, q in enumerate(query_points)}
    metrics = [
        ("objective_l2_norm", "Objective Weight L2 Norm ||p||_2", "weight_l2_norm_trials.png"),
        ("objective_l2_sq", "Objective Weight L2 Squared ||p||_2^2", "weight_l2_sq_trials.png"),
        ("effective_sample_size", "Effective Sample Size 1 / ||p||_2^2", "weight_effective_sample_size_trials.png"),
    ]

    for metric, ylabel, filename in metrics:
        arr = np.full((len(all_trial_weight_stats), len(query_points)), np.nan, dtype=float)
        for t, trial_data in enumerate(all_trial_weight_stats):
            for d in trial_data:
                if metric in d and "n_queries" in d:
                    arr[t, q_to_col[int(d["n_queries"])]] = float(d[metric])

        if np.all(np.isnan(arr)):
            continue

        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(query_points, mean, "-o", color="#2563eb", lw=2.0,
                label=f"Mean over {n_trials} trials")
        ax.fill_between(query_points, mean - std, mean + std,
                        color="#2563eb", alpha=0.18, label="±1 std")
        for row in arr:
            ax.plot(query_points, row, "-", color="#93c5fd", alpha=0.20, lw=0.8)

        ax.set_xlabel("Number of Queried Points", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel + " vs. Query Count", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, ls="--")
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()
        out_file = _generated_figure_path(out_dir, filename)
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        print(f"Saved weight-stat trials plot to {out_file}")


def _save_mp_trials_plot(auc_query_points, all_trial_mp_counts, out_dir, n_trials):
    """Plot queried MP fraction across trials with confidence region (mean ± std)."""
    max_len = max(len(t) for t in all_trial_mp_counts)
    padded = [t + [float('nan')] * (max_len - len(t)) for t in all_trial_mp_counts]
    counts = np.array(padded, dtype=float)  # (n_trials, n_snapshots)

    # Convert cumulative MP counts to fractions: mp_count / total_queries
    queries_arr = np.array(auc_query_points[:max_len], dtype=float)
    fractions = counts / queries_arr[np.newaxis, :]

    mean_frac = np.nanmean(fractions, axis=0)
    std_frac  = np.nanstd(fractions, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(auc_query_points, mean_frac, 'o-', color='#E07070', lw=2, markersize=6,
            label='Mean MP Fraction')
    ax.fill_between(auc_query_points, mean_frac - std_frac, mean_frac + std_frac,
                    alpha=0.25, color='#E07070', label='±1 std')

    # Overlay individual trial lines faintly
    for t in range(n_trials):
        ax.plot(auc_query_points, fractions[t], '-', color='#999999', alpha=0.3, lw=0.8)

    ax.set_xlabel('Number of Queries', fontsize=12)
    ax.set_ylabel('MP Fraction in Queried Samples', fontsize=12)
    ax.set_title(f'Queried MP Fraction across {n_trials} Trials', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_file = _generated_figure_path(out_dir, 'mp_fraction_trials.png')
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved MP fraction trials plot to {out_file}")


def generate_confusion_matrix(clf, X_full, y_full, out_dir):
    from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_recall_fscore_support
    from matplotlib.colors import LogNorm

    y_pred = clf.predict(X_full)

    acc = accuracy_score(y_full, y_pred)
    labels = [MP_LABEL, MR_LABEL]
    cm = confusion_matrix(y_full, y_pred, labels=labels)

    precision, recall, _, _ = precision_recall_fscore_support(
        y_full, y_pred, labels=labels, zero_division=0
    )

    print(f"\nOverall Accuracy on all data: {acc:.4%}")
    print(f"Class MP (0): Precision = {precision[0]:.4f}, Recall = {recall[0]:.4f}")
    print(f"Class MR (1): Precision = {precision[1]:.4f}, Recall = {recall[1]:.4f}\n")

    print("Confusion Matrix:")
    print("                 | Pred MP (0) | Pred MR (1)")
    print("--------------------------------------------")
    print(f"True MP (0)     | {cm[0, 0]:11d} | {cm[0, 1]:11d}")
    print(f"True MR (1)     | {cm[1, 0]:11d} | {cm[1, 1]:11d}")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['MP (Fe/H < -2)', 'MR (Fe/H >= -2)'])
    disp.plot(cmap='Blues', ax=ax, values_format=',')
    try:
        disp.im_.set_norm(LogNorm(vmin=max(cm.min(), 1), vmax=cm.max()))
    except Exception:
        pass

    title_str = f'Overall Evaluation\nAcc: {acc:.2%}  MP(P:{precision[0]:.3f}, R:{recall[0]:.3f}) MR(P:{precision[1]:.3f}, R:{recall[1]:.3f})'
    plt.title(title_str, fontsize=11)

    plt.tight_layout()
    out_file = _generated_figure_path(out_dir, 'confusion_matrix_all_data.png')
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix plot to {out_file}.")


def generate_pr_curve(clf_list, X_full, y_full, out_dir):
    """Plot one or more Precision-Recall curves on the same figure.

    Parameters
    ----------
    clf_list : list of (label, clf) tuples
        Each entry is a (human-readable label, trained classifier) pair.
        E.g. [("Halfway (2500 queries)", clf_half), ("Final (5000 queries)", clf_final)].
    """
    from sklearn.metrics import precision_recall_curve, auc

    colors = ['#E07070', '#4A90D9', '#5A9E7A', '#D4A24E', '#9B59B6']
    y_true_mp = mp_target(y_full)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (label, clf) in enumerate(clf_list):
        if hasattr(clf, "predict_proba"):
            y_scores = mp_probability(clf, X_full)
        else:
            y_scores = -clf.decision_function(X_full)

        precision, recall, _ = precision_recall_curve(y_true_mp, y_scores)
        # Drop the sklearn sentinel point (recall=0, precision=1) at the end
        precision, recall = precision[:-1], recall[:-1]
        pr_auc = auc(recall, precision) if len(recall) >= 2 else 0.0

        color = colors[i % len(colors)]
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{label} (AUC = {pr_auc:.3f})')

    ax.set_xlabel('Recall (MP Class)', fontsize=12)
    ax.set_ylabel('Precision (MP Class)', fontsize=12)
    ax.set_title('Precision-Recall Curve for MP Class', fontsize=14, fontweight='bold')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)

    fig.tight_layout()
    out_file = _generated_figure_path(out_dir, 'pr_curve_mp.png')
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved PR curve plot to {out_file}.")


def save_final_model_summary(clf, full_data_file, out_dir):
    """Save linear weights or tree feature importances for the final classifier."""
    with h5py.File(full_data_file, "r") as f:
        cols = _feature_cols(list(f.keys()))

    out_file = os.path.join(out_dir, "final_weights.csv")
    if hasattr(clf, "coef_") and hasattr(clf, "intercept_"):
        w, b = clf.coef_.flatten(), clf.intercept_[0]
        with open(out_file, "w") as f:
            f.write("feature,weight\n" + f"BIAS,{b}\n")
            f.writelines(f"{name},{wv}\n" for name, wv in zip(cols, w))
        return

    importances = getattr(clf, "feature_importances_", None)
    if importances is not None:
        importances = np.asarray(importances, dtype=np.float64).ravel()
        with open(out_file, "w") as f:
            f.write("feature,importance\n")
            f.writelines(f"{name},{val}\n" for name, val in zip(cols, importances))
        return

    with open(out_file, "w") as f:
        f.write("feature,value\n")
        f.write("MODEL_HAS_NO_LINEAR_WEIGHTS_OR_FEATURE_IMPORTANCES,nan\n")
