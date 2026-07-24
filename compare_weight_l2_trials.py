#!/usr/bin/env python3
"""Compare reweighting concentration metrics across active-learning runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_LABELS = {
    "objective_l2_norm": "Objective Weight L2 Norm ||p||_2",
    "objective_l2_sq": "Objective Weight L2 Squared ||p||_2^2",
    "effective_sample_size": "Effective Sample Size 1 / ||p||_2^2",
    "effective_sample_fraction": "Effective Sample Fraction",
    "max_mass": "Maximum Single-Point Mass",
    "top10_mass": "Top-10 Mass",
    "top100_mass": "Top-100 Mass",
}


def _load_weight_stats(directory: Path, metric: str):
    stats_path = directory / "weight_stats_trials.json"
    if stats_path.exists():
        with stats_path.open() as f:
            data = json.load(f)
    else:
        auc_path = directory / "auc_trials.json"
        with auc_path.open() as f:
            data = json.load(f)

    trial_stats = data.get("trial_weight_stats", [])
    if not trial_stats:
        raise ValueError(f"No trial_weight_stats found in {directory}")

    query_points = sorted({
        int(d["n_queries"])
        for trial in trial_stats
        for d in trial
        if "n_queries" in d and metric in d
    })
    if not query_points:
        raise ValueError(f"Metric {metric!r} not found in {directory}")

    q_to_col = {q: i for i, q in enumerate(query_points)}
    arr = np.full((len(trial_stats), len(query_points)), np.nan, dtype=float)
    for row, trial in enumerate(trial_stats):
        for d in trial:
            if "n_queries" in d and metric in d:
                arr[row, q_to_col[int(d["n_queries"])]] = float(d[metric])

    return {
        "directory": directory,
        "query_points": np.asarray(query_points, dtype=float),
        "values": arr,
        "mean": np.nanmean(arr, axis=0),
        "std": np.nanstd(arr, axis=0),
        "n_trials": arr.shape[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--metric", default="objective_l2_norm",
                        choices=sorted(METRIC_LABELS),
                        help="Weight concentration metric to plot.")
    parser.add_argument("--out", type=Path, default=Path("weight_l2_comparison.png"))
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    labels = args.labels
    if labels is not None and len(labels) != len(args.directories):
        parser.error("--labels must have the same length as directories.")
    if labels is None:
        labels = [p.name for p in args.directories]

    experiments = [_load_weight_stats(p, args.metric) for p in args.directories]
    ylabel = METRIC_LABELS[args.metric]
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(experiments)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    print(f"{'Experiment':35s} {'n_trials':>8s} {'Final Mean':>12s} {'Final Std':>10s}")
    print("-" * 70)
    for exp, label, color in zip(experiments, labels, colors):
        q = exp["query_points"]
        mean = exp["mean"]
        std = exp["std"]
        ax.plot(q, mean, "-o", lw=2.0, ms=4, color=color, label=label)
        ax.fill_between(q, mean - std, mean + std, color=color, alpha=0.16)
        print(f"{label:35s} {exp['n_trials']:8d} {mean[-1]:12.6g} {std[-1]:10.6g}")

    ax.set_xlabel("Number of Queried Points")
    ax.set_ylabel(ylabel)
    ax.set_title(args.title or f"{ylabel} vs. Query Count")
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"\nSaved plot to: {args.out}")


if __name__ == "__main__":
    main()
