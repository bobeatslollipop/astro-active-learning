#!/usr/bin/env python3
"""
compare_auc_trials.py
---------------------
Read auc_trials.json (the raw data behind auc_trials.png) from multiple
experiment directories and plot all PR-AUC learning curves on the same figure
for side-by-side comparison.

Usage
-----
  # Auto-discover all sibling directories that contain auc_trials.json:
  python compare_auc_trials.py

  # Specify directories explicitly (relative or absolute paths are fine):
  python compare_auc_trials.py al_random_6k al_uncertainty_6k al_kmedianpp_6k

  # Custom output filename:
  python compare_auc_trials.py --out my_comparison.png al_random_6k al_uncertainty_6k

  # Custom legend labels (must match the number of directories):
  python compare_auc_trials.py --labels "Random" "Uncertainty" al_random_6k al_uncertainty_6k

  # Overlay individual trial lines for each experiment:
  python compare_auc_trials.py --show-trials

  # Set figure size (width x height in inches):
  python compare_auc_trials.py --figsize 14 7
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Color palette (supports up to 16 experiments; cycles if exceeded) ─────────
PALETTE = [
    "#4A90D9",  # blue
    "#E07070",  # red
    "#5A9E7A",  # green
    "#D4A24E",  # orange
    "#9B59B6",  # purple
    "#1ABC9C",  # cyan
    "#E67E22",  # dark orange
    "#2ECC71",  # bright green
    "#E74C3C",  # bright red
    "#3498DB",  # sky blue
    "#F39C12",  # yellow
    "#8E44AD",  # dark purple
    "#16A085",  # dark cyan
    "#D35400",  # brown-orange
    "#27AE60",  # dark green
    "#2980B9",  # dark blue
]

DEFAULT_FIGSIZE = (10.0, 5.8)
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 15
TITLE_FONTSIZE = 18


def natural_sort_key(path: Path) -> list:
    """Key function for natural/alphanumeric sorting of paths."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.name)]


def find_json_dirs(base_dir: Path) -> list[Path]:
    """Return all subdirectories of *base_dir* that contain auc_trials.json, sorted naturally by name."""
    found = [p.parent for p in base_dir.glob("*/auc_trials.json")]
    return sorted(found, key=natural_sort_key)


def load_experiment(directory: Path) -> dict | None:
    """
    Load auc_trials.json from *directory*.

    Returns a dict with keys:
        {
            "query_points": list[int],
            "trial_aucs":   list[list[float]],   # shape (n_trials, n_snapshots)
            "mean_auc":     np.ndarray,           # shape (n_snapshots,)
            "std_auc":      np.ndarray,           # shape (n_snapshots,)
        }
    Returns None (with a warning) if the file is missing or cannot be parsed.
    """
    json_path = directory / "auc_trials.json"
    if not json_path.exists():
        print(f"  [Warning] {json_path} not found — skipping.", file=sys.stderr)
        return None

    try:
        with open(json_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [Warning] Failed to parse {json_path}: {e} — skipping.", file=sys.stderr)
        return None

    query_points = data.get("auc_query_points", [])
    trial_aucs   = data.get("trial_aucs", [])

    if not query_points or not trial_aucs:
        print(f"  [Warning] {json_path} contains no data — skipping.", file=sys.stderr)
        return None

    # Pad shorter trials with NaN in case a trial was interrupted early
    max_len = max(len(t) for t in trial_aucs)
    padded = np.array(
        [t + [float("nan")] * (max_len - len(t)) for t in trial_aucs],
        dtype=float,
    )  # (n_trials, n_snapshots)

    # Trim query_points to match the actual snapshot count
    query_points = query_points[:max_len]

    result = {
        "query_points": query_points,
        "trial_aucs":   trial_aucs,
        "aucs_array":   padded,
        "mean_auc":     np.nanmean(padded, axis=0),
        "std_auc":      np.nanstd(padded, axis=0),
        "n_trials":     len(trial_aucs),
        "has_mp_data":  False,
    }

    # --- MP count (optional field, absent in older JSON files) ---
    trial_mp_counts = data.get("trial_mp_counts", [])
    if trial_mp_counts:
        mp_padded = np.array(
            [t + [float("nan")] * (max_len - len(t)) for t in trial_mp_counts],
            dtype=float,
        )
        result["has_mp_data"]    = True
        result["mp_counts"]      = mp_padded
        result["mean_mp_count"]  = np.nanmean(mp_padded, axis=0)
        result["std_mp_count"]   = np.nanstd(mp_padded, axis=0)

    return result


def make_label(directory: Path) -> str:
    """Generate a human-readable label from the directory name (strips the 'al_' prefix)."""
    name = directory.name
    if name.startswith("al_"):
        name = name[3:]
    # e.g. random_6k → Random 6k,  kmedianpp_hard_6k → Kmedianpp Hard 6k
    return name.replace("_", " ").title()


def plot_comparison(
    experiments: list[dict],
    labels: list[str],
    out_path: Path,
    show_trials: bool = False,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    title: str = "",
    cmap_runs: str | None = None,
) -> None:
    """Plot all experiments on the same axes with mean ± 1σ shading."""

    fig, ax = plt.subplots(figsize=figsize)

    if cmap_runs == "none":
        cmap_runs = None

    n_exps = len(experiments)
    for i, (exp, label) in enumerate(zip(experiments, labels)):
        if cmap_runs:
            val = i / max(1, n_exps - 1)
            color = plt.get_cmap(cmap_runs)(val)
        else:
            color = PALETTE[i % len(PALETTE)]
        qp    = exp["query_points"]
        mean  = exp["mean_auc"]
        std   = exp["std_auc"]
        n     = exp["n_trials"]

        # Mean curve (thick line + markers)
        ax.plot(
            qp, mean,
            "o-",
            color=color,
            lw=2.4,
            markersize=5.5,
            label=f"{label}  (n={n})",
            zorder=3,
        )

        # ±1σ confidence band
        ax.fill_between(
            qp,
            mean - std,
            mean + std,
            alpha=0.18,
            color=color,
        )

        # Optional: overlay individual trial lines in light color
        if show_trials:
            for trial_row in exp["aucs_array"]:
                ax.plot(
                    qp, trial_row,
                    "-",
                    color=color,
                    alpha=0.20,
                    lw=0.8,
                    zorder=1,
                )

    # Axis labels, title, grid
    ax.set_xlabel("Number of Queries", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("PR-AUC (MP Class)", fontsize=AXIS_LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)
    ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.85, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=TICK_FONTSIZE)

    # Print a summary table of final AUC values
    print(f"\n{'Experiment':<35} {'n_trials':>8} {'Final AUC Mean':>14} {'Final AUC Std':>13}")
    print("-" * 74)
    for label, exp in zip(labels, experiments):
        print(
            f"{label:<35} {exp['n_trials']:>8} "
            f"{exp['mean_auc'][-1]:>14.4f} {exp['std_auc'][-1]:>13.4f}"
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\n✓ Comparison plot saved to: {out_path}")


def plot_mp_count_comparison(
    experiments: list[dict],
    labels: list[str],
    out_path: Path,
    show_trials: bool = False,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    title: str = "",
    cmap_runs: str | None = None,
) -> None:
    """Plot MP count curves for all experiments that have MP data."""

    has_any = any(exp.get("has_mp_data") for exp in experiments)
    if not has_any:
        print("  [Info] No MP data found in any experiment — skipping MP plot.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    if cmap_runs == "none":
        cmap_runs = None

    n_exps = len(experiments)
    for i, (exp, label) in enumerate(zip(experiments, labels)):
        if not exp.get("has_mp_data"):
            continue
        if cmap_runs:
            val = i / max(1, n_exps - 1)
            color = plt.get_cmap(cmap_runs)(val)
        else:
            color = PALETTE[i % len(PALETTE)]
        qp    = exp["query_points"]
        mean  = exp["mean_mp_count"]
        std   = exp["std_mp_count"]
        n     = exp["n_trials"]

        ax.plot(
            qp, mean, "o-", color=color, lw=2.4, markersize=5.5,
            label=f"{label}  (n={n})", zorder=3,
        )
        ax.fill_between(qp, mean - std, mean + std, alpha=0.18, color=color)

        if show_trials:
            for trial_row in exp["mp_counts"]:
                ax.plot(qp, trial_row, "-", color=color, alpha=0.20, lw=0.8, zorder=1)

    ax.set_xlabel("Number of Queries", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Number of MP Samples in Queries", fontsize=AXIS_LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)
    ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.85, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=TICK_FONTSIZE)

    # Print summary
    print(f"\n{'Experiment':<35} {'n_trials':>8} {'Final MP Count':>14} {'Std':>8}")
    print("-" * 68)
    for label, exp in zip(labels, experiments):
        if exp.get("has_mp_data"):
            print(
                f"{label:<35} {exp['n_trials']:>8} "
                f"{exp['mean_mp_count'][-1]:>14.2f} {exp['std_mp_count'][-1]:>8.2f}"
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\n✓ MP count plot saved to: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read auc_trials.json from multiple experiment directories and plot them together.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "dirs",
        nargs="*",
        metavar="DIR",
        help="Experiment directories (each must contain auc_trials.json). "
             "If omitted, all qualifying subdirectories are auto-discovered.",
    )
    p.add_argument(
        "--labels", "-l",
        nargs="*",
        metavar="LABEL",
        help="Legend labels for each experiment (must match the number of dirs). "
             "Defaults to labels derived from directory names.",
    )
    p.add_argument(
        "--out", "-o",
        default="auc_comparison.png",
        metavar="FILE",
        help="Output image path (default: auc_comparison.png).",
    )
    p.add_argument(
        "--show-trials",
        action="store_true",
        help="Overlay individual trial lines on top of the mean curve.",
    )
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=list(DEFAULT_FIGSIZE),
        metavar=("W", "H"),
        help="Figure size in inches: width height (default: 10 5.8).",
    )
    p.add_argument(
        "--title",
        default="",
        help="Plot title. Default is empty, matching the compact poster-style layout.",
    )
    p.add_argument(
        "--base-dir",
        default=".",
        metavar="DIR",
        help="Root directory for auto-discovery mode (default: current directory).",
    )
    p.add_argument(
        "--cmap-runs",
        default="coolwarm",
        metavar="CMAP",
        help="Use a continuous Matplotlib colormap (e.g., viridis, plasma, coolwarm) "
             "to color the different experiment lines smoothly. Perfect for parameter sweeps. "
             "Use 'none' to disable and use the discrete color palette (default: coolwarm).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Determine the list of directories to load
    if args.dirs:
        directories = [Path(d) for d in args.dirs]
    else:
        base = Path(args.base_dir).resolve()
        directories = find_json_dirs(base)
        if not directories:
            print(
                f"[Error] No subdirectories containing auc_trials.json found under {base}.\n"
                "Please specify directories explicitly, e.g.:\n"
                "  python compare_auc_trials.py al_random_6k al_uncertainty_6k",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-discovered {len(directories)} experiment director(ies):")
        for d in directories:
            print(f"  {d}")

    # 2. Load data from each directory
    experiments = []
    valid_dirs  = []
    for d in directories:
        exp = load_experiment(d)
        if exp is not None:
            experiments.append(exp)
            valid_dirs.append(d)

    if not experiments:
        print("[Error] No experiment data could be loaded — exiting.", file=sys.stderr)
        sys.exit(1)

    # 3. Resolve legend labels
    if args.labels:
        if len(args.labels) != len(experiments):
            print(
                f"[Error] Number of --labels ({len(args.labels)}) does not match "
                f"number of valid experiments ({len(experiments)}).",
                file=sys.stderr,
            )
            sys.exit(1)
        labels = args.labels
    else:
        labels = [make_label(d) for d in valid_dirs]

    # 4. Plot AUC comparison
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(args.base_dir) / out_path
    plot_comparison(
        experiments=experiments,
        labels=labels,
        out_path=out_path,
        show_trials=args.show_trials,
        figsize=tuple(args.figsize),
        title=args.title,
        cmap_runs=args.cmap_runs,
    )

    # 5. Plot MP count comparison (auto-generated alongside AUC plot)
    mp_out = out_path.with_name(
        out_path.stem + "_mp_count" + out_path.suffix
    )
    plot_mp_count_comparison(
        experiments=experiments,
        labels=labels,
        out_path=mp_out,
        show_trials=args.show_trials,
        figsize=tuple(args.figsize),
        cmap_runs=args.cmap_runs,
    )


if __name__ == "__main__":
    main()
