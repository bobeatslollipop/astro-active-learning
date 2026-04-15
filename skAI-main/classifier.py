import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

BG = "#F5F0E8"        # beige
GRID = "#E0D8CC"      # slightly darker beige for grid
PASTELS = ["#E07070", "#E09A50", "#5A9E7A", "#6A7EC2", "#C26A8A", "#8A6AC2"]

def train(X, y):
    clf = SVC(kernel="linear", C=1e6, max_iter=100_000).fit(X, y)
    print("Accuracy:", clf.score(X, y))
    return clf

def get_limits(X2, margin=1):
    lim_min = min(X2[:,0].min(), X2[:,1].min()) - margin
    lim_max = max(X2[:,0].max(), X2[:,1].max()) + margin
    return lim_min, lim_max

def _scatter(X2, y, s=70, marker="o", edge="#A09080", lw=0.8, suffix=""):
    for i, cls in enumerate(np.unique(y)):
        mask = y == cls
        plt.scatter(X2[mask,0], X2[mask,1], color=PASTELS[i % len(PASTELS)],
                    edgecolors=edge, linewidths=lw, s=s, marker=marker,
                    label=f"Class {cls}{suffix}", zorder=3)

def _draw_line(clf, lim_min, lim_max):
    w = clf.coef_[0, :2]
    b = clf.intercept_[0]
    x_vals = np.linspace(lim_min, lim_max, 100)
    plt.plot(x_vals, -(w[0] * x_vals + b) / w[1],
             color="#5A4A3A", linewidth=2, linestyle="--", label="Decision boundary", zorder=4)

def _style_ax(lim_min, lim_max, title):
    ax = plt.gca()
    ax.set_facecolor(BG)
    ax.grid(color=GRID, linewidth=1)
    ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.tick_params(colors="#5A4A3A")
    for spine in ax.spines.values():
        spine.set_edgecolor("#C0B090")
    plt.xlabel("Feature 1", color="#3A2A1A")
    plt.ylabel("Feature 2", color="#3A2A1A")
    plt.title(title, fontsize=13, fontweight="bold", color="#3A2A1A")
    plt.legend(framealpha=0.9, facecolor=BG, edgecolor="#C0B090", labelcolor="#3A2A1A")
    plt.tight_layout()

def plot(clf, X, y):
    X2 = X[:, :2]
    lim_min, lim_max = get_limits(X2)
    fig = plt.figure(figsize=(7, 7), facecolor=BG)
    _scatter(X2, y)
    _draw_line(clf, lim_min, lim_max)
    _style_ax(lim_min, lim_max, "Linear SVM — separating line")
    plt.savefig("classifier.png", dpi=150); plt.show()

def plot_test(clf, S, y_train, T, y_test, savepath=None, show=True):
    S2, T2 = S[:, :2], T[:, :2]
    lim_min, lim_max = get_limits(np.vstack([S2, T2]))
    fig = plt.figure(figsize=(7, 7), facecolor=BG)
    _scatter(T2, y_test, s=70, suffix=" (test)")
    _scatter(S2, y_train, s=350, marker="*", edge="#2A1A0A", lw=1.5, suffix=" (train)")
    _draw_line(clf, lim_min, lim_max)
    _style_ax(lim_min, lim_max, "Test set with highlighted training points")
    plt.savefig(savepath if savepath else "classifier_test.png", dpi=150)
    if show: plt.show()
    else: plt.close()

BG = "#F5F0E8"  # redeclare after functions so it's available at module level too

# X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
#                             n_classes=3, n_clusters_per_class=1, random_state=42)
# clf = train(X, y)
# plot(clf, X, y)