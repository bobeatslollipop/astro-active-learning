from find_Set import *
from classifier import *
import os
from PIL import Image

IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

def iterative_training(s, t, n, steps=8, new_points_per_step=5):
    T = np.random.randn(n, t)

    Q, _ = np.linalg.qr(np.random.randn(n, n))
    n_dirs, scale = 1, 0.001
    Cov = np.eye(n) + sum((scale - 1) * np.outer(Q[:, i], Q[:, i]) for i in range(n_dirs))
    S = np.linalg.cholesky(Cov) @ np.random.randn(n, s)

    S = S[:, np.abs(S[0, :]) > 0.1]
    T = T[:, np.abs(T[0, :]) > 0.1]

    labels_T = np.zeros(T.shape[1])
    labels_T[T[0, :] > 0] = 1

    current_S = S.copy()
    accuracies, n_points, frame_paths = [], [], []

    for step in range(steps):
        labels_S = np.zeros(current_S.shape[1])
        labels_S[current_S[0, :] > 0] = 1

        clf = train(current_S.T, labels_S)
        acc = clf.score(T.T, labels_T)
        accuracies.append(acc); n_points.append(current_S.shape[1])

        path = os.path.join(IMG_DIR, f"step_{step+1:02d}_n={current_S.shape[1]}.png")
        print(f"\n--- Step {step+1} | n={current_S.shape[1]} | acc={acc:.3f} ---")
        plot_test(clf, current_S.T, labels_S, T.T, labels_T, savepath=path, show=False)
        frame_paths.append(path)

        current_S = find_Set(current_S, T, new_points_per_step)

    # final
    labels_S = np.zeros(current_S.shape[1])
    labels_S[current_S[0, :] > 0] = 1
    clf = train(current_S.T, labels_S)
    acc = clf.score(T.T, labels_T)
    accuracies.append(acc); n_points.append(current_S.shape[1])
    path = os.path.join(IMG_DIR, f"step_final_n={current_S.shape[1]}.png")
    print(f"\n--- Final | n={current_S.shape[1]} | acc={acc:.3f} ---")
    plot_test(clf, current_S.T, labels_S, T.T, labels_T, savepath=path, show=False)
    frame_paths.append(path)

    # accuracy curve
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#F5F0E8")
    ax.set_facecolor("#F5F0E8")
    ax.plot(n_points, accuracies, color="#E07070", linewidth=2, marker="o",
            markersize=7, markeredgecolor="#2A1A0A", markeredgewidth=1)
    ax.set_xlabel("Number of training points", color="#3A2A1A")
    ax.set_ylabel("Test accuracy", color="#3A2A1A")
    ax.set_title("Test accuracy vs training set size", fontsize=13, fontweight="bold", color="#3A2A1A")
    ax.grid(color="#E0D8CC"); ax.tick_params(colors="#5A4A3A")
    for spine in ax.spines.values(): spine.set_edgecolor("#C0B090")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "accuracy_curve.png"), dpi=150); plt.close()

    # gif
    frames = [Image.open(p) for p in frame_paths]
    gif_path = os.path.join(IMG_DIR, "training.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=800, loop=0)
    print(f"\nGIF saved to {gif_path}")

if __name__ == "__main__":
    iterative_training(s=100, t=1000, n=2, steps=8, new_points_per_step=4)