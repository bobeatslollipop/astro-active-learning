import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


def two_gaussian_blobs(w_true, n_samples=1000, oracle=None):
    from sklearn.datasets import make_blobs
    X_pool, _ = make_blobs(n_samples=n_samples, centers=[[-2, -2], [2, 2]], cluster_std=0.8)
    X_init = np.array([[-2, -2], [2, 2], [1.5, 1.5]])
    y_init = oracle(X_init)
    return X_pool, X_init, y_init

def uniform_grid(w_true, n_points=32, oracle=None):
    x1 = np.linspace(-3, 3, n_points)
    x2 = np.linspace(-3, 3, n_points)
    xx, yy = np.meshgrid(x1, x2)
    X_pool = np.c_[xx.ravel(), yy.ravel()]
    X_init = np.array([[2, 2], [-2, -2], [2, -1]])
    y_init = oracle(X_init)
    return X_pool, X_init, y_init

def wide_margin(w_true, n_samples=2000, margin=1.0, oracle=None):
    X_pool = np.random.randn(n_samples, 2)
    scores = X_pool @ w_true[:2] + w_true[2]
    X_pool = X_pool[np.abs(scores) > margin]
    X_init = np.array([[2, 1], [-2, -1], [1.5, 0]])
    y_init = oracle(X_init)
    return X_pool, X_init, y_init

def skewed_imbalanced(w_true, n_samples=1000, margin=0.2, oracle=None):
    X_pool = np.random.randn(n_samples, 2) + np.array([1.5, -1.0])
    scores = X_pool @ w_true[:2] + w_true[2]
    X_pool = X_pool[np.abs(scores) > margin]
    X_init = np.array([[3, 0], [-1, -2], [2, -1]])
    y_init = oracle(X_init)
    return X_pool, X_init, y_init

def two_moons(w_true, n_samples=50, noise=0.0, oracle=None):
    """Two interleaving half-circles, linearly separable by shifting them apart."""
    from sklearn.datasets import make_moons
    X_pool, _ = make_moons(n_samples=n_samples, noise=noise)
    # Shift moons to be linearly separable
    X_pool[_ == 0] += np.array([-1.5, 1.0])
    X_pool[_ == 1] += np.array([1.5, -1.0])
    X_init = np.array([[-2, 1.5], [2, -1.5], [-1, 0.5]])
    y_init = oracle(X_init)
    return X_pool, X_init, y_init

def diagonal_stripe(w_true, n_samples=1000, oracle=None):
    """Points sampled in a diagonal band, naturally aligned with a linear boundary."""
    t = np.random.uniform(-3, 3, n_samples)
    perp = np.random.uniform(-0.5, 0.5, n_samples)
    X_pool = np.column_stack([t + perp, t - perp])
    X_init = np.array([[2, 2], [-2, -2], [1, -1]])
    y_init = oracle(X_init)
    return X_pool, X_init, y_init

def concentric_clusters(w_true, n_samples=1000, oracle=None):
    """Multiple Gaussian blobs on each side of the boundary."""
    from sklearn.datasets import make_blobs
    centers_pos = [[ 3,  1], [ 1,  3], [ 2, -1]]
    centers_neg = [[-3, -1], [-1, -3], [-2,  1]]
    X_pos, _ = make_blobs(n_samples=n_samples // 2, centers=centers_pos, cluster_std=0.4)
    X_neg, _ = make_blobs(n_samples=n_samples // 2, centers=centers_neg, cluster_std=0.4)
    X_pool = np.vstack([X_pos, X_neg])
    X_init = np.array([[3, 1], [-3, -1], [1, 3]])
    y_init = oracle(X_init)
    return X_pool, X_init, y_init

def sparse_boundary(w_true, n_samples=1000, boundary_margin=2.0, oracle=None):
    """Most points far from boundary — very few near it."""
    X_pool = np.random.randn(n_samples, 2) * 3
    scores = X_pool @ w_true[:2] + w_true[2]
    X_pool = X_pool[np.abs(scores) > boundary_margin]
    X_init = np.array([[4, 2], [-4, -2], [3, -2]])
    y_init = oracle(X_init)
    return X_pool, X_init, y_init

def dense_boundary(w_true, n_samples=1000, boundary_margin=0.3, oracle=None):
    """Most points clustered near the boundary — hard for active learners."""
    X_pool = np.random.randn(n_samples, 2) * 0.5
    scores = X_pool @ w_true[:2] + w_true[2]
    X_pool = X_pool[np.abs(scores) < boundary_margin]
    X_init = np.array([[-10, 0.1], [-0.2, -0.1], [0.1, -0.1]])
    y_init = oracle(X_init)
    return X_pool, X_init, y_init

def anisotropic_gaussian(w_true, n_samples=1000, oracle=None):
    """Stretched Gaussians — one class elongated along x, other along y."""
    cov_pos = [[2.0, 0.0], [0.0, 0.2]]
    cov_neg = [[0.2, 0.0], [0.0, 2.0]]
    X_pos = np.random.multivariate_normal([ 2,  1], cov_pos, n_samples // 2)
    X_neg = np.random.multivariate_normal([-2, -1], cov_neg, n_samples // 2)
    X_pool = np.vstack([X_pos, X_neg])
    X_init = np.array([[2, 1], [-2, -1], [1, 0]])
    y_init = oracle(X_init)
    return X_pool, X_init, y_init

def uniform_disk(w_true, n_samples=1000, radius=3.0, oracle=None):
    """Points sampled uniformly within a circle."""
    r = radius * np.sqrt(np.random.uniform(0, 1, n_samples))
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    X_pool = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    X_init = np.array([[2, 0], [-2, 0], [0, 2]])
    y_init = oracle(X_init)
    return X_pool, X_init, y_init

def correlated_features(w_true, n_samples=1000, correlation=0.9, oracle=None):
    """Features are highly correlated — tests robustness to collinearity."""
    cov = [[1.0, correlation], [correlation, 1.0]]
    X_pos = np.random.multivariate_normal([ 2,  2], cov, n_samples // 2)
    X_neg = np.random.multivariate_normal([-2, -2], cov, n_samples // 2)
    X_pool = np.vstack([X_pos, X_neg])
    X_init = np.array([[2, 2], [-2, -2], [1, 1]])
    y_init = oracle(X_init)
    return X_pool, X_init, y_init
# -----------------------------
# Helper: add bias
# -----------------------------
def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


# -----------------------------
# Fit linear model (ridge)
# -----------------------------
def fit_ridge(X, y, lam=1e-2):
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w, A

def fit_svm(X, y, C=1e6, lam=1e-2):
    X_no_bias = X[:, :-1]

    clf = LinearSVC(C=C, loss='hinge', fit_intercept=True, max_iter=10000)
    clf.fit(X_no_bias, y)

    w = clf.coef_.flatten()
    b = clf.intercept_[0]

    # append bias to match our format
    w_full = np.append(w, b)
    A = X.T @ X + lam * np.eye(X.shape[1])
    return w_full, A


# -----------------------------
# KWIK-style learner
# -----------------------------
class KWIKActiveLinear:
    def __init__(self, X_labeled, y_labeled, X_pool, oracle, lam=1e-2, beta=1.0):
        self.X_l = add_bias(X_labeled)
        self.y_l = 2 * y_labeled - 1  # convert to {-1, +1}

        self.X_pool = add_bias(X_pool)
        self.oracle = oracle

        self.lam = lam
        self.beta = beta
        self.query_count = 0
        self.fig = None

    def predict(self, X, w):
        return np.sign(X @ w)

    def uncertainty(self, X, A_inv):
        # x^T A^{-1} x
        return np.sum(X @ A_inv * X, axis=1)

    def select_query(self, w, A_inv):
        preds = self.X_pool @ w
        uncert = np.sqrt(self.uncertainty(self.X_pool, A_inv))

        # Avoid division by zero
        uncert = np.maximum(uncert, 1e-8)

        # normalized margin (KWIK condition)
        score = np.abs(preds) / uncert

        # pick most uncertain (smallest score)
        idx = np.argmin(score)

        return idx, score

    def plot(self, w, step, final=False, lims = (-3, 3)):
        if self.fig == None:
            self.fig = plt.figure(figsize=(6, 6))
        else:
            self.fig.clf()
        # plt.figure(figsize=(6, 6))

        # split pool
        y_pool = self.oracle(self.X_pool[:, :-1])

        # pool points
        plt.scatter(
            self.X_pool[y_pool == 0, 0],
            self.X_pool[y_pool == 0, 1],
            marker='o',
            alpha=0.4,
            label='Pool 0'
        )
        plt.scatter(
            self.X_pool[y_pool == 1, 0],
            self.X_pool[y_pool == 1, 1],
            marker='o',
            alpha=0.4,
            label='Pool 1'
        )

        # labeled points
        y_lab = (self.y_l + 1) // 2
        plt.scatter(
            self.X_l[y_lab == 0, 0],
            self.X_l[y_lab == 0, 1],
            marker='*',
            s=150,
            label='Labeled 0'
        )
        plt.scatter(
            self.X_l[y_lab == 1, 0],
            self.X_l[y_lab == 1, 1],
            marker='*',
            s=150,
            label='Labeled 1'
        )

        # decision boundary
        x_vals = np.linspace(-3, 3, 100)
        if abs(w[1]) > 1e-6:
            y_vals = -(w[0] * x_vals + w[2]) / w[1]
            plt.plot(x_vals, y_vals, label='Classifier')
        else:
            x0 = -w[2] / w[0]
            plt.axvline(x=x0, label='Classifier')

        plt.title(f"Step {step}")
        plt.xlim(lims[0], lims[1])
        plt.ylim(lims[0], lims[1])
        plt.grid()
        plt.legend()
        if not final:
            plt.pause(0.1)
        else:
            plt.show()

    def step(self, plot=False):
        # fit model
        w, A = fit_svm(self.X_l, self.y_l, lam=self.lam)
        A_inv = np.linalg.inv(A)

        # pick query
        idx, score = self.select_query(w, A_inv)

        x_query = self.X_pool[idx, :-1]
        y_query = self.oracle(x_query)

        self.query_count += 1

        # update
        self.X_l = np.vstack([self.X_l, self.X_pool[idx]])
        self.y_l = np.append(self.y_l, 2 * y_query - 1)

        self.X_pool = np.delete(self.X_pool, idx, axis=0)

        if plot:
            self.plot(w, self.query_count)
        # print(w)

        return score[idx], w

    def run(self, max_queries=10, threshold=0.0, plot=True):
        for t in range(max_queries):
            if len(self.X_pool) == 0:
                break

            score, w = self.step(plot=plot)
            print(f"Query {self.query_count}: score={score:.4f}")

            if score < threshold:
                print("Stopping: confident")
                break

        # w, _ = fit_svm(self.X_l, self.y_l, lam=self.lam)
        if plot:
            self.plot(w, 0, final=True)
        return w


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    np.random.seed(1)

    # true separator
    w_true = np.array([1.5, 1, 0.5])

    def oracle(x):
        return (x @ w_true[:2] + w_true[2] > 0).astype(int)
    
    # Pick one:
    # X_pool, X_init, y_init = two_gaussian_blobs(w_true, oracle=oracle)
    # X_pool, X_init, y_init = uniform_grid(w_true, oracle=oracle)
    # X_pool, X_init, y_init = wide_margin(w_true, oracle=oracle)
    # X_pool, X_init, y_init = skewed_imbalanced(w_true, oracle=oracle)
    # X_pool, X_init, y_init = two_moons(w_true, oracle=oracle)
    # X_pool, X_init, y_init = diagonal_stripe(w_true, oracle=oracle)
    # X_pool, X_init, y_init = concentric_clusters(w_true, oracle=oracle)
    # X_pool, X_init, y_init = sparse_boundary(w_true, oracle=oracle)
    # X_pool, X_init, y_init = dense_boundary(w_true, oracle=oracle)
    # X_pool, X_init, y_init = anisotropic_gaussian(w_true, oracle=oracle)
    # X_pool, X_init, y_init = uniform_disk(w_true, oracle=oracle)
    X_pool, X_init, y_init = correlated_features(w_true, oracle=oracle)

    learner = KWIKActiveLinear(X_init, y_init, X_pool, oracle)

    w_learned = learner.run(plot=True)

    print("Learned w:", w_learned)
    print("True w:", w_true)
