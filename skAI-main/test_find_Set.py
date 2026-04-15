import numpy as np
import matplotlib.pyplot as plt
from find_Set import *
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


def generate_points_on_circle(n_clusters, num_points):
    # Generate cluster centers on a circle
    # n_clusters = 5
    angles = np.linspace(0, 2*np.pi, n_clusters, endpoint=False)
    centers = 10 * np.column_stack([np.cos(angles), np.sin(angles)])

    # Generate points around each center
    T = np.vstack([np.random.normal(c, 0.5, (num_points//n_clusters, 2)) for c in centers])
    return T.T

def plot_points(S,T,new_S):
    plt.figure(figsize=(10, 10))

    plt.scatter(T[0, :], T[1, :],
                marker='.', s=80,           # Bigger markers
                color='crimson',            # High-contrast red
                alpha=0.4,                  # Transparency for dense overlap
                label='T', zorder=1)

    # Find columns in new_S that are NOT in S
    mask = ~np.isin(new_S.T, S.T).all(axis=1)
    exclusive_new_S = new_S[:, mask]

    plt.scatter(exclusive_new_S[0, :], exclusive_new_S[1, :],
                marker='^', s=200,          # Large & prominent
                color='limegreen',          # High-contrast green
                edgecolors='darkgreen', linewidths=1.0,
                label='New points selected', zorder=3)

    plt.scatter(S[0, :], S[1, :],
                marker='o', s=80,          # Bigger markers
                color='dodgerblue',         # High-contrast blue
                alpha=1,                  # Transparency for dense overlap
                edgecolors='darkblue', linewidths=0.6,
                label='S', zorder=2)

    plt.title('Point Sets', fontsize=16, fontweight='bold', pad=12)
    plt.xlabel('X', fontsize=13)
    plt.ylabel('Y', fontsize=13)
    plt.legend(fontsize=12, markerscale=1.4, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def different_covariance_test(s, t, n , new_points = 10):
    # T: standard normal
    T = np.random.randn(n, t)

    # S: lopsided covariance — identity but shrunk along random directions
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    # n_dirs, scale = 1, 0.05  # scale < 1 to shrink variance
    n_dirs, scale = 1, 0.00  # scale < 1 to shrink variance
    Cov = np.eye(n) + sum((scale - 1) * np.outer(Q[:, i], Q[:, i]) for i in range(n_dirs))
    S = np.linalg.cholesky(Cov) @ np.random.randn(n, s)

    new_S = find_Set(S,T, new_points)
    plot_points(S,T,new_S)

    # kmeans = KMeans(n_clusters = new_points).fit(T.T)
    kmeans = KMedoids(n_clusters = new_points).fit(T.T)
    plot_points(S, T, kmeans.cluster_centers_.T)



def cluster_test(s, t, n, new_points = 10):
    # S = np.random.normal(0,1,(n,s))
    S = generate_points_on_circle(2, s)
    T = generate_points_on_circle(4, t)

    new_S = find_Set(S,T, new_points)
    plot_points(S,T,new_S)

    kmeans = KMedoids(n_clusters = new_points).fit(T.T)
    # kmeans = KMeans(n_clusters = new_points).fit(T.T)
    plot_points(S, T, kmeans.cluster_centers_.T)



def long_tail_test(s, t, n, new_points = 10):
    S = np.random.randn(n, s)
    Z = np.random.randn(n, t)
    nu = 2
    u = np.random.chisquare(nu, size=(1, t))
    T = Z / np.sqrt(u / nu)

    new_S = find_Set(S, T, new_points)
    plot_points(S, T, new_S)

    # kmeans = KMeans(n_clusters = new_points).fit(T.T)
    kmeans = KMedoids(n_clusters = new_points).fit(T.T)
    plot_points(S, T, kmeans.cluster_centers_.T)


if __name__ == "__main__":
    cluster_test(10,1000,4)
    different_covariance_test(50,1000,2)
    long_tail_test(10,3000,4)

