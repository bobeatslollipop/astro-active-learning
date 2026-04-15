from find_Set import *
from classifier import *


def different_covariance_test(s, t, n , new_points = 20):
    # T: standard normal
    T = np.random.randn(n, t)

    # S: lopsided covariance — identity but shrunk along random directions
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    n_dirs, scale = 1, 0.001  # scale < 1 to shrink variance
    Cov = np.eye(n) + sum((scale - 1) * np.outer(Q[:, i], Q[:, i]) for i in range(n_dirs))
    S = np.linalg.cholesky(Cov) @ np.random.randn(n, s)

    S = S[:, np.abs(S[0, :]) > 0.1]
    T = T[:, np.abs(T[0, :]) > 0.1]

    new_S = find_Set(S,T, new_points)
    # plot_points(S,T,new_S)

    labels_T = np.zeros(T.shape[1])
    labels_T[T[0,:] > 0] = 1


    labels = np.zeros(S.shape[1])
    labels[S[0,:] > 0] = 1

    clf = train(S.T, labels)
    # plot(clf, S.T, labels)
    plot_test(clf, S.T, labels, T.T, labels_T)

    labels = np.zeros(new_S.shape[1])
    labels[new_S[0,:] > 0] = 1
    clf = train(new_S.T, labels)
    # plot(clf, new_S.T, labels)
    plot_test(clf, new_S.T, labels, T.T, labels_T)

if __name__ == "__main__":
    different_covariance_test(100,1000, 2, new_points = 10)