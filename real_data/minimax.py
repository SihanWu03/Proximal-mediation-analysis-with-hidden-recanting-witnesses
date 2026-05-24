import numpy as np
from numba import njit

@njit
def kkt_solve(kh1, kf0, kf1, kf2, kh, kf, g1, g2, lambda1, lambda2):
    """
    Core solver for minimax optimization.
    """
    n = kh.shape[0]
    kkt_matrix = np.zeros((2 * n, 2 * n))
    kkt_matrix[:n, :n] = 2 * lambda1 * kh
    kkt_matrix[:n, n : 2 * n] = np.ascontiguousarray(kh1.T).dot(
        np.ascontiguousarray((g1 * kf1.T).T)
    )
    kkt_matrix[n : 2 * n, :n] = kkt_matrix[:n, n : 2 * n].T
    kkt_matrix[n : 2 * n, n : 2 * n] = -2 * (kf2.T.dot(kf2) + lambda2 * kf)

    kkt_vec = np.zeros(2 * n)
    kkt_vec[n : 2 * n] = -(kf0.T.dot(np.ascontiguousarray(g2)))

    jitter = 1e-8
    for i in range(2 * n):
        kkt_matrix[i, i] += jitter


    sol = np.linalg.solve(kkt_matrix, kkt_vec)

    alpha, beta = sol[:n], sol[n : 2 * n]
    return alpha, beta


@njit
def score_nuisance_function(h1, kf0, kf1, kf2, kf, g1, g2, lambda2):
    """
    Solver for scoring the nuisance function.
    """
    kkt_matrix = kf2.T.dot(kf2) + lambda2 * kf
    kkt_vector = (kf0.T.dot(g2) + kf1.T.dot(g1 * h1)) * 0.5


    n = kkt_matrix.shape[0]
    jitter = 1e-8
    for i in range(n):
        kkt_matrix[i, i] += jitter


    beta = np.linalg.solve(kkt_matrix, kkt_vector)

    f_values_0 = kf0.dot(beta)
    f_values_1 = kf1.dot(beta)
    f_values_2 = kf2.dot(beta)

    metric = (
        np.mean(g1 * h1 * f_values_1)
        + np.mean(g2 * f_values_0)
        - np.mean(f_values_2**2)
    )
    return metric