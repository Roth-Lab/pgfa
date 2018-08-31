import numpy as np

from pgfa.math_utils import do_metropolis_hastings_accept_reject


def do_collapsed_mh_singletons_update(row, density, alpha, X, Z):
    K = Z.shape[1]
    N = Z.shape[0]

    m = np.sum(Z, axis=0)

    m -= Z[row]

    cols = np.atleast_1d(np.squeeze(np.where(m > 0)))

    K_non_singleton = len(cols)

    K_new = K_non_singleton + np.random.poisson(alpha / N)

    if K_new == K:
        return Z

    Z_new = np.zeros((N, K_new), dtype=np.int64)

    Z_new[:, :K_non_singleton] = Z[:, cols]

    Z_new[row, K_non_singleton:] = 1

    log_p_old = density.log_p(X, Z)

    log_p_new = density.log_p(X, Z_new)

    if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
        Z = Z_new

    return Z
