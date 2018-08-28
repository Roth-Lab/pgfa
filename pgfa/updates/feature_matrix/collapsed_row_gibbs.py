import itertools
import numba
import numpy as np

from pgfa.math_utils import discrete_rvs, log_normalize


def do_collapsed_row_gibbs_update(marginal_density, a, b, cols, row, X, Z):
    Zs = list(map(np.array, itertools.product([0, 1], repeat=len(cols))))

    Zs = np.array(Zs, dtype=np.int)

    return _do_collapsed_row_gibbs_update(marginal_density, a, b, cols, row, X, Z, Zs)


@numba.jit
def _do_collapsed_row_gibbs_update(marginal_density, a, b, cols, row, X, Z, Zs):
    log_a = np.log(a[cols])

    log_b = np.log(b[cols])

    log_p = np.zeros(len(Zs))

    for idx in range(len(Zs)):
        Z[row, cols] = Zs[idx]

        log_p[idx] = np.sum(Zs[idx] * log_a) + np.sum((1 - Zs[idx]) * log_b) + marginal_density.log_p(X, Z)

    log_p = log_normalize(log_p)

    idx = discrete_rvs(np.exp(log_p))

    Z[row, cols] = Zs[idx]

    return Z[row]
