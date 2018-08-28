import numba
import numpy as np

from pgfa.math_utils import discrete_rvs, log_normalize


@numba.jit
def do_collaped_gibbs_update(marginal_density, a, b, cols, row, X, Z):
    log_p = np.zeros(2)

    for k in cols:
        Z[row, k] = 0

        log_p[0] = np.log(b[k]) + marginal_density.log_p(X, Z)

        Z[row, k] = 1

        log_p[1] = np.log(a[k]) + marginal_density.log_p(X, Z)

        log_p = log_normalize(log_p)

        Z[row, k] = discrete_rvs(np.exp(log_p))

    return Z[row]
