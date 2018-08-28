import numba
import numpy as np

from pgfa.math_utils import discrete_rvs, log_normalize


@numba.jit
def do_gibbs_update(density, a, b, cols, x, z, V):
    log_p = np.zeros(2)

    for k in cols:
        z[k] = 0

        log_p[0] = np.log(b[k]) + density.log_p(x, z, V)

        z[k] = 1

        log_p[1] = np.log(a[k]) + density.log_p(x, z, V)

        log_p = log_normalize(log_p)

        z[k] = discrete_rvs(np.exp(log_p))

    return z
