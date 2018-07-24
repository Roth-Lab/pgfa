import numba
import numpy as np

from pgfa.distributions.base import Distribution
from pgfa.math_utils import log_gamma


class DirichletDistribution(Distribution):
    def __init__(self, dim):
        self.data_dim = dim

        self.params_dim = dim

    def get_log_p_fn(self):
        return log_dirichlet_pdf

    def rvs(self, params, size=None):
        params = self._get_params(params)

        params = unpack_dirichlet_params(params)

        return self._get_data(np.random.dirichlet(params, size=size))


@numba.jit(nopython=True)
def log_dirichlet_pdf(x, params):
    x = unpack_dirichlet_data(x)

    params = unpack_dirichlet_params(params)

    K = len(params)

    log_p = log_gamma(np.sum(params))

    for k in range(K):
        log_p += (params[k] - 1) * np.log(x[k]) - log_gamma(params[k])

    return log_p


@numba.jit(nopython=True)
def unpack_dirichlet_data(x):
    return x


@numba.jit(nopython=True)
def unpack_dirichlet_params(params):
    return params
