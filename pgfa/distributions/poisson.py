import numba
import numpy as np

from pgfa.distributions.base import Distribution
from pgfa.math_utils import log_factorial


class PoissonDistribution(Distribution):
    data_dim = 1

    params_dim = 1

    def get_grad_log_p_wrt_params_fn(self):
        return grad_log_poisson_pdf_wrt_params

    def get_log_p_fn(self):
        return log_poisson_pdf

    def rvs(self, params=None, size=None):
        params = self._get_params(params)

        rate = unpack_poisson_params(params)

        x = self._get_data(np.random.poisson(rate, size=size))

        if size is not None:
            x = x.reshape((size, self.data_dim))

        return x


@numba.jit(nopython=True)
def grad_log_poisson_pdf_wrt_params(x, params):
    x = unpack_poisson_data(x)

    if not data_is_valid(x):
        return -np.inf

    rate = unpack_poisson_params(params)

    if rate == 0:
        return np.nan

    return x / rate - 1


@numba.jit(nopython=True)
def log_poisson_pdf(x, params):
    x = unpack_poisson_data(x)

    if not data_is_valid(x):
        return -np.inf

    rate = unpack_poisson_params(params)

    if rate == 0:
        if x == 0:
            log_p = 0

        else:
            log_p = -np.inf

    else:
        log_p = x * np.log(rate) - rate - log_factorial(x)

    return log_p


@numba.jit(nopython=True)
def data_is_valid(x):
    return (x >= 0)


@numba.jit(nopython=True)
def unpack_poisson_data(x):
    return x[0]


@numba.jit(nopython=True)
def unpack_poisson_params(params):
    rate = params[0]

    return rate
