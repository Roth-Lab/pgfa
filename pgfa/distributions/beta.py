import numba
import numpy as np

from pgfa.distributions.base import Distribution
from pgfa.math_utils import log_beta, psi


class BetaDistribution(Distribution):
    data_dim = 1

    params_dim = 2

    def get_grad_log_p_wrt_data_fn(self):
        return grad_log_beta_pdf_wrt_data

    def get_grad_log_p_wrt_params_fn(self):
        return grad_log_beta_pdf_wrt_params

    def get_log_p_fn(self):
        return log_beta_pdf

    def rvs(self, params, size=None):
        a, b = unpack_beta_params(params)

        x = self._get_data(np.random.beta(a, b, size=size))

        if size is not None:
            x = x.reshape((size, self.data_dim))

        return x

    def _get_default_params(self):
        return np.array([1.0, 1.0])


@numba.jit(nopython=True)
def grad_log_beta_pdf_wrt_data(x, params):
    x = unpack_beta_data(x)

    if not data_is_valid(x):
        return np.ones((1, )) * np.nan

    a, b = unpack_beta_params(params)

    grad = np.empty((1,))

    grad[0] = ((a - 1) / x) - ((b - 1) / (1 - x))

    return grad


@numba.jit(nopython=True)
def grad_log_beta_pdf_wrt_params(x, params):
    x = unpack_beta_data(x)

    if not data_is_valid(x):
        return np.ones((2, )) * np.nan

    a, b = unpack_beta_params(params)

    grad = np.empty((2,))

    grad_norm = psi(a + b)

    grad[0] = grad_norm - psi(a) + np.log(x)

    grad[1] = grad_norm - psi(b) + np.log(1 - x)

    return grad


@numba.jit(nopython=True)
def log_beta_pdf(x, params):
    x = unpack_beta_data(x)

    if not data_is_valid(x):
        return -np.inf

    a, b = unpack_beta_params(params)

    return -log_beta(a, b) + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x)


@numba.jit(nopython=True)
def data_is_valid(x):
    return (x > 0) and (x < 1)


@numba.jit(nopython=True)
def unpack_beta_data(x):
    return x[0]


@numba.jit(nopython=True)
def unpack_beta_params(params):
    a = params[0]

    b = params[1]

    return a, b
