import numba
import numpy as np

from pgfa.distributions.base import Distribution
from pgfa.math_utils import log_gamma, psi


class GammaDistribution(Distribution):
    data_dim = 1

    params_dim = 2

    def get_grad_log_p_wrt_data_fn(self):
        return grad_log_gamma_pdf_wrt_data

    def get_grad_log_p_wrt_params_fn(self):
        return grad_log_gamma_pdf_wrt_params

    def get_log_p_fn(self):
        return log_gamma_pdf

    def rvs(self, params, size=None):
        params = self._get_params(params)

        shape, scale = unpack_gamma_params(params)

        x = np.random.gamma(shape, scale=scale, size=size)

        x = self._get_data(x)

        if size is not None:
            x = x.reshape((size, self.data_dim))

        return x


@numba.jit(nopython=True)
def grad_log_gamma_pdf_wrt_data(x, params):
    x = unpack_gamma_data(x)

    shape, scale = unpack_gamma_params(params)

    grad = np.empty((1,))

    grad[0] = ((shape - 1) / x) - (1 / scale)

    return grad


@numba.jit(nopython=True)
def grad_log_gamma_pdf_wrt_params(x, params):
    x = unpack_gamma_data(x)

    shape, scale = unpack_gamma_params(params)

    grad = np.zeros((2,))

    grad[0] = np.log(x) - psi(shape) - np.log(scale)

    grad[1] = (1 / scale) * (x / scale - shape)

    return grad


@numba.jit(nopython=True)
def log_gamma_pdf(x, params):
    x = unpack_gamma_data(x)

    shape, scale = unpack_gamma_params(params)

    return (shape - 1) * np.log(x) - (x / scale) - log_gamma(shape) - shape * np.log(scale)


@numba.jit(nopython=True)
def unpack_gamma_data(x):
    return x[0]


@numba.jit(nopython=True)
def unpack_gamma_params(params):
    shape = params[0]

    scale = params[1]

    return shape, scale
