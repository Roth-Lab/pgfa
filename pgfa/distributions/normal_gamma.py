import numba
import numpy as np

from pgfa.distributions.base import Distribution
from pgfa.math_utils import log_gamma, psi


class NormalGammaDistribution(Distribution):
    data_dim = 2

    params_dim = 4

    def get_grad_log_p_wrt_data_fn(self):
        return grad_log_normal_gamma_pdf_wrt_data

    def get_grad_log_p_wrt_params_fn(self):
        return grad_log_normal_gamma_pdf_wrt_params

    def get_log_p_fn(self):
        return log_normal_gamma_pdf

    def rvs(self, params, size=None):
        params = self._get_params(params)

        m, l, a, b = unpack_normal_gamma_params(params)

        t = np.random.gamma(a, 1 / b, size=size)

        x = np.random.normal(m, 1 / (l * t), size=size)

        return np.column_stack([x, t])


@numba.jit(nopython=True)
def grad_log_normal_gamma_pdf_wrt_data(data, params):
    x, t = unpack_normal_gamma_data(data)

    m, l, a, b = unpack_normal_gamma_params(params)

    grad = np.zeros(2)

    grad[0] = - l * t * (x - m)

    grad[1] = (a - 0.5) / t - b - 0.5 * l * (x - m) ** 2

    return grad


@numba.jit(nopython=True)
def grad_log_normal_gamma_pdf_wrt_params(data, params):
    x, t = unpack_normal_gamma_data(data)

    m, l, a, b = unpack_normal_gamma_params(params)

    grad = np.zeros(4)

    grad[0] = l * t * (x - m)

    grad[1] = 0.5 / l - 0.5 * t * (x - m) ** 2

    grad[2] = np.log(b) - psi(a) + np.log(t)

    grad[3] = a / b - t

    return grad


@numba.jit(nopython=True)
def log_normal_gamma_pdf(data, params):
    x, t = unpack_normal_gamma_data(data)

    m, l, a, b = unpack_normal_gamma_params(params)

    log_norm = a * np.log(b) + 0.5 * np.log(l) - log_gamma(a) - 0.5 * np.log(2 * np.pi)

    log_data = (a - 0.5) * np.log(t) - b * t - 0.5 * l * t * (x - m) ** 2

    return log_norm + log_data


@numba.jit(nopython=True)
def unpack_normal_gamma_data(x):
    return x[0], x[1]


@numba.jit(nopython=True)
def unpack_normal_gamma_params(params):
    return params[0], params[1], params[2], params[3]
