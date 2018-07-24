import numba
import numpy as np

from pgfa.distributions.base import Distribution

import pgfa.distributions.gamma
import pgfa.distributions.normal


class NormalGammaProductDistribution(Distribution):
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

        norm_params, gamma_params = _unpack_params(params)

        t = np.random.gamma(gamma_params[0], gamma_params[1], size=size)

        x = np.random.normal(norm_params[0], 1 / norm_params[1], size=size)

        return np.column_stack([x, t])


@numba.jit(nopython=True)
def grad_log_normal_gamma_pdf_wrt_data(data, params):
    x, s = _unpack_data(data)

    norm_params, gamma_params = _unpack_params(params)

    grad = np.zeros(2)

    norm_grad = pgfa.distributions.normal.grad_log_normal_pdf_wrt_data(x, norm_params)

    gamma_grad = pgfa.distributions.gamma.grad_log_gamma_pdf_wrt_data(s, gamma_params)

    grad[0] = norm_grad[0]

    grad[1] = gamma_grad[0]

    return grad


@numba.jit(nopython=True)
def grad_log_normal_gamma_pdf_wrt_params(data, params):
    x, s = _unpack_data(data)

    norm_params, gamma_params = _unpack_params(params)

    grad = np.zeros(4)

    norm_grad = pgfa.distributions.normal.grad_log_normal_pdf_wrt_params(x, norm_params)

    gamma_grad = pgfa.distributions.gamma.grad_log_gamma_pdf_wrt_params(s, gamma_params)

    grad[0] = norm_grad[0]

    grad[1] = norm_grad[1]

    grad[2] = gamma_grad[0]

    grad[3] = gamma_grad[1]

    return grad


@numba.jit(nopython=True)
def log_normal_gamma_pdf(data, params):
    x, s = _unpack_data(data)

    norm_params, gamma_params = _unpack_params(params)

    return pgfa.distributions.normal.log_normal_pdf(x, norm_params) + \
        pgfa.distributions.gamma.log_gamma_pdf(s, gamma_params)


@numba.jit(nopython=True)
def _unpack_data(x):
    return x[0:1], x[1:]


@numba.jit(nopython=True)
def _unpack_params(params):
    return params[0:2], params[2:]
