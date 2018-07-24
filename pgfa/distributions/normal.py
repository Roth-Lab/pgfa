import numba
import numpy as np

from pgfa.distributions.base import Distribution


class NormalDistribution(Distribution):
    data_dim = 1

    params_dim = 2

    def get_grad_log_p_wrt_data_fn(self):
        return grad_log_normal_pdf_wrt_data

    def get_grad_log_p_wrt_params_fn(self):
        return grad_log_normal_pdf_wrt_params

    def get_log_p_fn(self):
        return log_normal_pdf

    def rvs(self, params, size=None):
        params = self._get_params(params)

        mean, precision = unpack_normal_params(params)

        std_dev = 1 / precision

        x = self._get_data(np.random.normal(mean, scale=std_dev, size=size))

        if size is not None:
            x = x.reshape((size, self.data_dim))

        return x


@numba.jit(nopython=True)
def grad_log_normal_pdf_wrt_data(x, params):
    x = unpack_normal_data(x)

    mean, precision = unpack_normal_params(params)

    grad = np.empty((1,))

    grad[0] = - precision * (x - mean)

    return grad


@numba.jit(nopython=True)
def grad_log_normal_pdf_wrt_params(x, params):
    x = unpack_normal_data(x)

    mean, precision = unpack_normal_params(params)

    grad = np.zeros((2,))

    grad[0] = precision * (x - mean)

    grad[1] = 0.5 * (1 / precision) - 0.5 * (x - mean)**2

    return grad


@numba.jit(nopython=True)
def log_normal_pdf(x, params):
    x = unpack_normal_data(x)

    mean, precision = unpack_normal_params(params)

    return 0.5 * np.log(precision) - 0.5 * np.log(2 * np.pi) - 0.5 * precision * (x - mean)**2


@numba.jit(nopython=True)
def unpack_normal_data(x):
    return x[0]


@numba.jit(nopython=True)
def unpack_normal_params(params):
    mean = params[0]

    precision = params[1]

    return mean, precision
