import numba
import numpy as np

from pgfa.distributions.base import Distribution


class ProductDistribution(Distribution):
    def __init__(self, base_dist, dim):
        self.base_dist = base_dist

        self.dim = dim

        self.data_dim = dim * base_dist.data_dim

        self.params_dim = dim * base_dist.params_dim

        self._grad_log_p_wrt_data_fn = get_product_grad_log_p_wrt_data_fn(self)

        self._grad_log_p_wrt_params_fn = get_product_grad_log_p_wrt_params_fn(self)

        self._log_p_fn = get_log_p_fn(self)

    def get_grad_log_p_wrt_data_fn(self):
        if self._grad_log_p_wrt_data_fn is None:
            raise NotImplementedError()

        return self._grad_log_p_wrt_data_fn

    def get_grad_log_p_wrt_params_fn(self):
        if self._grad_log_p_wrt_params_fn is None:
            raise NotImplementedError()

        return self._grad_log_p_wrt_params_fn

    def get_log_p_fn(self):
        return self._log_p_fn

    def rvs(self, params, size=None):
        params = self._get_standard_params(params)

        D = self.base_dist.data_dim

        K = self._get_num_components(params)

        if size is None:
            size = 1

        result = np.zeros((size, K, D))

        for k in range(K):
            result[:, k] = self.base_dist.rvs(params[k], size=size)

        result = result.reshape((size, K * D))

        return result

    def _get_standard_data_params(self, data, params):
        """ Return array with shape KxNxD.
        """
        params = self._get_standard_params(params)

        data = np.atleast_2d(data)

        if data.ndim == 2:
            D = self.base_dist.data_dim

            K = self._get_num_components(params)

            N = data.shape[0]

            if data.ndim == 1:
                assert D == 1

            data = data.reshape((N, K, D))

            data = np.swapaxes(data, 0, 1)

        return data, params

    def _get_num_components(self, params):
        if params.ndim == 2:
            num_components = params.shape[1] // self.base_dist.params_dim

        else:
            num_components = params.shape[0]

        return num_components

    def _get_standard_params(self, params):
        """ Return Kx1xD array of parameters.
        """
        params = np.atleast_2d(params)

        if params.ndim == 2:
            D = self.base_dist.params_dim

            num_components = self._get_num_components(params)

            params = params.reshape((num_components, 1, D))

        return params


def get_product_grad_log_p_wrt_data_fn(dist):
    @numba.jit(nopython=True)
    def grad_log_p_wrt_data_fn(x, params):
        x = x.reshape((dim, data_dim))

        params = params.reshape((dim, params_dim))

        grad = np.zeros((dim, data_dim))

        for d in range(dim):
            grad[d] = dist_grad_log_p_fn(x[d], params[d])

        grad = grad.flatten()

        return grad

    dim = dist.dim

    data_dim = dist.base_dist.data_dim

    params_dim = dist.base_dist.params_dim

    try:
        dist_grad_log_p_fn = dist.base_dist.get_grad_log_p_wrt_data_fn()

    except NotImplementedError:
        return None

    return grad_log_p_wrt_data_fn


def get_product_grad_log_p_wrt_params_fn(dist):
    @numba.jit(nopython=True)
    def grad_log_p_wrt_params_fn(x, params):
        x = x.reshape((dim, data_dim))

        params = params.reshape((dim, params_dim))

        grad = np.zeros((dim, params_dim))

        for d in range(dim):
            grad[d] = dist_grad_log_p_fn(x[d], params[d])

        grad = grad.flatten()

        return grad

    dim = dist.dim

    data_dim = dist.base_dist.data_dim

    params_dim = dist.base_dist.params_dim

    try:
        dist_grad_log_p_fn = dist.base_dist.get_grad_log_p_wrt_params_fn()

    except NotImplementedError:
        return None

    return grad_log_p_wrt_params_fn


def get_log_p_fn(dist):
    @numba.jit(nopython=True)
    def log_p_fn(x, params):
        x = x.reshape((dim, data_dim))

        params = params.reshape((dim, params_dim))

        log_p = 0

        for d in range(dim):
            log_p += dist_log_p_fn(x[d], params[d])

        return log_p

    dim = dist.dim

    data_dim = dist.base_dist.data_dim

    params_dim = dist.base_dist.params_dim

    dist_log_p_fn = dist.base_dist.get_log_p_fn()

    return log_p_fn


def log_product_pdf(f, dim):
    @numba.jit(nopython=True)
    def log_p_fn(x, params):
        log_p = 0

        for d in range(dim):
            log_p += f(x[:, d], params[:, d])

        return log_p

    return log_p_fn
