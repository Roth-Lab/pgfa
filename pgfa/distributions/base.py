import numba
import numpy as np


class Distribution(object):

    def get_grad_log_p_wrt_data_fn(self):
        raise NotImplementedError()

    def get_grad_log_p_wrt_params_fn(self):
        raise NotImplementedError()

    def get_log_p_fn(self):
        raise NotImplementedError()

    def rvs(self, params, size=None):
        raise NotImplementedError()

    def grad_log_p_wrt_data(self, data, params, bulk_sum=True):
        data = self._get_data(data)

        params = self._get_params(params)

        f = self.get_grad_log_p_wrt_data_fn()

        if self._is_bulk(data):
            if bulk_sum:
                grad = _grad_log_p_wrt_data_bulk_sum(f, data, params)

            else:
                grad = _grad_log_p_wrt_data_bulk(f, data, params)

        else:
            grad = f(data, params)

        return np.atleast_2d(grad)

    def grad_log_p_wrt_params(self, data, params, bulk_sum=True):
        data = self._get_data(data)

        params = self._get_params(params)

        f = self.get_grad_log_p_wrt_params_fn()

        if self._is_bulk(data):
            if bulk_sum:
                grad = _grad_log_p_wrt_params_bulk_sum(f, data, params)

            else:
                grad = _grad_log_p_wrt_params_bulk(f, data, params)

        else:
            grad = f(data, params)

        return np.atleast_2d(grad)

    def log_p(self, data, params, bulk_sum=True):
        data = self._get_data(data)

        params = self._get_params(params)

        f = self.get_log_p_fn()

        if self._is_bulk(data):
            if bulk_sum:
                log_p = _log_p_bulk_sum(f, data, params)

            else:
                log_p = _log_p_bulk(f, data, params)

        else:
            log_p = f(data[0], params)

        return log_p

    def _get_data(self, data):
        return np.atleast_2d(data)

    def _get_params(self, params):
        return np.atleast_1d(np.squeeze(params))

    def _is_bulk(self, data):
        return data.shape[0] > 1


@numba.jit(nopython=True)
def _log_p_bulk(f, data, params):
    N = data.shape[0]

    result = np.zeros(N)

    for n in range(N):
        result[n] = f(data[n], params)

    return result


@numba.jit(nopython=True)
def _log_p_bulk_sum(f, data, params):
    N = data.shape[0]

    result = 0

    for n in range(N):
        result += f(data[n], params)

    return result


@numba.jit(nopython=True)
def _grad_log_p_wrt_data_bulk(f, data, params):
    D = data.shape[1]

    N = data.shape[0]

    result = np.zeros((N, D))

    for n in range(N):
        result[n] = f(data[n], params)

    return result


@numba.jit(nopython=True)
def _grad_log_p_wrt_data_bulk_sum(f, data, params):
    D = data.shape[1]

    N = data.shape[0]

    result = np.zeros((D,))

    for n in range(N):
        result += f(data[n], params)

    return result


@numba.jit(nopython=True)
def _grad_log_p_wrt_params_bulk(f, data, params):
    D = len(params)

    N = data.shape[0]

    result = np.zeros((N, D))

    for n in range(N):
        result[n] = f(data[n], params)

    return result


@numba.jit(nopython=True)
def _grad_log_p_wrt_params_bulk_sum(f, data, params):
    D = len(params)

    N = data.shape[0]

    result = np.zeros((D,))

    for n in range(N):
        result += f(data[n], params)

    return result
