import ctypes
import numba
import numpy as np


@numba.jit(nopython=True)
def get_linear_sum_params(z, params):
    """ Return the $\sum_{k} z_{k} \times \theta_{k}$
    """
    t = len(z)

    D = params.shape[1]

    f = np.zeros((D,))

    for d in range(D):
        f[d] = np.sum(params[:t, d] * z) + 1

    return f


@numba.jit(nopython=True)
def log_beta(a, b):
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)


@numba.jit(nopython=True)
def log_factorial(x):
    return log_gamma(x + 1)


@numba.jit(nopython=True)
def log_gamma(x):
    return np.math.lgamma(x)


@numba.jit(nopython=True)
def log_sum_exp(log_X):
    max_exp = np.max(log_X)

    if np.isinf(max_exp):
        return max_exp

    total = 0

    for x in log_X:
        total += np.exp(x - max_exp)

    return np.log(total) + max_exp


@numba.jit(nopython=True)
def log_normalize(log_p):
    return log_p - log_sum_exp(log_p)


def _get_psi():
    addr = numba.extending.get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1psi")

    functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    _fn = functype(addr)

    @numba.vectorize('float64(float64)')
    def _vec_fn(x):
        return _fn(x)

    @numba.jit(nopython=True)
    def fn(x):
        return _vec_fn(x)

    return fn


psi = _get_psi()
