import math
import numba
import numpy as np


@numba.njit(cache=True)
def bernoulli_rvs(p):
    return discrete_rvs(np.array([1 - p, p]))


@numba.njit(cache=True)
def discrete_rvs(p):
    p = p / np.sum(p)
    return np.random.multinomial(1, p).argmax()


def discrete_rvs_gumbel_trick(log_p):
    U = np.random.gumbel(size=len(log_p))
    return np.argmax(log_p + U)


@numba.njit(cache=True)
def do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
    u = np.random.random()

    diff = (log_p_new - log_q_new) - (log_p_old - log_q_old)

    if diff >= np.log(u):
        accept = True

    else:
        accept = False

    return accept


@numba.jit(nopython=True)
def log_beta(a, b):
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)


@numba.jit(nopython=True)
def log_factorial(x):
    return log_gamma(x + 1)


@numba.njit(cache=True)
def log_binomial_coefficient(n, x):
    return log_factorial(n) - log_factorial(x) - log_factorial(n - x)


@numba.vectorize([numba.float64(numba.float64)])
def log_gamma(x):
    return np.math.lgamma(x)


@numba.jit(cache=True, nopython=True)
def log_sum_exp(log_X):
    max_exp = np.max(log_X)

    if np.isinf(max_exp):
        return max_exp

    total = 0

    for x in log_X:
        total += np.exp(x - max_exp)

    return np.log(total) + max_exp


@numba.jit(cache=True, nopython=True)
def log_normalize(log_p):
    return log_p - log_sum_exp(log_p)


@numba.njit(cache=True)
def exp_normalize(log_p):
    return np.exp(log_normalize(log_p))


def ffa_rvs(a, b, K, N):
    p = np.random.beta(a, b, size=K)

    Z = np.zeros((N, K), dtype=np.int64)

    for k in range(K):
        Z[:, k] = np.random.multinomial(1, [1 - p[k], p[k]], size=N).argmax(axis=1)

    return Z


def ibp_rvs(alpha, N):
    K = np.random.poisson(alpha)

    Z = np.ones((1, K), dtype=np.int64)

    for n in range(1, N):
        K = Z.shape[1]

        z = np.zeros(K)

        m = np.sum(Z, axis=0)

        for k in range(K):
            p = np.array([n - m[k], m[k]])

            p = p / p.sum()

            z[k] = discrete_rvs(p)

        Z = np.row_stack([Z, z])

        k_new = np.random.poisson(alpha / (n + 1))

        if k_new > 0:
            Z = np.column_stack([Z, np.zeros((Z.shape[0], k_new))])

            Z[n, K:] = 1

    return Z.astype(np.int64)


def log_ffa_pdf(a_0, b_0, Z):
    K = Z.shape[1]

    N = Z.shape[0]

    m = np.sum(Z, axis=0)

    a = a_0 + m

    b = b_0 + (N - m)

    log_p = 0

    for k in range(K):
        log_p += log_beta(a[k], b[k]) - log_beta(a_0, b_0)

    return log_p


def log_ibp_pdf(alpha, Z):
    K = Z.shape[1]

    if K == 0:
        return 0

    N = Z.shape[0]

    H = np.sum(1 / np.arange(1, N + 1))

    log_p = K * np.log(alpha) - H * alpha

    histories, history_counts = np.unique(Z, axis=1, return_counts=True)

    m = histories.sum(axis=0)

    num_histories = histories.shape[1]

    for h in range(num_histories):
        K_h = history_counts[h]

        log_p -= log_factorial(K_h)

        log_p += K_h * log_factorial(m[h] - 1) + K_h * log_factorial(N - m[h])

        log_p -= history_counts[h] * log_factorial(N)

    return log_p


@numba.jit(cache=True, nopython=True)
def cholesky_update(L, x, alpha=1, inplace=True):
    """ Rank one update of a Cholesky factorized matrix.
    """
    dim = len(x)

    x = x.copy()

    if not inplace:
        L = L.copy()

    for i in range(dim):
        r = np.sqrt(L[i, i] ** 2 + alpha * x[i] ** 2)

        c = r / L[i, i]

        s = x[i] / L[i, i]

        L[i, i] = r

        idx = i + 1

        L[idx:dim, i] = (L[idx:dim, i] + alpha * s * x[idx:dim]) / c

        x[idx:dim] = c * x[idx:dim] - s * L[idx:dim, i]

    return L


@numba.jit(cache=True, nopython=True)
def cholesky_log_det(X):
    return 2 * np.sum(np.log(np.diag(X)))


@numba.njit(cache=True)
def conditional_multinomial_resampling(log_w, num_resampled):
    W = exp_normalize(log_w)

    multiplicities = np.random.multinomial(num_resampled - 1, W)

    multiplicities[0] += 1

    indexes = []

    for i, m in enumerate(multiplicities):
        indexes.extend([i] * m)

    return indexes


@numba.njit(cache=True)
def multinomial_resampling(log_w, num_resampled):
    W = exp_normalize(log_w)

    multiplicities = np.random.multinomial(num_resampled, W)

    indexes = []

    for i, m in enumerate(multiplicities):
        indexes.extend([i] * m)

    return indexes


@numba.jit(cache=True)
def conditional_stratified_resampling(log_w, num_resampled):
    """ Perform conditional stratified resampling.

    This will enforce that 0 is always in the resampled index set.

    Parameters
    ----------
    log_w: (ndarray) Log weights of particles. Can be unnormalized.
    num_resampled: (int) Number of indexes to resample.

    Returns
    -------
    indexes: (ndarray) Indexes of resampled values.

    Reference: Algorithm 4 of On particle Gibbs sampling

    TODO: Update ref and fix num_sample being ignored
    """
    W = exp_normalize(log_w) + 1e-10

    W = W / np.sum(W)

    N = len(W)

    NW0 = N * W[0]

    U = np.zeros(N)

    if NW0 <= 1:
        U[0] = np.random.uniform(0, NW0)

    else:
        r = NW0 - np.floor(NW0)

        p = r * (np.floor(NW0) + 1) / NW0

        if bernoulli_rvs(p):
            U[0] = np.random.uniform(0, r)

        else:
            U[0] = np.random.uniform(r, 1)

    U[1:] = np.random.uniform(0, 1, size=N - 1)

    SU = (U + np.arange(N)) / N

    idxs = inverse_cdf(SU, W)

    assert len(idxs) == N

    assert idxs[0] == 0

    return idxs


@numba.njit(cache=True)
def stratified_resampling(log_w, num_resampled):
    """ Perform stratified resampling.

    Parameters
    ----------
    log_w: (ndarray) Log weights of particles. Can be unnormalized.
    num_resampled: (int) Number of indexes to resample.

    Returns
    -------
    indexes: (ndarray) Indexes of resampled values.
    """
    su = (random.rand(num_resampled) + np.arange(num_resampled)) / num_resampled
    return inverse_cdf(su, W)


@numba.njit(cache=True)
def inverse_cdf(su, W):
    """Inverse CDF algorithm for a finite distribution.
        Parameters
        ----------
        su: (M,) ndarray
            M sorted uniform variates (i.e. M ordered points in [0,1]).
        W: (N,) ndarray
            a vector of N normalized weights (>=0 and sum to one)
        Returns
        -------
        A: (M,) ndarray
            a vector of M indices in range 0, ..., N-1
    """
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, dtype=np.int64)
    for n in range(M):
        while su[n] > s:
            j += 1
            s += W[j]
        A[n] = j
    return A
