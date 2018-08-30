import numba
import numpy as np


@numba.njit
def discrete_rvs(p):
    p = p / np.sum(p)
    P = np.cumsum(p)
    u = np.random.random_sample()
    return np.digitize(np.array(u), P).max()


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

    return Z


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
