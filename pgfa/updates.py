import itertools
import numba
import numpy as np

from pgfa.math_utils import log_normalize, log_sum_exp
from pgfa.smc import resample
from pgfa.stats import discrete_rvs


# @numba.jit
def do_collaped_gibbs_update(density, a, b, cols, row, X, Z):
    log_p = np.zeros(2)

    for k in cols:
        Z[row, k] = 0

        log_p[0] = np.log(b[k]) + density.log_p(X, Z)

        Z[row, k] = 1

        log_p[1] = np.log(a[k]) + density.log_p(X, Z)

        log_p = log_normalize(log_p)

        Z[row, k] = discrete_rvs(np.exp(log_p))

    return Z[row]


def do_collapsed_row_gibbs_update(density, a, b, cols, row, X, Z):
    Zs = list(map(np.array, itertools.product([0, 1], repeat=len(cols))))

    Zs = np.array(Zs, dtype=np.int)

    return _do_collapsed_row_gibbs_update(density, a, b, cols, row, X, Z, Zs)


# @numba.jit(nopython=True)
def _do_collapsed_row_gibbs_update(density, a, b, cols, row, X, Z, Zs):
    log_a = np.log(a[cols])

    log_b = np.log(b[cols])

    log_p = np.zeros(len(Zs))

    for idx in range(len(Zs)):
        Z[row, cols] = Zs[idx]

        log_p[idx] = np.sum(Zs[idx] * log_a) + np.sum((1 - Zs[idx]) * log_b) + density.log_p(X, Z)

    log_p = log_normalize(log_p)

    idx = discrete_rvs(np.exp(log_p))

    Z[row, cols] = Zs[idx]

    return Z[row]


@numba.jit
def do_gibbs_update(density, a, b, cols, x, z, V):
    log_p = np.zeros(2)

    for k in cols:
        z[k] = 0

        log_p[0] = np.log(b[k]) + density.log_p(x, z, V)

        z[k] = 1

        log_p[1] = np.log(a[k]) + density.log_p(x, z, V)

        log_p = log_normalize(log_p)

        z[k] = discrete_rvs(np.exp(log_p))

    return z


@numba.jit
def do_particle_gibbs_update(density, a, b, cols, x, z, V, annealed=True, num_particles=10, resample_threshold=0.5):
    T = len(cols)

    log_p = np.zeros(num_particles)

    log_W = np.zeros(num_particles)

    particles = np.zeros((num_particles, T), dtype=np.int64)

    z_test = z.copy()

    z_test[cols] = 0

    for t in range(T):
        particles[0, t] = z[cols[t]]

        log_W = log_normalize(log_W)

        log_p_old = log_p

        if t > 0:
            log_W, particles = resample(log_W, particles, conditional=True, threshold=resample_threshold)

        for i in range(num_particles):
            if i == 0:
                idx = particles[0, t]

            else:
                idx = -1

            z_test[cols[:t]] = particles[i, :t]

            if annealed:
                particles[i, t], log_p[i], log_norm = propose_annealed(
                    cols[:(t + 1)], density, a, b, x, z_test, V, T, idx=idx
                )

            else:
                particles[i, t], log_p[i], log_norm = propose(
                    cols[:(t + 1)], density, a, b, x, z_test, V, idx=idx
                )

            log_w = log_norm - log_p_old[i]

            log_W[i] = log_W[i] + log_w

    log_W = log_normalize(log_W)

    W = np.exp(log_W)

    idx = discrete_rvs(W)

    z[cols] = particles[idx]

    return z


@numba.jit(nopython=True)
def log_target_pdf_annealed(cols, density, a, b, x, z, V, T):
    t = len(cols)
    log_p = 0
    log_p += np.sum(z[cols] * np.log(a[cols]))
    log_p += np.sum((1 - z[cols]) * np.log(b[cols]))
    if t > 1:
        log_p += ((t - 1) / (T - 1)) * density.log_p(x, z, V)
    return log_p


@numba.jit
def propose_annealed(cols, density, a, b, x, z, V, T, idx=-1):
    log_p = np.zeros(2)

    z[cols[-1]] = 0

    log_p[0] = log_target_pdf_annealed(cols, density, a, b, x, z, V, T)

    z[cols[-1]] = 1

    log_p[1] = log_target_pdf_annealed(cols, density, a, b, x, z, V, T)

    log_norm = log_sum_exp(log_p)

    log_p = log_p - log_norm

    if idx == -1:
        p = np.exp(log_p)

        idx = discrete_rvs(p)

    idx = int(idx)

    return idx, log_p[idx], log_norm


@numba.jit
def log_target_pdf(cols, density, a, b, x, z, V):
    log_p = 0
    log_p += np.sum(z[cols] * np.log(a[cols]))
    log_p += np.sum((1 - z[cols]) * np.log(b[cols]))
    log_p += density.log_p(x, z, V)
    return log_p


@numba.jit
def propose(cols, density, a, b, x, z, V, idx=-1):
    log_p = np.zeros(2)

    z[cols[-1]] = 0

    log_p[0] = log_target_pdf(cols, density, a, b, x, z, V)

    z[cols[-1]] = 1

    log_p[1] = log_target_pdf(cols, density, a, b, x, z, V)

    log_norm = log_sum_exp(log_p)

    log_p = log_p - log_norm

    if idx == -1:
        p = np.exp(log_p)

        idx = discrete_rvs(p)

    idx = int(idx)

    return idx, log_p[idx], log_norm


def do_row_gibbs_update(density, a, b, cols, x, z, V):
    Zs = list(map(np.array, itertools.product([0, 1], repeat=len(cols))))

    Zs = np.array(Zs, dtype=np.int)

    return _do_row_gibbs_update(density, a, b, cols, x, z, V, Zs)


@numba.jit(nopython=True)
def _do_row_gibbs_update(density, a, b, cols, x, z, V, Zs):
    log_a = np.log(a[cols])

    log_b = np.log(b[cols])

    log_p = np.zeros(len(Zs))

    for idx in range(len(Zs)):
        z[cols] = Zs[idx]

        log_p[idx] = np.sum(Zs[idx] * log_a) + np.sum((1 - Zs[idx]) * log_b) + density.log_p(x, z, V)

    log_p = log_normalize(log_p)

    idx = discrete_rvs(np.exp(log_p))

    z[cols] = Zs[idx]

    return z


def get_cols(m, include_singletons=False):
    K = len(m)

    if include_singletons:
        cols = np.arange(K)

    else:
        cols = np.atleast_1d(np.squeeze(np.where(m > 0)))

    np.random.shuffle(cols)

    return cols


def get_rows(N):
    rows = np.arange(N)

    np.random.shuffle(rows)

    return rows