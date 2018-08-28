import numpy as np
import numba

from pgfa.math_utils import log_sum_exp
from pgfa.stats import discrete_rvs


@numba.jit
def get_ess(log_W):
    W = np.exp(log_W)

    return 1 / np.sum(np.square(W))


@numba.jit
def log_target_pdf(cols, density, a, b, x, z, V):
    log_a = z * np.log(a)

    log_b = (1 - z) * np.log(b)

    return np.sum(log_a[cols]) + np.sum(log_b[cols]) + density.log_p(x, z, V)


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


@numba.jit
def resample(log_W, particles, conditional=True, threshold=0.5):
    num_features = len(log_W)

    num_particles = particles.shape[0]

    if (get_ess(log_W) / num_particles) <= threshold:
        new_particles = np.zeros(particles.shape, dtype=np.int64)

        W = np.exp(log_W)

        W = W + 1e-10

        W = W / np.sum(W)

        if conditional:
            new_particles[0] = particles[0]

            multiplicity = np.random.multinomial(num_particles - 1, W)

            idx = 1

        else:
            multiplicity = np.random.multinomial(num_particles, W)

            idx = 0

        for k in range(num_features):
            for _ in range(multiplicity[k]):
                new_particles[idx] = particles[k]

                idx += 1

        log_W = np.ones(num_particles) * np.log(1 / num_particles)

        particles = new_particles

    return log_W, particles
