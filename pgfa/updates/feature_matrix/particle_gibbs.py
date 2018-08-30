import numba
import numpy as np

from pgfa.math_utils import discrete_rvs, log_normalize, log_sum_exp


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
            log_W, particles = _resample(log_W, particles, conditional=True, threshold=resample_threshold)

        for i in range(num_particles):
            if i == 0:
                idx = particles[0, t]

            else:
                idx = -1

            z_test[cols[:t]] = particles[i, :t]

            if annealed:
                particles[i, t], log_p[i], log_norm = _propose_annealed(
                    cols[:(t + 1)], density, a, b, x, z_test, V, T, idx=idx
                )

            else:
                particles[i, t], log_p[i], log_norm = _propose(
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
def _log_target_pdf_annealed(cols, density, a, b, x, z, V, T):
    t = len(cols)
    log_p = 0
    log_p += np.sum(z[cols] * np.log(a[cols]))
    log_p += np.sum((1 - z[cols]) * np.log(b[cols]))
    if t > 1:
        log_p += ((t - 1) / (T - 1)) * density.log_p(x, z, V)
    return log_p


@numba.jit
def _propose_annealed(cols, density, a, b, x, z, V, T, idx=-1):
    log_p = np.zeros(2)

    z[cols[-1]] = 0

    log_p[0] = _log_target_pdf_annealed(cols, density, a, b, x, z, V, T)

    z[cols[-1]] = 1

    log_p[1] = _log_target_pdf_annealed(cols, density, a, b, x, z, V, T)

    log_norm = log_sum_exp(log_p)

    log_p = log_p - log_norm

    if idx == -1:
        p = np.exp(log_p)

        idx = discrete_rvs(p)

    idx = int(idx)

    return idx, log_p[idx], log_norm


@numba.jit
def _log_target_pdf(cols, density, a, b, x, z, V):
    log_p = 0
    log_p += np.sum(z[cols] * np.log(a[cols]))
    log_p += np.sum((1 - z[cols]) * np.log(b[cols]))
    log_p += density.log_p(x, z, V)
    return log_p


@numba.jit
def _propose(cols, density, a, b, x, z, V, idx=-1):
    log_p = np.zeros(2)

    z[cols[-1]] = 0

    log_p[0] = _log_target_pdf(cols, density, a, b, x, z, V)

    z[cols[-1]] = 1

    log_p[1] = _log_target_pdf(cols, density, a, b, x, z, V)

    log_norm = log_sum_exp(log_p)

    log_p = log_p - log_norm

    if idx == -1:
        p = np.exp(log_p)

        idx = discrete_rvs(p)

    idx = int(idx)

    return idx, log_p[idx], log_norm


@numba.jit
def _get_ess(log_W):
    W = np.exp(log_W)

    return 1 / np.sum(np.square(W))


@numba.jit
def _resample(log_W, particles, conditional=True, threshold=0.5):
    num_features = len(log_W)

    num_particles = particles.shape[0]

    if (_get_ess(log_W) / num_particles) <= threshold:
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

        log_W = -np.log(num_particles) * np.ones(num_particles)

        particles = new_particles

    return log_W, particles