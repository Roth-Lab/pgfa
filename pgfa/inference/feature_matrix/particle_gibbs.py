import numba
import numpy as np

from pgfa.math_utils import get_linear_sum_params, log_normalize, log_sum_exp
from pgfa.stats import discrete_rvs


class ParticleGibbsFeatureAllocationMatrixKernel(object):
    def update(self, model):
        feat_mat = model.latent_values.copy()

        log_p_fn = model.feature_dist.get_log_p_fn()

        return update_feature_matrix_by_particle_gibbs_collapsed(
            log_p_fn,
            model.feature_weight_params,
            model.feature_params,
            model.data,
            feat_mat
        )


@numba.jit(nopython=True)
def update_feature_matrix_by_particle_gibbs_collapsed(log_p_fn, rho_priors, theta, X, Z):
    K = Z.shape[1]

    N = Z.shape[0]

    rows = np.arange(N)

    np.random.shuffle(rows)

    for n in rows:
        m = np.sum(Z, axis=0)

        m -= Z[n]

        a = m + rho_priors[0]

        b = (N - 1 - m) + rho_priors[1]

        z = Z[n]

        sigma = np.arange(K)

        np.random.shuffle(sigma)

        z = update_feature_matrix_by_particle_gibbs_collapsed_single_row(
            log_p_fn,
            a[sigma],
            b[sigma],
            theta[sigma],
            X[n],
            z[sigma]
        )

        for k in range(K):
            Z[n, sigma[k]] = z[k]

    return Z


@numba.jit(nopython=True, parallel=True)
def update_feature_matrix_by_particle_gibbs_collapsed_single_row(
        log_p_fn,
        a,
        b,
        params,
        x,
        z,
        annealed=True,
        num_particles=5):

    K = len(z)

    log_w = np.zeros((num_particles, K))

    log_W = np.zeros(num_particles)

    particles = np.zeros((num_particles, K), dtype=np.int64)

    particles[0] = get_conditional_path(z)

    for i in numba.prange(1, num_particles):
        if annealed:
            particles[i, 0] = propose_annealed(log_p_fn, a, b, params, K, x, particles[i, :1])

        else:
            particles[i, 0] = propose(log_p_fn, a, b, params, x, particles[i, :1])

    for i in numba.prange(num_particles):
        if annealed:
            log_w[i, 0] = get_log_weight_annealed(log_p_fn, a[0], b[0], params, K, x, particles[i, :1])

        else:
            log_w[i, 0] = get_log_weight(log_p_fn, a[0], b[0], params, x, particles[i, :1])

        log_W[i] = log_w[i, 0]

    for k in range(1, K):
        log_W = log_normalize(log_W)

        log_W, particles = resample(log_W, particles, conditional=True)

        for i in numba.prange(1, num_particles):
            if annealed:
                particles[i, k] = propose_annealed(log_p_fn, a, b, params, K, x, particles[i, :(k + 1)])

            else:
                particles[i, k] = propose(log_p_fn, a, b, params, x, particles[i, :(k + 1)])

        for i in numba.prange(num_particles):
            if annealed:
                log_w[i, k] = get_log_weight_annealed(log_p_fn, a[k], b[k], params, K, x, particles[i, :(k + 1)])

            else:
                log_w[i, k] = get_log_weight(log_p_fn, a[k], b[k], params, x, particles[i, :(k + 1)])

            log_W[i] = log_W[i] + log_w[i, k]

    log_W = log_normalize(log_W)

    W = np.exp(log_W)

    idx = discrete_rvs(W)

    return particles[idx]


@numba.jit(nopython=True)
def get_conditional_path(z):
    return z


@numba.jit(nopython=True)
def get_ess(log_W):
    W = np.exp(log_W)

    return 1 / np.sum(np.square(W))


@numba.jit(nopython=True)
def get_log_weight(log_p_fn, a, b, params, x, z):
    if len(z) == 1:
        f_0 = get_linear_sum_params(1 - z, params)

        f_1 = get_linear_sum_params(z, params)

        return log_sum_exp(np.array([
            log_p_fn(x, f_0) + np.log(a),
            log_p_fn(x, f_1) + np.log(b)
        ]))

    else:
        f_old = get_linear_sum_params(z[:-1], params)

        f_new = get_linear_sum_params(z, params)

        return log_sum_exp(np.array([
            log_p_fn(x, f_new) - log_p_fn(x, f_old) + np.log(a),
            np.log(b)
        ]))


@numba.jit(nopython=True)
def get_log_weight_annealed(log_p_fn, a, b, params, T, x, z):
    if len(z) == 1:
        return log_sum_exp(np.array([
            np.log(a),
            np.log(b)
        ]))

    else:
        t = len(z)

        f_old = get_linear_sum_params(z[:-1], params)

        f_new = get_linear_sum_params(z, params)

        log_p_old = log_p_fn(x, f_old)

        log_p_new = log_p_fn(x, f_new)

        if np.isinf(log_p_old):
            return -np.inf

        else:
            return log_sum_exp(np.array([
                ((t - 1) / (T - 1)) * log_p_new - ((t - 2) / (T - 1)) * log_p_old + np.log(a),
                (1 / (T - 1)) * log_p_old + np.log(b)
            ]))


@numba.jit(nopython=True)
def get_normalized_weights(log_weights):
    return log_normalize(log_weights)


@numba.jit(nopython=True)
def log_target_pdf(log_p_fn, a, b, params, x, z):
    t = len(z)

    a = a[:t]

    b = b[:t]

    f = get_linear_sum_params(z, params)

    return log_p_fn(x, f) + np.sum(z * np.log(a)) + np.sum((1 - z) * np.log(b))


@numba.jit(nopython=True)
def log_target_pdf_annealed(log_p_fn, a, b, params, T, x, z):
    t = len(z)

    a = a[:t]

    b = b[:t]

    f = get_linear_sum_params(z, params)

    log_p = log_p_fn(x, f)

    if t == 1:
        return np.sum(z * np.log(a)) + np.sum((1 - z) * np.log(b))

    else:
        return ((t - 1) / (T - 1)) * log_p + np.sum(z * np.log(a)) + np.sum((1 - z) * np.log(b))


@numba.jit(nopython=True)
def propose(log_p_fn, a, b, params, x, z):
    log_p = np.zeros(2)

    z[-1] = 0

    log_p[0] = log_target_pdf(log_p_fn, a, b, params, x, z)

    z[-1] = 1

    log_p[1] = log_target_pdf(log_p_fn, a, b, params, x, z)

    log_p = log_normalize(log_p)

    p = np.exp(log_p)

    return discrete_rvs(p)


@numba.jit(nopython=True)
def propose_annealed(log_p_fn, a, b, params, T, x, z):
    log_p = np.zeros(2)

    z[-1] = 0

    log_p[0] = log_target_pdf_annealed(log_p_fn, a, b, params, T, x, z)

    z[-1] = 1

    log_p[1] = log_target_pdf_annealed(log_p_fn, a, b, params, T, x, z)

    log_p = log_normalize(log_p)

    p = np.exp(log_p)

    return discrete_rvs(p)


@numba.jit(nopython=True)
def resample(log_W, particles, conditional=True, threshold=0.5):
    num_features = len(log_W)

    num_particles = particles.shape[0]

    if get_ess(log_W) <= threshold:
        new_particles = np.zeros(particles.shape, dtype=np.int64)

        W = np.exp(log_W)

        W = W + 1e-10

        W = W / np.sum(W)

        if conditional:
            new_particles[0] = particles[0]  # .copy()

            multiplicity = np.random.multinomial(num_particles - 1, W)

            idx = 1

        else:
            multiplicity = np.random.multinomial(num_particles, W)

            idx = 0

        for k in range(num_features):
            for _ in range(multiplicity[k]):
                new_particles[idx] = particles[k]  # .copy()

                idx += 1

        log_W = np.ones(num_particles) * (1 / num_particles)

        particles = new_particles

    return log_W, particles
