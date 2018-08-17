import numba
import numpy as np

from pgfa.math_utils import get_linear_sum_params, log_normalize, log_sum_exp
from pgfa.stats import discrete_rvs


class ParticleGibbsFeatureAllocationMatrixKernel(object):
    def __init__(self, annealed=True, num_particles=10, resample_threshold=0.5):
        self.annealed = annealed

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

    def update(self, model):
        feat_mat = model.latent_values.copy()

        log_p_fn = model.feature_dist.get_log_p_fn()

        return update_feature_matrix_by_particle_gibbs_collapsed(
            log_p_fn,
            model.feature_weight_params,
            model.feature_params,
            model.data,
            feat_mat,
            annealed=self.annealed,
            num_particles=self.num_particles,
            resample_threshold=self.resample_threshold
        )


@numba.jit(nopython=True)
def update_feature_matrix_by_particle_gibbs_collapsed(
        log_p_fn,
        rho_priors,
        theta,
        X,
        Z,
        annealed=True,
        num_particles=10,
        resample_threshold=0.5):

    K = Z.shape[1]

    N = Z.shape[0]

    rows = np.arange(N)

    np.random.shuffle(rows)

    cols = np.arange(K)

    np.random.shuffle(cols)

    for n in rows:
        m = np.sum(Z, axis=0)

        m -= Z[n]

        a = m + rho_priors[0]

        b = (N - 1 - m) + rho_priors[1]

        z = Z[n]

        z = update_feature_matrix_by_particle_gibbs_collapsed_single_row(
            log_p_fn,
            a[cols],
            b[cols],
            theta[cols],
            X[n],
            z[cols],
            annealed=annealed,
            num_particles=num_particles,
            resample_threshold=resample_threshold
        )

        for k in range(K):
            Z[n, cols[k]] = z[k]

    return Z


@numba.jit(nopython=True, parallel=False)
def update_feature_matrix_by_particle_gibbs_collapsed_single_row(
        log_p_fn,
        a,
        b,
        params,
        x,
        z,
        annealed=True,
        num_particles=10,
        resample_threshold=0.5):

    K = len(z)

    log_norm = np.zeros(num_particles)

    log_p = np.zeros(num_particles)

    log_w = np.zeros((num_particles, K))

    log_W = np.zeros(num_particles)

    particles = np.zeros((num_particles, K), dtype=np.int64)

    particles[0] = get_conditional_path(z)

    for k in range(K):
        log_W = log_normalize(log_W)

        log_p_old = log_p

        if k > 0:
            log_W, particles = resample(log_W, particles, conditional=True, threshold=resample_threshold)

        for i in numba.prange(num_particles):
            if i == 0:
                idx = particles[i, k]

            else:
                idx = -1

            if annealed:
                particles[i, k], log_p[i], log_norm[i] = propose_annealed(
                    log_p_fn, a, b, params, K, x, particles[i, :(k + 1)], idx=idx
                )

            else:
                particles[i, k], log_p[i], log_norm[i] = propose(
                    log_p_fn, a, b, params, x, particles[i, :(k + 1)], idx=idx
                )

            log_w[i, k] = log_norm[i] - log_p_old[i]

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
def propose(log_p_fn, a, b, params, x, z, idx=-1):
    log_p = np.zeros(2)

    z[-1] = 0

    log_p[0] = log_target_pdf(log_p_fn, a, b, params, x, z)

    z[-1] = 1

    log_p[1] = log_target_pdf(log_p_fn, a, b, params, x, z)

    log_norm = log_sum_exp(log_p)

    log_p = log_p - log_norm

    if idx == -1:
        p = np.exp(log_p)

        idx = discrete_rvs(p)

    return idx, log_p[idx], log_norm


@numba.jit(nopython=True)
def propose_annealed(log_p_fn, a, b, params, T, x, z, idx=-1):
    log_p = np.zeros(2)

    z[-1] = 0

    log_p[0] = log_target_pdf_annealed(log_p_fn, a, b, params, T, x, z)

    z[-1] = 1

    log_p[1] = log_target_pdf_annealed(log_p_fn, a, b, params, T, x, z)

    log_norm = log_sum_exp(log_p)

    log_p = log_p - log_norm

    if idx == -1:
        p = np.exp(log_p)

        idx = discrete_rvs(p)

    return idx, log_p[idx], log_norm


@numba.jit(nopython=True)
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
