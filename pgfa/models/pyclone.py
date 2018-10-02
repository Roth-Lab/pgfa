import math
import numba
import numpy as np

from pgfa.math_utils import ffa_rvs, ibp_rvs

import pgfa.updates.feature_matrix


class PyCloneFeatureAllocationModel(object):
    def __init__(self, data, K=None, params=None, priors=None):
        self.data = data

        self.ibp = (K is None)

        if params is None:
            params = get_params_from_data(data, K=K)

        self.params = params

        if priors is None:
            priors = Priors()

        self.priors = priors

    @property
    def log_p(self):
        return 0

    def update(self, num_particles=10, resample_threshold=0.5, update_type='g'):
#         self.params = update_eta(self.data, self.params)

        self.params = update_Z(
            self.data,
            self.params,
            self.priors,
            ibp=self.ibp,
            num_particles=num_particles,
            resample_threshold=resample_threshold,
            update_type=update_type
        )


#=========================================================================
# Updates
#=========================================================================
def update_alpha(params, priors):
    a = priors.alpha[0] + params.K

    b = priors.alpha[1] + np.sum(1 / np.arange(1, params.N + 1))

    params.alpha = np.random.gamma(a, 1 / b)

    return params


# def update_eta(data, params):
#     params_old = params
#
#     params_new = params.copy()
#
#     density = FullDensity()
#
#     for k in range(params.K):
#         for d in range(params.D):
#             params_new.eta[d, k] = np.random.gamma(0.25 * params_old.eta[d, k], 4)
#
#             log_p_old = density.log_p(data, params_old)
#
#             log_p_new = density.log_p(data, params_new)
#
#             u = np.random.rand()
#
#             if np.log(u) <= (log_p_new - log_p_old):
#                 params_old.eta[d, k] = params_new.eta[d, k]
#
#             else:
#                 params_new.eta[d, k] = params_old.eta[d, k]
#
#     return params_new

def update_eta(data, params):
    params_old = params

    params_new = params.copy()

    density = FullDensity()

    for d in range(params.D):
        params_new.eta[d] = np.random.dirichlet(params_old.eta[d] + 1e-6)

        log_p_old = density.log_p(data, params_old)

        log_p_new = density.log_p(data, params_new)

        u = np.random.rand()

        if np.log(u) <= (log_p_new - log_p_old):
            params_old.eta[d] = params_new.eta[d]

        else:
            params_new.eta[d] = params_old.eta[d]

    return params_new


def update_Z(data, params, priors, ibp=False, num_particles=10, resample_threshold=0.5, update_type='g'):
    for row_idx in pgfa.updates.feature_matrix.get_rows(params.N):
        density = Density(row_idx)

        m = np.sum(params.Z, axis=0)

        m -= params.Z[row_idx]

        if ibp:
            a = m

            b = (params.N - m)

        else:
            a = priors.Z[1] + m

            b = priors.Z[0] + (params.N - 1 - m)

        cols = pgfa.updates.feature_matrix.get_cols(m, include_singletons=(not ibp))

        if update_type == 'g':
            params = pgfa.updates.feature_matrix.do_gibbs_update(density, a, b, cols, row_idx, data, params)

        elif update_type == 'pg':
            params = pgfa.updates.feature_matrix.do_particle_gibbs_update(
                density, a, b, cols, row_idx, data, params,
                annealed=False, num_particles=num_particles, resample_threshold=resample_threshold
            )

        elif update_type == 'pga':
            params = pgfa.updates.feature_matrix.do_particle_gibbs_update(
                density, a, b, cols, row_idx, data, params,
                annealed=True, num_particles=num_particles, resample_threshold=resample_threshold
            )

        elif update_type == 'rg':
            params = pgfa.updates.feature_matrix.do_row_gibbs_update(density, a, b, cols, row_idx, data, params)

        if ibp:
            proposal = Proposal(row_idx)

            params = pgfa.updates.feature_matrix.do_mh_singletons_update(
                density, proposal, data, params
            )

    return params


#=========================================================================
# Densities and proposals
#=========================================================================
@numba.jitclass([
    ('row_idx', numba.int64)
])
class Density(object):
    def __init__(self, row_idx):
        self.row_idx = row_idx

    def log_p(self, data, params):
        phi = params.phi
        x = data[self.row_idx]
        z = params.Z[self.row_idx].astype(np.float64)

        log_p = 0

        for d in range(params.D):
            f = np.sum(z * phi[d])

            f = min(1 - 1e-6, max(1e-6, f))

            log_p += binomial_log_likelihood(x[d, 0], x[d, 1], f)

        return log_p


@numba.jitclass([
])
class FullDensity(object):
    def __init__(self):
        pass

    def log_p(self, data, params):
        phi = params.phi
        X = data
        Z = params.Z

        log_p = 0

        for n in range(params.N):
            for d in range(params.D):
                f = np.sum(Z[n] * phi[d])

                f = min(1 - 1e-6, max(1e-6, f))

                log_p += binomial_log_likelihood(X[n, d, 0], X[n, d, 1], f)

        return log_p


@numba.jit(nopython=True)
def binomial_log_likelihood(n, x, p):
    if p <= 0:
        if x == 0:
            return 0

        else:
            return -np.inf

    elif p >= 1:
        if x == n:
            return 0

        else:
            return -np.inf

    else:
        log_p = x * np.log(p) + (n - x) * np.log(1 - p)

    return log_p


@numba.jit(nopython=True)
def binomial_log_pdf(n, x, p):
    return log_binomial_coefficient(n, x) + binomial_log_likelihood(n, x, p)


@numba.jit(cache=True, nopython=True)
def log_binomial_coefficient(n, x):
    return log_factorial(n) - log_factorial(x) - log_factorial(n - x)


@numba.jit(cache=True, nopython=True)
def log_factorial(x):
    return log_gamma(x + 1)


@numba.vectorize(["float64(float64)", "int64(float64)"])
def log_gamma(x):
    return math.lgamma(x)


class Proposal(object):
    def __init__(self, row_idx):
        self.row_idx = row_idx

    def rvs(self, data, params, num_singletons):
        m = np.sum(params.Z, axis=0)

        m -= params.Z[self.row_idx]

        non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))

        num_non_singleton = len(non_singletons_idxs)

        K_new = num_non_singleton + num_singletons

        Z_new = np.zeros((params.N, K_new), dtype=np.int64)

        Z_new[:, :num_non_singleton] = params.Z[:, non_singletons_idxs]

        Z_new[self.row_idx, num_non_singleton:] = 1

        eta_new = np.zeros((params.D, K_new))

        eta_new[:, :num_non_singleton] = params.eta[:, non_singletons_idxs]

        eta_new[:, num_non_singleton:] = np.random.gamma(params.kappa, 1, size=(params.D, num_singletons))

        return Parameters(
            params.alpha,
            params.kappa,
            eta_new,
            Z_new
        )


#=========================================================================
# Container classes
#=========================================================================
def get_params_from_data(data, K=None):
    D = data.shape[1]
    N = data.shape[0]

    if K is None:
        Z = ibp_rvs(1, N)

    else:
        Z = ffa_rvs(1, 1, K, N)

    K = Z.shape[1]

    phi = np.random.dirichlet(np.ones(K), size=D)

    return Parameters(1, 1, phi, Z)


@numba.jitclass([
    ('alpha', numba.float64),
    ('kappa', numba.float64),
    ('eta', numba.float64[:, :]),
    ('Z', numba.int64[:, :])
])
class Parameters(object):
    def __init__(self, alpha, kappa, eta, Z):
        self.alpha = alpha

        self.kappa = kappa

        self.eta = eta

        self.Z = Z

    @property
    def phi(self):
        phi = np.zeros(self.eta.shape)

        for d in range(self.D):
            phi[d] = self.eta[d] / np.sum(self.eta[d])

        return phi

    @property
    def D(self):
        return self.eta.shape[0]

    @property
    def K(self):
        return self.Z.shape[1]

    @property
    def N(self):
        return self.Z.shape[0]

    def copy(self):
        return Parameters(self.alpha, self.kappa, self.phi.copy(), self.Z.copy())


class Priors(object):
    def __init__(self, alpha=None, Z=None):
        if alpha is None:
            alpha = np.ones(2)

        self.alpha = alpha

        if Z is None:
            Z = np.ones(2)

        self.Z = Z
