import itertools
import numpy as np
import numba
import scipy.stats

from pgfa.math_utils import ffa_rvs, ibp_rvs, log_ffa_pdf, log_ibp_pdf

import pgfa.updates.feature_matrix


class NonparametricSparaseFactorAnalysisModel(object):
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
        params = self.params
        priors = self.priors

        alpha = params.alpha
        gamma = params.gamma
        F = params.F
        S = params.S
        Z = params.Z
        V = params.V
        W = params.W
        X = self.data

        log_p = 0

        # Likelihood
        for n in range(params.N):
            log_p += scipy.stats.multivariate_normal.logpdf(X[:, n], np.dot(W, F[:, n]), np.linalg.inv(S))

        # Binary matrix prior
        if priors.alpha is None:
            log_p += log_ffa_pdf(priors.Z[0], priors.Z[1], Z)

        else:
            log_p += log_ibp_pdf(alpha, Z)

            # Prior on alpha
            log_p += scipy.stats.gamma.logpdf(alpha, priors.alpha[0], scale=(1 / priors.alpha[1]))

        # Common factors prior
        for k in range(params.K):
            log_p += scipy.stats.multivariate_normal.logpdf(
                V[:, k], np.zeros(params.D), (1 / gamma) * np.eye(params.D)
            )

        # Factor loadings prior
        for n in range(params.N):
            log_p += scipy.stats.multivariate_normal.logpdf(F[:, n], np.zeros(params.K), np.eye(params.K))

        # Noise covariance
        for d in range(params.D):
            log_p += scipy.stats.gamma.logpdf(S[d, d], priors.S[0], scale=(1 / priors.S[1]))

        log_p += scipy.stats.gamma.logpdf(1 / gamma, priors.gamma[0], scale=(1 / priors.gamma[1]))

        return log_p

    def log_predictive_pdf(self, data):
        S = self.params.S
        W = self.params.W

        mean = np.zeros(self.params.D)

        covariance = np.linalg.inv(S) + W @ W.T

        print(self.params.D)

        try:
            log_p = np.sum(scipy.stats.multivariate_normal.logpdf(data.T, mean, covariance))

        except np.linalg.LinAlgError as e:
            print(W)
            raise e

        return log_p

    def update(self, num_particles=10, resample_threshold=0.5, update_type='g'):
        update_Z(
            self.data,
            self.params,
            self.priors,
            ibp=self.ibp,
            num_particles=num_particles,
            resample_threshold=resample_threshold,
            update_type=update_type
        )

        update_V(self.data, self.params)

        update_F(self.data, self.params)

        update_gamma(self.params, self.priors)

        update_S(self.data, self.params, self.priors)

        update_alpha(self.params, self.priors)


#=========================================================================
# Updates
#=========================================================================
def update_alpha(params, priors):
    a = params.K + priors.alpha[0]

    b = np.sum(1 / np.arange(1, params.D + 1)) + priors.alpha[0]

    params.alpha = np.random.gamma(a, 1 / b)


def update_gamma(params, priors):
    a = priors.gamma[0] + 0.5 * np.sum(params.Z)

    b = priors.gamma[1] + 0.5 * np.sum(np.square(params.W))

    params.gamma = np.random.gamma(a, 1 / b)


def update_F(data, params):
    S = params.S
    W = params.W
    X = data

    A = np.eye(params.K) + W.T @ S @ W

    b = np.linalg.inv(A) @ W.T @ S @ X

    cov = np.linalg.inv(A)

    cov_chol = np.linalg.cholesky(cov)

    eps = np.random.normal(0, 1, size=(params.K, params.N))

    params.F = b + cov_chol @ eps


def update_S(data, params, priors):
    S = np.zeros((params.D, params.D))

    F = params.F
    W = params.W
    X = data

    R = X - np.dot(W, F)

    a = priors.S[0] + 0.5 * params.N

    b = priors.S[1] + 0.5 * np.sum(np.square(R), axis=1)

    for d in range(params.D):
        S[d, d] = np.random.gamma(a, 1 / b[d])

    params.S = S


def update_V(data, params):
    gamma = params.gamma
    F = params.F
    S = params.S
    Z = params.Z
    V = params.V
    X = data

    FF = np.square(F).sum(axis=1)

    R = X - np.dot(Z * V, F)

    for d in range(params.D):
        for k in range(params.K):
            rk = R[d] + Z[d, k] * V[d, k] * F[k]

            prec = gamma + Z[d, k] * S[d, d] * FF[k]

            mu = Z[d, k] * (S[d, d] / prec) * np.dot(F[k], rk)

            V[d, k] = np.random.normal(mu, 1 / prec)

            R[d] = rk - Z[d, k] * V[d, k] * F[k]

    params.V = V


def update_Z(data, params, priors, ibp=False, num_particles=10, resample_threshold=0.5, update_type='g'):
    alpha = params.alpha
    F = params.F
    S = params.S
    Z = params.Z
    V = params.V
    X = data

    proposal = MultivariateGaussianProposal(np.zeros(params.D), (1 / params.gamma) * np.eye(params.D))

    for row_idx in pgfa.updates.feature_matrix.get_rows(params.D):
        density = Density(row_idx, S[row_idx, row_idx], F)

        x = X[row_idx]

        z = Z[row_idx]

        m = np.sum(Z, axis=0)

        m -= z

        a = priors.Z[0] + m

        b = priors.Z[1] + (params.D - 1 - m)

        cols = pgfa.updates.feature_matrix.get_cols(m, include_singletons=(not ibp))

        if update_type == 'g':
            Z[row_idx] = pgfa.updates.feature_matrix.do_gibbs_update(density, a, b, cols, x, z, V)

        elif update_type == 'pg':
            Z[row_idx] = pgfa.updates.feature_matrix.do_particle_gibbs_update(
                density, a, b, cols, x, z, V,
                annealed=False, num_particles=num_particles, resample_threshold=resample_threshold
            )

        elif update_type == 'pga':
            Z[row_idx] = pgfa.updates.feature_matrix.do_particle_gibbs_update(
                density, a, b, cols, x, z, V,
                annealed=True, num_particles=num_particles, resample_threshold=resample_threshold
            )

        elif update_type == 'rg':
            Z[row_idx] = pgfa.updates.feature_matrix.do_row_gibbs_update(density, a, b, cols, x, z, V)

        if ibp:
            V, Z = pgfa.updates.feature_matrix.do_mh_singletons_update(
                row_idx, density, proposal, alpha, V, X, Z
            )

        params.V = V

        params.Z = Z


#=========================================================================
# Container classes
#=========================================================================
def get_params_from_data(data, K=None):
    D = data.shape[0]
    N = data.shape[1]

    if K is None:
        Z = ibp_rvs(1, D)

    else:
        Z = ffa_rvs(1, 1, K, D)

    K = Z.shape[1]

    F = np.random.normal(0, 1, size=(K, N))

    S = np.eye(D)

    V = np.random.multivariate_normal(np.zeros(K), np.eye(K), size=D)

    return Parameters(1, 1, F, S, V, Z)


class Parameters(object):
    def __init__(self, alpha, gamma, F, S, V, Z):
        self.alpha = alpha

        self.gamma = gamma

        self.F = F

        self.S = S

        self.V = V

        self.Z = Z

    @property
    def D(self):
        return self.V.shape[0]

    @property
    def K(self):
        return self.F.shape[0]

    @property
    def N(self):
        return self.F.shape[1]

    @property
    def W(self):
        return self.Z * self.V

    def copy(self):
        return Parameters(self.alpha, self.gamma, self.F.copy(), self.S.copy(), self.V.copy(), self.Z.copy())


class Priors(object):
    def __init__(self, alpha=None, gamma=None, S=None, Z=None):
        if alpha is None:
            alpha = np.ones(2)

        self.alpha = alpha

        if gamma is None:
            gamma = np.ones(2)

        self.gamma = gamma

        if S is None:
            S = np.ones(2)

        self.S = S

        if Z is None:
            Z = np.ones(2)

        self.Z = Z


#=========================================================================
# Densities and proposals
#=========================================================================
class Density(object):
    def __init__(self, row_idx, s, F):
        self.row_idx = row_idx

        self.s = s

        self.F = F

    def log_p(self, x, z, V):
        N = self.F.shape[1]

        log_p = 0

        w = z * V[self.row_idx]

        for n in range(N):
            f = self.F[:, n]

            mean = np.sum(w * f)

            log_p += log_normal_likelihood(x[n], mean, self.s)

        return log_p


class MultivariateGaussianProposal(object):
    def __init__(self, mean, covariance):
        self.mean = mean

        self.covariance = covariance

    def rvs(self, size=None):
        return np.random.multivariate_normal(self.mean, self.covariance, size=size)


@numba.njit
def log_normal_likelihood(x, mean, precision):
    return -0.5 * precision * (x - mean)**2


#=========================================================================
# Benchmark utils
#=========================================================================
def get_rmse(data, params):
    X_true = data

    X_pred = np.dot(params.W, params.F)

    return np.sqrt(np.mean((X_true - X_pred)**2))


def get_min_error(params_pred, params_true):
    W_pred = params_pred.W

    W_true = params_true.W

    K_pred = W_pred.shape[1]

    K_true = W_true.shape[1]

    min_error = float('inf')

    for perm in itertools.permutations(range(K_pred)):
        error = np.sqrt(np.mean((W_pred[:, perm[:K_true]] - W_true)**2))

        if error < min_error:
            min_error = error

    return min_error
