import itertools
import numpy as np
import numba
import scipy.linalg
import scipy.stats

from pgfa.math_utils import do_metropolis_hastings_accept_reject, ffa_rvs, ibp_rvs, log_ffa_pdf, log_ibp_pdf

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
            log_p += scipy.stats.multivariate_normal.logpdf(
                X[:, n], W @ F[:, n], np.diag(1 / S), allow_singular=True
            )

        # Binary matrix prior
        if self.ibp:
            log_p += log_ibp_pdf(alpha, Z)

            # Prior on alpha
            log_p += scipy.stats.gamma.logpdf(alpha, priors.alpha[0], scale=(1 / priors.alpha[1]))

        else:
            log_p += log_ffa_pdf(priors.Z[0], priors.Z[1], Z)

        # Common factors prior
        log_p += np.sum(
            scipy.stats.multivariate_normal.logpdf(V.T, np.zeros(params.D), (1 / gamma) * np.eye(params.D))
        )

        if params.K > 0:
            # Factor loadings prior
            log_p += np.sum(
                scipy.stats.multivariate_normal.logpdf(F.T, np.zeros(params.K), np.eye(params.K))
            )

        # Noise covariance
        log_p += np.sum(
            scipy.stats.gamma.logpdf(S, priors.S[0], scale=(1 / priors.S[1]))
        )

        log_p += scipy.stats.gamma.logpdf(1 / gamma, priors.gamma[0], scale=(1 / priors.gamma[1]))

        return log_p

    @property
    def rmse(self):
        X_true = self.data

        X_pred = self.params.W @ self.params.F

        return np.sqrt(np.mean((X_true - X_pred)**2))

    def log_predictive_pdf(self, data):
        S = self.params.S
        W = self.params.W

        mean = np.zeros(self.params.D)

        covariance = np.diag(1 / S) + W @ W.T

        try:
            log_p = np.sum(
                scipy.stats.multivariate_normal.logpdf(data.T, mean, covariance, allow_singular=True)
            )

        except np.linalg.LinAlgError as e:
            print(W)
            print(np.sum(self.params.Z, axis=0))
            print((W @ W.T).shape)
            raise e

        return log_p

    def update(self, num_particles=10, resample_threshold=0.5, update_type='g'):
        self.params = update_Z(
            self.data,
            self.params,
            self.priors,
            ibp=self.ibp,
            num_particles=num_particles,
            resample_threshold=resample_threshold,
            update_type=update_type
        )

#         self.params = update_V(self.data, self.params)
# 
#         if self.params.K > 0:
#             self.params = update_F(self.data, self.params)
# 
#         self.params = update_gamma(self.params, self.priors)
# 
#         self.params = update_S(self.data, self.params, self.priors)
# 
#         if self.ibp:
#             self.params = update_alpha(self.params, self.priors)


#=========================================================================
# Updates
#=========================================================================
def update_alpha(params, priors):
    a = priors.alpha[0] + params.K

    b = priors.alpha[1] + np.sum(1 / np.arange(1, params.D + 1))

    params.alpha = scipy.stats.gamma.rvs(a, scale=(1 / b))

    return params


def update_gamma(params, priors):
    a = priors.gamma[0] + 0.5 * params.K * params.D  # np.sum(params.Z)

    b = priors.gamma[1] + 0.5 * np.sum(np.square(params.V))

    params.gamma = scipy.stats.gamma.rvs(a, scale=(1 / b))

    return params


def update_F(data, params):
    S = np.diag(params.S)
    W = params.W
    X = data

    A = np.eye(params.K) + W.T @ S @ W

#     A_inv = np.linalg.inv(A)
#
#     b = A_inv @ W.T @ S @ X
#
#     chol = np.linalg.cholesky(A_inv)
#
#     eps = np.random.normal(0, 1, size=(params.K, params.N))
#
#     params.F = b + chol @ eps
#
#     return params

    A_chol = scipy.linalg.cho_factor(A)

    b = scipy.linalg.cho_solve(A_chol, W.T @ S @ X)

    eps = np.random.normal(0, 1, size=(params.K, params.N))

    params.F = b + scipy.linalg.solve_triangular(A_chol[0], eps, lower=A_chol[1])

    return params


def update_S(data, params, priors):
    S = np.zeros(params.D)

    F = params.F
    W = params.W
    X = data

    R = X - W @ F

    a = priors.S[0] + 0.5 * params.N

    b = priors.S[1] + 0.5 * np.sum(np.square(R), axis=1)

    for d in range(params.D):
        S[d] = scipy.stats.gamma.rvs(a, scale=(1 / b[d]))

    params.S = S

    return params


@numba.njit
def update_V(data, params):
    gamma = params.gamma
    F = params.F
    S = params.S
    Z = params.Z
    V = params.V
    X = data

    FF = np.sum(np.square(F), axis=1)

    for d in range(params.D):
        R = X[d] - (Z[d] * V[d]) @ F

        for k in range(params.K):
            rk = R + Z[d, k] * V[d, k] * F[k]

            prec = gamma + Z[d, k] * S[d] * FF[k]

            mean = Z[d, k] * (S[d] / prec) * (F[k] @ rk.T)

            std = 1 / np.sqrt(prec)

            V[d, k] = np.random.normal(mean, std)

            R = rk - Z[d, k] * V[d, k] * F[k]

    params.V = V

    return params


def update_Z(data, params, priors, ibp=False, num_particles=10, resample_threshold=0.5, update_type='g'):
    for row_idx in pgfa.updates.feature_matrix.get_rows(params.D):
        density = Density(row_idx)

        m = np.sum(params.Z, axis=0)

        m -= params.Z[row_idx]

        if ibp:
            a = m

            b = params.D - m

        else:
            a = priors.Z[1] + m

            b = priors.Z[0] + (params.D - 1 - m)

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
            params = update_Z_singletons(data, params, row_idx)

    return params


# def update_Z_singletons(data, params, row_idx):
#     m = np.sum(params.Z, axis=0)
#
#     m -= params.Z[row_idx]
#
#     non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))
#
#     singletons_idxs = np.atleast_1d(np.squeeze(np.where(m == 0)))
#
#     num_non_singleton = len(non_singletons_idxs)
#
#     e = data[row_idx] - params.V[row_idx, non_singletons_idxs] @ params.F[non_singletons_idxs]
#
#     e = np.atleast_2d(e)
#
#     assert e.shape == (1, params.N)
#
#     s = params.S[row_idx]
#
#     k_old = len(singletons_idxs)
#
#     v_old = params.V[row_idx, singletons_idxs]
#
#     v_old = np.atleast_2d(v_old).T
#
#     assert v_old.shape == (k_old, 1)
#
#     log_p_old = get_log_p_singleton(e, k_old, s, v_old)
#
#     k_new = np.random.poisson(params.alpha / params.D)
#
#     v_new = np.random.normal(0, 1, size=(k_new, 1)) * (1 / np.sqrt(params.gamma))
#
#     assert v_new.shape == (k_new, 1)
#
#     log_p_new = get_log_p_singleton(e, k_new, s, v_new)
#
#     if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
# #         print(log_p_new, log_p_old, k_new, k_old)
#         K_new = num_non_singleton + k_new
#
#         F = np.zeros((K_new, params.N))
#         V = np.zeros((params.D, K_new))
#         Z = np.zeros((params.D, K_new), dtype=np.int64)
#
#         F[:num_non_singleton] = params.F[non_singletons_idxs]
#         V[:, :num_non_singleton] = params.V[:, non_singletons_idxs]
#         Z[:, :num_non_singleton] = params.Z[:, non_singletons_idxs]
#
#         if k_new > 0:
#             F[num_non_singleton:] = get_F_singleton(e, k_new, s, v_new)
#             V[:, num_non_singleton:] = np.random.normal(0, 1, size=(params.D, k_new)) * (1 / np.sqrt(params.gamma))
#             V[row_idx, num_non_singleton:] = np.squeeze(v_new)
#             Z[row_idx, num_non_singleton:] = 1
#
#         # Update V from posterior
#         e = data[row_idx] - Z[row_idx] * V[row_idx] @ F
#
#         e = np.atleast_2d(e)
#
#         assert e.shape == (1, params.N)
#
#         FF = np.sum(np.square(F), axis=1)
#
#         for k in range(num_non_singleton, K_new):
#             ek = e + V[row_idx, k] * F[k]
#
#             prec = params.gamma + s * FF[k]
#
#             mean = (s / prec) * F[k] @ ek.T
#
#             std = 1 / np.sqrt(prec)
#
#             V[row_idx, k] = np.random.normal(mean, std)
#
#         params = Parameters(
#             params.alpha,
#             params.gamma,
#             F,
#             params.S,
#             V,
#             Z
#         )
#
#     return params
def update_Z_singletons(data, params, row_idx):
    params_old = params
    singletons_idxs = get_singletons_idxs(params_old, row_idx)
    k_old = len(singletons_idxs)
    v_old = params_old.V[row_idx, singletons_idxs]

    log_p_old = 0
    log_p_old += log_likelihood(data, params_old)
    log_p_old += np.sum(scipy.stats.norm.logpdf(v_old, np.zeros(k_old), (1 / params.gamma) * np.eye(k_old)))
    log_q_old = get_log_p_F_singleton(data, params_old, row_idx)

    k_new = np.random.poisson(params.alpha / params.D)
    params_new = propose_new_params(data, k_new, params, row_idx)
    singletons_idxs = get_singletons_idxs(params_new, row_idx)
    v_new = params_new.V[row_idx, singletons_idxs]

    log_p_new = 0
    log_p_new += log_likelihood(data, params_new)
    log_p_new += np.sum(scipy.stats.norm.logpdf(v_new, np.zeros(k_new), (1 / params.gamma) * np.eye(k_new)))
    log_q_new = get_log_p_F_singleton(data, params_new, row_idx)

    if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
        params = params_new

    else:
        params = params_old

    return params


def propose_new_params(data, k_new, params, row_idx):
    non_singletons_idxs = get_non_singletons_idxs(params, row_idx)

    num_non_singleton = len(non_singletons_idxs)

    K_new = num_non_singleton + k_new

    V = np.zeros((params.D, K_new))
    V[:, :num_non_singleton] = params.V[:, non_singletons_idxs]
    V[:, num_non_singleton:] = np.random.normal(0, 1, size=(params.D, k_new)) * (1 / np.sqrt(params.gamma))

    Z = np.zeros((params.D, K_new), dtype=np.int64)
    Z[:, :num_non_singleton] = params.Z[:, non_singletons_idxs]
    Z[row_idx, num_non_singleton:] = 1

    F = np.zeros((K_new, params.N))
    F[:num_non_singleton] = params.F[non_singletons_idxs]

    params = Parameters(
        params.alpha,
        params.gamma,
        F,
        params.S,
        V,
        Z
    )

    params.F[num_non_singleton:] = get_F_singleton(data, params, row_idx)

    return params


def get_non_singletons_idxs(params, row_idx):
    m = np.sum(params.Z, axis=0)

    m -= params.Z[row_idx]

    return np.atleast_1d(np.squeeze(np.where(m > 0)))


def get_singletons_idxs(params, row_idx):
    m = np.sum(params.Z, axis=0)

    m -= params.Z[row_idx]

    return np.atleast_1d(np.squeeze(np.where(m == 0)))


def get_log_p_singleton(e, k, s, v):
    """
    Parameters
    ----------
    e: ndarray (1xN)
    k: int
    s: float
    v: ndarray (Kx1)
    """
    N = len(e)

    prec = s * (v @ v.T) + np.eye(k)

    mean = s * np.linalg.solve(prec, v) @ e

    log_p = 0

    log_p -= 0.5 * N * np.linalg.slogdet(prec)[1]

    log_p += 0.5 * np.sum(mean.T @ prec @ mean)

    return log_p


def get_F_singleton(data, params, row_idx):
    non_singletons_idxs = get_non_singletons_idxs(params, row_idx)

    singleton_idxs = get_singletons_idxs(params, row_idx)

    k = len(singleton_idxs)

    e = data[row_idx] - params.V[row_idx, non_singletons_idxs] @ params.F[non_singletons_idxs]
    e = np.atleast_2d(e)

    s = params.S[row_idx]

    v = params.V[row_idx, singleton_idxs].reshape((k, 1))

    prec = s * (v @ v.T) + np.eye(k)

    mean = s * np.linalg.solve(prec, v) @ e

    chol = np.linalg.cholesky(prec)

    eps = np.random.normal(0, 1, size=(k, params.N))

    F = mean + np.linalg.solve(chol, eps)

    return F


def get_log_p_F_singleton(data, params, row_idx):
    non_singletons_idxs = get_non_singletons_idxs(params, row_idx)

    singleton_idxs = get_singletons_idxs(params, row_idx)

    k = len(singleton_idxs)

    e = data[row_idx] - params.V[row_idx, non_singletons_idxs] @ params.F[non_singletons_idxs]
    e = np.atleast_2d(e)

    s = params.S[row_idx]

    v = params.V[row_idx, singleton_idxs].reshape((k, 1))

    f = params.F[singleton_idxs]

    prec = s * (v @ v.T) + np.eye(k)

    mean = s * np.linalg.solve(prec, v) @ e

    diff = (f - mean)

    log_p = 0
    log_p += 0.5 * np.linalg.slogdet(prec)[1]
    for n in range(params.N):
        log_p -= 0.5 * diff[:, n].T @ prec @ diff[:, n]

    return log_p


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

    S = np.ones(D)

    if K == 0:
        V = np.zeros((D, K))

    else:
        V = np.random.multivariate_normal(np.zeros(K), np.eye(K), size=D)

    return Parameters(1, 1, F, S, V, Z)


@numba.jitclass([
    ('alpha', numba.float64),
    ('gamma', numba.float64),
    ('F', numba.float64[:, :]),
    ('S', numba.float64[:]),
    ('V', numba.float64[:, :]),
    ('Z', numba.int64[:, :])
])
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
        return self.Z.shape[0]

    @property
    def K(self):
        return self.Z.shape[1]

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
@numba.jitclass([
    ('row_idx', numba.int64),
])
class Density(object):
    def __init__(self, row_idx):
        self.row_idx = row_idx

    def log_p(self, data, params):
        log_p = 0

        w = params.Z[self.row_idx] * params.V[self.row_idx]

        for n in range(params.N):
            f = params.F[:, n]

            mean = np.sum(w * f)

            log_p += log_normal_likelihood(
                data[self.row_idx, n], mean, params.S[self.row_idx]
            )

        return log_p


@numba.njit
def log_likelihood(data, params):
    log_p = 0

    mean = params.W @ params.F

    for d in range(params.D):
        prec = params.S[d]

        for n in range(params.N):
            log_p += log_normal_likelihood(data[d, n], mean[d, n], prec)

    return log_p


@numba.njit
def log_normal_likelihood(x, mean, prec):
    return -0.5 * prec * (x - mean)**2


class CollapsedDensity(object):
    def __init__(self, row_idx):
        self.row_idx = row_idx

    def log_p(self, data, params):
        m = np.sum(params.Z, axis=0)

        m -= params.Z[self.row_idx]

        non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))

        singletons_idxs = np.atleast_1d(np.squeeze(np.where(m == 0)))

        k = len(singletons_idxs)

        if k == 0:
            return 0

        w = params.Z[self.row_idx, non_singletons_idxs] * params.V[self.row_idx, non_singletons_idxs]

        r = data[self.row_idx] - w @ params.F[non_singletons_idxs]

        s = params.S[self.row_idx]

        v = params.V[self.row_idx, singletons_idxs]

        M = s * v @ v.T + np.eye(k)

        m = s * np.linalg.solve(M, v)[:, np.newaxis] * r[np.newaxis, :]

#         m = s * np.linalg.solve(M, v) @ r

        log_p = 0
        log_p -= 0.5 * params.N * np.log(np.linalg.det(M))
        log_p += 0.5 * np.trace(m.T @ M @ m)

        return log_p


class Proposal(object):
    def __init__(self, row_idx):
        self.row_idx = row_idx

    def rvs(self, data, params, num_singletons):
        m = np.sum(params.Z, axis=0)

        m -= params.Z[self.row_idx]

        non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))

        num_non_singleton = len(non_singletons_idxs)

        K_new = num_non_singleton + num_singletons

        V_new = np.zeros((params.D, K_new))

        Z_new = np.zeros((params.D, K_new), dtype=np.int64)

        V_new[:, :num_non_singleton] = params.V[:, non_singletons_idxs]

        Z_new[:, :num_non_singleton] = params.Z[:, non_singletons_idxs]

        V_new[self.row_idx, num_non_singleton:] = np.random.normal(
            0, 1, size=num_singletons) * (1 / np.sqrt(params.gamma))

        Z_new[self.row_idx, num_non_singleton:] = 1

        F_new = np.zeros((K_new, params.N))

        new_params = Parameters(
            params.alpha,
            params.gamma,
            F_new,
            params.S,
            V_new,
            Z_new
        )

        new_params = update_F(data, new_params)

        new_params.F[:num_non_singleton] = params.F[non_singletons_idxs]

        return new_params


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
