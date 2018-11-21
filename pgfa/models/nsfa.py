import numpy as np
import numba
import scipy.linalg
import scipy.stats

from pgfa.math_utils import do_metropolis_hastings_accept_reject

import pgfa.models.base


class Model(pgfa.models.base.AbstractModel):
    @staticmethod
    def get_default_params(data, feat_alloc_dist):
        D = data.shape[0]
        N = data.shape[1]

        Z = feat_alloc_dist.rvs(1, D)

        K = Z.shape[1]

        F = np.random.normal(0, 1, size=(K, N))

        S = np.ones(D)

        if K == 0:
            V = np.zeros((D, K))

        else:
            V = np.random.multivariate_normal(np.zeros(K), np.eye(K), size=D)

        return Parameters(1, np.ones(2), 1, np.ones(2), F, S, np.ones(2), V, Z)

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

        log_p = 0

        for d in range(self.params.D):
            X_d = self.data[d]

            X_d = X_d[~np.isnan(X_d)]

            log_p += np.sum(
                scipy.stats.norm.logpdf(X_d, mean[d], np.sqrt(covariance[d, d]))
            )

#         try:
#             log_p = np.sum(
#                 scipy.stats.multivariate_normal.logpdf(data.T, mean, covariance, allow_singular=True)
#             )
#
#         except np.linalg.LinAlgError as e:
#             print(W)
#             print(np.sum(self.params.Z, axis=0))
#             print((W @ W.T).shape)
#             raise e

        return log_p

    def _init_joint_dist(self, feat_alloc_dist):
        self.joint_dist = pgfa.models.base.JointDistribution(
            DataDistribution(), feat_alloc_dist, ParametersDistribution()
        )


class ModelUpdater(pgfa.models.base.AbstractModelUpdater):
    def _update_model_params(self, model):
        update_V(model)

        update_F(model)

        update_gamma(model)

        update_S(model)


class Parameters(pgfa.models.base.AbstractParameters):
    def __init__(self, alpha, alpha_prior, gamma, gamma_prior, F, S, S_prior, V, Z):
        self.alpha = alpha

        self.alpha_prior = alpha_prior

        self.gamma = gamma

        self.gamma_prior = gamma_prior

        self.F = F

        self.S = S

        self.S_prior = S_prior

        self.V = V

        self.Z = Z

    @property
    def param_shapes(self):
        return {
            'alpha': (),
            'alpha_prior': (2,),
            'gamma': (),
            'gamma_prior': (2,),
            'F': ('K', 'N'),
            'S': ('D',),
            'S_prior': (2,),
            'V': ('D', 'K'),
            'Z': ('D', 'K')
        }

    @property
    def D(self):
        return self.Z.shape[0]

    @property
    def N(self):
        return self.F.shape[1]

    @property
    def W(self):
        return self.Z * self.V

    def copy(self):
        return Parameters(
            self.alpha,
            self.alpha_prior.copy(),
            self.gamma,
            self.gamma_prior.copy(),
            self.F.copy(),
            self.S.copy(),
            self.S_prior.copy(),
            self.V.copy(),
            self.Z.copy()
        )


#=========================================================================
# Updates
#=========================================================================
def update_gamma(model):
    params = model.params

    a = params.gamma_prior[0] + 0.5 * params.D * params.K

    b = params.gamma_prior[1] + 0.5 * np.sum(np.square(params.V))

    params.gamma = scipy.stats.gamma.rvs(a, scale=(1 / b))

    model.params = params


def update_F(model):
    params = model.params

    if params.K == 0:
        return

    S = np.diag(params.S)
    W = params.W
    X = np.nan_to_num(model.data)

    A = np.eye(params.K) + W.T @ S @ W

    A_chol = scipy.linalg.cho_factor(A)

    b = scipy.linalg.cho_solve(A_chol, W.T @ S @ X)

    eps = np.random.normal(0, 1, size=(params.K, params.N))

    params.F = b + scipy.linalg.solve_triangular(A_chol[0], eps, lower=A_chol[1])

    model.params = params


def update_S(model):
    params = model.params

    F = params.F
    W = params.W
    X = model.data

    R = X - W @ F

    a = params.S_prior[0] + 0.5 * params.N

    b = params.S_prior[1] + 0.5 * np.nansum(np.square(R), axis=1)

    S = np.zeros(params.D)

    for d in range(params.D):
        S[d] = scipy.stats.gamma.rvs(a, scale=(1 / b[d]))

    params.S = S

    model.params = params


def update_V(model):
    params = model.params

    params.V = _update_V(model.data, params.gamma, params.F, params.S, params.V, params.Z)

    model.params = params


# @numba.njit(cache=True)
def _update_V(data, gamma, F, S, V, Z):
    X = data

    D = Z.shape[0]
    K = Z.shape[1]

    for d in range(D):
        idxs = ~np.isnan(X[d])

        F_temp = F[:, idxs]

        FF = np.sum(np.square(F_temp), axis=1)

        R = X[d, idxs] - (Z[d] * V[d]) @ F_temp

        for k in range(K):
            rk = R + Z[d, k] * V[d, k] * F_temp[k]

            prec = gamma + Z[d, k] * S[d] * FF[k]

            mean = Z[d, k] * (S[d] / prec) * (F_temp[k] @ rk.T)

            std = 1 / np.sqrt(prec)

            V[d, k] = np.random.normal(mean, std)

            R = rk - Z[d, k] * V[d, k] * F_temp[k]

    return V


#=========================================================================
# Densities and proposals
#=========================================================================
class DataDistribution(pgfa.models.base.AbstractDataDistribution):
    def log_p(self, data, params):
        D = params.D
        N = params.N
        F = params.F
        S = params.S
        W = params.W
        X = data

        diff = X - (W @ F)

        log_p = 0

        # TODO: N need to be replace by number non-nan entries
        log_p += 0.5 * N * (np.log(np.prod(S)) - D * np.log(2 * np.pi))

        log_p -= 0.5 * np.nansum(S[:, np.newaxis] * np.square(diff))

        return log_p

    def log_p_row(self, data, params, row_idx):
        N = params.N
        F = params.F
        S = params.S
        V = params.V
        Z = params.Z
        X = data

        return log_p_row(F, N, S, X, V, Z, row_idx)


@numba.njit(cache=True)
def log_p_row(F, N, S, X, V, Z, row_idx):
    w = Z[row_idx] * V[row_idx]

    diff = (X[row_idx] - w @ F)

    log_p = 0

    # TODO: N need to be replace by number non-nan entries
    log_p += 0.5 * N * (np.log(S[row_idx]) - np.log(2 * np.pi))

    log_p -= 0.5 * np.nansum(S[row_idx] * np.square(diff))

    return log_p


class ParametersDistribution(pgfa.models.base.AbstractParametersDistribution):
    def log_p(self, params):
        alpha = params.alpha
        gamma = params.gamma
        F = params.F
        S = params.S
        V = params.V

        log_p = 0

        # Gamma prior on $\alpha$
        a = params.alpha_prior[0]
        b = params.alpha_prior[1]
        log_p += scipy.stats.gamma.logpdf(alpha, a, scale=(1 / b))

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
            scipy.stats.gamma.logpdf(S, params.S_prior[0], scale=(1 / params.S_prior[1]))
        )

        log_p += scipy.stats.gamma.logpdf(1 / gamma, params.gamma_prior[0], scale=(1 / params.gamma_prior[1]))

        return log_p


#=========================================================================
# Singletons updaters
#=========================================================================
class PriorSingletonsUpdater(object):
    def __init__(self, num_iters=1):
        self.num_iters = num_iters

    def update_row(self, model, row_idx):
        for _ in range(self.num_iters):
            model.params = self._update_row(model, row_idx)

        return model.params

    def _update_row(self, model, row_idx):
        alpha = model.params.alpha
        gamma = model.params.gamma

        D = model.params.D
        N = model.params.N

        k_old = len(get_singletons_idxs(model.params.Z, row_idx))

        k_new = scipy.stats.poisson.rvs(alpha / D)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = get_non_singletons_idxs(model.params.Z, row_idx)

        num_non_singletons = len(non_singleton_idxs)

        K_new = num_non_singletons + k_new

        params_new = model.params.copy()

        params_new.F = np.zeros((K_new, N))

        params_new.F[:num_non_singletons] = model.params.F[non_singleton_idxs]

        params_new.F[num_non_singletons:] = np.random.normal(0, 1, size=(k_new, N))

        params_new.V = np.zeros((D, K_new))

        params_new.V[:, :num_non_singletons] = model.params.V[:, non_singleton_idxs]

        params_new.V[:, num_non_singletons:] = np.random.multivariate_normal(
            np.zeros(D), (1 / gamma) * np.eye(D), size=k_new
        ).T

        params_new.Z = np.zeros((D, K_new), dtype=np.int64)

        params_new.Z[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

        params_new.Z[row_idx, num_non_singletons:] = 1

        log_p_new = model.data_dist.log_p_row(model.data, params_new, row_idx)

        log_p_old = model.data_dist.log_p_row(model.data, model.params, row_idx)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            params = params_new

        else:
            params = model.params

        return params


class CollapsedSingletonsUpdater(object):
    def update_row(self, model, row_idx):
        D = model.params.D
        N = model.params.N
        gamma = model.params.gamma
        F = model.params.F
        S = model.params.S
        V = model.params.V
        Z = model.params.Z
        X = model.data

        singleton_idxs = get_singletons_idxs(Z, row_idx)

        k_old = len(singleton_idxs)

        k_new = model.feat_alloc_prior.sample_num_singletons(model.params.Z)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = get_non_singletons_idxs(model.params.Z, row_idx)

        f = F[non_singleton_idxs]

        s = S[row_idx]

        v = V[row_idx, non_singleton_idxs]

        z = Z[row_idx, non_singleton_idxs]

        x = X[row_idx]

        E = x - (v * z) @ f

        if k_old == 0:
            log_p_old = 0

        else:
            v_old = model.params.V[row_idx, singleton_idxs]

            prec_old = s * (v_old @ v_old.T) + np.eye(k_old)

            mean_old = s * np.linalg.solve(prec_old, v_old)

            log_p_old = 0
            log_p_old -= 0.5 * N * np.log(np.linalg.det(prec_old))
            log_p_old += 0.5 * np.sum(np.square(E) * (mean_old.T @ prec_old @ mean_old))

        if k_new == 0:
            log_p_new = 0

        else:
            v_new = np.atleast_2d(np.random.normal(0, 1 / np.sqrt(gamma), size=k_new))

            prec_new = s * (v_new @ v_new.T) + np.eye(k_new)

            mean_new = s * np.linalg.solve(prec_new, v_new.T)

            log_p_new = 0
            log_p_new -= 0.5 * N * np.log(np.linalg.det(prec_new))
            log_p_new += 0.5 * np.sum(np.square(E) * (mean_new.T @ prec_new @ mean_new))

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            num_non_singletons = len(non_singleton_idxs)

            K_new = num_non_singletons + k_new

            F_new = np.zeros((K_new, N))

            F_new[:num_non_singletons] = model.params.F[non_singleton_idxs]

            V_new = np.zeros((D, K_new))

            V_new[:, :num_non_singletons] = model.params.V[:, non_singleton_idxs]

            if k_new > 0:
                chol = np.linalg.cholesky(prec_new)

                eps = np.random.normal(0, 1, size=(k_new, N))

                F_new[num_non_singletons:] = mean_new * E + np.linalg.solve(chol, eps)

                V_new[:, num_non_singletons:] = v_new

            Z_new = np.zeros((D, K_new), dtype=np.int64)

            Z_new[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

            Z_new[row_idx, num_non_singletons:] = 1

            params = Parameters(gamma, F_new, S, V_new, Z_new)

            params = update_V(X, params)

        else:
            params = model.params

        return params


def get_non_singletons_idxs(Z, row_idx):
    m = np.sum(Z, axis=0)

    m -= Z[row_idx]

    return np.atleast_1d(np.squeeze(np.where(m > 0)))


def get_singletons_idxs(Z, row_idx):
    m = np.sum(Z, axis=0)

    m -= Z[row_idx]

    return np.atleast_1d(np.squeeze(np.where(m == 0)))
