import numpy as np
import numba
import scipy.linalg
import scipy.stats

from pgfa.math_utils import do_metropolis_hastings_accept_reject

import pgfa.models.base


class Model(pgfa.models.base.AbstractModel):

    def __init__(self, data, feat_alloc_dist, params=None):
        self._data = data

        if params is None:
            params = self.get_default_params(data, feat_alloc_dist)

        self.params = params

        self._init_joint_dist(feat_alloc_dist)

        self.impute_data()

    @staticmethod
    def get_default_params(data, feat_alloc_dist):
        N, D = data.shape

        Z = feat_alloc_dist.rvs(1, D)

        K = Z.shape[1]

        F = scipy.stats.norm.rvs(0, 1, size=(K, N))

        tau_v = np.ones(K)

        tau_x = np.ones(D)

        V = np.zeros((D, K))

        if K > 0:
            for k in range(K):
                V[:, k] = scipy.stats.multivariate_normal.rvs(np.zeros(D), (1 / tau_v[k]) * np.eye(D))

        return Parameters(1, np.ones(2), tau_v, np.ones(2), tau_x, np.array([1, 1]), F, V, Z)

    def impute_data(self):
        data = self._data.copy()

        idxs = np.isnan(data)

        if np.any(idxs):
            data[idxs] = self.params.Y[idxs]

        self.data = data

    def _init_joint_dist(self, feat_alloc_dist):
        self.joint_dist = pgfa.models.base.JointDistribution(
            DataDistribution(), feat_alloc_dist, ParametersDistribution()
        )


class ModelUpdater(pgfa.models.base.AbstractModelUpdater):

    def update(self, model, alpha_updates=1, feat_alloc_updates=1, param_updates=1):
        """ Update all parameters in a feature allocation model.
        """
        model.impute_data()

        for _ in range(feat_alloc_updates):
            self.feat_alloc_updater.update(model)

        for _ in range(param_updates):
            self._update_model_params(model)

        for _ in range(alpha_updates):
            pgfa.feature_allocation_distributions.update_alpha(model)

    def _update_model_params(self, model):
        update_F(model)

        update_V(model)

        update_tau_x(model)

        update_tau_v(model)


class Parameters(pgfa.models.base.AbstractParameters):

    def __init__(self, alpha, alpha_prior, tau_v, tau_v_prior, tau_x, tau_x_prior, F, V, Z):
        self.alpha = alpha

        self.alpha_prior = alpha_prior

        self.tau_v = tau_v

        self.tau_v_prior = tau_v_prior

        self.tau_x = tau_x

        self.tau_x_prior = tau_x_prior

        self.F = F

        self.V = V

        self.Z = Z

    @property
    def param_shapes(self):
        return {
            'alpha': (),
            'alpha_prior': (2,),
            'tau_v': ('K',),
            'tau_v_prior': (2,),
            'tau_x': ('D',),
            'tau_x_prior': (2,),
            'F': ('K', 'N'),
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

    @property
    def Y(self):
        return (self.W @ self.F).T

    def copy(self):
        return Parameters(
            self.alpha,
            self.alpha_prior.copy(),
            self.tau_v,
            self.tau_v_prior.copy(),
            self.tau_x.copy(),
            self.tau_x_prior.copy(),
            self.F.copy(),
            self.V.copy(),
            self.Z.copy()
        )


#=========================================================================
# Updates
#=========================================================================
def update_tau_v(model):
    params = model.params

    for k in range(params.K):
        a = params.tau_v_prior[0] + 0.5 * np.sum(params.Z[:, k])

        b = params.tau_v_prior[1] + 0.5 * np.sum(np.square(params.W[:, k]))

        params.tau_v[k] = scipy.stats.gamma.rvs(a, scale=(1 / b))

    model.params = params


def update_F(model):
    params = model.params

    if params.K == 0:
        return

    tau_x = np.diag(params.tau_x)
    W = params.W
    X = model.data

    A = np.eye(params.K) + W.T @ tau_x @ W

    A_chol = scipy.linalg.cho_factor(A)

    b = scipy.linalg.cho_solve(A_chol, W.T @ tau_x @ X.T)

    eps = scipy.stats.norm.rvs(0, 1, size=(params.K, params.N))

    params.F = b + scipy.linalg.solve_triangular(A_chol[0], eps, lower=A_chol[1])

    model.params = params


def update_tau_x(model):
    params = model.params

    X = model.data

    R = X - model.params.Y

    a = params.tau_x_prior[0] + 0.5 * params.N

    b = params.tau_x_prior[1] + 0.5 * np.nansum(np.square(R), axis=1)

    tau_x = np.zeros(params.D)

    for d in range(params.D):
        tau_x[d] = scipy.stats.gamma.rvs(a, scale=(1 / b[d]))

    params.tau_x = tau_x

    model.params = params


def update_V(model):
    params = model.params

    params.V = _update_V(model.data, params.tau_v, params.tau_x, params.F, params.V, params.Z)

    model.params = params


@numba.njit(cache=True)
def _update_V(data, tau_v, tau_x, F, V, Z):
    X = data

    D, K = Z.shape

    FF = np.sum(np.square(F), axis=1)

    for d in np.random.permutation(D):
        R = X[:, d] - (Z[d] * V[d]) @ F

        for k in np.random.permutation(K):
            rk = R + Z[d, k] * V[d, k] * F[k]

            prec = tau_v[k] + Z[d, k] * tau_x[d] * FF[k]

            mean = Z[d, k] * (tau_x[d] / prec) * (F[k] @ rk.T)

            std = 1 / np.sqrt(prec)

            V[d, k] = np.random.normal(mean, std)

            R = rk - Z[d, k] * V[d, k] * F[k]

    return V


#=========================================================================
# Densities and proposals
#=========================================================================
class DataDistribution(pgfa.models.base.AbstractDataDistribution):

    def log_p(self, data, params):
        tau_x = params.tau_x
        F = params.F
        W = params.W
        X = data

        D = params.D
        N = params.N

        return 0.5 * N * (np.sum(np.log(tau_x)) - D * np.log(2 * np.pi)) - 0.5 * np.sum(tau_x[:, np.newaxis] * np.square(X.T - (W @ F)))

    def log_p_row(self, data, params, row_idx):
        tau_x = params.tau_x
        F = params.F
        V = params.V
        Z = params.Z
        X = data

        N = params.N

        t_x = tau_x[row_idx]

        w = Z[row_idx] * V[row_idx]

        x = X[:, row_idx]

        return 0.5 * N * (np.log(t_x) - np.log(2 * np.pi)) - 0.5 * t_x * np.sum(np.square(x - w @ F))


class ParametersDistribution(pgfa.models.base.AbstractParametersDistribution):

    def log_p(self, params):
        alpha = params.alpha
        tau_v = params.tau_v
        tau_x = params.tau_x
        F = params.F
        V = params.V

        log_p = 0

        # Gamma prior on $\alpha$
        a = params.alpha_prior[0]
        b = params.alpha_prior[1]
        log_p += scipy.stats.gamma.logpdf(alpha, a, scale=(1 / b))

        # Common factors prior
        for k in range(params.K):
            log_p += scipy.stats.multivariate_normal.logpdf(
                V[:, k], np.zeros(params.D), (1 / tau_v[k]) * np.eye(params.D)
            )

        if params.K > 0:
            # Factor loadings prior
            log_p += np.sum(
                scipy.stats.multivariate_normal.logpdf(F.T, np.zeros(params.K), np.eye(params.K))
            )

        # Noise covariance
        a = params.tau_x_prior[0]
        b = params.tau_x_prior[1]
        log_p += np.sum(
            scipy.stats.gamma.logpdf(tau_x, a, scale=(1 / b))
        )

        #
        a = params.tau_v_prior[0]
        b = params.tau_v_prior[1]
        log_p += np.sum(scipy.stats.gamma.logpdf(tau_v, a, scale=(1 / b)))

        return log_p


#=========================================================================
# tau_xingletons updaters
#=========================================================================
class CollapsedSingletonsUpdater(object):

    def update_row(self, model, row_idx):
        alpha = model.params.alpha
        tau_v_prior = model.params.tau_v_prior
        tau_x = model.params.tau_x
        F = model.params.F
        V = model.params.V
        Z = model.params.Z
        X = model.data

        D = model.params.D
        N = model.params.N

        singleton_idxs = get_singletons_idxs(Z, row_idx)

        k_old = len(singleton_idxs)

        k_new = scipy.stats.poisson.rvs(alpha / D)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = get_non_singletons_idxs(model.params.Z, row_idx)

        f = F[non_singleton_idxs]

        s = tau_x[row_idx]

        v = V[row_idx, non_singleton_idxs]

        z = Z[row_idx, non_singleton_idxs]

        x = X[:, row_idx]

        E = x - (v * z) @ f

        if k_old == 0:
            log_p_old = 0

        else:
            v_old = model.params.V[row_idx, singleton_idxs]

            prec_old = s * (v_old @ v_old.T) + np.eye(k_old)

            M_old = s * np.linalg.solve(prec_old, v_old)
            M_old = M_old[:, np.newaxis] @ E[np.newaxis, :]

            log_p_old = 0
            log_p_old -= 0.5 * N * np.log(np.linalg.det(prec_old))
            log_p_old += 0.5 * np.trace(M_old.T @ prec_old @ M_old)

        if k_new == 0:
            log_p_new = 0

        else:
            tau_v_new = scipy.stats.gamma.rvs(tau_v_prior[0], scale=(1 / tau_v_prior[1]), size=k_new)

            v_new = scipy.stats.norm.rvs(0, 1, size=k_new) * (1 / np.sqrt(tau_v_new))

            prec_new = s * (v_new @ v_new.T) + np.eye(k_new)

            M_new = s * np.linalg.solve(prec_new, v_new.T)
            M_new = M_new[:, np.newaxis] @ E[np.newaxis, :]

            log_p_new = 0
            log_p_new -= 0.5 * N * np.log(np.linalg.det(prec_new))
            log_p_new += 0.5 * np.trace(M_new.T @ prec_new @ M_new)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            params = model.params.copy()

            num_non_singletons = len(non_singleton_idxs)

            K_new = num_non_singletons + k_new

            params.tau_v = np.zeros(K_new)

            params.tau_v[:num_non_singletons] = model.params.tau_v[non_singleton_idxs]

            params.F = np.zeros((K_new, N))

            params.F[:num_non_singletons] = model.params.F[non_singleton_idxs]

            params.V = np.zeros((D, K_new))

            params.V[:, :num_non_singletons] = model.params.V[:, non_singleton_idxs]

            if k_new > 0:
                chol = np.linalg.cholesky(prec_new)

                eps = scipy.stats.norm.rvs(0, 1, size=(k_new, N))

                params.tau_v[num_non_singletons:] = tau_v_new

                params.F[num_non_singletons:] = M_new + np.linalg.solve(chol, eps)

                params.V[:, num_non_singletons:] = v_new

            params.Z = np.zeros((D, K_new), dtype=np.int64)

            params.Z[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

            params.Z[row_idx, num_non_singletons:] = 1

            model.params = params

            update_V(model)


def get_non_singletons_idxs(Z, row_idx):
    m = np.sum(Z, axis=0)

    m -= Z[row_idx]

    return np.atleast_1d(np.squeeze(np.where(m > 0)))


def get_singletons_idxs(Z, row_idx):
    m = np.sum(Z, axis=0)

    m -= Z[row_idx]

    return np.atleast_1d(np.squeeze(np.where(m == 0)))


def log_held_out_pdf(data, idxs, params):
    if not np.any(idxs):
        return 0

    X_pred = params.Y

    log_p = 0

    for d in range(params.D):
        X_obs_d = data[idxs[d], d]

        X_pred_d = X_pred[idxs[d], d]

        dmv = np.sum(idxs[d])

        if dmv == 0:
            continue

        log_p += scipy.stats.multivariate_normal.logpdf(X_obs_d, X_pred_d, (1 / params.tau_x[d]) * np.eye(dmv))

    return log_p


def log_predictive_pdf(data, params):
    tau_x = params.tau_x
    W = params.W

    mean = np.zeros(params.D)

    covariance = np.diag(1 / tau_x) + W @ W.T

    log_p = 0

    for d in range(params.D):
        X_d = data[d]

        X_d = X_d[~np.isnan(X_d)]

        log_p += np.sum(
            scipy.stats.norm.logpdf(X_d, mean[d], np.sqrt(covariance[d, d]))
        )

    return log_p


def rmse(data, params):
    return np.sqrt(np.nanmean((params.Y - data) ** 2))
