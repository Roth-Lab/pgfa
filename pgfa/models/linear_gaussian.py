import numba
import numpy as np
import scipy.linalg
import scipy.stats

from pgfa.math_utils import do_metropolis_hastings_accept_reject

import pgfa.models.base


def get_model(data, K=None):
    if K is None:
        feat_alloc_dist = pgfa.feature_allocation_distributions.IndianBuffetProcessDistribution()

    else:
        feat_alloc_dist = pgfa.feature_allocation_distributions.BetaBernoulliFeatureAllocationDistribution(K)

    return Model(data, feat_alloc_dist)


def simulate_data(params, prop_missing=0):
    data_true = scipy.stats.matrix_normal.rvs(
        mean=params.Z @ params.V,
        rowcov=(1 / params.tau_x) * np.eye(params.N),
        colcov=np.eye(params.D)
    )

    mask = np.random.uniform(0, 1, size=data_true.shape) <= prop_missing

    data = data_true.copy()

    data[mask] = np.nan

    return data, data_true


def simulate_params(alpha=1, tau_v=1, tau_x=1, D=1, K=None, N=100):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K=K)

    Z = feat_alloc_dist.rvs(alpha, N)

    K = Z.shape[1]

    V = scipy.stats.matrix_normal.rvs(
        mean=np.zeros((K, D)),
        rowcov=(1 / tau_v) * np.eye(K),
        colcov=np.eye(D)
    )

    return Parameters(alpha, np.ones(2), tau_v, np.ones(2), tau_x, np.ones(2), V, Z)


class Model(pgfa.models.base.AbstractModel):

    @staticmethod
    def get_default_params(data, feat_alloc_dist):
        D = data.shape[1]
        N = data.shape[0]

        Z = feat_alloc_dist.rvs(1, N)

        K = Z.shape[1]

        V = scipy.stats.matrix_normal.rvs(
            mean=np.zeros((K, D)),
            rowcov=np.eye(K),
            colcov=np.eye(D)
        )

        return Parameters(1, np.ones(2), 1, np.ones(2), 1, np.ones(2), V, Z)

    def _init_joint_dist(self, feat_alloc_dist):
        self.joint_dist = pgfa.models.base.JointDistribution(
            DataDistribution(), feat_alloc_dist, ParametersDistribution()
        )


class ModelUpdater(pgfa.models.base.AbstractModelUpdater):

    def _update_model_params(self, model):
        if model.params.K > 0:
            update_V(model)

        update_tau_v(model)

        update_tau_x(model)


class Parameters(pgfa.models.base.AbstractParameters):

    def __init__(self, alpha, alpha_prior, tau_v, tau_v_prior, tau_x, tau_x_prior, V, Z):
        self.alpha = float(alpha)

        self.alpha_prior = np.array(alpha_prior, dtype=np.float64)

        self.tau_v = float(tau_v)

        self.tau_v_prior = np.array(tau_v_prior, dtype=np.float64)

        self.tau_x = float(tau_x)

        self.tau_x_prior = np.array(tau_x_prior, dtype=np.float64)

        self.V = np.array(V, dtype=np.float64)

        self.Z = np.array(Z, dtype=np.int8)

    @property
    def param_shapes(self):
        return {
            'alpha': (),
            'alpha_prior': (2,),
            'tau_v': (),
            'tau_v_prior': (2,),
            'tau_x': (),
            'tau_x_prior': (2,),
            'V': ('K', 'D'),
            'Z': ('N', 'K')
        }

    @property
    def D(self):
        return self.V.shape[1]

    @property
    def N(self):
        return self.Z.shape[0]

    def copy(self):
        return Parameters(
            self.alpha,
            self.alpha_prior.copy(),
            self.tau_v,
            self.tau_v_prior.copy(),
            self.tau_x,
            self.tau_x_prior.copy(),
            self.V.copy(),
            self.Z.copy()
        )


# =========================================================================
# Updates
# =========================================================================
def update_V(model):
    data = model.data
    params = model.params

    t_v = params.tau_v
    t_x = params.tau_x
    Z = params.Z.astype(np.float64)
    X = data

    D = params.D

    V = np.zeros(params.V.shape)

    for d in range(D):
        idxs = ~np.isnan(X[:, d])

        X_tmp = X[idxs, d]

        Z_tmp = Z[idxs]

        M = Z_tmp.T @ Z_tmp + (t_v / t_x) * np.eye(params.K)

        V[:, d] = scipy.stats.multivariate_normal.rvs(
            scipy.linalg.solve(M, Z_tmp.T @ X_tmp, assume_a='pos'),
            (1 / t_x) * scipy.linalg.inv(M)
        )

    model.params.V = V


def update_tau_v(model):
    params = model.params

    V = params.V

    a = params.tau_v_prior[0] + 0.5 * params.K * params.D

    b = params.tau_v_prior[1] + 0.5 * np.sum(np.square(V))

    params.tau_v = scipy.stats.gamma.rvs(a, scale=(1 / b))

    model.params = params


def update_tau_x(model):
    data = model.data
    params = model.params

    V = params.V
    Z = params.Z
    X = data

    idxs = ~np.isnan(X)

    Y = X - Z @ V

    a = params.tau_x_prior[0] + 0.5 * np.sum(idxs)

    b = params.tau_x_prior[1] + 0.5 * np.sum(np.square(Y[idxs]))

    params.tau_x = scipy.stats.gamma.rvs(a, scale=(1 / b))

    model.params = params


# =========================================================================
# Densities and proposals
# =========================================================================
class DataDistribution(pgfa.models.base.AbstractDataDistribution):

    def _log_p(self, data, params):
        t_x = params.tau_x

        V = params.V
        Z = params.Z.astype(np.float64)
        X = data

        idxs = ~np.isnan(X)

        resid = X - Z @ V

        log_p = 0.5 * np.sum(idxs) * (np.log(t_x) - np.log(2 * np.pi))

        log_p -= 0.5 * t_x * np.sum(np.square(resid[idxs]))

        return log_p

    def _log_p_row(self, data, params, row_idx):
        return _log_p_row(params.tau_x, data[row_idx], params.Z[row_idx].astype(float), params.V)


class ParametersDistribution(pgfa.models.base.AbstractParametersDistribution):

    def log_p(self, params):
        if params.K == 0:
            return -np.inf

        alpha = params.alpha
        t_v = params.tau_v
        t_x = params.tau_x
        V = params.V

        D = params.D
        K = params.K

        log_p = 0

        # Gamma prior on $\alpha$
        a = params.alpha_prior[0]
        b = params.alpha_prior[1]
        log_p += scipy.stats.gamma.logpdf(alpha, a, scale=(1 / b))

        # Gamma prior on $\tau_{a}$
        a = params.tau_v_prior[0]
        b = params.tau_v_prior[1]
        log_p += scipy.stats.gamma.logpdf(t_v, a, scale=(1 / b))

        # Gamma prior on $\tau_{x}$
        a = params.tau_x_prior[0]
        b = params.tau_x_prior[1]
        log_p += scipy.stats.gamma.logpdf(t_x, a, scale=(1 / b))

        # Prior on V
        log_p += scipy.stats.matrix_normal.logpdf(
            V,
            mean=np.zeros((K, D)),
            colcov=np.eye(D),
            rowcov=(1 / t_v) * np.eye(K)
        )

        return log_p


@numba.njit(cache=True)
def _log_p_row(t_x, x, z, V):
    D = V.shape[1]

    log_p = 0

    m = z @ V

    for d in range(D):
        if np.isnan(x[d]):
            continue

        log_p += 0.5 * (np.log(t_x) - np.log(2 * np.pi))

        log_p -= 0.5 * t_x * np.square(x[d] - m[d])

    return log_p


# =========================================================================
# Singletons updaters
# =========================================================================
class PriorSingletonsUpdater(object):

    def update_row(self, model, row_idx):
        alpha = model.params.alpha
        tau_v = model.params.tau_v

        D = model.params.D
        N = model.params.N

        k_old = len(self._get_singleton_idxs(model.params.Z, row_idx))

        k_new = scipy.stats.poisson.rvs(alpha / N)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = self._get_non_singleton_idxs(model.params.Z, row_idx)

        num_non_singletons = len(non_singleton_idxs)

        K_new = len(non_singleton_idxs) + k_new

        params_old = model.params.copy()

        params_new = model.params.copy()

        params_new.V = np.zeros((K_new, D))

        params_new.V[:num_non_singletons] = model.params.V[non_singleton_idxs]

        if k_new > 0:
            params_new.V[num_non_singletons:] = scipy.stats.matrix_normal.rvs(
                mean=np.zeros((k_new, D)),
                rowcov=(1 / tau_v) * np.eye(k_new),
                colcov=np.eye(D)
            )

        params_new.Z = np.zeros((N, K_new), dtype=np.int8)

        params_new.Z[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

        params_new.Z[row_idx, num_non_singletons:] = 1

        log_p_new = model.data_dist.log_p_row(model.data, params_new, row_idx)

        log_p_old = model.data_dist.log_p_row(model.data, model.params, row_idx)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            model.params = params_new

        else:
            model.params = params_old

    def _get_column_counts(self, Z, row_idx):
        m = np.sum(Z, axis=0)

        m -= Z[row_idx]

        return m

    def _get_non_singleton_idxs(self, Z, row_idx):
        m = self._get_column_counts(Z, row_idx)

        return np.atleast_1d(np.squeeze(np.where(m > 0)))

    def _get_singleton_idxs(self, Z, row_idx):
        m = self._get_column_counts(Z, row_idx)

        return np.atleast_1d(np.squeeze(np.where(m == 0)))


class CollapsedSingletonsUpdater(object):

    def update_row(self, model, row_idx):
        alpha = model.params.alpha
        t_v = model.params.tau_v
        t_x = model.params.tau_x
        V = model.params.V
        Z = model.params.Z
        X = model.data

        D = model.params.D
        N = model.params.N

        m = np.sum(Z, axis=0)

        m -= Z[row_idx]

        k_old = np.sum(m == 0)

        k_new = np.random.poisson(alpha / N)

        if k_new == k_old:
            return model.params

        non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))

        xmo = np.square(X[row_idx] - Z[row_idx, non_singletons_idxs] @ V[non_singletons_idxs])

        log_p_old = 0
        log_p_new = 0

        for d in range(D):
            if np.isnan(xmo[d]):
                continue

            log_p_old -= 0.5 * np.log((1 / t_x) + k_old * (1 / t_v))
            log_p_old -= 0.5 * np.sum(1 / ((1 / t_x) + k_old * (1 / t_v)) * xmo[d])

            log_p_new -= 0.5 * np.log((1 / t_x) + k_new * (1 / t_v))
            log_p_new -= 0.5 * np.sum(1 / ((1 / t_x) + k_new * (1 / t_v)) * xmo[d])

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))

            num_non_singletons = len(non_singletons_idxs)

            K = num_non_singletons + k_new

            Z = np.zeros((N, K), dtype=np.int64)

            Z[:, :num_non_singletons] = model.params.Z[:, non_singletons_idxs]

            Z[row_idx, num_non_singletons:] = 1

            V = np.zeros((K, D))

            V[:num_non_singletons] = model.params.V[non_singletons_idxs]

            if k_new > 0:
                V[num_non_singletons:] = self._sample_new_V(k_new, model.data, model.params)

            model.params.Z = Z

            model.params.V = V

    def _sample_new_V(self, k, data, params):
        D = params.D
        N = params.N
        V = params.V
        Z = params.Z
        t_v = params.tau_v
        t_x = params.tau_x
        X = data

        Z_new = np.zeros((N, k))

        V_new = np.zeros((k, D))

        for d in range(D):
            idxs = ~np.isnan(X[:, d])

            Z_new_tmp = Z_new[idxs]

            M = Z_new_tmp.T @ Z_new_tmp + (t_v / t_x) * np.eye(k)

            V_new[:, d] = scipy.stats.multivariate_normal.rvs(
                scipy.linalg.solve(M, Z_new_tmp.T @ (X[idxs, d] - Z[idxs] @ V[:, d]), assume_a='pos'),
                (1 / t_x) * scipy.linalg.inv(M)
            )

        return V_new
