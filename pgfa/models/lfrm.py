import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import bernoulli_rvs, do_metropolis_hastings_accept_reject


class LatentFactorRelationalModel(object):
    def __init__(self, data, feat_alloc_prior, params=None, priors=None, symmetric=True):
        self.data = data

        self.feat_alloc_prior = feat_alloc_prior

        if params is None:
            params = self.get_params_from_data()

        self.params = params

        if priors is None:
            priors = Priors()

        self.priors = priors

        self.data_dist = DataDistribution()

        self.joint_dist = JointDistribution(feat_alloc_prior, priors, symmetric=symmetric)

        self.symmetric = symmetric

    @property
    def log_p(self):
        return self.joint_dist.log_p(self.data, self.params)

    def get_params_from_data(self):
        N = self.data.shape[1]

        Z = self.feat_alloc_prior.rvs(N)

        K = Z.shape[1]

        if K == 0:
            V = np.zeros((K, K))

        else:
            V = np.random.normal(0, 1, size=(K, K))

            V = np.triu(V)

            V = V + V.T - np.diag(np.diag(V))

        return Parameters(1, V, Z)

    def predict(self, method='max'):
        V = self.params.V
        Z = self.params.Z
        X = np.zeros((self.params.N, self.params.N))

        for i in range(self.params.N):
            for j in range(self.params.N):
                m = Z[i].T @ V @ Z[j]

                r = np.exp(-m)

                p = 1 / (1 + r)

                if method == 'max':
                    X[i, j] = int(p >= 0.5)

                elif method == 'random':
                    X[i, j] = bernoulli_rvs(p)

                else:
                    X[i, j] = p

        return X


#=========================================================================
# Updates
#=========================================================================
class LatentFactorRelationalModelUpdater(object):
    def __init__(self, feat_alloc_updater):
        self.feat_alloc_updater = feat_alloc_updater

    def update(self, model):
        self.feat_alloc_updater.update(model)

        model.params = update_V(model.data, model.params, symmetric=model.symmetric)

        model.params = update_tau(model.params, model.priors, symmetric=model.symmetric)

        model.feat_alloc_prior.update(model.params.Z)


def update_tau(params, priors, symmetric=True):
    if symmetric:
        a = priors.tau[0] + 0.5 * (0.5 * params.K * (params.K + 1))

        b = priors.tau[1] + 0.5 * np.sum(np.square(np.triu(params.V)))

    else:
        a = priors.tau[0] + 0.5 * params.K ** 2

        b = priors.tau[1] + 0.5 * np.sum(np.square(params.V.flatten()))

    params.tau = np.random.gamma(a, 1 / b)

    return params


def update_V(data, params, proposal_precision=1, symmetric=True):
    density = DataDistribution()

    proposal_std = 1 / np.sqrt(proposal_precision)

    if symmetric:
        for i in np.random.permutation(params.K):
            for j in np.random.permutation(np.arange(i, params.K)):
                v_old = params.V[i, j]

                v_new = np.random.normal(v_old, proposal_std)

                params.V[i, j] = v_old

                params.V[j, i] = v_old

                log_p_old = density.log_p(data, params)

                log_q_old = scipy.stats.norm.logpdf(v_old, v_new, proposal_std)

                params.V[i, j] = v_new

                params.V[j, i] = v_new

                log_p_new = density.log_p(data, params)

                log_q_new = scipy.stats.norm.logpdf(v_new, v_new, proposal_std)

                if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
                    params.V[i, j] = v_new

                    params.V[j, i] = v_new

                else:
                    params.V[i, j] = v_old

                    params.V[j, i] = v_old

    else:
        for i in np.random.permutation(params.K):
            for j in np.random.permutation(params.K):
                v_old = params.V[i, j]

                v_new = np.random.normal(v_old, proposal_std)

                params.V[i, j] = v_old

                log_p_old = density.log_p(data, params)

                log_q_old = scipy.stats.norm.logpdf(v_old, v_new, proposal_std)

                params.V[i, j] = v_new

                log_p_new = density.log_p(data, params)

                log_q_new = scipy.stats.norm.logpdf(v_new, v_new, proposal_std)

                if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
                    params.V[i, j] = v_new

                else:
                    params.V[i, j] = v_old

    return params


#=========================================================================
# Densities and proposals
#=========================================================================
class DataDistribution(object):
    def __init__(self):
        pass

    def log_p(self, data, params):
        log_p = 0

        for row_idx in range(params.N):
            log_p += self.log_p_row(data, params, row_idx)

        return log_p

    def log_p_row(self, data, params, row_idx):
        return _log_p_row(params.V, data, params.Z.astype(np.float64), row_idx)


class JointDistribution(object):
    def __init__(self, feat_alloc_prior, priors, symmetric=True):
        self.data_dist = DataDistribution()

        self.feat_alloc_prior = feat_alloc_prior

        self.priors = priors

        self.symmetric = symmetric

    def log_p(self, data, params):
        log_p = 0

        # Binary matrix prior
        log_p += self.feat_alloc_prior.log_p(params.Z)

        # Data
        log_p += self.data_dist.log_p(data, params)

        # Prior
        if self.symmetric:
            log_p += np.sum(scipy.stats.norm.logpdf(
                np.squeeze(params.V[np.triu_indices(params.K)]), 0, 1 / np.sqrt(params.tau)
            ))

        else:
            log_p += np.sum(scipy.stats.norm.logpdf(
                params.V.flatten(), 0, 1 / np.sqrt(params.tau)
            ))

        return log_p


@numba.njit(cache=True)
def _log_p_row(V, X, Z, row_idx):
    N = X.shape[0]

    log_p = 0

    log_p += _get_log_p_sigmoid(row_idx, row_idx, X, V, Z)

    for i in range(N):
        if i == row_idx:
            continue

        log_p += _get_log_p_sigmoid(i, row_idx, X, V, Z)

        log_p += _get_log_p_sigmoid(row_idx, i, X, V, Z)

    return log_p


@numba.njit(cache=True)
def _get_log_p_sigmoid(i, j, X, V, Z):
    if np.isnan(X[i, j]):
        log_p = 0

    else:
        m = Z[i].T @ V @ Z[j]

        f = np.exp(-m)

        if X[i, j] == 0:
            log_p = -m - np.log1p(f)

        else:
            log_p = -np.log1p(f)

    return log_p


#=========================================================================
# Container classes
#=========================================================================
class Parameters(object):
    def __init__(self, tau, V, Z):
        self.tau = tau

        self.V = V

        self.Z = Z

    @property
    def K(self):
        return self.Z.shape[1]

    @property
    def N(self):
        return self.Z.shape[0]

    def copy(self):
        return Parameters(
            self.tau,
            self.V.copy(),
            self.Z.copy()
        )


class Priors(object):
    def __init__(self, tau=None):
        if tau is None:
            tau = np.ones(2)

        self.tau = tau


#=========================================================================
# Singletons updaters
#=========================================================================
class PriorSingletonsUpdater(object):
    def update_row(self, model, row_idx):
        k_old = len(get_singleton_idxs(model.params.Z, row_idx))

        k_new = model.feat_alloc_prior.sample_num_singletons(model.params.Z)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = get_non_singleton_idxs(model.params.Z, row_idx)

        num_non_singletons = len(non_singleton_idxs)

        K_new = len(non_singleton_idxs) + k_new

        Z_new = np.zeros((model.params.N, K_new), dtype=np.int64)

        Z_new[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

        Z_new[row_idx, num_non_singletons:] = 1

        V_new = np.zeros((K_new, K_new))

        for i in range(num_non_singletons):
            for j in range(num_non_singletons):
                V_new[i, j] = model.params.V[non_singleton_idxs[i], non_singleton_idxs[j]]

        if model.symmetric:
            for i in range(num_non_singletons, K_new):
                for j in range(i, K_new):
                    V_new[i, j] = np.random.normal(0, 1 / np.sqrt(model.params.tau))

                    V_new[j, i] = V_new[i, j]

        else:
            for i in range(num_non_singletons, K_new):
                for j in range(num_non_singletons, K_new):
                    V_new[j, i] = np.random.normal(0, 1 / np.sqrt(model.params.tau))

        params_new = Parameters(model.params.tau, V_new, Z_new)

        log_p_new = model.data_dist.log_p_row(model.data, params_new, row_idx)

        log_p_old = model.data_dist.log_p_row(model.data, model.params, row_idx)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            params = params_new

        else:
            params = model.params

        return params


def get_column_counts(Z, row_idx):
    m = np.sum(Z, axis=0)

    m -= Z[row_idx]

    return m


def get_non_singleton_idxs(Z, row_idx):
    m = get_column_counts(Z, row_idx)

    return np.atleast_1d(np.squeeze(np.where(m > 0)))


def get_singleton_idxs(Z, row_idx):
    m = get_column_counts(Z, row_idx)

    return np.atleast_1d(np.squeeze(np.where(m == 0)))
