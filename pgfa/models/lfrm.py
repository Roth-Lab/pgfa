import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import bernoulli_rvs, do_metropolis_hastings_accept_reject

import pgfa.models.base


def get_model(data, K=None, symmetric=False):
    if K is None:
        feat_alloc_dist = pgfa.feature_allocation_distributions.IndianBuffetProcessDistribution()

    else:
        feat_alloc_dist = pgfa.feature_allocation_distributions.BetaBernoulliFeatureAllocationDistribution(K)

    return Model(data, feat_alloc_dist, symmetric=symmetric)


def simulate_data(params, prop_missing=0, symmetric=False):
    data_true = np.zeros((params.N, params.N))

    if symmetric:
        for i in range(params.N):
            for j in range(i, params.N):
                m = params.Z[i].T @ params.V @ params.Z[j]

                f = np.exp(-m)

                p = 1 / (1 + f)

                data_true[i, j] = bernoulli_rvs(p)

                data_true[j, i] = data_true[i, j]

    else:
        for i in range(params.N):
            for j in range(params.N):
                m = params.Z[i].T @ params.V @ params.Z[j]

                f = np.exp(-m)

                p = 1 / (1 + f)

                data_true[i, j] = bernoulli_rvs(p)

    data = data_true.copy()

    for i in range(params.N):
        for j in range(params.N):
            if np.random.random() < prop_missing:
                data[i, j] = np.nan

    return data, data_true


def simulate_params(N, K=None, alpha=1, tau=1):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K=K)

    Z = feat_alloc_dist.rvs(alpha, N)

    K = Z.shape[1]

    V = scipy.stats.norm.rvs(0, 1 / np.sqrt(tau), size=(K, K))

    return Parameters(alpha, np.ones(2), tau, np.ones(2), V, Z)


class Model(pgfa.models.base.AbstractModel):

    @staticmethod
    def get_default_params(data, feat_alloc_dist):
        N = data.shape[1]

        Z = feat_alloc_dist.rvs(1, N)

        K = Z.shape[1]

        if K == 0:
            V = np.zeros((K, K))

        else:
            V = scipy.stats.norm.rvs(0, 1, size=(K, K))

            V = np.triu(V)

            V = V + V.T - np.diag(np.diag(V))

        return Parameters(1, np.ones(2), 1, np.ones(2), V, Z)

    def __init__(self, data, feat_alloc_dist, params=None, symmetric=True):
        self.symmetric = symmetric

        super().__init__(data, feat_alloc_dist, params=params)

    def _init_joint_dist(self, feat_alloc_dist):
        self.joint_dist = pgfa.models.base.JointDistribution(
            DataDistribution(symmetric=self.symmetric),
            feat_alloc_dist,
            ParametersDistribution(symmetric=self.symmetric)
        )

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

                elif method == 'prob':
                    X[i, j] = p

        return X


class ModelUpdater(pgfa.models.base.AbstractModelUpdater):

    def _update_model_params(self, model):
        update_V(model)

        update_tau(model)


class Parameters(pgfa.models.base.AbstractParameters):

    def __init__(self, alpha, alpha_prior, tau, tau_prior, V, Z):
        self.alpha = float(alpha)

        self.alpha_prior = np.array(alpha_prior, dtype=np.float64)

        self.tau = float(tau)

        self.tau_prior = np.array(tau_prior, dtype=np.float64)

        self.V = np.array(V, dtype=np.float64)

        self.Z = np.array(Z, dtype=np.int8)

    @property
    def param_shapes(self):
        return {
            'alpha': (),
            'alpha_prior': (2,),
            'tau': (),
            'tau_prior': (2,),
            'V': ('K', 'K'),
            'Z': ('N', 'K')
        }

    @property
    def D(self):
        return self.N

    @property
    def N(self):
        return self.Z.shape[0]

    def copy(self):
        return Parameters(
            self.alpha,
            self.alpha_prior.copy(),
            self.tau,
            self.tau_prior.copy(),
            self.V.copy(),
            self.Z.copy()
        )


# =========================================================================
# Updates
# =========================================================================
def update_V(model, proposal_precision=1):
    if model.symmetric:
        _update_V_symmetric(model, proposal_precision=proposal_precision)

    else:
        _update_V_full(model, proposal_precision=proposal_precision)


def _update_V_full(model, proposal_precision=1):
    for i in np.random.permutation(model.params.K):
        for j in np.random.permutation(model.params.K):
            _update_V_element(i, j, model, proposal_precision)


def _update_V_element(i, j, model, proposal_precision):
    proposal_std = 1 / np.sqrt(proposal_precision)

    v_old = model.params.V[i, j]

    log_p_old = model.log_p

    v_new = scipy.stats.norm.rvs(v_old, proposal_std)

    model.params.V[i, j] = v_new

    log_p_new = model.log_p

    log_q_old = scipy.stats.norm.logpdf(v_old, v_new, proposal_std)

    log_q_new = scipy.stats.norm.logpdf(v_new, v_old, proposal_std)

    if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
        model.params.V[i, j] = v_new

    else:
        model.params.V[i, j] = v_old


def _update_V_symmetric(model, proposal_precision=1):
    for i in np.random.permutation(model.params.K):
        for j in np.random.permutation(np.arange(i, model.params.K)):
            _update_V_element_symmetric(i, j, model, proposal_precision)


def _update_V_element_symmetric(i, j, model, proposal_precision):
    proposal_std = 1 / np.sqrt(proposal_precision)

    v_old = model.params.V[i, j]

    log_p_old = model.log_p

    v_new = scipy.stats.norm.rvs(v_old, proposal_std)

    model.params.V[i, j] = v_new

    model.params.V[j, i] = v_new

    log_p_new = model.log_p

    log_q_old = scipy.stats.norm.logpdf(v_old, v_new, proposal_std)

    log_q_new = scipy.stats.norm.logpdf(v_new, v_old, proposal_std)

    if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
        model.params.V[i, j] = v_new

        model.params.V[j, i] = v_new

    else:
        model.params.V[i, j] = v_old

        model.params.V[j, i] = v_old


def update_tau(model):
    params = model.params
    symmetric = model.symmetric

    if symmetric:
        a = params.tau_prior[0] + 0.5 * (0.5 * params.K * (params.K + 1))

        b = params.tau_prior[1] + 0.5 * np.sum(np.square(np.triu(params.V)))

    else:
        a = params.tau_prior[0] + 0.5 * params.K ** 2

        b = params.tau_prior[1] + 0.5 * np.sum(np.square(params.V.flatten()))

    params.tau = scipy.stats.gamma.rvs(a, scale=(1 / b))

    model.params = params


# =========================================================================
# Densities and proposals
# =========================================================================
class DataDistribution(pgfa.models.base.AbstractDataDistribution):

    def __init__(self, annealing_power=1.0, symmetric=False):
        self.annealing_power = annealing_power

        self.symmetric = symmetric

    def _log_p(self, data, params):
        if params.Z.shape[1] == 0:
            log_p = -np.inf

        else:
            if self.symmetric:
                log_p = _log_p_symmetric(data, params.V, params.Z.astype(np.float64))

            else:
                log_p = _log_p(data, params.V, params.Z.astype(np.float64))

        return log_p

    def _log_p_row(self, data, params, row_idx):
        if params.Z.shape[1] == 0:
            log_p = -np.inf

        else:
            if self.symmetric:
                log_p = _log_p_symmetric_row(data, params.V, params.Z.astype(np.float64), row_idx)

            else:
                log_p = _log_p_row(data, params.V, params.Z.astype(np.float64), row_idx)

        return log_p


class ParametersDistribution(pgfa.models.base.AbstractParametersDistribution):

    def __init__(self, symmetric=True):
        self.symmetric = symmetric

    def log_p(self, params):
        log_p = 0

        # Gamma prior on $\alpha$
        a = params.alpha_prior[0]
        b = params.alpha_prior[1]
        log_p += scipy.stats.gamma.logpdf(params.alpha, a, scale=(1 / b))

        # Gamma prior on $\tau$
        a = params.tau_prior[0]
        b = params.tau_prior[1]
        log_p += scipy.stats.gamma.logpdf(params.tau, a, scale=(1 / b))

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
def _log_p_symmetric(X, V, Z):
    N = X.shape[0]

    log_p = 0

    for i in range(N):
        for j in range(i, N):
            if np.isnan(X[i, j]):
                continue

            m = Z[i] @ V @ Z[j].T

            log_p += log_sigmoid(X[i, j], m)

    return log_p


@numba.njit(cache=True)
def _log_p(X, V, Z):
    N = X.shape[0]

    log_p = 0

    for i in range(N):
        for j in range(N):
            if np.isnan(X[i, j]):
                continue

            m = Z[i] @ V @ Z[j].T

            log_p += log_sigmoid(X[i, j], m)

    return log_p


@numba.njit(cache=True)
def _log_p_symmetric_row(X, V, Z, row_idx):
    N = X.shape[0]

    ZV = Z[row_idx] @ V

    log_p = 0

    for i in range(N):
        if np.isnan(X[row_idx, i]):
            continue

        m = ZV @ Z[i].T

        log_p += log_sigmoid(X[row_idx, i], m)

    return log_p


@numba.njit(cache=True)
def _log_p_row(X, V, Z, row_idx):
    N = X.shape[0]

    VZ = V @ Z[row_idx].T

    ZV = Z[row_idx] @ V

    log_p = 0

    for i in range(N):
        if not np.isnan(X[row_idx, i]):
            m = ZV @ Z[i].T

            log_p += log_sigmoid(X[row_idx, i], m)

        if i == row_idx:
            continue

        if not np.isnan(X[i, row_idx]):
            m = Z[i] @ VZ

            log_p += log_sigmoid(X[i, row_idx], m)

    return log_p


@numba.njit(cache=True)
def log_sigmoid(x, m):
    r = np.exp(-m)

    if x == 0:
        return -m - np.log1p(r)

    else:
        return -np.log1p(r)

# =========================================================================
# Singletons updaters
# =========================================================================


class PriorSingletonsUpdater(object):

    def update_row(self, model, row_idx):
        singleton_idxs = get_singleton_idxs(model.params.Z, row_idx)

        k_old = len(singleton_idxs)

        k_new = scipy.stats.poisson.rvs(model.params.alpha / model.params.N)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = get_non_singleton_idxs(model.params.Z, row_idx)

        num_non_singletons = len(non_singleton_idxs)

        K_new = len(non_singleton_idxs) + k_new

        Z_new = np.zeros((model.params.N, K_new), dtype=np.int64)

        Z_new[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

        Z_new[row_idx, num_non_singletons:] = 1

        V_new = np.zeros((K_new, K_new))

        # Copy over old values
        for i in range(num_non_singletons):
            for j in range(num_non_singletons):
                V_new[i, j] = model.params.V[non_singleton_idxs[i], non_singleton_idxs[j]]

        # Propose new values for V
        std = 1 / np.sqrt(model.params.tau)

        if model.symmetric:
            for i in range(num_non_singletons, K_new):
                V_new[i, i] = scipy.stats.norm.rvs(0, std)

                for j in range(num_non_singletons):
                    V_new[i, j] = scipy.stats.norm.rvs(0, std)

                    V_new[j, i] = V_new[i, j]

            assert np.all(V_new.T == V_new)

        else:
            for i in range(num_non_singletons, K_new):
                for j in range(K_new):
                    V_new[i, j] = scipy.stats.norm.rvs(0, std)

                    if i != j:
                        V_new[j, i] = scipy.stats.norm.rvs(0, std)

        for i in range(num_non_singletons):
            assert np.all(
                V_new[i, :num_non_singletons] == model.params.V[non_singleton_idxs[i]][non_singleton_idxs]
            )

        params_new = model.params.copy()

        params_new.V = V_new

        params_new.Z = Z_new

        log_p_new = model.data_dist.log_p(model.data, params_new)

        log_p_old = model.data_dist.log_p(model.data, model.params)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            model.params = params_new


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
