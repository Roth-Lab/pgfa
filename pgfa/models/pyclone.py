import math
import numba
import numpy as np
import scipy.optimize
import scipy.stats

from pgfa.math_utils import do_metropolis_hastings_accept_reject

import pgfa.models.base


class Model(pgfa.models.base.AbstractModel):
    @staticmethod
    def get_default_params(data, feat_alloc_dist):
        D = data.shape[1]
        N = data.shape[0]

        Z = feat_alloc_dist.rvs(1, N)

        K = Z.shape[1]

        eta = np.random.gamma(1, 1, size=(K, D))

        return Parameters(1, np.ones(2), eta, Z)

    def _init_joint_dist(self, feat_alloc_dist):
        self.joint_dist = pgfa.models.base.JointDistribution(
            DataDistribution(), feat_alloc_dist, ParametersDistribution()
        )


class ParallelTemperingUpdater(pgfa.models.base.AbstractModelUpdater):
    def __init__(self, data, feat_alloc_dist, feat_alloc_updater, num_chains=10):
        self.feat_alloc_updater = feat_alloc_updater

        self.temps = np.linspace(0, 1, num_chains)  # ** 3

        self.iter = 0

        self.params = []

        for _ in self.temps:
            self.params.append(Model.get_default_params(data, feat_alloc_dist))

    def update(self, model, alpha_updates=1, feat_alloc_updates=1, param_updates=1):
        for idx, (p, t) in enumerate(zip(self.params, self.temps)):
            model.params = p

            model.data_dist.temp = t

            for _ in range(feat_alloc_updates):
                self.feat_alloc_updater.update(model)

            for _ in range(param_updates):
                self._update_model_params(model)

            for _ in range(alpha_updates):
                pgfa.feature_allocation_distributions.update_alpha(model)

            self.params[idx] = model.params.copy()

        T = len(self.temps)

        if self.iter % 2 == 0:
            pairs = zip(range(0, T - 1, 2), range(1, T, 2))

        else:
            pairs = zip(range(1, T - 3, 2), range(2, T - 2, 2))

        model.data_dist.temp = 1

        for idx_l, idx_h in pairs:
            t_l = self.temps[idx_l]

            t_h = self.temps[idx_h]

            log_p_new = (t_h - t_l) * model.data_dist.log_p(model.data, self.params[idx_l])

            log_p_old = (t_h - t_l) * model.data_dist.log_p(model.data, self.params[idx_h])

            if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
                print('Accept', idx_l, idx_h)

                temp = self.params[idx_l]

                self.params[idx_l] = self.params[idx_h]

                self.params[idx_h] = temp

        model.params = self.params[-1]

        self.iter += 1

    def _update_model_params(self, model):
        update_eta_independent(model)


class ModelUpdater(pgfa.models.base.AbstractModelUpdater):
    def _update_model_params(self, model):
        update_eta_hmc(model)


class Parameters(pgfa.models.base.AbstractParameters):
    def __init__(self, alpha, alpha_prior, eta, Z):
        self.alpha = alpha

        self.alpha_prior = alpha_prior

        self.eta = eta

        self.Z = Z

    @property
    def phi(self):
        return self.eta / np.sum(self.eta, axis=0)

    @property
    def D(self):
        return self.eta.shape[1]

    @property
    def N(self):
        return self.Z.shape[0]

    def copy(self):
        return Parameters(self.alpha, self.alpha_prior.copy(), self.eta.copy(), self.Z.copy())


#=========================================================================
# Updates
#=========================================================================
def update_eta(model, precision=10):
    data = model.data
    params = model.params

    data_dist = DataDistribution()

    for d in np.random.permutation(params.D):
        params_new = params.copy()

        params_new.eta[:, d] = scipy.stats.dirichlet.rvs(params.phi[:, d] * precision) * precision

        log_p_old = data_dist.log_p(data, params)

        log_p_new = data_dist.log_p(data, params_new)

        log_q_old = scipy.stats.dirichlet.logpdf(params.phi[:, d], params_new.phi[:, d] * precision)

        log_q_new = scipy.stats.dirichlet.logpdf(params_new.phi[:, d], params.phi[:, d] * precision)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
            print('Accept')
            params.eta[:, d] = params_new.eta[:, d]

    model.params = params


def update_eta_independent(model):
    data = model.data
    params = model.params

    data_dist = model.data_dist

    params_old = params.copy()

    params_new = params.copy()

    for k in np.random.permutation(params.K):
        for d in np.random.permutation(params.D):
            params_new.eta[k, d] = np.random.gamma(1, 1)

            log_p_old = data_dist.log_p(data, params_old)

            log_p_new = data_dist.log_p(data, params_new)

            if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
                params_old.eta[k, d] = params_new.eta[k, d]

            else:
                params_new.eta[k, d] = params_old.eta[k, d]

    model.params = params_new


def update_eta_map(model):
    V = model.params.eta
    Z = model.params.Z.astype(np.float64)
    X = model.data

    for d in range(model.params.D):
        def f(v): return -1 * log_p(X[:, d, :], v, Z)

        def g(v): return -1 * grad(X[:, d, :], v, Z)

        b = np.tile([1e-6, np.inf], (model.params.K, 1))

        res = scipy.optimize.minimize(f, V[:, d], jac=g, bounds=b)

        V[:, d] = res['x']

    model.params.eta = V


def update_eta_hmc(model):
    V = model.params.eta
    Z = model.params.Z.astype(np.float64)
    X = model.data

    for d in range(model.params.D):
        def f(v): return -1 * log_p(X[:, d, :], v, Z)

        def g(v): return -1 * grad(X[:, d, :], v, Z)

        V[:, d] = do_hmc(f, g, V[:, d], L=np.random.randint(0, 11))

    model.params.eta = V


def do_hmc(log_p, grad_log_p, params, eps=1e-3, L=10):
    dim = len(params)

    q = params
    q_old = params
    p = mvn_rvs(dim)
    p_old = p

    p = p - eps * grad_log_p(q) / 2

    for i in range(L):
        q = q + eps * p

        if i < (L - 1):
            p = p - eps * grad_log_p(q)

    p = p - eps * grad_log_p(q) / 2

    p = -p

    U_new = log_p(q)
    K_new = np.sum(p ** 2) / 2
    U_old = log_p(q_old)
    K_old = np.sum(p_old ** 2) / 2

    diff = (U_new + K_new) - (U_old + K_old)

    u = np.random.random()

    if np.log(u) < diff:
        q_new = q

    else:
        q_new = q_old

    return q_new


def mvn_rvs(dim, size=1):
    return np.atleast_1d(
        scipy.stats.multivariate_normal.rvs(np.zeros(dim), np.eye(dim), size=size)
    )


@numba.njit
def log_p(X, v, Z):
    N = Z.shape[0]
    M = _get_M(v, Z)
    log_p = 0
    for n in range(N):
        x = X[n, 1]
        y = X[n, 0]
        log_p += x * np.log(M[n]) + (y - x) * np.log(1 - M[n])
    return log_p


@numba.njit
def grad(X, v, Z):
    K = Z.shape[1]
    N = Z.shape[0]
    M = _get_M(v, Z)
    g = np.zeros(K)
    for n in range(N):
        g += _grad(X[n, 1], X[n, 0], M[n], v, Z[n])
    return g


@numba.njit
def _grad(x, y, m, v, z):
    K = len(v)
    grad_log_p_wrt_m = (x - y * m) / (m * (1 - m))
    grad_m_wrt_v = np.zeros(K)
    for k in range(K):
        grad_m_wrt_v[k] = np.sum((z[k] - z) * v) / np.sum(v) ** 2
    return grad_log_p_wrt_m * grad_m_wrt_v


@numba.njit
def _get_M(v, Z):
    w = v / np.sum(v)
    M = Z @ w
    M = 0.01 * 1e-3 + 0.99 * M
    return M


#=========================================================================
# Densities and proposals
#=========================================================================
class DataDistribution(pgfa.models.base.AbstractDataDistribution):
    def __init__(self, temp=1):
        self.temp = temp

    def log_p(self, data, params):
        return self.temp * _log_p(params.phi, data, params.Z)

    def log_p_row(self, data, params, row_idx):
        phi = params.phi
        x = data[row_idx]
        z = params.Z[row_idx].astype(np.float64)

        return self.temp * _log_p_row(phi, x, z)


class ParametersDistribution(pgfa.models.base.AbstractParametersDistribution):
    def log_p(self, params):
        return 0


@numba.njit(cache=True)
def _log_p(phi, X, Z):
    N = Z.shape[0]

    log_p = 0

    for row_idx in range(N):
        log_p += _log_p_row(phi, X[row_idx], Z[row_idx])

    return log_p


@numba.njit(cache=True)
def _log_p_row(phi, x, z, eps=1e-6):
    log_p = 0

    for d in range(len(x)):
        f = np.sum(phi[:, d] * z)

        f = min(1 - eps, max(eps, f))

        log_p += binomial_log_likelihood(x[d, 0], x[d, 1], f)

    return log_p


@numba.njit(cache=True)
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


@numba.njit(cache=True)
def binomial_log_pdf(n, x, p):
    return log_binomial_coefficient(n, x) + binomial_log_likelihood(n, x, p)


@numba.njit(cache=True)
def log_binomial_coefficient(n, x):
    return log_factorial(n) - log_factorial(x) - log_factorial(n - x)


@numba.njit(cache=True)
def log_factorial(x):
    return log_gamma(x + 1)


@numba.vectorize(["float64(float64)", "int64(float64)"], cache=True)
def log_gamma(x):
    return math.lgamma(x)


#=========================================================================
# Singletons updaters
#=========================================================================
class PriorSingletonsUpdater(object):
    def update_row(self, model, row_idx):
        alpha = model.params.alpha

        D = model.params.D
        N = model.params.N

        k_old = len(get_singleton_idxs(model.params.Z, row_idx))

        k_new = np.random.poisson(alpha / N)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = get_non_singleton_idxs(model.params.Z, row_idx)

        num_non_singletons = len(non_singleton_idxs)

        K_new = len(non_singleton_idxs) + k_new

        params_new = model.params.copy()

        params_new.eta = np.zeros((K_new, D))

        params_new.eta[:num_non_singletons] = model.params.eta[non_singleton_idxs]

        params_new.eta[num_non_singletons:] = np.random.gamma(1, 1, size=(k_new, D))

        params_new.Z = np.zeros((N, K_new), dtype=np.int64)

        params_new.Z[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

        params_new.Z[row_idx, num_non_singletons:] = 1

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
