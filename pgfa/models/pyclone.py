import math
import numba
import numpy as np

from pgfa.math_utils import do_metropolis_hastings_accept_reject


class PyCloneFeatureAllocationModel(object):
    def __init__(self, data, feat_alloc_prior, params=None, priors=None):
        self.data = data

        self.feat_alloc_prior = feat_alloc_prior

        if params is None:
            params = self.get_params_from_data()

        self.params = params

        if priors is None:
            priors = Priors()

        self.priors = priors

        self.data_dist = DataDistribution()

        self.joint_dist = JointDistribution(feat_alloc_prior, priors)

    @property
    def log_p(self):
        return self.joint_dist.log_p(self.data, self.params)

    def get_params_from_data(self):
        D = self.data.shape[1]
        N = self.data.shape[0]

        Z = self.feat_alloc_prior.rvs(N)

        K = Z.shape[1]

        phi = np.random.dirichlet(np.ones(K), size=D)

        return Parameters(1, phi, Z)


#=========================================================================
# Updates
#=========================================================================
class PyCloneFeatureAllocationModelUpdater(object):
    def __init__(self, feat_alloc_updater):
        self.feat_alloc_updater = feat_alloc_updater

    def update(self, model):
        self.feat_alloc_updater.update(model)

        model.params = update_eta(model.data, model.params)

        model.feat_alloc_prior.update(model.params.Z)


def update_eta(data, params):
    data_dist = DataDistribution()

    params_old = params.copy()

    params_new = params.copy()

    for k in np.random.permutation(params.K):
        for d in np.random.permutation(params.D):
            params_new.eta[d, k] = np.random.gamma(1, 1)

            log_p_old = data_dist.log_p(data, params_old)

            log_p_new = data_dist.log_p(data, params_new)

            if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
                params_old.eta[d, k] = params_new.eta[d, k]

            else:
                params_new.eta[d, k] = params_old.eta[d, k]

    return params_new


#=========================================================================
# Densities and proposals
#=========================================================================
class DataDistribution(object):
    def log_p(self, data, params):
        return _log_p(params.phi, data, params.Z)

    def log_p_row(self, data, params, row_idx):
        phi = params.phi
        x = data[row_idx]
        z = params.Z[row_idx].astype(np.float64)

        return _log_p_row(phi, x, z)


@numba.njit(cache=True)
def _log_p(phi, X, Z):
    N = Z.shape[0]

    log_p = 0

    for row_idx in range(N):
        log_p += _log_p_row(phi, X[row_idx], Z[row_idx])

    return log_p


@numba.njit(cache=True)
def _log_p_row(phi, x, z):
    log_p = 0

    for d in range(len(phi)):
        f = np.sum(z * phi[d])

        f = min(1 - 1e-6, max(1e-6, f))

        log_p += binomial_log_likelihood(x[d, 0], x[d, 1], f)

    return log_p


class JointDistribution(object):
    def __init__(self, feat_alloc_prior, priors):
        self.data_dist = DataDistribution()

        self.feat_alloc_prior = feat_alloc_prior

        self.priors = priors

    def log_p(self, data, params):
        log_p = 0

        # Binary matrix prior
        log_p += self.feat_alloc_prior.log_p(params.Z)

        # Data
        log_p += self.data_dist.log_p(data, params)

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
# Container classes
#=========================================================================
class Parameters(object):
    def __init__(self, kappa, eta, Z):
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
        return Parameters(self.kappa, self.phi.copy(), self.Z.copy())


class Priors(object):
    pass


#=========================================================================
# Singletons updaters
#=========================================================================
class PriorSingletonsUpdater(object):
    def update_row(self, model, row_idx):
        D = model.params.D
        N = model.params.N

        k_old = len(get_singleton_idxs(model.params.Z, row_idx))

        k_new = model.feat_alloc_prior.sample_num_singletons(model.params.Z)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = get_non_singleton_idxs(model.params.Z, row_idx)

        num_non_singletons = len(non_singleton_idxs)

        K_new = len(non_singleton_idxs) + k_new

        eta_new = np.zeros((D, K_new))

        eta_new[:, :num_non_singletons] = model.params.eta[:, non_singleton_idxs]

        eta_new[:, num_non_singletons:] = np.random.gamma(1, 1, size=(D, k_new))

        Z_new = np.zeros((N, K_new), dtype=np.int64)

        Z_new[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

        Z_new[row_idx, num_non_singletons:] = 1

        params_new = Parameters(model.params.kappa, eta_new, Z_new)

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
