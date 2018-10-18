import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import do_metropolis_hastings_accept_reject


class LinearGaussianModel(object):
    def __init__(self, data, feat_alloc_prior, collapsed=False, params=None, priors=None):
        self.data = data

        self.feat_alloc_prior = feat_alloc_prior

        if params is None:
            params = self.get_params_from_data()

        self.params = params

        if priors is None:
            priors = Priors()

        self.priors = priors

        if collapsed:
            self.data_dist = CollapsedDataDistribution()

            self.joint_dist = CollapsedJointDistribution(feat_alloc_prior, priors)

        else:
            self.data_dist = UncollapsedDataDistribution()

            self.joint_dist = UncollapsedJointDistribution(feat_alloc_prior, priors)

    @property
    def log_p(self):
        """ Log of joint pdf.
        """
        return self.joint_dist.log_p(self.data, self.params)

    def get_params_from_data(self):
        D = self.data.shape[1]
        N = self.data.shape[0]

        Z = self.feat_alloc_prior.rvs(N)

        K = Z.shape[1]

        V = np.random.multivariate_normal(np.zeros(D), np.eye(D), size=K)

        return Parameters(1, 1, V, Z)


#=========================================================================
# Updates
#=========================================================================
class LinearGaussianModelUpdater(object):
    def __init__(self, feat_alloc_updater):
        self.feat_alloc_updater = feat_alloc_updater

    def update(self, model):
        self.feat_alloc_updater.update(model)

        model.params = update_V(model.data, model.params)

        model.params = update_tau_a(model.params, model.priors)

        model.params = update_tau_x(model.data, model.params, model.priors)

        model.feat_alloc_prior.update(model.params.Z)


def update_V(data, params):
    t_v = params.tau_v
    t_x = params.tau_x
    Z = params.Z
    X = np.nan_to_num(data)

    M = np.linalg.inv(Z.T @ Z + (t_v / t_x) * np.eye(params.K))

    params.V = scipy.stats.matrix_normal.rvs(
        mean=M @ Z.T @ X,
        rowcov=(1 / t_x) * M,
        colcov=np.eye(params.D)
    )

    return params


def update_tau_a(params, priors):
    V = params.V

    a = priors.tau_v[0] + 0.5 * params.K * params.D

    b = priors.tau_v[1] + 0.5 * np.trace(V.T @ V)

    params.tau_v = np.random.gamma(a, 1 / b)

    return params


def update_tau_x(data, params, priors):
    V = params.V
    Z = params.Z
    X = np.nan_to_num(data)

    a = priors.tau_x[0] + 0.5 * params.N * params.D

    Y = X - Z @ V

    b = priors.tau_x[1] + 0.5 * np.trace(Y.T @ Y)

    params.tau_x = np.random.gamma(a, 1 / b)

    return params


#=========================================================================
# Container classes
#=========================================================================
class Parameters(object):
    def __init__(self, tau_v, tau_x, V, Z):
        self.tau_v = tau_v

        self.tau_x = tau_x

        self.V = V

        self.Z = Z

    @property
    def D(self):
        return self.V.shape[1]

    @property
    def K(self):
        return self.Z.shape[1]

    @property
    def N(self):
        return self.Z.shape[0]

    def copy(self):
        return Parameters(
            self.tau_v,
            self.tau_x,
            self.V.copy(),
            self.Z.copy()
        )


class Priors(object):
    def __init__(self, tau_v=None, tau_x=None):
        if tau_v is None:
            tau_v = np.ones(2)

        self.tau_v = tau_v

        if tau_x is None:
            tau_x = np.ones(2)

        self.tau_x = tau_x


#=========================================================================
# Densities and proposals
#=========================================================================
class CollapsedDataDistribution(object):
    def log_p(self, data, params):
        return self.log_p_row(data, params, None)

    def log_p_row(self, data, params, row_idx):
        t_v = params.tau_v
        t_x = params.tau_x
        Z = params.Z
        X = data

        D = params.D
        K = params.K
        N = params.N

        M = np.linalg.inv(Z.T @ Z + (t_v / t_x) * np.eye(K))

        log_p = 0

        log_p += 0.5 * (N - K) * D * t_x

        log_p += 0.5 * K * D * t_v

        log_p -= 0.5 * N * D * np.log(2 * np.pi)

        log_p += 0.5 * D * np.log(np.linalg.det(M))

        log_p -= 0.5 * t_x * np.trace(X.T @ (np.eye(N) - Z @ M @ Z.T) @ X)

        return log_p


class CollapsedJointDistribution(object):
    def __init__(self, feat_alloc_prior, priors):
        self.data_dist = CollapsedDataDistribution()

        self.feat_alloc_prior = feat_alloc_prior

        self.priors = priors

    def log_p(self, data, params):
        t_v = params.tau_v
        t_x = params.tau_x

        log_p = 0

        # Binary matrix prior
        log_p += self.feat_alloc_prior.log_p(params.Z)

        # Data
        log_p += self.data_dist.log_p(data, params)

        # Gamma prior on $\tau_{a}$
        a = self.priors.tau_v[0]
        b = self.priors.tau_v[1]
        log_p += scipy.stats.gamma.logpdf(t_v, a, scale=(1 / b))

        # Gamma prior on $\tau_{x}$
        a = self.priors.tau_x[0]
        b = self.priors.tau_x[1]
        log_p += scipy.stats.gamma.logpdf(t_x, a, scale=(1 / b))

        return log_p


class UncollapsedDataDistribution(object):
    def __init__(self):
        pass

    def log_p(self, data, params):
        t_x = params.tau_x

        D = params.D
        N = params.N
        V = params.V
        Z = params.Z
        X = data

        resid = (X - Z @ V)

        log_p = 0.5 * N * D * (np.log(t_x) - np.log(2 * np.pi))

        log_p -= 0.5 * t_x * np.nansum(np.square(resid))

        return log_p

    def log_p_row(self, data, params, row_idx):
        return _log_p_row(params.tau_x, data[row_idx], params.Z[row_idx].astype(float), params.V)


@numba.njit(cache=True)
def _log_p_row(t_x, x, z, V):
    D = V.shape[1]

    log_p = 0

    log_p += 0.5 * D * (np.log(t_x) - np.log(2 * np.pi))

    log_p -= 0.5 * t_x * np.nansum(np.square(x - z @ V))

    return log_p


class UncollapsedJointDistribution(object):
    def __init__(self, feat_alloc_prior, priors):
        self.data_dist = UncollapsedDataDistribution()

        self.feat_alloc_prior = feat_alloc_prior

        self.priors = priors

    def log_p(self, data, params):
        t_v = params.tau_v
        t_x = params.tau_x
        V = params.V

        D = params.D
        K = params.K

        log_p = 0

        # Binary matrix prior
        log_p += self.feat_alloc_prior.log_p(params.Z)

        # Data
        log_p += self.data_dist.log_p(data, params)

        # Gamma prior on $\tau_{a}$
        a = self.priors.tau_v[0]
        b = self.priors.tau_v[1]
        log_p += scipy.stats.gamma.logpdf(t_v, a, scale=(1 / b))

        # Gamma prior on $\tau_{x}$
        a = self.priors.tau_x[0]
        b = self.priors.tau_x[1]
        log_p += scipy.stats.gamma.logpdf(t_x, a, scale=(1 / b))

        # Prior on V
        log_p += scipy.stats.matrix_normal.logpdf(
            V,
            mean=np.zeros((K, D)),
            colcov=np.eye(D),
            rowcov=(1 / t_v) * np.eye(K)
        )

        return log_p


#=========================================================================
# Singletons updaters
#=========================================================================
class PriorSingletonsUpdater(object):
    def update_row(self, model, row_idx):
        D = model.params.D
        N = model.params.N
        t_v = model.params.tau_v
        t_x = model.params.tau_x

        k_old = len(self._get_singleton_idxs(model.params.Z, row_idx))

        k_new = model.feat_alloc_prior.sample_num_singletons(model.params.Z)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = self._get_non_singleton_idxs(model.params.Z, row_idx)

        num_non_singletons = len(non_singleton_idxs)

        K_new = len(non_singleton_idxs) + k_new

        V_new = np.zeros((K_new, D))

        V_new[:num_non_singletons] = model.params.V[non_singleton_idxs]

        V_new[num_non_singletons:] = np.random.multivariate_normal(np.zeros(D), (1 / t_v) * np.eye(D), size=k_new)

        Z_new = np.zeros((N, K_new))

        Z_new[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

        Z_new[row_idx, num_non_singletons:] = 1

        params_new = Parameters(t_v, t_x, V_new, Z_new)

        log_p_new = model.data_dist.log_p_row(model.data, params_new, row_idx)

        log_p_old = model.data_dist.log_p_row(model.data, model.params, row_idx)

        diff = log_p_new - log_p_old

        if diff > np.log(np.random.rand()):
            params = params_new

        else:
            params = model.params

        return params

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


class CollapsedSingletonUpdater(object):
    def update_row(self, model, row_idx):
        alpha = model.feat_alloc_prior.alpha
        t_v = model.params.tau_v
        t_x = model.params.tau_x
        D = model.params.D
        N = model.params.N
        V = model.params.V
        Z = model.params.Z
        X = model.data

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
                V[num_non_singletons:] = self._sample_new_V(k_new, model.data, model.params, row_idx)

            params = Parameters(model.params.tau_v, model.params.tau_x, V, Z)

        else:
            params = model.params

        return params

    def _sample_new_V(self, k, data, params, row_idx):
        D = params.D
        N = params.N
        V = params.V
        Z = params.Z
        t_v = params.tau_v
        t_x = params.tau_x
        X = data

        m = np.sum(Z)

        m -= Z[row_idx]

        non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))

        V_old = V[non_singletons_idxs]

        Z_old = Z[:, non_singletons_idxs]

        Z_new = np.zeros((N, k))

        Z_new[row_idx] = 1

        M = np.linalg.inv(Z_new.T @ Z_new + (t_v / t_x) * np.eye(k))

        return scipy.stats.matrix_normal.rvs(
            mean=M @ Z_new.T @ (X - Z_old @ V_old),
            rowcov=(1 / t_x) * M,
            colcov=np.eye(D)
        )

    def _get_log_p(self, k, data, params, row_idx):
        D = params.D
        N = params.N
        V = params.V
        Z = params.Z
        t_v = params.tau_v
        t_x = params.tau_x
        X = data

        m = np.sum(Z)

        m -= Z[row_idx]

        non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))

        V_old = V[non_singletons_idxs]

        Z_old = Z[:, non_singletons_idxs]

        Z_new = np.zeros((N, k))

        Z_new[row_idx] = 1

        M = np.linalg.inv(Z_new.T @ Z_new + (t_v / t_x) * np.eye(k))

        temp = Z_new.T @ (X - Z_old @ V_old)

        log_p = 0
        log_p += 0.5 * k * D * (np.log(t_v) - np.log(t_x))
        log_p += 0.5 * D * np.log(np.linalg.det(M))
        log_p += 0.5 * t_x * np.trace(temp.T @ M @ temp)

        return log_p
