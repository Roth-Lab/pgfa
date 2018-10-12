import numba
import numpy as np
import scipy.stats


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
    X = data

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
    X = data

    a = priors.tau_x[0] + 0.5 * params.N * params.D

    Y = X - Z @ V

    b = priors.tau_x[1] + 0.5 * np.trace(Y.T @ Y)

    params.tau_x = np.random.gamma(a, 1 / b)

    return params


#=========================================================================
# Container classes
#=========================================================================
@numba.jitclass([
    ('tau_v', numba.float64),
    ('tau_x', numba.float64),
    ('V', numba.float64[:, :]),
    ('Z', numba.int64[:, :])
])
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
        return self.log_p_row(data, params, -1)

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


@numba.jitclass([])
class UncollapsedDataDistribution(object):
    def __init__(self):
        pass

    def log_p(self, data, params):
        log_p = 0

        for row_idx in range(params.N):
            log_p += self.log_p_row(data, params, row_idx)

        return log_p

    def log_p_row(self, data, params, row_idx):
        t_x = params.tau_x
        V = params.V
        x = data[row_idx]
        z = params.Z[row_idx]

        D = params.D

        log_p = 0.5 * D * (np.log(t_x) - np.log(2 * np.pi))

        for d in range(D):
            m = np.sum(V[:, d] * z)

            log_p -= 0.5 * t_x * (x[d] - m)**2

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


class SingletonsProposal(object):
    def rvs(self, data, params, num_singletons, row_idx):
        m = np.sum(params.Z, axis=0)

        m -= params.Z[row_idx]

        non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))

        num_non_singleton = len(non_singletons_idxs)

        K_new = num_non_singleton + num_singletons

        Z_new = np.zeros((params.N, K_new), dtype=np.int64)

        Z_new[:, :num_non_singleton] = params.Z[:, non_singletons_idxs]

        Z_new[row_idx, num_non_singleton:] = 1

        V_new = np.zeros((K_new, params.D))

        V_new[:num_non_singleton] = params.V[non_singletons_idxs]

        V_new[num_non_singleton:] = np.random.multivariate_normal(
            np.zeros(params.D), (1 / params.tau_v) * np.eye(params.D), size=num_singletons
        )

        return Parameters(
            params.tau_v,
            params.tau_x,
            V_new,
            Z_new
        )
