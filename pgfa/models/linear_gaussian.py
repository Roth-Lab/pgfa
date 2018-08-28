import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import ffa_rvs, ibp_rvs, log_ffa_pdf, log_ibp_pdf

import pgfa.updates.feature_matrix


class CollapsedLinearGaussianModel(object):
    def log_p(self, data, params):
        """ Log of joint pdf.
        """
        pass

    def log_predictive_p(self, data, params):
        """ Log of predicitve pdf.
        """
        pass

    def update(self, update_type='gibbs'):
        update_Z_collapsed(self.data, self.params, ibp=self.ibp, update_type=update_type)

        update_alpha(self.data, self.params, self.priors)

        update_A(self.data, self.params)

        update_tau_a(self.params, self.priors)

        update_tau_x(self.data, self.params, self.priors)


class LinearGaussianModel(object):
    def __init__(self, data, K=None, params=None, priors=None):
        self.data = data

        self.ibp = (K is None)

        if params is None:
            params = LinearGaussianParameters.get_from_data(data, K=K)

        self.params = params

        if priors is None:
            priors = LinearGaussianPriors()

        self.priors = priors

    @property
    def log_p(self):
        """ Log of joint pdf.
        """
        alpha = self.params.alpha
        t_a = self.params.tau_a
        t_x = self.params.tau_x
        A = self.params.A
        Z = self.params.Z
        X = self.data

        D = X.shape[1]
        K = Z.shape[1]
        N = Z.shape[0]

        log_p = 0

        # Gamma prior on $\tau_{a}$
        a = self.priors.tau_a[0]
        b = self.priors.tau_a[1]
        log_p += scipy.stats.gamma.logpdf(t_a, a, scale=(1 / b))

        # Gamma prior on $\tau_{x}$
        a = self.priors.tau_x[0]
        b = self.priors.tau_x[1]
        log_p += scipy.stats.gamma.logpdf(t_x, a, scale=(1 / b))

        # Prior on Z
        if self.ibp:
            log_p += log_ibp_pdf(self.params.alpha, self.params.Z)

        else:
            log_p += log_ffa_pdf(alpha / K, 1, Z)

        # Prior on A
        log_p += scipy.stats.matrix_normal.logpdf(
            A,
            mean=np.zeros((K, D)),
            colcov=np.eye(D),
            rowcov=(1 / t_x) * np.eye(K)
        )

        log_p += scipy.stats.matrix_normal.logpdf(
            X,
            mean=Z @ A,
            colcov=np.eye(D),
            rowcov=(1 / t_x) * np.eye(N)
        )

        return log_p

    def log_predictive_p(self, data, params):
        """ Log of predicitve pdf.
        """
        pass

    def update(self, num_particles=10, resample_threshold=0.5, update_type='g'):
        update_Z(
            self.data,
            self.params,
            ibp=self.ibp,
            num_particles=num_particles,
            resample_threshold=resample_threshold,
            update_type=update_type
        )

        update_alpha(self.data, self.params, self.priors)

        update_A(self.data, self.params)

        update_tau_a(self.params, self.priors)

        update_tau_x(self.data, self.params, self.priors)


#=========================================================================
# Updates
#=========================================================================
def update_alpha(data, params, priors):
    D = data.shape[0]
    K = params.Z.shape[1]

    a = K + priors.alpha[0]

    b = np.sum(1 / np.arange(1, D + 1)) + priors.alpha[1]

    params.alpha = np.random.gamma(a, 1 / b)


def update_A(data, params):
    t_a = params.tau_a
    t_x = params.tau_x
    Z = params.Z
    X = data

    D = X.shape[1]
    K = Z.shape[1]

    M = np.linalg.inv(Z.T @ Z + (t_a / t_x) * np.eye(K))

    params.A = scipy.stats.matrix_normal.rvs(
        mean=M @ Z.T @ X,
        rowcov=(1 / t_x) * M,
        colcov=np.eye(D)
    )


def update_tau_a(params, priors):
    A = params.A

    D = A.shape[1]
    K = A.shape[0]

    a = priors.tau_a[0] + 0.5 * K * D

    b = priors.tau_a[1] + 0.5 * np.trace(A.T @ A)

    params.tau_a = np.random.gamma(a, 1 / b)


def update_tau_x(data, params, priors):
    A = params.A
    Z = params.Z
    X = data

    D = X.shape[1]
    N = X.shape[0]

    a = priors.tau_x[0] + 0.5 * N * D

    Y = X - Z @ A

    b = priors.tau_x[1] + 0.5 * np.trace(Y.T @ Y)

    params.tau_x = np.random.gamma(a, 1 / b)


def update_Z(data, params, ibp=False, num_particles=10, resample_threshold=0.5, update_type='g'):
    alpha = params.alpha
    A = params.A
    Z = params.Z
    X = data

    D = X.shape[1]
    K = Z.shape[1]
    N = Z.shape[0]

    a_0, b_0 = _get_feature_priors(alpha, K, ibp)

    density = LinearGaussianDensity(params.tau_x)

    proposal = MultivariateGaussianProposal(np.zeros(D), 1 / params.tau_a * np.eye(D))

    for row_idx in pgfa.updates.feature_matrix.get_rows(N):
        x = X[row_idx]

        z = Z[row_idx]

        m = np.sum(Z, axis=0)

        m -= z

        a = a_0 + m

        b = b_0 + (N - m)

        cols = pgfa.updates.feature_matrix.get_cols(m, include_singletons=(not ibp))

        if update_type == 'g':
            Z[row_idx] = pgfa.updates.feature_matrix.do_gibbs_update(density, a, b, cols, x, z, A)

        elif update_type == 'pg':
            Z[row_idx] = pgfa.updates.feature_matrix.do_particle_gibbs_update(
                density, a, b, cols, x, z, A,
                annealed=False, num_particles=num_particles, resample_threshold=resample_threshold
            )

        elif update_type == 'pga':
            Z[row_idx] = pgfa.updates.feature_matrix.do_particle_gibbs_update(
                density, a, b, cols, x, z, A,
                annealed=True, num_particles=num_particles, resample_threshold=resample_threshold
            )

        elif update_type == 'rg':
            Z[row_idx] = pgfa.updates.feature_matrix.do_row_gibbs_update(density, a, b, cols, x, z, A)

        if ibp:
            A, Z = pgfa.updates.feature_matrix.do_mh_singletons_update(
                row_idx, density, proposal, alpha, A, X, Z
            )

    params.A = A

    params.Z = Z


def update_Z_collapsed(data, params, ibp=False, update_type='gibbs'):
    alpha = params.alpha
    Z = params.Z
    X = data

    K = Z.shape[1]
    N = Z.shape[0]

    a_0, b_0 = _get_feature_priors(alpha, K, ibp)

    density = LinearGaussianMarginalDensity(params.tau_a, params.tau_x)

    for row_idx in pgfa.updates.feature_matrix.get_rows(N):
        m = np.sum(Z, axis=0)

        m -= Z[row_idx]

        a = a_0 + m

        b = b_0 + (N - m)

        cols = pgfa.updates.feature_matrix.get_cols(m, include_singletons=(not ibp))

        if update_type == 'g':
            Z[row_idx] = pgfa.updates.feature_matrix.do_collaped_gibbs_update(density, a, b, cols, row_idx, X, Z)

        elif update_type == 'rg':
            Z[row_idx] = pgfa.updates.feature_matrix.do_collapsed_row_gibbs_update(density, a, b, cols, row_idx, X, Z)

        if ibp:
            Z = pgfa.updates.feature_matrix.do_collapsed_mh_singletons_update(row_idx, density, alpha, X, Z)

    params.Z = Z


def _get_feature_priors(alpha, K, ibp):
    if ibp:
        a_0 = 0
        b_0 = 0

    else:
        a_0 = alpha / K
        b_0 = 1

    return a_0, b_0


#=========================================================================
# Container classes
#=========================================================================
class LinearGaussianParameters(object):
    @staticmethod
    def get_from_data(data, K=None):
        D = data.shape[1]
        N = data.shape[0]

        if K is None:
            Z = ibp_rvs(1, N)

        else:
            Z = ffa_rvs(1, 1, K, N)

        K = Z.shape[1]

        A = np.random.multivariate_normal(np.zeros(D), np.eye(D), size=K)

        return LinearGaussianParameters(1, 1, 1, A, Z)

    def __init__(self, alpha, tau_a, tau_x, A, Z):
        self.alpha = alpha

        self.tau_a = tau_a

        self.tau_x = tau_x

        self.A = A

        self.Z = Z

    def copy(self):
        return LinearGaussianParameters(
            self.alpha,
            self.tau_a,
            self.tau_x,
            self.A.copy(),
            self.Z.copy()
        )


class LinearGaussianPriors(object):
    def __init__(self, alpha=None, tau_a=None, tau_x=None):
        if alpha is None:
            alpha = np.ones(2)

        self.alpha = alpha

        if tau_a is None:
            tau_a = np.ones(2)

        self.tau_a = tau_a

        if tau_x is None:
            tau_x = np.ones(2)

        self.tau_x = tau_x


#=========================================================================
# Densities and proposals
#=========================================================================
@numba.jitclass([
    ('t_x', numba.float64)
])
class LinearGaussianDensity(object):
    def __init__(self, t_x):
        self.t_x = t_x

    def log_p(self, x, z, V):
        t_x = self.t_x

        D = V.shape[1]

        log_p = 0.5 * D * (np.log(t_x) - np.log(2 * np.pi))

        for d in range(D):
            m = np.sum(V[:, d] * z)

            log_p -= 0.5 * t_x * (x[d] - m)**2

        return log_p


class LinearGaussianMarginalDensity(object):
    def __init__(self, t_a, t_x):
        self.t_a = t_a

        self.t_x = t_x

    def log_p(self, X, Z):
        t_a = self.t_a
        t_x = self.t_x

        D = X.shape[1]
        K = Z.shape[1]
        N = X.shape[0]

        M = np.linalg.inv(Z.T @ Z + (t_a / t_x) * np.eye(K))

        XZ = X.T @ Z

        XX = X.T @ X

        log_p = 0
        log_p += 0.5 * (N - K) * D * np.log(t_x)
        log_p -= 0.5 * N * D * np.log(2 * np.pi)
        log_p -= 0.5 * D * np.log(np.linalg.det(M))
        log_p -= 0.5 * t_x * np.trace(XX - XZ @ M @ XZ.T)

        return log_p


class MultivariateGaussianProposal(object):
    def __init__(self, mean, covariance):
        self.mean = mean

        self.covariance = covariance

    def rvs(self, size=None):
        return np.random.multivariate_normal(self.mean, self.covariance, size=size)
