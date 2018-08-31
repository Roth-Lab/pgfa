import numba
import numpy as np
import scipy.linalg
import scipy.stats

from pgfa.math_utils import cholesky_log_det, cholesky_update, ffa_rvs, ibp_rvs, log_ffa_pdf, log_ibp_pdf

import pgfa.updates.feature_matrix


class CollapsedLinearGaussianModel(object):
    def __init__(self, data, K=None, params=None, priors=None):
        self.data = data

        self.ibp = (K is None)

        if params is None:
            params = get_params_from_data(data, K=K)

        self.params = params

        if priors is None:
            priors = LinearGaussianPriors()

        self.priors = priors

    def log_p(self, data, params):
        """ Log of joint pdf.
        """
        pass

    def log_predictive_p(self, data, params):
        """ Log of predicitve pdf.
        """
        pass

    def update(self, num_particles=10, resample_threshold=0.5, update_type='g'):
        update_Z_collapsed(
            self.data,
            self.params,
            ibp=self.ibp,
            num_particles=num_particles,
            resample_threshold=resample_threshold,
            update_type=update_type
        )

        update_alpha(self.params, self.priors)

        update_V(self.data, self.params)

        update_tau_a(self.params, self.priors)

        update_tau_x(self.data, self.params, self.priors)


class LinearGaussianModel(object):
    def __init__(self, data, K=None, params=None, priors=None):
        self.data = data

        self.ibp = (K is None)

        if params is None:
            params = get_params_from_data(data, K=K)

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
        V = self.params.V
        Z = self.params.Z
        X = self.data

        D = self.params.D
        K = self.params.K
        N = self.params.N

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

        # Prior on V
        log_p += scipy.stats.matrix_normal.logpdf(
            V,
            mean=np.zeros((K, D)),
            colcov=np.eye(D),
            rowcov=(1 / t_x) * np.eye(K)
        )

        log_p += scipy.stats.matrix_normal.logpdf(
            X,
            mean=Z @ V,
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

        update_alpha(self.params, self.priors)

        update_V(self.data, self.params)

        update_tau_a(self.params, self.priors)

        update_tau_x(self.data, self.params, self.priors)


#=========================================================================
# Updates
#=========================================================================
def update_alpha(params, priors):
    a = params.K + priors.alpha[0]

    b = np.sum(1 / np.arange(1, params.N + 1)) + priors.alpha[1]

    params.alpha = np.random.gamma(a, 1 / b)


def update_V(data, params):
    t_a = params.tau_a
    t_x = params.tau_x
    Z = params.Z
    X = data

    if params.K == 0:
        return

    M = np.linalg.inv(Z.T @ Z + (t_a / t_x) * np.eye(params.K))

    params.V = scipy.stats.matrix_normal.rvs(
        mean=M @ Z.T @ X,
        rowcov=(1 / t_x) * M,
        colcov=np.eye(params.D)
    )


def update_tau_a(params, priors):
    V = params.V

    a = priors.tau_a[0] + 0.5 * params.K * params.D

    b = priors.tau_a[1] + 0.5 * np.trace(V.T @ V)

    params.tau_a = np.random.gamma(a, 1 / b)


def update_tau_x(data, params, priors):
    V = params.V
    Z = params.Z
    X = data

    a = priors.tau_x[0] + 0.5 * params.N * params.D

    Y = X - Z @ V

    b = priors.tau_x[1] + 0.5 * np.trace(Y.T @ Y)

    params.tau_x = np.random.gamma(a, 1 / b)


def update_Z(data, params, ibp=False, num_particles=10, resample_threshold=0.5, update_type='g'):
    alpha = params.alpha
    V = params.V
    Z = params.Z
    X = data

    a_0, b_0 = _get_feature_priors(alpha, params.K, ibp)

    density = LinearGaussianDensity(params.tau_x)

    proposal = MultivariateGaussianProposal(np.zeros(params.D), (1 / params.tau_a) * np.eye(params.D))

    for row_idx in pgfa.updates.feature_matrix.get_rows(params.N):
        x = X[row_idx]

        z = Z[row_idx]

        m = np.sum(Z, axis=0)

        m -= z

        a = a_0 + m

        b = b_0 + (params.N - 1 - m)

        cols = pgfa.updates.feature_matrix.get_cols(m, include_singletons=(not ibp))

        if update_type == 'g':
            Z[row_idx] = pgfa.updates.feature_matrix.do_gibbs_update(density, a, b, cols, x, z, V)

        elif update_type == 'pg':
            Z[row_idx] = pgfa.updates.feature_matrix.do_particle_gibbs_update(
                density, a, b, cols, x, z, V,
                annealed=False, num_particles=num_particles, resample_threshold=resample_threshold
            )

        elif update_type == 'pga':
            Z[row_idx] = pgfa.updates.feature_matrix.do_particle_gibbs_update(
                density, a, b, cols, x, z, V,
                annealed=True, num_particles=num_particles, resample_threshold=resample_threshold
            )

        elif update_type == 'rg':
            Z[row_idx] = pgfa.updates.feature_matrix.do_row_gibbs_update(density, a, b, cols, x, z, V)

        if ibp:
            V, Z = pgfa.updates.feature_matrix.do_mh_singletons_update(
                row_idx, density, proposal, alpha, V, X, Z
            )

    params.V = V

    params.Z = Z


def update_Z_collapsed(data, params, ibp=False, num_particles=10, resample_threshold=0.5, update_type='g'):
    alpha = params.alpha
    Z = params.Z
    X = data

    a_0, b_0 = _get_feature_priors(alpha, params.K, ibp)

    for row_idx in pgfa.updates.feature_matrix.get_rows(params.N):
        density = LinearGaussianMarginalDensity(row_idx, params.tau_a, params.tau_x, Z, rank_one=False)

        m = np.sum(Z, axis=0)

        m -= Z[row_idx]

        a = a_0 + m

        b = b_0 + (params.N - 1 - m)

        cols = pgfa.updates.feature_matrix.get_cols(m, include_singletons=(not ibp))

        if update_type == 'g':
            Z[row_idx] = pgfa.updates.feature_matrix.do_collaped_gibbs_update(density, a, b, cols, row_idx, X, Z)

        elif update_type == 'pg':
            Z[row_idx] = pgfa.updates.feature_matrix.do_collapsed_particle_gibbs_update(
                density, a, b, cols, row_idx, X, Z,
                annealed=False, num_particles=num_particles, resample_threshold=resample_threshold
            )

        elif update_type == 'pga':
            Z[row_idx] = pgfa.updates.feature_matrix.do_collapsed_particle_gibbs_update(
                density, a, b, cols, row_idx, X, Z,
                annealed=True, num_particles=num_particles, resample_threshold=resample_threshold
            )

        elif update_type == 'rg':
            Z[row_idx] = pgfa.updates.feature_matrix.do_collapsed_row_gibbs_update(density, a, b, cols, row_idx, X, Z)

        if ibp:
            density = LinearGaussianMarginalDensity(row_idx, params.tau_a, params.tau_x, Z, rank_one=False)

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
def get_params_from_data(data, K=None):
    D = data.shape[1]
    N = data.shape[0]

    if K is None:
        Z = ibp_rvs(1, N)

    else:
        Z = ffa_rvs(1, 1, K, N)

    K = Z.shape[1]

    V = np.random.multivariate_normal(np.zeros(D), np.eye(D), size=K)

    return LinearGaussianParameters(1, 1, 1, V, Z)


class LinearGaussianParameters(object):
    def __init__(self, alpha, tau_a, tau_x, V, Z):
        self.alpha = alpha

        self.tau_a = tau_a

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
        return LinearGaussianParameters(
            self.alpha,
            self.tau_a,
            self.tau_x,
            self.V.copy(),
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
    def __init__(self, row, t_a, t_x, Z, rank_one=False):
        self.row = row

        self.t_a = t_a

        self.t_x = t_x

        self.rank_one = rank_one

        if rank_one:
            K = Z.shape[1]

            M_inv = Z.T @ Z + (t_a / t_x) * np.eye(K)

            L, _ = scipy.linalg.cho_factor(M_inv, lower=True)

            z = Z[row]

            cholesky_update(L, z, alpha=-1, inplace=True)

            self.L_i = L

    def log_p(self, X, Z):
        t_a = self.t_a
        t_x = self.t_x

        D = X.shape[1]
        K = Z.shape[1]
        N = X.shape[0]

        log_p = 0
        log_p += 0.5 * N * D * np.log(2 * np.pi)
        log_p += 0.5 * (N - K) * D * t_x
        log_p += 0.5 * K * D * t_a

        if self.rank_one:
            L_i = self.L_i

            z = Z[self.row]

            L = cholesky_update(L_i, z, alpha=1, inplace=False)

            MZ = scipy.linalg.cho_solve((L, True), Z.T)

            log_det_M = -1 * cholesky_log_det(L)

            log_p += 0.5 * D * log_det_M
            log_p -= 0.5 * t_x * np.trace(X.T @ (np.eye(N) - Z @ MZ) @ X)

        else:
            M = np.linalg.inv(Z.T @ Z + (t_a / t_x) * np.eye(K))

            log_p += 0.5 * D * np.prod(np.linalg.slogdet(M))
            log_p -= 0.5 * t_x * np.trace(X.T @ (np.eye(N) - Z @ M @ Z.T) @ X)

        return log_p


class MultivariateGaussianProposal(object):
    def __init__(self, mean, covariance):
        self.mean = mean

        self.covariance = covariance

    def rvs(self, size=None):
        return np.random.multivariate_normal(self.mean, self.covariance, size=size)
