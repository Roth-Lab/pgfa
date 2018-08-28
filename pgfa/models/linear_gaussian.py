import numba
import numpy as np
import scipy.stats

from pgfa.stats import ffa_rvs, ibp_rvs
from pgfa.updates import do_gibbs_update, do_row_gibbs_update, do_particle_gibbs_update, do_collaped_gibbs_update, \
    do_collapsed_row_gibbs_update, get_cols, get_rows


def get_params(D, N, K=None):
    if K is None:
        Z = ibp_rvs(1, N)

    else:
        Z = ffa_rvs(1, 1, K, N)

    K = Z.shape[1]

    return LinearGaussianParameters(
        1,
        1,
        1,
        np.random.multivariate_normal(np.zeros(D), np.eye(D), size=K),
        Z
    )


def get_priors():
    return LinearGaussianPriors(
        np.ones(2),
        np.ones(2),
        np.ones(2)
    )


class LinearGaussianParameters(object):
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
    def __init__(self, alpha, tau_a, tau_x):
        self.alpha = alpha

        self.tau_a = tau_a

        self.tau_x = tau_x


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


class LinearGaussianModel(object):
    def __init__(self, data, K=None, params=None, priors=None):
        self.data = data

        self.K = K

        if params is None:
            params = get_params(data.shape[1], data.shape[0], K=self.K)

        if priors is None:
            priors = get_priors()

        self.params = params

        self.priors = priors

    @property
    def feature_priors(self):
        K = self.params.Z.shape[1]

        if self.ibp:
            a_0 = 0
            b_0 = 0

        else:
            a_0 = self.params.alpha / K
            b_0 = 1

        return a_0, b_0

    @property
    def ibp(self):
        return self.K is None

    def get_log_p_X(self, A, X, Z):
        X = np.atleast_2d(X)
        Z = np.atleast_2d(Z)

        density = LinearGaussianDensity(self.params.tau_x)

        log_p = 0

        for x, z in zip(X, Z):
            log_p += density.log_p(x, z, A)

        return log_p

    def get_log_p_X_collapsed(self, X, Z):
        X = np.atleast_2d(X)
        Z = np.atleast_2d(Z)

        density = LinearGaussianMarginalDensity(self.params.tau_a, self.params.tau_x)

        return density.log_p(X, Z)

    def _update_A(self):
        t_a = self.params.tau_a
        t_x = self.params.tau_x
        Z = self.params.Z
        X = self.data

        D = X.shape[1]
        K = Z.shape[1]

        M = np.linalg.inv(Z.T @ Z + (t_a / t_x) * np.eye(K))

        self.params.A = scipy.stats.matrix_normal.rvs(
            mean=M @ Z.T @ X,
            rowcov=(1 / t_x) * M,
            colcov=np.eye(D)
        )

    def _update_tau_a(self):
        A = self.params.A

        D = A.shape[1]
        K = A.shape[0]

        a = self.priors.tau_a[0] + 0.5 * K * D

        b = self.priors.tau_a[1] + 0.5 * np.trace(A.T @ A)

        self.params.tau_a = np.random.gamma(a, 1 / b)

    def _update_tau_x(self):
        A = self.params.A
        Z = self.params.Z
        X = self.data

        D = X.shape[1]
        N = X.shape[0]

        a = self.priors.tau_x[0] + 0.5 * N * D

        Y = X - Z @ A

        b = self.priors.tau_x[1] + 0.5 * np.trace(Y.T @ Y)

        self.params.tau_x = np.random.gamma(a, 1 / b)


class LinearGaussianUncollapsedModel(LinearGaussianModel):
    def update(self, num_particles=10, resample_threshold=0.5, update_type='g'):
        self._update_Z(num_particles=num_particles, resample_threshold=resample_threshold, update_type=update_type)

        self._update_A()

        self._update_tau_a()

        self._update_tau_x()

    def _update_Z(self, num_particles=10, resample_threshold=0.5, update_type='g'):
        V = self.params.A
        Z = self.params.Z
        X = self.data

        N = self.params.Z.shape[0]

        a_0, b_0 = self.feature_priors

        density = LinearGaussianDensity(self.params.tau_x)

        for n in get_rows(N):
            x = X[n]

            z = Z[n]

            m = np.sum(Z, axis=0)

            m -= z

            a = a_0 + m

            b = b_0 + (N - m)

            cols = get_cols(m, include_singletons=(not self.ibp))

            if update_type == 'g':
                Z[n] = do_gibbs_update(density, a, b, cols, x, z, V)

            elif update_type == 'pg':
                Z[n] = do_particle_gibbs_update(
                    density, a, b, cols, x, z, V,
                    annealed=False, num_particles=num_particles, resample_threshold=resample_threshold
                )

            elif update_type == 'pga':
                Z[n] = do_particle_gibbs_update(
                    density, a, b, cols, x, z, V,
                    annealed=True, num_particles=num_particles, resample_threshold=resample_threshold
                )

            elif update_type == 'rg':
                Z[n] = do_row_gibbs_update(density, a, b, cols, x, z, V)

            if self.ibp:
                self._update_Z_singletons(n)

    def _update_Z_singletons(self, row):
        alpha = self.params.alpha
        t_a = self.params.tau_a
        A = self.params.A
        Z = self.params.Z
        X = self.data

        D = X.shape[1]
        N = Z.shape[0]

        m = np.sum(Z, axis=0)

        m -= Z[row]

        non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))

        K_non_singleton = len(non_singletons_idxs)

        K_new = K_non_singleton + np.random.poisson(alpha / N)

        z_A = Z[row, non_singletons_idxs]

        # Current state
        log_p_old = self.get_log_p_X(A, X[row], Z[row])

        # New state
        z_new = np.ones(K_new)

        z_new[:K_non_singleton] = z_A

        A_new = np.zeros((K_new, D))

        A_new[:K_non_singleton] = A[non_singletons_idxs]

        A_new[K_non_singleton:] = np.random.multivariate_normal(np.zeros(D), (1 / t_a) * np.eye(D))

        log_p_new = self.get_log_p_X(A_new, X[row], z_new)

        # MH step
        diff = log_p_new - log_p_old

        u = np.random.rand()

        if diff > np.log(u):
            self.params.A = A_new

            Z_new = np.zeros((N, K_new))

            Z_new[:, :K_non_singleton] = Z[:, non_singletons_idxs]

            Z_new[row, K_non_singleton:] = 1

            self.params.Z = Z_new


class LinearGaussianCollapsedModel(LinearGaussianModel):
    def update(self, update_type='gibbs'):
        self._update_Z(update_type=update_type)

        self._update_A()

        self._update_tau_a()

        self._update_tau_x()

    def _update_Z(self, update_type='gibbs'):
        Z = self.params.Z
        X = self.data

        N = Z.shape[0]

        a_0, b_0 = self.feature_priors

        density = LinearGaussianMarginalDensity(self.params.tau_a, self.params.tau_x)

        for n in get_rows(N):
            m = np.sum(Z, axis=0)

            m -= Z[n]

            a = a_0 + m

            b = b_0 + (N - m)

            cols = get_cols(m, include_singletons=(not self.ibp))

            if update_type == 'gibbs':
                Z[n] = do_collaped_gibbs_update(density, a, b, cols, n, X, Z)

            elif update_type == 'row-gibbs':
                Z[n] = do_collapsed_row_gibbs_update(density, a, b, cols, n, X, Z)

            if self.ibp:
                self._update_Z_singletons(n)

    def _update_Z_singletons(self, row):
        alpha = self.params.alpha
        X = self.data
        Z = self.params.Z

        K = Z.shape[1]
        N = Z.shape[0]

        m = np.sum(Z, axis=0)

        m -= Z[row]

        cols = np.atleast_1d(np.squeeze(np.where(m > 0)))

        K_non_singleton = len(cols)

        K_new = K + np.random.poisson(alpha / N)

        Z_new = np.zeros((N, K_new), dtype=np.int64)

        Z_new[:, :K_non_singleton] = Z[:, cols]

        Z_new[row, K_non_singleton:] = 1

        log_p_old = self.get_log_p_X_collapsed(X, Z)

        log_p_new = self.get_log_p_X_collapsed(X, Z_new)

        diff = log_p_new - log_p_old

        u = np.random.rand()

        if diff > np.log(u):
            self.params.Z = Z_new
