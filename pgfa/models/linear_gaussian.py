import itertools
import numpy as np
import scipy.stats

from pgfa.math_utils import log_normalize
from pgfa.stats import discrete_rvs, ffa_rvs, ibp_rvs


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
    def ibp(self):
        return self.K is None

    def get_log_p_X(self, X, Z):
        A = self.params.A
        t_x = self.params.tau_x

        X = np.atleast_2d(X)
        Z = np.atleast_2d(Z)

        D = X.shape[1]
        N = X.shape[0]

        Y = X - np.dot(Z, A)

        return sum([
            0.5 * N * D * np.log(t_x),
            -0.5 * N * D * np.log(2 * np.pi),
            -0.5 * t_x * np.trace(np.dot(Y.T, Y))
        ])

    def get_log_p_X_collapsed(self, X, Z):
        t_a = self.params.tau_a
        t_x = self.params.tau_x

        D = X.shape[1]
        K = Z.shape[1]
        N = X.shape[0]

        M = self._get_M(t_a, t_x, Z)

        XZ = np.dot(X.T, Z)

        XX = np.dot(X.T, X)

        return sum([
            0.5 * (N - K) * D * np.log(t_x),
            0.5 * K * D * np.log(t_a),
            -0.5 * N * D * np.log(2 * np.pi),
            -0.5 * D * np.log(np.linalg.det(M)),
            -0.5 * t_x * np.trace(XX - np.dot(XZ, np.dot(M, XZ.T)))
        ])

    def _get_cols(self, m):
        K = len(m)

        if self.ibp:
            cols = np.atleast_1d(np.squeeze(np.where(m > 0)))

        else:
            cols = np.arange(K)

        np.random.shuffle(cols)

        return cols

    def _get_rows(self):
        rows = np.arange(self.data.shape[0])

        np.random.shuffle(rows)

        return rows

    def _get_M(self, t_a, t_x, Z):
        K = Z.shape[1]

        return np.linalg.inv(np.dot(Z.T, Z) + (t_a / t_x) * np.eye(K))

    def _remove_empty_rows(self):
        m = np.sum(self.params.Z, axis=0)

        cols = np.atleast_1d(np.squeeze(np.where(m > 0)))

        self.params.Z = self.params.Z[:, cols]

    def _update_A(self):
        t_a = self.params.tau_a
        t_x = self.params.tau_x
        Z = self.params.Z
        X = self.data

        D = X.shape[1]

        M = self._get_M(t_a, t_x, Z)

        self.params.A = scipy.stats.matrix_normal.rvs(
            mean=np.dot(M, np.dot(Z.T, X)),
            rowcov=(1 / t_x) * M,
            colcov=np.eye(D)
        )

    def _update_tau_a(self):
        A = self.params.A

        D = A.shape[1]
        K = A.shape[0]

        a = self.priors.tau_a[0] + 0.5 * K * D

        b = self.priors.tau_a[1] + 0.5 * np.trace(np.dot(A.T, A))

        self.params.tau_a = np.random.gamma(a, 1 / b)

    def _update_tau_x(self):
        A = self.params.A
        Z = self.params.Z
        X = self.data

        D = X.shape[1]
        N = X.shape[0]

        a = self.priors.tau_x[0] + 0.5 * N * D

        Y = X - np.dot(Z, A)

        b = self.priors.tau_x[1] + 0.5 * np.trace(np.dot(Y.T, Y))

        self.params.tau_x = np.random.gamma(a, 1 / b)


class LinearGaussianCollapsed(LinearGaussianModel):
    def update(self, kernel='gibbs'):
        if kernel == 'gibbs':
            self._update_Z()

        elif kernel == 'row_gibbs':
            self._update_Z_row()

        self._update_A()

        self._update_tau_a()

        self._update_A()

        self._update_tau_x()

    def _update_Z(self):
        alpha = self.params.alpha
        t_a = self.params.tau_a
        t_x = self.params.tau_x
        X = self.data

        K = self.params.Z.shape[1]
        N = self.params.Z.shape[0]

        if self.ibp:
            a_0 = 0
            b_0 = 0

        else:
            a_0 = alpha / K
            b_0 = 1

        log_p = np.zeros(2)

        rows = self._get_rows()

        for n in rows:
            Z = self.params.Z

            m = np.sum(Z, axis=0)

            m -= Z[n]

            a = a_0 + m

            b = b_0 + (N - m)

            cols = self._get_cols(m)

            for k in cols:
                Z[n, k] = 0

                log_p[0] = np.log(b[k]) + self.get_log_p_X_collapsed(X, Z)

                Z[n, k] = 1

                log_p[1] = np.log(a[k]) + self.get_log_p_X_collapsed(X, Z)

                log_p = log_normalize(log_p)

                Z[n, k] = discrete_rvs(np.exp(log_p))

            self.params.Z = Z

            if self.ibp:
                self._update_Z_singletons(n)

            self._remove_empty_rows()

    def _update_Z_row(self):
        alpha = self.params.alpha
        t_a = self.params.tau_a
        t_x = self.params.tau_x
        X = self.data

        K = self.params.Z.shape[1]
        N = self.params.Z.shape[0]

        if self.ibp:
            a_0 = 0
            b_0 = 0

        else:
            a_0 = alpha / K
            b_0 = 1

        log_p = np.zeros(2)

        rows = self._get_rows()

        for n in rows:
            Z = self.params.Z

            m = np.sum(Z, axis=0)

            m -= Z[n]

            a = a_0 + m

            b = b_0 + (N - m)

            cols = self._get_cols(m)

            K_non_singleton = len(cols)

            Zs = list(map(np.array, itertools.product([0, 1], repeat=K_non_singleton)))

            Zs = np.array(Zs, dtype=np.int)

            log_p = np.zeros(len(Zs))

            for idx, z in enumerate(Zs):
                Z[n, cols] = z

                log_p[idx] = np.sum(Z[n] * np.log(a)) + np.sum((1 - Z[n]) * np.log(b)) + \
                    self.get_log_p_X_collapsed(X, Z)

            log_p = log_normalize(log_p)

            idx = discrete_rvs(np.exp(log_p))

            Z[n, cols] = Zs[idx]

            self.params.Z = Z

            if self.ibp:
                self._update_Z_singletons(n)

            self._remove_empty_rows()

    def _update_Z_singletons(self, row):
        alpha = self.params.alpha
        t_a = self.params.tau_a
        t_x = self.params.tau_x
        X = self.data
        Z = self.params.Z

        K = Z.shape[1]
        N = Z.shape[0]

        m = np.sum(Z, axis=0)
        m -= Z[row]

        cols = np.atleast_1d(np.squeeze(np.where(m > 0)))

        K_non_singleton = len(cols)

        k_old = K - K_non_singleton

        k_new = np.random.poisson(alpha / N)

        Z = Z[:, cols]

        Z_old = np.column_stack([Z, np.zeros((N, k_old))])

        Z_old[row, K_non_singleton:] = 1

        Z_new = np.column_stack([Z, np.zeros((N, k_new))])

        Z_new[row, K_non_singleton:] = 1

        log_p_old = self.get_log_p_X_collapsed(X, Z_old)

        log_p_new = self.get_log_p_X_collapsed(X, Z_new)

        diff = log_p_new - log_p_old

        u = np.random.rand()

        if diff > np.log(u):
            Z = Z_new

        else:
            Z = Z_old

        self.params.Z = Z
