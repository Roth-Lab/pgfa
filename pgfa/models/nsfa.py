import itertools
import numpy as np
import numba
import scipy.special
import scipy.stats

from pgfa.math_utils import log_factorial, log_normalize
from pgfa.stats import discrete_rvs


class NSFA(object):
    def get_init_params(self, data, num_latent_dim, seed=None):
        if seed is not None:
            np.random.seed(seed)

        D = data.shape[0]
        K = num_latent_dim
        N = data.shape[1]

        # IBP params
        alpha = 1

        # Common factor variance
        gamma = 1

        # Factor loadings
        F = np.random.normal(0, 1, size=(K, N))

        # Covariance of observation
        S = np.eye(D)

        # Sparse matrix for common factors
        U = np.random.randint(0, 2, size=(D, K))

        # Common factors
        V = np.random.multivariate_normal(np.zeros(K), gamma * np.eye(K), size=D)

        return Parameters(alpha, gamma, F, S, U, V)

    def get_priors(self, ibp=False):
        gamma = np.array([1, 1])

        if ibp:
            alpha = np.array([1, 1])

            U = np.array([0, 0])

        else:
            alpha = None

            U = np.array([1, 1])

        S = np.array([1, 1])

        return Priors(alpha, gamma, S, U)

    def log_pdf(self, data, params, priors):
        alpha = params.alpha
        gamma = params.gamma
        F = params.F
        S = params.S
        U = params.U
        V = params.V
        W = params.W
        X = data

        log_p = 0

        # Likelihood
        for n in range(params.N):
            log_p += scipy.stats.multivariate_normal.logpdf(X[:, n], np.dot(W, F[:, n]), S)

        # Binary matrix prior
        m = np.sum(U, axis=0)

        a = priors.U[0] + m

        b = priors.U[1] + (params.D - m)

        if priors.alpha is None:
            for k in range(params.K):
                log_p += scipy.special.betaln(a[k], b[k]) - scipy.special.betaln(priors.U[0], priors.U[1])

        else:
            H = np.sum(1 / np.arange(1, params.D + 1))

            log_p += params.K * np.log(alpha) - H * alpha

            histories, history_counts = np.unique(params.U, axis=1, return_counts=True)

            m = histories.sum(axis=0)

            num_histories = histories.shape[1]

            for h in range(num_histories):
                K_h = history_counts[h]

                log_p -= log_factorial(K_h)

                log_p += K_h * log_factorial(m[h] - 1) + K_h * log_factorial(params.D - m[h])

                log_p -= history_counts[h] * log_factorial(params.D)

            log_p += scipy.stats.gamma.logpdf(alpha, priors.alpha[0], scale=(1 / priors.alpha[1]))

        # Common factors prior
        for k in range(params.K):
            log_p += scipy.stats.multivariate_normal.logpdf(V[:, k], np.zeros(params.D), gamma * np.eye(params.D))

        # Factor loadings prior
        for n in range(params.N):
            log_p += scipy.stats.multivariate_normal.logpdf(F[:, n], np.zeros(params.K), np.eye(params.K))

        # Noise covariance
        for d in range(params.D):
            log_p += scipy.stats.gamma.logpdf(S[d, d], priors.S[0], scale=(1 / priors.S[1]))

        log_p += scipy.stats.gamma.logpdf(1 / gamma, priors.gamma[0], scale=(1 / priors.gamma[1]))

        return log_p

    def log_predictive_pdf(self, data, params):
        S = params.S
        W = params.W

        mean = np.zeros(params.D)

        covariance = S + np.dot(W, W.T)

        try:
            log_p = np.sum(scipy.stats.multivariate_normal.logpdf(data.T, mean, covariance))

        except np.linalg.LinAlgError as e:
            print(W)
            raise e

        return log_p

    def update_params(self, data, params, priors, u_update='gibbs'):
        params = params.copy()

        if u_update == 'gibbs':
            self._update_U(data, params, priors)

        elif u_update == 'row_gibbs':
            self._update_U_row(data, params, priors)

        self._update_V(data, params)

        if priors.alpha is not None:
            self._update_alpha(params, priors)

        self._update_gamma(params, priors)

        self._update_F(data, params)

        self._update_S(data, params, priors)

        return params

    def _update_alpha(self, params, priors):
        a = params.K + priors.alpha[0]

        b = np.sum(1 / np.arange(1, params.D + 1)) + priors.alpha[0]

        params.alpha = np.random.gamma(a, 1 / b)

    def _update_gamma(self, params, priors):
        a = priors.gamma[0] + 0.5 * np.sum(params.U)

        b = priors.gamma[1] + 0.5 * np.sum(np.square(params.W))

        params.gamma = 1 / np.random.gamma(a, 1 / b)

    def _update_F(self, data, params):
        S = params.S
        W = params.W
        X = data

        S_inv = np.linalg.inv(S)

        sigma = np.linalg.inv(np.eye(params.K) + np.dot(W.T, np.dot(S_inv, W)))

        temp = np.dot(sigma, np.dot(W.T, S_inv))

        mu = np.dot(temp, X)

        chol = np.linalg.cholesky(sigma)

        params.F = mu + np.dot(chol, np.random.normal(0, 1, size=(params.K, params.N)))

    def _update_S(self, data, params, priors):
        S = np.zeros((params.D, params.D))

        F = params.F
        W = params.W
        X = data

        R = X - np.dot(W, F)

        a = priors.S[0] + 0.5 * params.N

        b = priors.S[1] + 0.5 * np.sum(np.square(R), axis=1)

        for d in range(params.D):
            S[d, d] = 1 / np.random.gamma(a, 1 / b[d])

        params.S = S

    def _update_U(self, data, params, priors):
        F = params.F
        S = params.S
        U = params.U
        V = params.V
        X = data

        log_p = np.zeros(2)

        prec = 1 / np.diag(S)

        rows = np.arange(params.D)

        np.random.shuffle(rows)

        cols = np.arange(params.K)

        np.random.shuffle(cols)

        for d in rows:
            m = np.sum(U, axis=0)

            m -= U[d]

            a = m + priors.U[0]

            b = (params.D - m) + priors.U[1]

            for k in cols:
                if m[k] == 0:
                    continue

                U[d, k] = 0

                log_p[0] = np.log(b[k]) + log_p_fn(prec[d], U[d], V[d], X[d], F)

                U[d, k] = 1

                log_p[1] = np.log(a[k]) + log_p_fn(prec[d], U[d], V[d], X[d], F)

                log_p = log_normalize(log_p)

                U[d, k] = discrete_rvs(np.exp(log_p))

            if priors.alpha is not None:
                self._update_U_singletons(d, X, params, priors)

    def _update_U_row(self, data, params, priors):
        F = params.F
        S = params.S
        U = params.U
        V = params.V
        X = data

        Us = list(map(np.array, itertools.product([0, 1], repeat=params.K)))

        Us = np.array(Us, dtype=np.int)

        for d in range(params.D):
            m = np.sum(U, axis=0)

            m -= U[d]

            a = m + priors.U[0]

            b = (params.D - m) + priors.U[1]

            if priors.alpha is not None:
                idxs = (m > 0)

                a = a[idxs]

                b = b[idxs]

                K = np.sum(idxs)

                Us = list(map(np.array, itertools.product([0, 1], repeat=K)))

                Us = np.array(Us, dtype=np.int)

            else:
                idxs = np.arange(params.K)

            log_p = np.zeros(len(Us))

            for i, u in enumerate(Us):
                log_p[i] = np.sum(u * np.log(a)) + np.sum((1 - u) * np.log(b)) + \
                    log_p_fn(1 / S[d, d], u, V[d, idxs], X[d], F[idxs])

            log_p = log_normalize(log_p)

            p = np.exp(log_p)

            idx = discrete_rvs(p)

            U[d, idxs] = Us[idx]

            if priors.alpha is not None:
                self._update_U_singletons(d, X, params, priors)

    def _update_U_singletons(self, d, data, params, priors):
        F = params.F
        S = params.S
        U = params.U
        V = params.V
        W = params.W
        X = data

        m = np.sum(U, axis=0)

        m -= U[d]

        idxs = (m == 0)

        r = X[d] - np.dot(W[d, ~idxs], F[~idxs])

        s = S[d, d]

        kappa_old = np.sum(idxs)

        # Old
        v_old = V[d, idxs]

        log_p_old = self._get_singleton_log_p(r, s, v_old)

        # New
        kappa_new = np.random.poisson(params.alpha / params.D)

        v_new = np.random.normal(0, np.sqrt(params.gamma), size=kappa_new)

        log_p_new = self._get_singleton_log_p(r, s, v_new)

        log_mh_ratio = log_p_new - log_p_old

        u = np.random.random()

        if np.log(u) < log_mh_ratio:
            params.U[d, idxs] = 0

            params = self._remove_empty_features(params)

            if kappa_old < kappa_new:
                K = params.K

                params.F = np.row_stack([params.F, self._get_singleton_F(r, s, v_new)])

                params.U = np.column_stack([params.U, np.zeros((params.D, kappa_new))])

                params.U[d, K:] = 1

                params.V = np.column_stack(
                    [params.V, np.random.normal(0, np.sqrt(params.gamma), size=(params.D, kappa_new))]
                )

                params.V[d, K:] = v_new

                self._update_V(
                    data, params, ds=[d], ks=range(K + 1, K + kappa_new)
                )

                # TODO: Resample V from posterior for new cols

    def _update_V(self, data, params, ds=None, ks=None):
        V = np.zeros(params.V.shape)

        gamma = params.gamma
        F = params.F
        S = params.S
        U = params.U
        W = params.W
        X = data

        FF = np.square(F).sum(axis=1)

        R = X - np.dot(W, F)

        if ds is None:
            ds = range(params.D)

        if ks is None:
            ks = range(params.K)

        for d in ds:
            for k in ks:
                if U[d, k] == 1:
                    rk = R[d] + W[d, k] * F[k]

                else:
                    rk = R[d]

                prec = (FF[k] / S[d, d]) + (1 / gamma)

                sigma = 1 / prec

                mu = (sigma / S[d, d]) * np.dot(F[k], rk)

                V[d, k] = np.random.normal(mu, sigma)

                if U[d, k] == 1:
                    W[d, k] = V[d, k]

                else:
                    W[d, k] = 0

                R[d] = rk - W[d, k] * F[k]

        params.V = V

    def _get_singleton_log_p(self, r, s, v):
        K = len(v)

        N = len(r)

        log_p = 0

        M = (1 / s) * np.dot(v, v.T) + np.eye(K)

        temp = (1 / s) * np.dot(np.linalg.inv(M), v)

        for n in range(N):
            m = temp * r[n]

            log_p += 0.5 * np.dot(m.T, np.dot(M, m))

        log_p -= 0.5 * N * np.log(np.linalg.det(M))

        return log_p

    def _get_singleton_F(self, r, s, v):
        K = len(v)

        N = len(r)

        prec = (1 / s) * np.dot(v, v.T) + np.eye(K)

        cov = np.linalg.inv(prec)

        temp = (1 / s) * np.dot(cov, v)

        cov_chol = np.linalg.cholesky(cov)

        mean = np.dot(temp[:, np.newaxis], r[np.newaxis, :])

        F = mean + np.dot(cov_chol, np.random.normal(0, 1, size=(K, N)))

        return F

    def _remove_empty_features(self, params):
        m = np.sum(params.U, axis=0)

        idxs = (m == 0)

        params.F = params.F[~idxs]

        params.U = params.U[:, ~idxs]

        params.V = params.V[:, ~idxs]

        return params


class Parameters(object):
    def __init__(self, alpha, gamma, F, S, U, V):
        self.alpha = alpha

        self.gamma = gamma

        self.F = F

        self.S = S

        self.U = U

        self.V = V

    @property
    def D(self):
        return self.V.shape[0]

    @property
    def K(self):
        return self.F.shape[0]

    @property
    def N(self):
        return self.F.shape[1]

    @property
    def W(self):
        return self.U * self.V

    def copy(self):
        return Parameters(self.alpha, self.gamma, self.F.copy(), self.S.copy(), self.U.copy(), self.V.copy())


class Priors(object):
    def __init__(self, alpha, gamma, S, U):
        self.alpha = alpha

        self.gamma = gamma

        self.S = S

        self.U = U


@numba.njit
def log_p_fn(precision, u, v, x, F):
    N = F.shape[1]

    log_p = 0

    w = u * v

    for n in range(N):
        f = F[:, n]

        mean = np.sum(w * f)

        log_p += log_normal_likelihood(x[n], mean, precision)

    return log_p


@numba.njit
def log_normal_likelihood(x, mean, precision):
    return -0.5 * precision * (x - mean)**2


def get_rmse(data, params):
    X_true = data

    X_pred = np.dot(params.W, params.F)

    return np.sqrt(np.mean((X_true - X_pred)**2))


def get_min_error(params_pred, params_true):
    W_pred = params_pred.W

    W_true = params_true.W

    K_pred = W_pred.shape[1]

    K_true = W_true.shape[1]

    min_error = float('inf')

    for perm in itertools.permutations(range(K_pred)):
        error = np.sqrt(np.mean((W_pred[:, perm[:K_true]] - W_true)**2))

        if error < min_error:
            min_error = error

    return min_error
