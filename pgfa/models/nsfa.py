import itertools
import numpy as np
import numba
import scipy.special
import scipy.stats

from pgfa.math_utils import log_normalize
from pgfa.stats import discrete_rvs


class NSFA(object):

    def get_init_params(self, data, num_latent_dim, seed=None):
        if seed is not None:
            np.random.seed(seed)

        D = data.shape[0]
        K = num_latent_dim
        N = data.shape[1]

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

        return Parameters(gamma, F, S, U, V)

    def get_priors(self):
        gamma = np.array([1, 1])

        S = np.array([1, 1])

        U = np.array([1, 1])

        return Priors(gamma, S, U)

    def log_pdf(self, data, params, priors):
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

        for k in range(params.K):
            log_p += scipy.special.betaln(a[k], b[k]) - scipy.special.betaln(priors.U[0], priors.U[1])

        # Common factors prior
        for k in range(params.K):
            log_p += scipy.stats.multivariate_normal.logpdf(V[:, k], np.zeros(params.D), gamma * np.eye(params.D))

        # Factor loadings prior
        for n in range(params.N):
            log_p += scipy.stats.multivariate_normal.logpdf(F[:, n], np.zeros(params.K), np.eye(params.K))

        # Noise covariance
        for d in range(params.D):
            log_p += scipy.stats.gamma.logpdf(S[d, d], priors.S[0], scale=(1 / priors.S[1]))

        log_p += scipy.stats.gamma.logpdf(gamma, priors.gamma[0], scale=(1 / priors.gamma[1]))

        return log_p

    def log_predictive_pdf(self, data, params):
        S = params.S
        W = params.W

        mean = np.zeros(params.D)

        covariance = S + np.dot(W, W.T)

        return np.sum(scipy.stats.multivariate_normal.logpdf(data.T, mean, covariance))

    def update_params(self, data, params, priors):
        params.gamma = self._update_gamma(params, priors)

        params.F = self._update_F(data, params)

        params.S = self._update_S(data, params, priors)

        params.U = self._update_U(data, params, priors)

        params.V = self._update_V(data, params)

        return params

    def _update_gamma(self, params, priors):
        a = priors.gamma[0] + 0.5 * np.sum(params.U)

        b = priors.gamma[1] + 0.5 * np.sum(np.square(params.W))

        return 1 / np.random.gamma(a, 1 / b)

    def _update_F(self, data, params):
        S = params.S
        W = params.W
        X = data

        S_inv = np.linalg.inv(S)

        sigma = np.linalg.inv(np.eye(params.K) + np.dot(W.T, np.dot(S_inv, W)))

        temp = np.dot(sigma, np.dot(W.T, S_inv))

        mu = np.dot(temp, X)

        chol = np.linalg.cholesky(sigma)

        F = mu + np.dot(chol, np.random.normal(0, 1, size=(params.K, params.N)))

        return F

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

        return S

    def _update_U(self, data, params, priors):
        U = params.U.copy()

        F = params.F
        S = params.S
        V = params.V
        X = data

        log_p = np.zeros(2)

        for d in range(params.D):
            for k in range(params.K):
                m = np.sum(U, axis=0)

                m[k] -= U[d, k]

                a = m + priors.U[0]

                b = (params.D - 1 - m) + priors.U[1]

                U[d, k] = 0

                log_p[0] = np.log(b[k]) + log_p_fn(1 / S[d, d], U[d], V[d], X[d], F)

                U[d, k] = 1

                log_p[1] = np.log(a[k]) + log_p_fn(1 / S[d, d], U[d], V[d], X[d], F)

                log_p = log_normalize(log_p)

                U[d, k] = discrete_rvs(np.exp(log_p))

        return U

    def _update_V(self, data, params):
        V = params.V.copy()

        gamma = params.gamma
        F = params.F
        S = params.S
        U = params.U
        X = data

        F_square = np.square(F).sum(axis=1)

        for d in range(params.D):
            for k in range(params.K):
                sigma = 1 / ((F_square[k] / S[d, d]) + (1 / gamma))

                R = X - np.dot((U * V), F)

                X_star = R[d] + (U[d, k] * V[d, k] * F[k])

                mu = (sigma / S[d, d]) * np.dot(F[k], X_star)

                V[d, k] = np.random.normal(mu, sigma)

        return V


class Parameters(object):
    def __init__(self, gamma, F, S, U, V):
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
        return Parameters(self.gamma, self.F.copy(), self.S.copy(), self.U.copy(), self.V.copy())


class Priors(object):
    def __init__(self, gamma, S, U):
        self.gamma = gamma

        self.S = S

        self.U = U


@numba.njit
def log_p_fn(precision, u, v, x, F):
    N = F.shape[0]

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
