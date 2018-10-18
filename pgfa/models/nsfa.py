import itertools
import numpy as np
import numba
import scipy.linalg
import scipy.stats

from pgfa.math_utils import do_metropolis_hastings_accept_reject


class NonparametricSparaseFactorAnalysisModel(object):
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

    @property
    def rmse(self):
        X_true = self.data

        X_pred = self.params.W @ self.params.F

        return np.sqrt(np.mean((X_true - X_pred)**2))

    def get_params_from_data(self):
        D = self.data.shape[0]
        N = self.data.shape[1]

        Z = self.feat_alloc_prior.rvs(D)

        K = Z.shape[1]

        F = np.random.normal(0, 1, size=(K, N))

        S = np.ones(D)

        if K == 0:
            V = np.zeros((D, K))

        else:
            V = np.random.multivariate_normal(np.zeros(K), np.eye(K), size=D)

        return Parameters(1, F, S, V, Z)

    def log_predictive_pdf(self, data):
        S = self.params.S
        W = self.params.W

        mean = np.zeros(self.params.D)

        covariance = np.diag(1 / S) + W @ W.T

        try:
            log_p = np.sum(
                scipy.stats.multivariate_normal.logpdf(data.T, mean, covariance, allow_singular=True)
            )

        except np.linalg.LinAlgError as e:
            print(W)
            print(np.sum(self.params.Z, axis=0))
            print((W @ W.T).shape)
            raise e

        return log_p


#=========================================================================
# Updates
#=========================================================================
class NonparametricSparaseFactorAnalysisModelUpdater(object):
    def __init__(self, feat_alloc_updater):
        self.feat_alloc_updater = feat_alloc_updater

    def update(self, model):
        self.feat_alloc_updater.update(model)

        model.params = update_V(model.data, model.params)

        if model.params.K > 0:
            model.params = update_F(model.data, model.params)

        model.params = update_gamma(model.params, model.priors)

        model.params = update_S(model.data, model.params, model.priors)

        model.feat_alloc_prior.update(model.params.Z)


def update_gamma(params, priors):
    a = priors.gamma[0] + 0.5 * params.K * params.D  # np.sum(params.Z)

    b = priors.gamma[1] + 0.5 * np.sum(np.square(params.V))

    params.gamma = scipy.stats.gamma.rvs(a, scale=(1 / b))

    return params


def update_F(data, params):
    S = np.diag(params.S)
    W = params.W
    X = data

    A = np.eye(params.K) + W.T @ S @ W

    A_chol = scipy.linalg.cho_factor(A)

    b = scipy.linalg.cho_solve(A_chol, W.T @ S @ X)

    eps = np.random.normal(0, 1, size=(params.K, params.N))

    params.F = b + scipy.linalg.solve_triangular(A_chol[0], eps, lower=A_chol[1])

    return params


def update_S(data, params, priors):
    F = params.F
    W = params.W
    X = data

    R = X - W @ F

    a = priors.S[0] + 0.5 * params.N

    b = priors.S[1] + 0.5 * np.sum(np.square(R), axis=1)

    S = np.zeros(params.D)

    for d in range(params.D):
        S[d] = scipy.stats.gamma.rvs(a, scale=(1 / b[d]))

    params.S = S

    return params


def update_V(data, params):
    params.V = _update_V(data, params.gamma, params.F, params.S, params.V, params.Z)

    return params


@numba.njit(cache=True)
def _update_V(data, gamma, F, S, V, Z):
    X = data

    D = Z.shape[0]
    K = Z.shape[1]

    FF = np.sum(np.square(F), axis=1)

    for d in range(D):
        R = X[d] - (Z[d] * V[d]) @ F

        for k in range(K):
            rk = R + Z[d, k] * V[d, k] * F[k]

            prec = gamma + Z[d, k] * S[d] * FF[k]

            mean = Z[d, k] * (S[d] / prec) * (F[k] @ rk.T)

            std = 1 / np.sqrt(prec)

            V[d, k] = np.random.normal(mean, std)

            R = rk - Z[d, k] * V[d, k] * F[k]

    return V


#=========================================================================
# Container classes
#=========================================================================
class Parameters(object):
    def __init__(self, gamma, F, S, V, Z):
        self.gamma = gamma

        self.F = F

        self.S = S

        self.V = V

        self.Z = Z

    @property
    def D(self):
        return self.Z.shape[0]

    @property
    def K(self):
        return self.Z.shape[1]

    @property
    def N(self):
        return self.F.shape[1]

    @property
    def W(self):
        return self.Z * self.V

    def copy(self):
        return Parameters(self.gamma, self.F.copy(), self.S.copy(), self.V.copy(), self.Z.copy())


class Priors(object):
    def __init__(self, gamma=None, S=None, Z=None):
        if gamma is None:
            gamma = np.ones(2)

        self.gamma = gamma

        if S is None:
            S = np.ones(2)

        self.S = S

        if Z is None:
            Z = np.ones(2)

        self.Z = Z


#=========================================================================
# Densities and proposals
#=========================================================================
class DataDistribution(object):
    def __init__(self):
        pass

    def log_p(self, data, params):
        D = params.D
        N = params.N
        F = params.F
        S = params.S
        W = params.W
        X = data

        diff = X - (W @ F)

        log_p = 0

        log_p += 0.5 * N * (np.log(np.prod(S)) - D * np.log(2 * np.pi))

        log_p -= 0.5 * np.sum(S[:, np.newaxis] * np.square(diff))

        return log_p

    def log_p_row(self, data, params, row_idx):
        N = params.N
        F = params.F
        S = params.S
        V = params.V
        Z = params.Z
        X = data

        return log_p_row(F, N, S, X, V, Z, row_idx)
#
#         w = Z[row_idx] * V[row_idx]
#
#         diff = (X[row_idx] - w @ F)
#
#         log_p = 0
#
#         log_p += 0.5 * N * (np.log(S[row_idx]) - np.log(2 * np.pi))
#
#         log_p -= 0.5 * np.sum(S[row_idx] * np.square(diff))
#
#         return log_p


@numba.njit(cache=True)
def log_p_row(F, N, S, X, V, Z, row_idx):
    w = Z[row_idx] * V[row_idx]

    diff = (X[row_idx] - w @ F)

    log_p = 0

    log_p += 0.5 * N * (np.log(S[row_idx]) - np.log(2 * np.pi))

    log_p -= 0.5 * np.sum(S[row_idx] * np.square(diff))

    return log_p


class JointDistribution(object):
    def __init__(self, feat_alloc_prior, priors):
        self.data_dist = DataDistribution()

        self.feat_alloc_prior = feat_alloc_prior

        self.priors = priors

    def log_p(self, data, params):
        gamma = params.gamma
        F = params.F
        S = params.S
        V = params.V

        log_p = 0

        # Binary matrix prior
        log_p += self.feat_alloc_prior.log_p(params.Z)

        # Data
        log_p += self.data_dist.log_p(data, params)

        # Common factors prior
        log_p += np.sum(
            scipy.stats.multivariate_normal.logpdf(V.T, np.zeros(params.D), (1 / gamma) * np.eye(params.D))
        )

        if params.K > 0:
            # Factor loadings prior
            log_p += np.sum(
                scipy.stats.multivariate_normal.logpdf(F.T, np.zeros(params.K), np.eye(params.K))
            )

        # Noise covariance
        log_p += np.sum(
            scipy.stats.gamma.logpdf(S, self.priors.S[0], scale=(1 / self.priors.S[1]))
        )

        log_p += scipy.stats.gamma.logpdf(1 / gamma, self.priors.gamma[0], scale=(1 / self.priors.gamma[1]))

        return log_p


class SingletonsProposal(object):
    def rvs(self, data, params, num_singletons, row_idx):
        m = np.sum(params.Z, axis=0)

        m -= params.Z[row_idx]

        non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))

        num_non_singleton = len(non_singletons_idxs)

        K_new = num_non_singleton + num_singletons

        V_new = np.zeros((params.D, K_new))

        Z_new = np.zeros((params.D, K_new), dtype=np.int64)

        V_new[:, :num_non_singleton] = params.V[:, non_singletons_idxs]

        Z_new[:, :num_non_singleton] = params.Z[:, non_singletons_idxs]

        V_new[row_idx, num_non_singleton:] = np.random.normal(
            0, 1, size=num_singletons) * (1 / np.sqrt(params.gamma))

        Z_new[row_idx, num_non_singleton:] = 1

        F_new = np.zeros((K_new, params.N))

        new_params = Parameters(
            params.gamma,
            F_new,
            params.S,
            V_new,
            Z_new
        )

        new_params = update_F(data, new_params)

        new_params.F[:num_non_singleton] = params.F[non_singletons_idxs]

        return new_params


#=========================================================================
# Singletons updaters
#=========================================================================
class PriorSingletonsUpdater(object):
    def update_row(self, model, row_idx):
        D = model.params.D
        N = model.params.N
        gamma = model.params.gamma
        S = model.params.S

        k_old = len(get_singletons_idxs(model.params.Z, row_idx))

        k_new = model.feat_alloc_prior.sample_num_singletons(model.params.Z)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = get_non_singletons_idxs(model.params.Z, row_idx)

        num_non_singletons = len(non_singleton_idxs)

        K_new = num_non_singletons + k_new

        F_new = np.zeros((K_new, N))

        F_new[:num_non_singletons] = model.params.F[non_singleton_idxs]

        F_new[num_non_singletons:] = np.random.normal(0, 1, size=(k_new, N))

        V_new = np.zeros((D, K_new))

        V_new[:, :num_non_singletons] = model.params.V[:, non_singleton_idxs]

        V_new[:, num_non_singletons:] = np.random.multivariate_normal(
            np.zeros(D), (1 / gamma) * np.eye(D), size=k_new
        ).T

        Z_new = np.zeros((D, K_new), dtype=np.int64)

        Z_new[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

        Z_new[row_idx, num_non_singletons:] = 1

        params_new = Parameters(gamma, F_new, S, V_new, Z_new)

        log_p_new = model.data_dist.log_p_row(model.data, params_new, row_idx)

        log_p_old = model.data_dist.log_p_row(model.data, model.params, row_idx)

        diff = log_p_new - log_p_old

        if diff > np.log(np.random.rand()):
            params = params_new

        else:
            params = model.params

        return params


class MetropolisHastingsSingletonsUpdater(object):
    def update_row(self, model, row_idx):
        data = model.data
        params = model.params

        # Old
        params_old = params

        singletons_idxs = get_singletons_idxs(params_old, row_idx)

        k_old = len(singletons_idxs)

        v_old = params_old.V[row_idx, singletons_idxs]

        log_p_old = 0

        log_p_old += log_likelihood(data, params_old)

        log_p_old += np.sum(scipy.stats.norm.logpdf(v_old, np.zeros(k_old), (1 / params.gamma) * np.eye(k_old)))

        log_q_old = self._get_log_p_F_singleton(data, params_old, row_idx)

        # New
        k_new = np.random.poisson(params.alpha / params.D)

        params_new = self._propose_new_params(data, k_new, params, row_idx)

        singletons_idxs = get_singletons_idxs(params_new, row_idx)

        v_new = params_new.V[row_idx, singletons_idxs]

        log_p_new = 0

        log_p_new += log_likelihood(data, params_new)

        log_p_new += np.sum(scipy.stats.norm.logpdf(v_new, np.zeros(k_new), (1 / params.gamma) * np.eye(k_new)))

        log_q_new = self._get_log_p_F_singleton(data, params_new, row_idx)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
            params = params_new

        else:
            params = params_old

        return params

    def _get_F_singleton(self, data, params, row_idx):
        non_singletons_idxs = get_non_singletons_idxs(params, row_idx)

        singleton_idxs = get_singletons_idxs(params, row_idx)

        k = len(singleton_idxs)

        e = data[row_idx] - params.V[row_idx, non_singletons_idxs] @ params.F[non_singletons_idxs]

        e = np.atleast_2d(e)

        s = params.S[row_idx]

        v = params.V[row_idx, singleton_idxs].reshape((k, 1))

        prec = s * (v @ v.T) + np.eye(k)

        mean = s * np.linalg.solve(prec, v) @ e

        chol = np.linalg.cholesky(prec)

        eps = np.random.normal(0, 1, size=(k, params.N))

        F = mean + np.linalg.solve(chol, eps)

        return F

    def _get_log_p_F_singleton(self, data, params, row_idx):
        non_singletons_idxs = get_non_singletons_idxs(params, row_idx)

        singleton_idxs = get_singletons_idxs(params, row_idx)

        k = len(singleton_idxs)

        e = data[row_idx] - params.V[row_idx, non_singletons_idxs] @ params.F[non_singletons_idxs]

        e = np.atleast_2d(e)

        s = params.S[row_idx]

        v = params.V[row_idx, singleton_idxs].reshape((k, 1))

        f = params.F[singleton_idxs]

        prec = s * (v @ v.T) + np.eye(k)

        mean = s * np.linalg.solve(prec, v) @ e

        diff = (f - mean)

        log_p = 0

        log_p += 0.5 * np.linalg.slogdet(prec)[1]

        for n in range(params.N):
            log_p -= 0.5 * diff[:, n].T @ prec @ diff[:, n]

        return log_p

    def _propose_new_params(self, data, k_new, params, row_idx):
        non_singletons_idxs = get_non_singletons_idxs(params, row_idx)

        num_non_singleton = len(non_singletons_idxs)

        K_new = num_non_singleton + k_new

        # V
        V = np.zeros((params.D, K_new))

        V[:, :num_non_singleton] = params.V[:, non_singletons_idxs]

        V[:, num_non_singleton:] = np.random.normal(0, 1, size=(params.D, k_new)) * (1 / np.sqrt(params.gamma))

        # Z
        Z = np.zeros((params.D, K_new), dtype=np.int64)

        Z[:, :num_non_singleton] = params.Z[:, non_singletons_idxs]

        Z[row_idx, num_non_singleton:] = 1

        # F
        F = np.zeros((K_new, params.N))

        F[:num_non_singleton] = params.F[non_singletons_idxs]

        params = Parameters(
            params.alpha,
            params.gamma,
            F,
            params.S,
            V,
            Z
        )

        params.F[num_non_singleton:] = self._get_F_singleton(data, params, row_idx)

        return params


class CollapsedSingletonsUpdater(object):
    def update_Z_singletons(self, model, row_idx):
        data = model.data
        params = model.params

        m = np.sum(params.Z, axis=0)

        m -= params.Z[row_idx]

        non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))

        singletons_idxs = np.atleast_1d(np.squeeze(np.where(m == 0)))

        num_non_singleton = len(non_singletons_idxs)

        e = data[row_idx] - params.V[row_idx, non_singletons_idxs] @ params.F[non_singletons_idxs]

        e = np.atleast_2d(e)

        assert e.shape == (1, params.N)

        s = params.S[row_idx]

        k_old = len(singletons_idxs)

        v_old = params.V[row_idx, singletons_idxs]

        v_old = np.atleast_2d(v_old).T

        assert v_old.shape == (k_old, 1)

        log_p_old = self._get_log_p_singleton(e, k_old, s, v_old)

        k_new = np.random.poisson(params.alpha / params.D)

        v_new = np.random.normal(0, 1, size=(k_new, 1)) * (1 / np.sqrt(params.gamma))

        assert v_new.shape == (k_new, 1)

        log_p_new = self._get_log_p_singleton(e, k_new, s, v_new)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            #         print(log_p_new, log_p_old, k_new, k_old)
            K_new = num_non_singleton + k_new

            F = np.zeros((K_new, params.N))
            V = np.zeros((params.D, K_new))
            Z = np.zeros((params.D, K_new), dtype=np.int64)

            F[:num_non_singleton] = params.F[non_singletons_idxs]
            V[:, :num_non_singleton] = params.V[:, non_singletons_idxs]
            Z[:, :num_non_singleton] = params.Z[:, non_singletons_idxs]

            if k_new > 0:
                F[num_non_singleton:] = self._get_F_singleton(e, k_new, s, v_new)
                V[:, num_non_singleton:] = np.random.normal(0, 1, size=(params.D, k_new)) * (1 / np.sqrt(params.gamma))
                V[row_idx, num_non_singleton:] = np.squeeze(v_new)
                Z[row_idx, num_non_singleton:] = 1

            # Update V from posterior
            e = data[row_idx] - Z[row_idx] * V[row_idx] @ F

            e = np.atleast_2d(e)

            assert e.shape == (1, params.N)

            FF = np.sum(np.square(F), axis=1)

            for k in range(num_non_singleton, K_new):
                ek = e + V[row_idx, k] * F[k]

                prec = params.gamma + s * FF[k]

                mean = (s / prec) * F[k] @ ek.T

                std = 1 / np.sqrt(prec)

                V[row_idx, k] = np.random.normal(mean, std)

            params = Parameters(
                params.alpha,
                params.gamma,
                F,
                params.S,
                V,
                Z
            )

        return params

    def _get_log_p_singleton(self, e, k, s, v):
        """
        Parameters
        ----------
        e: ndarray (1xN)
        k: int
        s: float
        v: ndarray (Kx1)
        """
        N = len(e)

        prec = s * (v @ v.T) + np.eye(k)

        mean = s * np.linalg.solve(prec, v) @ e

        log_p = 0

        log_p -= 0.5 * N * np.linalg.slogdet(prec)[1]

        log_p += 0.5 * np.sum(mean.T @ prec @ mean)

        return log_p


def get_non_singletons_idxs(Z, row_idx):
    m = np.sum(Z, axis=0)

    m -= Z[row_idx]

    return np.atleast_1d(np.squeeze(np.where(m > 0)))


def get_singletons_idxs(Z, row_idx):
    m = np.sum(Z, axis=0)

    m -= Z[row_idx]

    return np.atleast_1d(np.squeeze(np.where(m == 0)))


#=========================================================================
# Benchmark utils
#=========================================================================
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
