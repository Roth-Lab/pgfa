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
    a = priors.gamma[0] + 0.5 * params.D * params.K

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
    def __init__(self, num_iters=1):
        self.num_iters = num_iters

    def update_row(self, model, row_idx):
        for _ in range(self.num_iters):
            model.params = self._update_row(model, row_idx)

        return model.params

    def _update_row(self, model, row_idx):
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

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            params = params_new

        else:
            params = model.params

        return params


class CollapsedSingletonsUpdater(object):
    def update_row(self, model, row_idx):
        D = model.params.D
        N = model.params.N
        gamma = model.params.gamma
        F = model.params.F
        S = model.params.S
        V = model.params.V
        Z = model.params.Z
        X = model.data

        singleton_idxs = get_singletons_idxs(Z, row_idx)

        k_old = len(singleton_idxs)

        k_new = model.feat_alloc_prior.sample_num_singletons(model.params.Z)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = get_non_singletons_idxs(model.params.Z, row_idx)

        f = F[non_singleton_idxs]

        s = S[row_idx]

        v = V[row_idx, non_singleton_idxs]

        z = Z[row_idx, non_singleton_idxs]

        x = X[row_idx]

        E = x - (v * z) @ f

        if k_old == 0:
            log_p_old = 0

        else:
            v_old = model.params.V[row_idx, singleton_idxs]

            prec_old = s * (v_old @ v_old.T) + np.eye(k_old)

            mean_old = s * np.linalg.solve(prec_old, v_old)

            log_p_old = 0
            log_p_old -= 0.5 * N * np.log(np.linalg.det(prec_old))
            log_p_old += 0.5 * np.sum(np.square(E) * (mean_old.T @ prec_old @ mean_old))

        if k_new == 0:
            log_p_new = 0

        else:
            v_new = np.atleast_2d(np.random.normal(0, 1 / gamma, size=k_new))

            prec_new = s * (v_new @ v_new.T) + np.eye(k_new)

            mean_new = s * np.linalg.solve(prec_new, v_new.T)

            log_p_new = 0
            log_p_new -= 0.5 * N * np.log(np.linalg.det(prec_new))
            log_p_new += 0.5 * np.sum(np.square(E) * (mean_new.T @ prec_new @ mean_new))

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            num_non_singletons = len(non_singleton_idxs)

            K_new = num_non_singletons + k_new

            F_new = np.zeros((K_new, N))

            F_new[:num_non_singletons] = model.params.F[non_singleton_idxs]

            V_new = np.zeros((D, K_new))

            V_new[:, :num_non_singletons] = model.params.V[:, non_singleton_idxs]

            if k_new > 0:
                chol = np.linalg.cholesky(prec_new)

                eps = np.random.normal(0, 1, size=(k_new, N))

                F_new[num_non_singletons:] = mean_new * E + np.linalg.solve(chol, eps)

#                 F_new[num_non_singletons:] = np.random.normal(0, 1, size=(k_new, N))

                V_new[:, num_non_singletons:] = v_new

            Z_new = np.zeros((D, K_new), dtype=np.int64)

            Z_new[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

            Z_new[row_idx, num_non_singletons:] = 1

            params = Parameters(gamma, F_new, S, V_new, Z_new)

            params = update_V(X, params)

        else:
            params = model.params

        return params


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
