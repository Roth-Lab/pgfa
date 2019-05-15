import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import discrete_rvs, do_metropolis_hastings_accept_reject, log_factorial,\
    log_normalize

import pgfa.models.base


class Model(pgfa.models.base.AbstractModel):

    @staticmethod
    def get_default_params(data, feat_alloc_dist):
        N, D = data.shape

        Z = feat_alloc_dist.rvs(1, D)

        K = Z.shape[1]

        h = scipy.stats.gamma.rvs(10, 1, size=D)

        t = scipy.stats.uniform.rvs(0.5, 0.4, size=D)

        C = scipy.stats.randint.rvs(0, 8, size=(N, K))

        E = 2 * np.ones((N, D), dtype=np.int)

        V = scipy.stats.gamma.rvs(1, 1, size=(D, K))

        return Parameters(
            1, np.ones(2), h, np.ones(2), t, np.ones(2), C, np.array([0, 8]), E, V, np.ones(2), Z
        )

    def _init_joint_dist(self, feat_alloc_dist):
        self.joint_dist = pgfa.models.base.JointDistribution(
            DataDistribution(), feat_alloc_dist, ParametersDistribution()
        )


class ModelUpdater(pgfa.models.base.AbstractModelUpdater):

    def _update_model_params(self, model):
        update_C(model)

        update_V(model)

        update_h(model)

        update_t(model)


class Parameters(pgfa.models.base.AbstractParameters):

    def __init__(self, alpha, alpha_prior, h, h_prior, t, t_prior, C, C_prior, E, V, V_prior, Z):
        self.alpha = float(alpha)

        self.alpha_prior = np.array(alpha_prior, dtype=np.float64)

        self.h = np.array(h, dtype=np.float64)

        self.h_prior = np.array(h_prior, dtype=np.float64)

        self.t = np.array(t, dtype=np.float64)

        self.t_prior = np.array(t_prior, dtype=np.float64)

        self.C = np.array(C, dtype=np.int64)

        self.C_prior = np.array(C_prior, dtype=np.int64)

        self.E = np.array(E, dtype=np.int64)

        self.V = np.array(V, dtype=np.float64)

        self.V_prior = np.array(V_prior, dtype=np.float64)

        self.Z = np.array(Z, dtype=np.int8)

    @property
    def param_shapes(self):
        return {
            'alpha': (),
            'alpha_prior': (2,),
            'h': ('D',),
            'h_prior': (2,),
            't': ('D',),
            't_prior': (2,),
            'C': ('N', 'K'),
            'C_prior': (2,),
            'E': ('N', 'D'),
            'V': ('D', 'K'),
            'V_prior': (2,),
            'Z': ('D', 'K')
        }

    @property
    def D(self):
        return self.E.shape[1]

    @property
    def F(self):
        F = (self.V * self.Z) / np.sum(self.V * self.D, axis=1)[:, np.newaxis]

        F[np.isnan(F)] = 0

        return F

    @property
    def N(self):
        return self.E.shape[0]

    def copy(self):
        return Parameters(
            self.alpha,
            self.alpha_prior.copy(),
            self.h.copy(),
            self.h_prior.copy(),
            self.t.copy(),
            self.t_prior.copy(),
            self.C.copy(),
            self.C_prior.copy(),
            self.E.copy(),
            self.V.copy(),
            self.V_prior.copy(),
            self.Z.copy()
        )


#=========================================================================
# Updates
#=========================================================================
def update_h(model, variance=1):
    def log_p_func(x, d):
        model.params.h[d] = x

        return model.joint_dist.log_p(model.data, params)

    params = model.params.copy()

    for d in range(model.params.D):
        params.h[d] = _do_mh_gamma_update(lambda x: log_p_func(x, d), params.h[d], variance=variance)

    model.params = params


def update_t(model, variance=1):
    def log_p_func(x, d):
        model.params.t[d] = x

        return model.joint_dist.log_p(model.data, params)

    params = model.params.copy()

    for d in range(model.params.D):
        params.t[d] = _do_mh_beta_update(log_p_func, params.t[d], variance)

    model.params = params


def update_C(model):
    c_min, c_max = model.params.C_prior

    params = model.params.copy()

    cs = np.arange(c_min, c_max + 1, dtype=np.int)

    log_p = np.zeros(len(cs))

    for n in range(params.N):
        for k in range(params.K):
            for c in cs:
                params.C[n, k] = c

                log_p[c] = model.data_dist.log_p(model.data, params)

            log_p = log_normalize(log_p)

            idx = discrete_rvs(np.exp(log_p))

            params.C[n, k] = cs[idx]

    model.params = params


def update_V(model, variance=1):
    def log_p_func(x, d, k):
        model.params.V[d, k] = x

        return model.joint_dist.log_p(model.data, params)

    params = model.params.copy()

    for d in range(model.params.D):
        for k in range(model.params.K):
            params.V[d, k] = _do_mh_gamma_update(lambda x: log_p_func(x, d, k), params.V[d, k], variance=variance)

    model.params = params


def _do_mh_beta_update(log_p_func, x_old, variance=1):
    def proposal_func(x):
        a, b = _get_beta_params(x, variance)

        return scipy.stats.beta.rvs(a, b)

    def proposal_log_p_func(x_curr, x_prev):
        a, b = _get_gamma_params(x_prev, variance)

        return scipy.stats.beta.logpdf(x_curr, a, b)

    return _do_mh_update(log_p_func, proposal_func, proposal_log_p_func, x_old)


def _do_mh_gamma_update(log_p_func, x_old, variance=1):
    def proposal_func(x):
        a, b = _get_gamma_params(x, variance)

        return scipy.stats.gamma.rvs(a, scale=(1 / b))

    def proposal_log_p_func(x_curr, x_prev):
        a, b = _get_gamma_params(x_prev, variance)

        return scipy.stats.gamma.logpdf(x_curr, a, scale=(1 / b))

    return _do_mh_update(log_p_func, proposal_func, proposal_log_p_func, x_old)


def _do_mh_update(log_p_func, proposal_func, proposal_log_p_func, x_old):
    x_new = proposal_func(x_old)

    log_p_new = log_p_func(x_new)

    log_q_new = proposal_log_p_func(x_new, x_old)

    log_p_old = log_p_func(x_old)

    log_q_old = proposal_log_p_func(x_old, x_new)

    if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
        x = x_new

    else:
        x = x_old

    return x


def _get_beta_params(mean, variance):
    a_b = (mean * (1 - mean)) / variance - 1

    a = mean / a_b

    b = a_b - 1

    return a, b


def _get_gamma_params(mean, variance):
    b = mean / variance

    a = b * mean

    return a, b


#=========================================================================
# Densities and proposals
#=========================================================================
class DataDistribution(pgfa.models.base.AbstractDataDistribution):

    def log_p(self, data, params):
        return _log_p(
            params.h,
            params.t,
            params.C,
            params.E,
            params.V,
            data,
            params.Z.astype(np.float64)
        )

    def log_p_row(self, data, params, row_idx):
        return _log_p_row(
            params.h[row_idx],
            params.t[row_idx],
            params.E[:, row_idx],
            params.V[row_idx],
            data[:, row_idx],
            params.Z[row_idx].astype(np.float64),
            params.C
        )


class ParametersDistribution(pgfa.models.base.AbstractParametersDistribution):

    def log_p(self, params):
        log_p = 0

        # Gamma prior on $\alpha$
        a = params.alpha_prior[0]
        b = params.alpha_prior[1]
        log_p += scipy.stats.gamma.logpdf(params.alpha, a, scale=(1 / b))

        # Gamma prior on $h$
        a = params.h_prior[0]
        b = params.h_prior[1]
        log_p += np.sum(scipy.stats.gamma.logpdf(params.h, a, scale=(1 / b)))

        # Gamma prior on $t$
        a = params.t_prior[0]
        b = params.t_prior[1]
        log_p += np.sum(scipy.stats.beta.logpdf(params.t, a, b))

        # Unfiorm prior on $C$
        log_p += 0

        # Gamma prior on $V$
        a = params.V_prior[0]
        b = params.V_prior[1]
        log_p += np.sum(scipy.stats.gamma.logpdf(params.V, a, scale=(1 / b)))

        return log_p


# @numba.njit(cache=True)
def _log_p(h, t, C, E, V, X, Z):
    D = X.shape[1]

    log_p = 0

    for d in range(D):
        log_p += _log_p_row(h[d], t[d], E[:, d], V[d], X[:, d], Z[d], C)

    return log_p


# @numba.njit(cache=True)
def _log_p_row(h_d, t_d, e, v, x, z, C):
    N = len(e)

    log_p = 0

    f = v * z

    norm = np.sum(f)

    if norm == 0:
        f = 0.0

    else:
        f /= norm

    for n in range(N):
        m = h_d * (t_d * np.sum(f * C[n]) + (1 - t_d) * e[n])

        log_p += x[n] * np.log(m) - log_factorial(x[n]) - m

    return log_p


#=========================================================================
# Singletons updaters
#=========================================================================
class PriorSingletonsUpdater(object):

    def update_row(self, model, row_idx):
        alpha = model.params.alpha
        tau_v = model.params.tau_v

        D = model.params.D
        N = model.params.N

        k_old = len(self._get_singleton_idxs(model.params.Z, row_idx))

        k_new = scipy.stats.poisson.rvs(alpha / N)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = self._get_non_singleton_idxs(model.params.Z, row_idx)

        num_non_singletons = len(non_singleton_idxs)

        K_new = len(non_singleton_idxs) + k_new

        params_old = model.params.copy()

        params_new = model.params.copy()

        params_new.V = np.zeros((K_new, D))

        params_new.V[:num_non_singletons] = model.params.V[non_singleton_idxs]

        if k_new > 0:
            params_new.V[num_non_singletons:] = scipy.stats.matrix_normal.rvs(
                mean=np.zeros((k_new, D)),
                rowcov=(1 / tau_v) * np.eye(k_new),
                colcov=np.eye(D)
            )

        params_new.Z = np.zeros((N, K_new), dtype=np.int8)

        params_new.Z[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

        params_new.Z[row_idx, num_non_singletons:] = 1

        log_p_new = model.data_dist.log_p_row(model.data, params_new, row_idx)

        log_p_old = model.data_dist.log_p_row(model.data, model.params, row_idx)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            model.params = params_new

        else:
            model.params = params_old

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
