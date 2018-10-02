import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import do_metropolis_hastings_accept_reject, ffa_rvs, ibp_rvs, log_ffa_pdf, log_ibp_pdf,\
    bernoulli_rvs

import pgfa.updates.feature_matrix


class LatentFactorRelationalModel(object):
    def __init__(self, data, K=None, params=None, priors=None):
        self.data = data

        self.ibp = (K is None)

        if params is None:
            params = get_params_from_data(data, K=K)

        self.params = params

        if priors is None:
            priors = Priors()

        self.priors = priors

    @property
    def log_p(self):
        data = self.data
        params = self.params
        priors = self.priors

        log_p = 0

        density = Density()

        log_p += density.log_p(data, params)

        # Binary matrix prior
        if self.ibp:
            log_p += log_ibp_pdf(params.alpha, params.Z)

            # Prior on alpha
            log_p += scipy.stats.gamma.logpdf(params.alpha, priors.alpha[0], scale=(1 / priors.alpha[1]))

        else:
            log_p += log_ffa_pdf(priors.Z[0], priors.Z[1], params.Z)

        log_p += np.sum(scipy.stats.norm.logpdf(
            np.squeeze(params.V[np.triu_indices(params.K)]), 0, 1 / np.sqrt(params.tau)
        ))

        return log_p

    def predict(self, method='max'):
        V = self.params.V
        Z = self.params.Z
        X = np.zeros((self.params.N, self.params.N))

        for i in range(self.params.N):
            for j in range(self.params.N):
                m = Z[i].T @ V @ Z[j]

                r = np.exp(-m)

                p = 1 / (1 + r)

                if method == 'max':
                    X[i, j] = int(p >= 0.5)

                elif method == 'random':
                    X[i, j] = bernoulli_rvs(p)

                else:
                    X[i, j] = p

        return X

    def update(self, num_particles=10, resample_threshold=0.5, update_type='g'):
        self.params = update_Z(
            self.data,
            self.params,
            self.priors,
            ibp=self.ibp,
            num_particles=num_particles,
            resample_threshold=resample_threshold,
            update_type=update_type
        )

        self.params = update_V(self.data, self.params)

#         self.params = update_tau(self.params, self.priors)

        if self.ibp:
            self.params = update_alpha(self.params, self.priors)


#=========================================================================
# Updates
#=========================================================================
def update_alpha(params, priors):
    a = priors.alpha[0] + params.K

    b = priors.alpha[1] + np.sum(1 / np.arange(1, params.N + 1))

    params.alpha = np.random.gamma(a, 1 / b)

    return params


def update_tau(params, priors):
    a = priors.tau[0] + 0.5 * (0.5 * params.K * (params.K + 1))

    b = priors.tau[1] + 0.5 * np.sum(np.square(np.triu(params.V)))

    params.tau = np.random.gamma(a, 1 / b)

    return params


def update_V(data, params, proposal_precision=10):
    density = Density()

    proposal_std = 1 / np.sqrt(proposal_precision)

    for i in range(params.K):
        for j in range(i, params.K):
            v_old = params.V[i, j]

            v_new = np.random.normal(v_old, proposal_std)

            params.V[i, j] = v_old

            params.V[j, i] = v_old

            log_p_old = density.log_p(data, params)

            log_q_old = scipy.stats.norm.logpdf(v_old, v_new, proposal_std)

            params.V[i, j] = v_new

            params.V[j, i] = v_new

            log_p_new = density.log_p(data, params)

            log_q_new = scipy.stats.norm.logpdf(v_new, v_new, proposal_std)

            if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
                params.V[i, j] = v_new

                params.V[j, i] = v_new

            else:
                params.V[i, j] = v_old

                params.V[j, i] = v_old

    return params


def update_Z(data, params, priors, ibp=False, num_particles=10, resample_threshold=0.5, update_type='g'):
    for row_idx in pgfa.updates.feature_matrix.get_rows(params.N):
        density = RowDensity(row_idx)

        m = np.sum(params.Z, axis=0)

        m -= params.Z[row_idx]

        if ibp:
            a = m

            b = (params.N - m)

        else:
            a = priors.Z[1] + m

            b = priors.Z[0] + (params.N - 1 - m)

        cols = pgfa.updates.feature_matrix.get_cols(m, include_singletons=(not ibp))

        if update_type == 'g':
            params = pgfa.updates.feature_matrix.do_gibbs_update(density, a, b, cols, row_idx, data, params)

        elif update_type == 'pg':
            params = pgfa.updates.feature_matrix.do_particle_gibbs_update(
                density, a, b, cols, row_idx, data, params,
                annealed=False, num_particles=num_particles, resample_threshold=resample_threshold
            )

        elif update_type == 'pga':
            params = pgfa.updates.feature_matrix.do_particle_gibbs_update(
                density, a, b, cols, row_idx, data, params,
                annealed=True, num_particles=num_particles, resample_threshold=resample_threshold
            )

        elif update_type == 'rg':
            params = pgfa.updates.feature_matrix.do_row_gibbs_update(density, a, b, cols, row_idx, data, params)

        if ibp:
            proposal = Proposal(row_idx)

            params = pgfa.updates.feature_matrix.do_mh_singletons_update(
                density, proposal, data, params
            )

    return params


#=========================================================================
# Densities and proposals
#=========================================================================
@numba.jitclass([
    ('row_idx', numba.int64)
])
class RowDensity(object):
    def __init__(self, row_idx):
        self.row_idx = row_idx

    def log_p(self, data, params):
        V = params.V
        Z = params.Z.astype(np.float64)
        X = data

        log_p = _get_log_p_sigmoid(self.row_idx, self.row_idx, X, V, Z)

        for i in range(params.N):
            if i == self.row_idx:
                continue

            log_p += _get_log_p_sigmoid(i, self.row_idx, X, V, Z)

            log_p += _get_log_p_sigmoid(self.row_idx, i, X, V, Z)

        return log_p


@numba.jitclass([
])
class Density(object):
    def __init__(self):
        pass

    def log_p(self, data, params):
        V = params.V
        Z = params.Z.astype(np.float64)
        X = data

        log_p = 0

        for i in range(params.N):
            for j in range(params.N):
                log_p += _get_log_p_sigmoid(i, j, X, V, Z)

        return log_p

    def _get_log_p(self, i, j, X, V, Z):
        if np.isnan(X[i, j]):
            log_p = 0

        else:
            m = Z[i].T @ V @ Z[j]

            f = np.exp(-m)

            if X[i, j] == 0:
                log_p = -m - np.log1p(f)

            else:
                log_p = -np.log1p(f)

        return log_p


@numba.njit
def _get_log_p_sigmoid(i, j, X, V, Z):
    if np.isnan(X[i, j]):
        log_p = 0

    else:
        m = Z[i].T @ V @ Z[j]

        f = np.exp(-m)

        if X[i, j] == 0:
            log_p = -m - np.log1p(f)

        else:
            log_p = -np.log1p(f)

    return log_p


class Proposal(object):
    def __init__(self, row_idx):
        self.row_idx = row_idx

    def rvs(self, data, params, num_singletons):
        m = np.sum(params.Z, axis=0)

        m -= params.Z[self.row_idx]

        non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))

        num_non_singleton = len(non_singletons_idxs)

        K_new = num_non_singleton + num_singletons

        Z_new = np.zeros((params.N, K_new), dtype=np.int64)

        Z_new[:, :num_non_singleton] = params.Z[:, non_singletons_idxs]

        Z_new[self.row_idx, num_non_singleton:] = 1

        V_new = np.zeros((K_new, K_new))

        for i in range(num_non_singleton):
            for j in range(num_non_singleton):
                V_new[i, j] = params.V[non_singletons_idxs[i], non_singletons_idxs[j]]

        for i in range(num_non_singleton, K_new):
            for j in range(i, K_new):
                V_new[i, j] = np.random.normal(0, 1 / np.sqrt(params.tau))

                V_new[j, i] = V_new[i, j]

        return Parameters(
            params.alpha,
            params.tau,
            V_new,
            Z_new
        )


#=========================================================================
# Container classes
#=========================================================================
def get_params_from_data(data, K=None):
    N = data.shape[1]

    if K is None:
        Z = ibp_rvs(1, N)

    else:
        Z = ffa_rvs(1, 1, K, N)

    K = Z.shape[1]

    if K == 0:
        V = np.zeros((K, K))

    else:
        V = np.random.normal(0, 1, size=(K, K))
        V = np.triu(V)
        V = V + V.T - np.diag(np.diag(V))

    return Parameters(1, 1, V, Z)


@numba.jitclass([
    ('alpha', numba.float64),
    ('tau', numba.float64),
    ('V', numba.float64[:, :]),
    ('Z', numba.int64[:, :])
])
class Parameters(object):
    def __init__(self, alpha, tau, V, Z):
        self.alpha = alpha

        self.tau = tau

        self.V = V

        self.Z = Z

    @property
    def K(self):
        return self.Z.shape[1]

    @property
    def N(self):
        return self.Z.shape[0]

    def copy(self):
        return Parameters(
            self.alpha,
            self.tau,
            self.V.copy(),
            self.Z.copy()
        )


class Priors(object):
    def __init__(self, alpha=None, tau=None, Z=None):
        if alpha is None:
            alpha = np.ones(2)

            alpha = np.array([1, 10])

        self.alpha = alpha

        if tau is None:
            tau = np.ones(2)

        self.tau = tau

        if Z is None:
            Z = np.ones(2)

        self.Z = Z
