import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import log_beta, log_factorial, log_sum_exp

import pgfa.models.base
import pgfa.models.pyclone.param_updates as param_updates


class Model(pgfa.models.base.AbstractModel):

    @staticmethod
    def get_default_params(data, feat_alloc_dist):
        N = len(data)

        D = len(data[0].sample_data_points)

        Z = feat_alloc_dist.rvs(1, N)

        K = Z.shape[1]

        precision_prior = np.array(param_updates.get_gamma_params(400, 1000))

        precision = scipy.stats.gamma.rvs(precision_prior[0], scale=(1 / precision_prior[1]))

        V_prior = param_updates.get_gamma_params(100, 100)

        V = scipy.stats.gamma.rvs(V_prior[0], scale=(1 / V_prior[1]), size=(K, D))

        return Parameters(1, np.ones(2), precision, precision_prior, V, V_prior, Z)

    def _init_joint_dist(self, feat_alloc_dist):
        self.joint_dist = pgfa.models.base.JointDistribution(
            DataDistribution(), feat_alloc_dist, ParametersDistribution()
        )


class ModelUpdater(pgfa.models.base.AbstractModelUpdater):

    def _update_model_params(self, model):
        for _ in range(20):
            param_updates.update_precision(model)

        for _ in range(4):
            f = np.random.choice([
                param_updates.update_V,
                param_updates.update_V_block,
                param_updates.update_V_block_dim,
                param_updates.update_V_perm
            ])

            f(model)

        for _ in range(20):
            param_updates.update_V_random_grid_pairwise(model, num_points=5)


class Parameters(pgfa.models.base.AbstractParameters):

    def __init__(self, alpha, alpha_prior, precision, precision_prior, V, V_prior, Z):
        self.alpha = float(alpha)

        self.alpha_prior = np.array(alpha_prior, dtype=np.float64)

        self.precision = float(precision)

        self.precision_prior = np.array(precision_prior, dtype=np.float64)

        self.V = np.array(V, dtype=np.float64)

        self.V_prior = np.array(V_prior, dtype=np.float64)

        self.Z = np.array(Z, dtype=np.int8)

    @property
    def param_shapes(self):
        return {
            'alpha': (),
            'alpha_prior': (2,),
            'precision': (),
            'precision_prior': (2,),
            'V': ('K', 'D'),
            'V_prior': (2,),
            'Z': ('N', 'K')
        }

    @property
    def D(self):
        return self.V.shape[1]

    @property
    def F(self):
        return self.V / np.sum(self.V, axis=0)[np.newaxis, :]

    @property
    def N(self):
        return self.Z.shape[0]

    def copy(self):
        return Parameters(
            self.alpha,
            self.alpha_prior.copy(),
            self.precision,
            self.precision_prior,
            self.V.copy(),
            self.V_prior.copy(),
            self.Z.copy()
        )


#=========================================================================
# Densities and proposals
#=========================================================================
class DataDistribution(pgfa.models.base.AbstractDataDistribution):

    def _log_p(self, data, params):
        return _log_p(data, params.precision, params.F, params.Z.astype(float))

    def _log_p_row(self, data, params, row_idx):
        return _log_p_row(data[row_idx], params.precision, params.F, params.Z[row_idx].astype(float))


class ParametersDistribution(pgfa.models.base.AbstractParametersDistribution):

    def log_p(self, params):
        log_p = 0

        # Gamma prior on $\alpha$
        a, b = params.alpha_prior
        log_p += scipy.stats.gamma.logpdf(params.alpha, a, scale=(1 / b))

        # Gamma prior on $v_{k d}$
        a, b = params.V_prior
        log_p += np.sum(scipy.stats.gamma.logpdf(params.V, a, scale=(1 / b)))

        # Gamma prior on precision
        a, b = params.precision_prior
        log_p += scipy.stats.gamma.logpdf(params.precision, a, scale=(1 / b))

        return log_p


def _log_p(X, precision, F, Z):
    N = len(X)

    log_p = 0

    for n in range(N):
        log_p += _log_p_row(X[n], precision, F, Z[n])

    return log_p


def _log_p_row(x, precision, F, z):
    log_p = 0

    phi = z @ F

    D = len(phi)

    for d in range(D):
        log_p += _log_p_sample(x.sample_data_points[d], phi[d], precision)

    return log_p


@numba.njit(cache=False)
def _log_p_sample(x, phi, precision):
    G = len(x.log_pi)

    log_p = np.zeros(G)

    f = phi

    t = x.tumour_content

    for g in range(len(x.log_pi)):
        cn_n, cn_r, cn_v = x.cn[g]

        mu_n, mu_r, mu_v = x.mu[g]

        norm = (1 - t) * cn_n + t * (1 - f) * cn_r + t * f * cn_v

        prob = (1 - t) * cn_n * mu_n + t * (1 - f) * cn_r * mu_r + t * f * cn_v * mu_v

        prob /= norm

        log_p[g] = x.log_pi[g] + log_beta_binomial_pdf(x.d, x.b, prob, precision)

    return log_sum_exp(log_p)


@numba.njit(cache=True)
def get_beta_binomial_params(m, s):
    a = m * s

    b = s - a

    return a, b


@numba.njit(cache=True)
def log_beta_binomial_pdf(n, x, m, s):
    a, b = get_beta_binomial_params(m, s)

    return log_binomial_coefficient(n, x) + log_beta(a + x, b + n - x) - log_beta(a, b)


@numba.njit(cache=True)
def log_binomial_coefficient(n, x):
    return log_factorial(n) - log_factorial(x) - log_factorial(n - x)
