import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import bernoulli_rvs, log_beta, log_factorial, do_metropolis_hastings_accept_reject


def get_feature_allocation_distribution(K=None):
    if K is None:
        dist = IndianBuffetProcessDistribution()

    else:
        dist = BetaBernoulliFeatureAllocationDistribution(K)

    return dist


class BetaBernoulliFeatureAllocationDistribution(object):
    """ Finite Beta-Bernoulli feature allocation model with feature weights marginalized.

    Note: We define the Beta parameters as $a=\frac{\alpha}{K}$ and $b=1$.

    Parameters
    ----------
    K: int
        Number of features.
    """

    def __init__(self, K):
        self.K = K

    def get_feature_probs(self, params, row_idx):
        alpha = params.alpha
        Z = params.Z

        N = Z.shape[0]

        m = _get_conditional_counts(row_idx, Z)

        a0, b0 = self._get_beta_params(alpha)

        a = a0 + m

        b = b0 + (N - 1 - m)

        return a / (a + b)

    def get_update_cols(self, params, row_idx):
        cols = np.arange(self.K)

        np.random.shuffle(cols)

        return cols
    
    def log_p(self, params):
        alpha = params.alpha
        Z = params.Z

        if Z.shape[1] != self.K:
            return float('-inf')

        K = Z.shape[1]

        N = Z.shape[0]

        if K == 0:
            return 0

        m = np.sum(Z, axis=0)

        a0, b0 = self._get_beta_params(alpha)

        a = a0 + m

        b = b0 + (N - m)

        return np.sum(log_beta(a, b) - log_beta(a0, b0))

    def rvs(self, alpha, N):
        K = self.K

        a, b = self._get_beta_params(alpha)

        p = np.random.beta(a, b, size=K)

        Z = np.zeros((N, K), dtype=np.int64)

        for k in range(K):
            Z[:, k] = np.random.multinomial(1, [1 - p[k], p[k]], size=N).argmax(axis=1)

        return Z

    def _get_beta_params(self, alpha):
        return alpha / self.K, 1


class IndianBuffetProcessDistribution(object):
    """ IBP feature allocation distributions.
    """

    def get_feature_probs(self, params, row_idx):
        Z = params.Z

        N = Z.shape[0]

        m = _get_conditional_counts(row_idx, Z)

        return m / N

    def get_update_cols(self, params, row_idx):
        Z = params.Z

        m = _get_conditional_counts(row_idx, Z)

        cols = [k for k in range(Z.shape[1]) if (m[k] > 0)]

        np.random.shuffle(cols)

        return cols
    
    def log_p(self, params):
        alpha = params.alpha
        Z = params.Z
        
        Z = Z[:, np.sum(Z, axis=0) > 0]
        
        N, K = Z.shape

        if K == 0:
            return 0        
                
        m = np.sum(Z, axis=0)
        
        H = np.sum(1 / np.arange(1, N + 1))

        log_p = 0
        
        log_p -= log_factorial(K)
        
        log_p += K * np.log(alpha)
        
        log_p -= H * alpha
        
        log_p += np.sum(log_factorial(m - 1) + log_factorial(N - m) - log_factorial(N))
        
        return log_p

    def rvs(self, alpha, N):
        K = np.random.poisson(alpha)

        Z = np.ones((1, K), dtype=np.int64)

        for idx in range(1, N):
            n = idx + 1

            K = Z.shape[1]

            z = np.zeros(K)

            m = np.sum(Z, axis=0)

            for k in range(K):
                z[k] = bernoulli_rvs(m[k] / n)

            Z = np.row_stack([Z, z])

            k_new = np.random.poisson(alpha / n)

            if k_new > 0:
                Z = np.column_stack([Z, np.zeros((Z.shape[0], k_new))])

                Z[idx, K:] = 1

        return Z.astype(np.int64)


def update_alpha(model):
    """ Metropolis-Hastings update of the Beta-Bernoulli or IBP concentration parameter from a Gamma(1, 1) prior.

    Note: The model parameters will be updated in place.

    Parameters
    ----------
    model: pgfa.models.base.AbstractModel
    """
    a = model.params.alpha_prior[0]
    b = model.params.alpha_prior[1]

    alpha_old = model.params.alpha

    log_p_old = model.feat_alloc_dist.log_p(model.params)

    alpha_new = scipy.stats.gamma.rvs(a, scale=(1 / b))

    model.params.alpha = alpha_new

    log_p_new = model.feat_alloc_dist.log_p(model.params)

    if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
        model.params.alpha = alpha_new

    else:
        model.params.alpha = alpha_old


def update_alpha_gibbs(model):
    """ Gibbs update of the IBP concentration parameter.

    Note: The model parameters will be updated in place.
    Note: This update is not valid for the Beta-Bernoulli distribution.

    Parameters
    ----------
    model: pgfa.models.base.AbstractModel
    """
    params = model.params
    priors = model.priors

    Z = params.Z

    K = Z.shape[1]

    N = Z.shape[0]

    a = K + priors[0]

    b = np.sum(1 / np.arange(1, N + 1)) + priors[1]

    params.alpha = np.random.gamma(a, 1 / b)

    model.params = params


@numba.njit(cache=True)
def _get_conditional_counts(row_idx, Z):
    m = np.sum(Z, axis=0)

    m -= Z[row_idx]

    return m
