import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import do_metropolis_hastings_accept_reject, log_factorial

import pgfa.models.base


class Model(pgfa.models.base.AbstractModel):

    @staticmethod
    def get_default_params(data, feat_alloc_dist):
        N, D = data.shape

        Z = feat_alloc_dist.rvs(1, N)

        K = Z.shape[1]

        mu = scipy.stats.gamma.rvs(1, scale=1, size=(K, D))
        
        nu = scipy.stats.gamma.rvs(1, scale=1, size=(D,))
        
        return Parameters(1, np.ones(2), mu, np.ones(2), nu, np.ones(2), Z)

    def _init_joint_dist(self, feat_alloc_dist):
        self.joint_dist = pgfa.models.base.JointDistribution(
            DataDistribution(), feat_alloc_dist, ParametersDistribution()
        )


class ModelUpdater(pgfa.models.base.AbstractModelUpdater):

    def _update_model_params(self, model):
        update_mu(model)

        update_nu(model)


class Parameters(pgfa.models.base.AbstractParameters):

    def __init__(self, alpha, alpha_prior, mu, mu_prior, nu, nu_prior, Z):
        self.alpha = float(alpha)

        self.alpha_prior = np.array(alpha_prior, dtype=np.float64)

        self.mu = np.array(mu, dtype=np.float64)

        self.mu_prior = np.array(mu_prior, dtype=np.float64)

        self.nu = np.array(nu, dtype=np.float64)

        self.nu_prior = np.array(nu_prior, dtype=np.float64)

        self.Z = np.array(Z, dtype=np.int8)

    @property
    def param_shapes(self):
        return {
            'alpha': (),
            'alpha_prior': (2,),
            'mu': ('K', 'D'),
            'mu_prior': (2,),
            'nu': ('D',),
            'nu_prior': (2,),
            'Z': ('N', 'K')
        }

    @property
    def D(self):
        return self.mu.shape[1]

    @property
    def N(self):
        return self.Z.shape[0]

    def copy(self):
        return Parameters(
            self.alpha,
            self.alpha_prior.copy(),
            self.mu.copy(),
            self.mu_prior.copy(),
            self.nu,
            self.nu_prior.copy(),
            self.Z.copy()
        )


#=========================================================================
# Updates
#=========================================================================
def update_mu(model, variance=1):
    params = model.params.copy()
    
    for d in range(model.params.D):
        for k in range(model.params.K):
            mu_old = params.mu[k, d]
            
            a, b = _get_gamma_params(mu_old, variance)
    
            mu_new = scipy.stats.gamma.rvs(a, scale=(1 / b))
        
            params.mu[k, d] = mu_new
            
            log_p_new = model.joint_dist.log_p(model.data, params)
            
            log_q_new = scipy.stats.gamma.logpdf(mu_new, a, scale=(1 / b))
        
            a, b = _get_gamma_params(mu_new, variance)
            
            params.mu[k, d] = mu_old
            
            log_p_old = model.joint_dist.log_p(model.data, params)
            
            log_q_old = scipy.stats.gamma.logpdf(mu_old, a, scale=(1 / b))
            
            if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
                params.mu[k, d] = mu_new
            
            else:
                params.mu[k, d] = mu_old
    
    model.params = params


def update_nu(model, variance=1):
    params = model.params.copy()
    
    for d in range(model.params.D):
        nu_old = params.nu[d]
        
        a, b = _get_gamma_params(nu_old, variance)
    
        nu_new = scipy.stats.gamma.rvs(a, scale=(1 / b))
        
        params.nu[d] = nu_new
        
        log_p_new = model.joint_dist.log_p(model.data, params)
        
        log_q_new = scipy.stats.gamma.logpdf(nu_new, a, scale=(1 / b))
        
        a, b = _get_gamma_params(nu_new, variance)
    
        params.nu[d] = nu_old
        
        log_p_old = model.joint_dist.log_p(model.data, params)
        
        log_q_old = scipy.stats.gamma.logpdf(nu_old, a, scale=(1 / b))
        
        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
            params.nu[d] = nu_new
        
        else:
            params.nu[d] = nu_old
    
    model.params = params


def _get_gamma_params(mean, variance):
    b = mean / variance
    
    a = b * mean
    
    return a, b


#=========================================================================
# Densities and proposals
#=========================================================================
class DataDistribution(pgfa.models.base.AbstractDataDistribution):

    def log_p(self, data, params):
        return _log_p(params.mu, params.nu, data.astype(np.float), params.Z.astype(np.float))

    def log_p_row(self, data, params, row_idx):
        return _log_p_row(params.mu, params.nu, data[row_idx].astype(np.float), params.Z[row_idx].astype(np.float))


class ParametersDistribution(pgfa.models.base.AbstractParametersDistribution):

    def log_p(self, params):
        log_p = 0

        # Gamma prior on $\alpha$
        a = params.alpha_prior[0]
        b = params.alpha_prior[1]
        log_p += scipy.stats.gamma.logpdf(params.alpha, a, scale=(1 / b))

        # Gamma prior on $\mu$
        a = params.mu_prior[0]
        b = params.mu_prior[1]
        log_p += np.sum(scipy.stats.gamma.logpdf(params.mu, a, scale=(1 / b)))

        # Gamma prior on $\nu$
        a = params.nu_prior[0]
        b = params.nu_prior[1]
        log_p += np.sum(scipy.stats.gamma.logpdf(params.nu, a, scale=(1 / b)))
     
        return log_p


@numba.njit(cache=True)
def _log_p(mu, nu, X, Z):
    N = X.shape[0]
    
    log_p = 0
    
    for n in range(N):
        log_p += _log_p_row(mu, nu, X[n], Z[n])
    
    return log_p


@numba.njit(cache=True)
def _log_p_row(mu, nu, x, z):
    D = mu.shape[1] 
    
    log_p = 0
    
    for d in range(D): 
        mean = np.sum(z * mu[:, d]) + nu[d]
   
        log_p += x[d] * np.log(mean) - log_factorial(x[d]) - mean
    
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
