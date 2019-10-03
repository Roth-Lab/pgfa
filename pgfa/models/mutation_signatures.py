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
        
        S = scipy.stats.dirichlet.rvs(np.ones(D), size=K)
        
        V = scipy.stats.gamma.rvs(1, 1, size=(N, K))
        
        return Parameters(1.0, np.ones(2), S, np.ones(D), V, np.ones(2), Z)

    def _init_joint_dist(self, feat_alloc_dist):
        self.joint_dist = pgfa.models.base.JointDistribution(
            DataDistribution(), feat_alloc_dist, ParametersDistribution()
        )


class ModelUpdater(pgfa.models.base.AbstractModelUpdater):

    def _update_model_params(self, model):
        update_V(model)


class Parameters(pgfa.models.base.AbstractParameters):

    def __init__(self, alpha, alpha_prior, S, S_prior, V, V_prior, Z):
        self.alpha = float(alpha)

        self.alpha_prior = np.array(alpha_prior, dtype=np.float64)

        self.S = np.array(S, dtype=np.float64)

        self.S_prior = np.array(S_prior, dtype=np.float64)

        self.V = np.array(V, dtype=np.float64)

        self.V_prior = np.array(V_prior, dtype=np.float64)

        self.Z = np.array(Z, dtype=np.int8)

    @property
    def param_shapes(self):
        return {
            'alpha': (),
            'alpha_prior': (2,),
            'S': ('K', 'D'),
            'S_prior': ('D',),
            'V': ('N', 'K'),
            'V_prior': (2,),
            'Z': ('N', 'K')
        }
    
    @property
    def pi(self):
        return self.W @ self.S
    
    @property
    def D(self):
        return self.S.shape[1]

    @property
    def N(self):
        return self.Z.shape[0]
    
    @property
    def W(self):
        W = self.V * self.Z
        
        return W / np.sum(W, axis=1)[:, np.newaxis]
    
    def copy(self):
        return Parameters(
            self.alpha,
            self.alpha_prior.copy(),
            self.S.copy(),
            self.S_prior.copy(),
            self.V.copy(),
            self.V_prior.copy(),
            self.Z.copy()
        )


#=========================================================================
# Updates
#=========================================================================
def update_V(model, variance=1):
    params = model.params.copy()
    
    a_prior, b_prior = model.params.V_prior
    
    for n in range(model.params.N):
        for k in range(model.params.K):
            v_old = params.V[n, k]
            
            a, b = _get_gamma_params(v_old, variance)
    
            v_new = scipy.stats.gamma.rvs(a, scale=(1 / b))
        
            params.V[n, k] = v_new
            
            log_p_new = model.data_dist.log_p_row(model.data, params, n)
            
            log_p_new += scipy.stats.gamma.logpdf(v_new, a_prior, scale=(1 / b_prior))
            
            log_q_new = scipy.stats.gamma.logpdf(v_new, a, scale=(1 / b))
        
            a, b = _get_gamma_params(v_new, variance)
            
            params.V[n, k] = v_old
            
            log_p_old = model.data_dist.log_p_row(model.data, params, n)
            
            log_p_old += scipy.stats.gamma.logpdf(v_old, a_prior, scale=(1 / b_prior))
            
            log_q_old = scipy.stats.gamma.logpdf(v_old, a, scale=(1 / b))
            
            if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
                params.V[n, k] = v_new
            
            else:
                params.V[n, k] = v_old
    
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
        return _log_p(params.S, params.V, data, params.Z.astype(float))

    def log_p_row(self, data, params, row_idx):
        return _log_p_row(params.S, params.V[row_idx], data[row_idx], params.Z[row_idx].astype(float))


class ParametersDistribution(pgfa.models.base.AbstractParametersDistribution):

    def log_p(self, params):
        log_p = 0

        # Gamma prior on $\alpha$
        a = params.alpha_prior[0]
        b = params.alpha_prior[1]
        log_p += scipy.stats.gamma.logpdf(params.alpha, a, scale=(1 / b))

        # Gamma prior on $V$
        a = params.V_prior[0]
        b = params.V_prior[1]
        log_p += np.sum(scipy.stats.gamma.logpdf(params.V, a, scale=(1 / b)))
     
        return log_p


@numba.njit(cache=True)
def _log_p(S, V, X, Z):
    N = X.shape[0]
    
    log_p = 0
    
    for n in range(N):
        log_p += _log_p_row(S, V[n], X[n], Z[n])
    
    return log_p


@numba.njit(cache=True)
def _log_p_row(S, v, x, z):
    D = S.shape[1]
    
    w = v * z
    
    if np.sum(w) == 0:
        return -np.inf
    
    w = w / np.sum(w)
    
    pi = w @ S
    
    log_p = 0 
    
    for d in range(D):
        if np.isnan(x[d]):
            continue
        
        log_p -= log_factorial(x[d])
        
        if pi[d] > 0:
            log_p += x[d] * np.log(pi[d])
    
    log_p += log_factorial(np.nansum(x))
    
    return log_p

