import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import do_metropolis_hastings_accept_reject, log_factorial

import pgfa.models.base

from sampyl import Slice


class Model(pgfa.models.base.AbstractModel):

    @staticmethod
    def get_default_params(data, feat_alloc_dist):
        N, D = data.shape

        Z = feat_alloc_dist.rvs(1, N)

        K = Z.shape[1]
        
        nu = scipy.stats.gamma.rvs(1, 1)
        
        S = scipy.stats.gamma.rvs(1, 1, size=(K, D))
        
        V = scipy.stats.gamma.rvs(1, 1, size=(N, K))
                
        return Parameters(1.0, np.ones(2), nu, np.ones(2), S, np.ones(2), V, np.ones(2), Z)

    def _init_joint_dist(self, feat_alloc_dist):
        self.joint_dist = pgfa.models.base.JointDistribution(
            DataDistribution(), feat_alloc_dist, ParametersDistribution()
        )


class ModelUpdater(pgfa.models.base.AbstractModelUpdater):

    def _update_model_params(self, model):
#         update_nu(model)
        
        update_S(model)

        update_V(model)
        

class Parameters(pgfa.models.base.AbstractParameters):

    def __init__(self, alpha, alpha_prior, nu, nu_prior, S, S_prior, V, V_prior, Z):
        self.alpha = float(alpha)

        self.alpha_prior = np.array(alpha_prior, dtype=np.float64)
        
        self.nu = nu
        
        self.nu_prior = nu_prior

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
            'nu': (),
            'nu_prior': (2,),
            'S': ('K', 'D'),
            'S_prior': (2,),
            'V': ('N', 'K'),
            'V_prior': (2,),
            'Z': ('N', 'K')
        }
    
    @property
    def D(self):
        return self.S.shape[1]

    @property
    def N(self):
        return self.Z.shape[0]
    
    def copy(self):
        return Parameters(
            self.alpha,
            self.alpha_prior.copy(),
            self.nu,
            self.nu_prior.copy(),
            self.S.copy(),
            self.S_prior.copy(),
            self.V.copy(),
            self.V_prior.copy(),
            self.Z.copy()
        )


#=========================================================================
# Updates
#=========================================================================
def update_nu(model, variance=1):
    params = model.params.copy()
    
    prior = model.params.nu_prior
    
    old = params.nu

    a, b = _get_gamma_params(old, variance)
        
    new = scipy.stats.gamma.rvs(a, scale=(1 / b))

    params.nu = new
    
    log_p_new = model.data_dist.log_p(model.data, params)
    
    log_p_new += scipy.stats.gamma.logpdf(new, prior[0], scale=(1 / prior[1]))
    
    log_q_new = scipy.stats.gamma.logpdf(new, a, scale=(1 / b))
    
    a, b = _get_gamma_params(old, variance)

    params.nu = old
    
    log_p_old = model.data_dist.log_p(model.data, params)
    
    log_p_old += scipy.stats.gamma.logpdf(old, prior[0], scale=(1 / prior[1]))
    
    log_q_old = scipy.stats.gamma.logpdf(old, a, scale=(1 / b))
    
    if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
        params.nu = new
    
    else:
        params.nu = old
    
    model.params = params


def update_r(model, variance=1):
    params = model.params.copy()
    
    r_prior = model.params.r_prior
    
    for n in range(model.params.N):
        r_old = params.r[n]
    
        a, b = _get_gamma_params(r_old, variance)
            
        r_new = scipy.stats.gamma.rvs(a, scale=(1 / b))

        params.r[n] = r_new
        
        log_p_new = model.data_dist.log_p_row(model.data, params, n)
        
        log_p_new += scipy.stats.gamma.logpdf(r_new, r_prior[0], scale=(1 / r_prior[1]))
        
        log_q_new = scipy.stats.gamma.logpdf(r_new, a, scale=(1 / b))
        
        a, b = _get_gamma_params(r_old, variance)

        params.r[n] = r_old
        
        log_p_old = model.data_dist.log_p_row(model.data, params, n)
        
        log_p_old += scipy.stats.gamma.logpdf(r_old, r_prior[0], scale=(1 / r_prior[1]))
        
        log_q_old = scipy.stats.gamma.logpdf(r_old, a, scale=(1 / b))
        
        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
            params.r[n] = r_new
        
        else:
            params.r[n] = r_old
    
    model.params = params


def update_S(model, variance=1):
    
    def log_p(x, d):
        old = model.params.S[:, d]
        
        model.params.S[:, d] = x
        
        log_p = model.log_p
        
        model.params.S[:, d] = old
        
        return log_p
    
    for d in range(model.params.D):
        sampler = Slice(
            lambda x: log_p(x, d),
            model.params.S[:, d]
        )
        
        model.params.S[:, d] = sampler.sample(1)
    
#     params = model.params.copy()
#     
#     S_prior = model.params.S_prior
# 
#     for k in range(model.params.K):
#         for d in range(model.params.D):
#             S_old = params.S[k, d]
#         
#             a, b = _get_gamma_params(S_old, variance)
#                 
#             S_new = scipy.stats.gamma.rvs(a, scale=(1 / b))
#     
#             params.S[k, d] = S_new
#             
#             log_p_new = model.data_dist.log_p(model.data, params)
#             
#             log_p_new += scipy.stats.gamma.logpdf(S_new, S_prior[0], scale=(1 / S_prior[1]))
#             
#             log_q_new = scipy.stats.gamma.logpdf(S_new, a, scale=(1 / b))
#             
#             a, b = _get_gamma_params(S_old, variance)
#     
#             params.S[k, d] = S_old
#             
#             log_p_old = model.data_dist.log_p(model.data, params)
#             
#             log_p_old += scipy.stats.gamma.logpdf(S_old, S_prior[0], scale=(1 / S_prior[1]))
#             
#             log_q_old = scipy.stats.gamma.logpdf(S_old, a, scale=(1 / b))
#             
#             if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
#                 params.S[k, d] = S_new
#             
#             else:
#                 params.S[k, d] = S_old
#     
#     model.params = params


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
        return _log_p(params.nu, params.S, params.V, data, params.Z.astype(float))

    def log_p_row(self, data, params, row_idx):
        return _log_p_row(params.nu, params.S, params.V[row_idx], data[row_idx], params.Z[row_idx].astype(float))


class ParametersDistribution(pgfa.models.base.AbstractParametersDistribution):

    def log_p(self, params):
        log_p = 0

        # Gamma prior on $\alpha$
        a, b = params.alpha_prior
        log_p += scipy.stats.gamma.logpdf(params.alpha, a, scale=(1 / b))
        
        # Gamma prior on $nu$
        a, b = params.nu_prior
        log_p += scipy.stats.gamma.logpdf(params.nu, a, scale=(1 / b))
        
        # Gamma prior on $S$
        a, b = params.S_prior
        log_p += np.sum(scipy.stats.gamma.logpdf(params.S, a, scale=(1 / b)))
     
        # Gamma prior on $V$
        a, b = params.V_prior
        log_p += np.sum(scipy.stats.gamma.logpdf(params.V, a, scale=(1 / b)))
        
        return log_p


@numba.njit(cache=True)
def _log_p(nu, S, V, X, Z):
    N = X.shape[0]
    
    log_p = 0
    
    for n in range(N):
        log_p += _log_p_row(nu, S, V[n], X[n], Z[n])
    
    return log_p


@numba.njit(cache=True)
def _log_p_row(nu, S, v, x, z):
    D = S.shape[1]
    
    m = (z * v) @ S
    
    log_p = 0 
    
    for d in range(D):
        if np.isnan(x[d]):
            continue
        
        log_p = poisson_log_pdf(x[d], m[d])
        
#         log_p += negative_binomial_log_pdf(x[d], 1 - m[d] / (m[d] + nu), m[d] ** 2 / nu)
        
#         log_p = x[d] * m[d]
#         
#         log_p -= log_factorial(x[d])
#         
#         log_p -= m[d]
#     
    return log_p


@numba.njit(cache=True)
def negative_binomial_log_pdf(k, p, r):
    return log_factorial(k + r - 1) - log_factorial(k) - log_factorial(r - 1) + k * np.log(p) + r * np.log1p(-p)


@numba.njit(cache=True)
def poisson_log_pdf(k, m):
    return k * np.log(m) - log_factorial(k) - m
    
# 
# @numba.njit(cache=True)
# def _log_p_row(S, v, x, z):
#     K, D = S.shape
#     
#     v_pad = np.ones(len(v) + 1, dtype=np.float64)
#     
#     v_pad[1:] = v
#     
#     z_pad = np.ones(len(z) + 1, dtype=np.float64)
#     
#     z_pad[1:] = z
#     
#     w = v_pad * z_pad
#     
#     w = w / np.sum(w)
#     
#     S_pad = np.zeros((K + 1, D))
#     
#     S_pad[0] = 1 / D
#     
#     S_pad[1:] = S
#     
#     pi = w @ S_pad
#     
#     log_p = 0 
#     
#     for d in range(D):
#         log_p += log_factorial(x[d])
#         
#         if pi[d] > 0:
#             log_p += x[d] * np.log(pi[d])
#     
#     log_p -= log_factorial(np.sum(x))
#     
#     return log_p

