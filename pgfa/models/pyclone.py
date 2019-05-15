import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import do_metropolis_hastings_accept_reject, log_beta, log_factorial, log_normalize, \
    log_sum_exp, discrete_rvs

import pgfa.models.base


def get_data_point(a, b, cn_major, cn_minor, cn_normal=2, error_rate=1e-3, tumour_content=None):
    cn_total = cn_major + cn_minor

    cn = []

    mu = []

    log_pi = []

    # Consider all possible mutational genotypes consistent with mutation before CN change
    for x in range(1, cn_major + 1):
        cn.append((cn_normal, cn_normal, cn_total))

        mu.append((error_rate, error_rate, min(1 - error_rate, x / cn_total)))

        log_pi.append(0)

    # Consider mutational genotype of mutation before CN change if not already added
    mutation_after_cn = (cn_normal, cn_total, cn_total)

    if mutation_after_cn not in cn:
        cn.append(mutation_after_cn)

        mu.append((error_rate, error_rate, min(1 - error_rate, 1 / cn_total)))

        log_pi.append(0)

        assert len(set(cn)) == 2
    
    cn = np.array(cn, dtype=np.int)
    
    mu = np.array(mu, dtype=np.float)
    
    log_pi = log_normalize(np.array(log_pi, dtype=np.float64))
    
    if tumour_content is None:
        tumour_content = np.ones(len(a))
        
    return DataPoint(np.array(a), np.array(b), cn, mu, log_pi, np.array(tumour_content))


@numba.jitclass([
    ('b', numba.int64[:]),
    ('d', numba.int64[:]),
    ('cn', numba.int64[:, :]),
    ('mu', numba.float64[:, :]),
    ('log_pi', numba.float64[:]),
    ('tumour_content', numba.float64[:])
])
class DataPoint(object):

    def __init__(self, a, b, cn, mu, log_pi, tumour_content=1.0):
        self.b = b
        self.d = a + b
        self.cn = cn
        self.mu = mu
        self.log_pi = log_pi
        self.tumour_content = tumour_content


class Model(pgfa.models.base.AbstractModel):

    @staticmethod
    def get_default_params(data, feat_alloc_dist):
        N = len(data)
        
        D = len(data[0].d)

        Z = feat_alloc_dist.rvs(1, N)

        K = Z.shape[1]
        
        precision = scipy.stats.gamma.rvs(1e-2, 1e2)
                
        V = scipy.stats.gamma.rvs(100, 1, size=(K, D))
        
        return Parameters(1, np.ones(2), precision, np.array([1e-2, 1e-2]), V, np.array([100, 1]), Z)

    def _init_joint_dist(self, feat_alloc_dist):
        self.joint_dist = pgfa.models.base.JointDistribution(
            DataDistribution(), feat_alloc_dist, ParametersDistribution()
        )


class ModelUpdater(pgfa.models.base.AbstractModelUpdater):

    def _update_model_params(self, model):
        for _ in range(20):
            update_precision(model)
            
        for _ in range(4):            
            f = np.random.choice([update_V_perm, update_V_block, update_V_block_dim])
            
            f(model)
            
        for _ in range(20):
            update_V_random_grid_pairwise(model, num_points=5)


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
# Updates
#=========================================================================
def update_precision(model, variance=1):   
    old = model.params.precision
    
    a_new, b_new = _get_gamma_params(old, variance)
    
    new = scipy.stats.gamma.rvs(a_new, scale=(1 / b_new))
    
    a_old, b_old = _get_gamma_params(new, variance)
    
    model.params.precision = new
    
    log_p_new = model.joint_dist.log_p(model.data, model.params)
    
    log_q_new = scipy.stats.gamma.logpdf(new, a_new, scale=(1 / b_new))
    
    model.params.precision = old
    
    log_p_old = model.joint_dist.log_p(model.data, model.params)
    
    log_q_old = scipy.stats.gamma.logpdf(old, a_old, scale=(1 / b_old))
    
    if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
        model.params.precision = new
    
    else:
        model.params.precision = old


def update_V(model, variance=1):
    params = model.params.copy()
       
    a_prior, b_prior = model.params.V_prior
     
    Ds = np.random.permutation(model.params.D)
     
    Ks = np.random.permutation(model.params.K)
     
    for d in Ds:
        for k in Ks:
            old = params.V[k, d]
               
            a, b = _get_gamma_params(old, variance)
       
            new = scipy.stats.gamma.rvs(a, scale=(1 / b))
           
            params.V[k, d] = new
               
            log_p_new = model.data_dist.log_p(model.data, params)
               
            log_p_new += scipy.stats.gamma.logpdf(new, a_prior, scale=(1 / b_prior))
               
            log_q_new = scipy.stats.gamma.logpdf(new, a, scale=(1 / b))
           
            a, b = _get_gamma_params(new, variance)
               
            params.V[k, d] = old
               
            log_p_old = model.data_dist.log_p(model.data, params)
               
            log_p_old += scipy.stats.gamma.logpdf(old, a_prior, scale=(1 / b_prior))
               
            log_q_old = scipy.stats.gamma.logpdf(old, a, scale=(1 / b))
               
            if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
                params.V[k, d] = new
               
            else:
                params.V[k, d] = old
       
    model.params = params


def update_V_perm(model):
    params = model.params.copy()
    
    for d in np.random.permutation(model.params.D):
        old = params.V[:, d].copy()
        
        new = params.V[np.random.permutation(params.K), d]
        
        params.V[:, d] = new
         
        log_p_new = model.data_dist.log_p(model.data, params)
         
        params.V[:, d] = old
         
        log_p_old = model.data_dist.log_p(model.data, params)
              
        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
#             print('Permutation accept')
            params.V[:, d] = new
          
        else:
            params.V[:, d] = old
      
    model.params = params 


def update_V_random_grid_pairwise(model, num_points=10):
    if model.params.K < 2:
        return
    
    ka, kb = np.random.choice(model.params.K, 2, replace=False)
    
    params = model.params.copy()
    
    old = params.V[[ka, kb]].flatten()
    
    D = params.D
    
    dim = 2 * D
    
    e = scipy.stats.multivariate_normal.rvs(np.zeros(dim), np.eye(dim))
    
    e /= np.linalg.norm(e)
    
    r = scipy.stats.gamma.rvs(1, 1)
    
    grid = np.arange(1, num_points + 1)
    
    ys = old[np.newaxis, :] + grid[:, np.newaxis] * r * e[np.newaxis, :]
    
    log_p_new = np.zeros(num_points)
    
    for i in range(num_points):
        params.V[[ka, kb]] = ys[i].reshape((2, D))
        
        log_p_new[i] = model.joint_dist.log_p(model.data, params)
    
    idx = discrete_rvs(np.exp(0.5 * np.log(grid) + log_normalize(log_p_new)))
    
    new = ys[idx]
    
    xs = new[np.newaxis, :] - grid[:, np.newaxis] * r * e[np.newaxis, :]
        
    log_p_old = np.zeros(num_points)

    for i in range(num_points):
        params.V[[ka, kb]] = xs[i].reshape((2, D))
        
        log_p_old[i] = model.joint_dist.log_p(model.data, params)
    
    if do_metropolis_hastings_accept_reject(log_sum_exp(log_p_new), log_sum_exp(log_p_old), 0, 0):
#         print('Accept', idx)
        params.V[[ka, kb]] = new.reshape((2, D))
    
    else:
        params.V[[ka, kb]] = old.reshape((2, D))
    
    model.params = params


def update_V_random_grid(model, num_points=10):
    if model.params.K < 2:
        return 
    
    params = model.params.copy()
    
    old = params.V.flatten()
    
    K, D = params.V.shape
    
    dim = K * D
    
    e = scipy.stats.multivariate_normal.rvs(np.zeros(dim), np.eye(dim))
    
    e /= np.linalg.norm(e)
    
    r = scipy.stats.gamma.rvs(1, 1)
    
    grid = np.arange(1, num_points + 1)
    
    ys = old[np.newaxis, :] + grid[:, np.newaxis] * r * e[np.newaxis, :]
    
    log_p_new = np.zeros(num_points)
    
    for i in range(num_points):
        params.V = ys[i].reshape((K, D))
        
        log_p_new[i] = model.joint_dist.log_p(model.data, params)
    
    idx = discrete_rvs(np.exp(0.5 * np.log(grid) + log_normalize(log_p_new)))
    
    new = ys[idx]
    
    xs = new[np.newaxis, :] - grid[:, np.newaxis] * r * e[np.newaxis, :]
        
    log_p_old = np.zeros(num_points)

    for i in range(num_points):
        params.V = xs[i].reshape((K, D))
        
        log_p_old[i] = model.joint_dist.log_p(model.data, params)
    
    if do_metropolis_hastings_accept_reject(log_sum_exp(log_p_new), log_sum_exp(log_p_old), 0, 0):
#         print('Accept', idx)
        params.V = new.reshape((K, D))
    
    else:
        params.V = old.reshape((K, D))
    
    model.params = params

 
def update_V_block(model, variance=1):
    params = model.params.copy()
      
    a_prior, b_prior = model.params.V_prior
      
    for k in np.random.permutation(model.params.K):
        old = params.V[k].copy()
        
        new = np.zeros(params.D)
        
        log_p_new = 0
        
        log_q_new = 0
        
        log_p_old = 0
        
        log_q_old = 0

        for d in range(model.params.D):            
            a, b = _get_gamma_params(old[d], variance)
      
            new[d] = scipy.stats.gamma.rvs(a, scale=(1 / b))
              
            log_p_new += scipy.stats.gamma.logpdf(new[d], a_prior, scale=(1 / b_prior))
              
            log_q_new += scipy.stats.gamma.logpdf(new[d], a, scale=(1 / b))
          
            a, b = _get_gamma_params(new[d], variance)
              
            log_p_old += scipy.stats.gamma.logpdf(old[d], a_prior, scale=(1 / b_prior))
              
            log_q_old += scipy.stats.gamma.logpdf(old[d], a, scale=(1 / b))
         
        params.V[k] = new
         
        log_p_new += model.data_dist.log_p(model.data, params)
         
        params.V[k] = old
         
        log_p_old += model.data_dist.log_p(model.data, params)
              
        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
            params.V[k] = new
          
        else:
            params.V[k] = old
      
    model.params = params


def update_V_block_dim(model, variance=1):
    params = model.params.copy()
      
    a_prior, b_prior = model.params.V_prior
      
    for d in np.random.permutation(model.params.D):
        old = params.V[:, d].copy()
        
        new = np.zeros(params.K)
        
        log_p_new = 0
        
        log_q_new = 0
        
        log_p_old = 0
        
        log_q_old = 0

        for k in range(model.params.K):            
            a, b = _get_gamma_params(old[k], variance)
      
            new[k] = scipy.stats.gamma.rvs(a, scale=(1 / b))
              
            log_p_new += scipy.stats.gamma.logpdf(new[k], a_prior, scale=(1 / b_prior))
              
            log_q_new += scipy.stats.gamma.logpdf(new[k], a, scale=(1 / b))
          
            a, b = _get_gamma_params(new[k], variance)
              
            log_p_old += scipy.stats.gamma.logpdf(old[k], a_prior, scale=(1 / b_prior))
              
            log_q_old += scipy.stats.gamma.logpdf(old[k], a, scale=(1 / b))
         
        params.V[:, d] = new
         
        log_p_new += model.data_dist.log_p(model.data, params)
         
        params.V[:, d] = old
         
        log_p_old += model.data_dist.log_p(model.data, params)
              
        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
            params.V[:, d] = new
          
        else:
            params.V[:, d] = old
      
    model.params = params

                
def _get_gamma_params(mean, variance):
    b = mean / variance
    
    a = b * mean
    
    return a, b


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

# class PriorSingletonsUpdater(object):
#  
#     def update_row(self, model, row_idx):
#         alpha = model.params.alpha
#  
#         D = model.params.D
#         N = model.params.N
#  
#         k_old = len(self._get_singleton_idxs(model.params.Z, row_idx))
#  
#         k_new = scipy.stats.poisson.rvs(alpha / N)
#  
#         if (k_new == 0) and (k_old == 0):
#             return model.params
#  
#         non_singleton_idxs = self._get_non_singleton_idxs(model.params.Z, row_idx)
#  
#         num_non_singletons = len(non_singleton_idxs)
#  
#         K_new = len(non_singleton_idxs) + k_new
#  
#         params_new = model.params.copy()
#   
#         params_new.V = np.zeros((K_new, D))
#   
#         params_new.V[:num_non_singletons] = model.params.V[non_singleton_idxs]
#  
#         params_new.Z = np.zeros((N, K_new), dtype=np.int8)
#  
#         params_new.Z[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]
#  
#         params_new.Z[row_idx, num_non_singletons:] = 1
#  
#         params_old = model.params.copy()
#  
#         params_old.V = np.zeros((model.params.K, D))
#   
#         params_old.V[:num_non_singletons] = model.params.V[non_singleton_idxs]
#  
#         params_old.Z = np.zeros((N, model.params.K), dtype=np.int8)
#  
#         params_old.Z[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]
#  
#         params_old.Z[row_idx, num_non_singletons:] = 1
#                  
#         a, b = model.params.V_prior
#          
#         log_p_new = np.zeros(100)
#          
#         log_p_old = np.zeros(100)
#          
#         for i in range(100):
#             params_new.V = scipy.stats.gamma.rvs(a, scale=(1 / b), size=(params_new.K, model.params.D)) 
#              
#             log_p_new[i] = model.data_dist.log_p(model.data, params_new)
#      
#             params_old.V = scipy.stats.gamma.rvs(a, scale=(1 / b), size=(params_old.K, model.params.D))
#      
#             log_p_old[i] = model.data_dist.log_p(model.data, params_old)
#         
#         log_p_new = log_sum_exp(log_p_new)
#         
#         log_p_old = log_sum_exp(log_p_old)
#  
#         if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
#             model.params = params_new
#              
#             params_new.V = scipy.stats.gamma.rvs(a, scale=(1 / b), size=(K_new, model.params.D))
#  
# #         else:
# #             model.params = params_old
#  
#     def _get_column_counts(self, Z, row_idx):
#         m = np.sum(Z, axis=0)
#  
#         m -= Z[row_idx]
#  
#         return m
#  
#     def _get_non_singleton_idxs(self, Z, row_idx):
#         m = self._get_column_counts(Z, row_idx)
#  
#         return np.atleast_1d(np.squeeze(np.where(m > 0)))
#  
#     def _get_singleton_idxs(self, Z, row_idx):
#         m = self._get_column_counts(Z, row_idx)
#  
#         return np.atleast_1d(np.squeeze(np.where(m == 0)))

 
class PriorSingletonsUpdater(object):
  
    def update_row(self, model, row_idx):
        alpha = model.params.alpha
  
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
            a, b = model.params.V_prior
              
            params_new.V[num_non_singletons:] = scipy.stats.gamma.rvs(a, scale=(1 / b), size=(k_new, model.params.D)) 
  
        params_new.Z = np.zeros((N, K_new), dtype=np.int8)
  
        params_new.Z[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]
  
        params_new.Z[row_idx, num_non_singletons:] = 1
  
        log_p_new = model.data_dist.log_p(model.data, params_new)
  
        log_p_old = model.data_dist.log_p(model.data, model.params)
  
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


class SplitMergeUpdater(object):

    def __init__(self, annealing_factor=1):
        self.annealing_factor = annealing_factor

    def update(self, model):
        anchors = np.random.choice(model.params.N, replace=False, size=2)
        
        if (np.sum(model.params.Z[anchors[0]]) == 0) or ((np.sum(model.params.Z[anchors[1]]) == 0)):
            return
        
        features, log_q_feature_fwd = self._select_features(anchors, model.params.Z)
        
        if features[0] == features[1]:
            V_new, Z_new, log_q_sm_fwd = self._propose_split(anchors, features, model, model.params.V, model.params.Z)
            
            _, log_q_feature_rev = self._select_features(anchors, Z_new)
            
            K_new = Z_new.shape[1]
            
            _, _, log_q_sm_rev = self._propose_merge(anchors, np.array([K_new - 2, K_new - 1]), V_new, Z_new)
        
        else:
            V_new, Z_new, log_q_sm_fwd = self._propose_merge(anchors, features, model.params.V, model.params.Z)
            
            _, log_q_feature_rev = self._select_features(anchors, Z_new)

            K_new = Z_new.shape[1]            
            
            _, _, log_q_sm_rev = self._propose_split(anchors, np.array([K_new - 1, K_new - 1]), model, V_new, Z_new, Z_target=model.params.Z[:, features])    
        
        params_fwd = model.params.copy()
        
        params_fwd.V = V_new
        
        params_fwd.Z = Z_new
        
        log_p_fwd = model.joint_dist.log_p(model.data, params_fwd)
        
        params_rev = model.params
        
        log_p_rev = model.joint_dist.log_p(model.data, params_rev)
        
        if do_metropolis_hastings_accept_reject(log_p_fwd, log_p_rev, self.annealing_factor * (log_q_feature_fwd + log_q_sm_fwd), self.annealing_factor * (log_q_feature_rev + log_q_sm_rev)):          
            model.params = params_fwd
        
        else:
            model.params = params_rev
        
    def _propose_merge(self, anchors, features, V, Z):
        k_a, k_b = features
        
        _, D = V.shape
        
        N, K = Z.shape
        
        V_new = np.zeros((K - 1, D), dtype=V.dtype)
                
        Z_new = np.zeros((N, K - 1), dtype=Z.dtype)
        
        idx = 0
        
        for k in range(K):
            if k in features:
                continue
            
            V_new[idx] = V[k]
            
            Z_new[:, idx] = Z[:, k]
            
            idx += 1
        
        active_set = list(set(np.atleast_1d(np.squeeze(np.where(Z[:, k_a] == 1)))) | set(np.atleast_1d(np.squeeze(np.where(Z[:, k_b] == 1)))))  
        
        Z_new[active_set, -1] = 1
        
        V_new[-1] = V[k_a] + V[k_b]
        
        return V_new, Z_new, 0
        
    def _propose_split(self, anchors, features, model, V, Z, Z_target=None):
        k_m = features[0]
        
        i, j = anchors
        
        _, D = V.shape
        
        N, K = Z.shape
        
        V_new = np.zeros((K + 1, D), dtype=V.dtype)
                
        Z_new = np.zeros((N, K + 1), dtype=Z.dtype)
        
        idx = 0
        
        for k in range(K):
            if k in features:
                continue
            
            V_new[idx] = V[k]
            
            Z_new[:, idx] = Z[:, k]
            
            idx += 1
        
        weight = np.random.random(D)
        
        V_new[-1] = weight * V[k_m] 
        
        V_new[-2] = (1 - weight) * V[k_m]
        
        Z_new[i, -1] = 1
        
        Z_new[j, -2] = 1
        
        active_set = list(np.squeeze(np.where(Z[:, k_m] == 1)))
        
        active_set.remove(i)
        
        active_set.remove(j)
        
        np.random.shuffle(active_set)
        
        log_q = 0
        
        log_p = np.zeros(3)
        
        params = model.params.copy()
        
        params.V = V_new
        
        params.Z = Z_new
        
        N_prev = 2
        
        for idx in active_set:  # + [i, j]:
            if idx not in [i, j]:
                N_prev += 1
            
            m_a = np.sum(Z_new[:, -1])
            
            m_b = np.sum(Z_new[:, -2])
            
            params.Z[idx, -1] = 1
            
            params.Z[idx, -2] = 0
            
            log_p[0] = np.log(m_a) + np.log(N_prev - m_b) + model.data_dist.log_p_row(model.data, params, idx)

            params.Z[idx, -1] = 0
            
            params.Z[idx, -2] = 1
            
            log_p[1] = np.log(N_prev - m_a) + np.log(m_b) + model.data_dist.log_p_row(model.data, params, idx)

            params.Z[idx, -1] = 1
            
            params.Z[idx, -2] = 1
            
            log_p[2] = np.log(m_a) + np.log(m_b) + model.data_dist.log_p_row(model.data, params, idx)
            
            log_p = log_normalize(log_p)
            
            if Z_target is None:
                state = discrete_rvs(np.exp(log_p))
            
            else:
                if np.all(Z_target[idx] == np.array([1, 0])):
                    state = 0
                
                elif np.all(Z_target[idx] == np.array([0, 1])):
                    state = 1
                
                elif np.all(Z_target[idx] == np.array([1, 1])):
                    state = 2
                
                else:
                    raise Exception('Invalid')
            
            if state == 0:
                Z_new[idx, -1] = 1
                
                Z_new[idx, -2] = 0
                
            elif state == 1:
                Z_new[idx, -1] = 0
                
                Z_new[idx, -2] = 1
            
            elif state == 2:
                Z_new[idx, -1] = 1
            
                Z_new[idx, -2] = 1
            
            else:
                raise Exception('Invalid state')
            
            log_q += log_p[state]
        
        assert Z_new is params.Z
        
        return V_new, Z_new, log_q               
                
    def _select_features(self, anchors, Z):
        i, j = anchors
        
        log_q = 0
        
        k_a = np.random.choice(np.where(Z[i] == 1)[0])
        
        log_q -= np.log(np.sum(Z[i]))
        
        k_b = np.random.choice(np.where(Z[j] == 1)[0])
        
        log_q -= np.log(np.sum(Z[j]))
        
        return np.array([k_a, k_b]), log_q


@numba.njit(cache=False)
def _log_p(X, precision, F, Z):
    N = len(X)
    
    log_p = 0
    
    for n in range(N):
        log_p += _log_p_row(X[n], precision, F, Z[n])
    
    return log_p


@numba.njit(cache=False)        
def _log_p_row(x, precision, F, z):
    phi = z @ F
    
    D = len(phi)
    
    G = len(x.log_pi)
    
    log_p = 0
    
    log_p_sample = np.zeros(G)
    
    for d in range(D):
        f = phi[d]
        
        t = x.tumour_content[d]
        
        for g in range(len(x.log_pi)):
            cn_n, cn_r, cn_v = x.cn[g]
        
            mu_n, mu_r, mu_v = x.mu[g]
        
            norm = (1 - t) * cn_n + t * (1 - f) * cn_r + t * f * cn_v
        
            prob = (1 - t) * cn_n * mu_n + t * (1 - f) * cn_r * mu_r + t * f * cn_v * mu_v
        
            prob /= norm

            log_p_sample[g] = x.log_pi[g] + log_beta_binomial_pdf(x.d[d], x.b[d], prob, precision)
        
        log_p += log_sum_exp(log_p_sample)
    
    return log_p


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
def log_binomial_pdf(n, x, p):   
    if p == 0:
        if x == 0:
            return 0
        
        else:
            return -np.inf
    
    elif p == 1:
        if x == n:
            return 0
        
        else:
            return -np.inf
    
    else:
        return log_binomial_coefficient(n, x) + x * np.log(p) + (n - x) * np.log1p(-p)


@numba.njit(cache=True)
def log_binomial_coefficient(n, x):
    return log_factorial(n) - log_factorial(x) - log_factorial(n - x)

