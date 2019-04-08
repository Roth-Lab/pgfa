import numpy as np
import scipy.optimize

from pgfa.data_structures import Particle, ParticleSwarm
from pgfa.math_utils import bernoulli_rvs, log_sum_exp
from pgfa.updates.base import FeatureAllocationMatrixUpdater


class DicreteParticleFilterUpdater(FeatureAllocationMatrixUpdater):

    def __init__(self, annealing_power=0.0, max_particles=10, singletons_updater=None):
        self.annealing_power = annealing_power
        
        self.max_particles = max_particles

        self.singletons_updater = singletons_updater

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        T = len(cols)

        conditional_path = params.Z[row_idx, cols].copy()
        
        log_feat_probs = np.row_stack([np.log1p(-feat_probs), np.log(feat_probs)])

        swarm = ParticleSwarm()
        
        swarm.add_particle(0, None)
        
        for t in range(T):
            if swarm.num_particles > self.max_particles:
                swarm = self._resample(swarm)
        
            new_swarm = ParticleSwarm()

            annealing_factor = self._get_annealing_factor(t, T)
            
            col = cols[t]
            
            states = [conditional_path[t], 1 - conditional_path[t]]

            for log_W, parent_particle in zip(swarm.log_weights, swarm.particles):
                if parent_particle is not None:
                    params.Z[row_idx, cols[:t]] = parent_particle.path
          
                for s in states:
                    particle = self._get_new_particle(
                        annealing_factor, col, data, dist, log_feat_probs, params, parent_particle, row_idx, s
                    )
                    
                    new_swarm.add_particle(log_W + particle.log_w, particle)

            swarm = new_swarm

        assert np.all(swarm[0].path == conditional_path)

        params.Z[row_idx, cols] = swarm.sample().path

        return params

    def _get_annealing_factor(self, t, T):
        if self.annealing_power == 'K':
            power = T
        
        else:
            power = self.annealing_power
        
        annealing_factor = ((t + 1) / T) ** power
            
        return annealing_factor

    def _get_new_particle(self, annealing_factor, col, data, dist, log_feat_probs, params, parent, row_idx, value):
        if parent is None:
            parent_log_p = 0
            
            parent_path = []

        else:
            parent_log_p = parent.log_p
            
            parent_path = parent.path

        params.Z[row_idx, col] = value
        
        prior = log_feat_probs[value, col]

        log_p = annealing_factor * dist.log_p_row(data, params, row_idx)
        
        if np.isneginf(log_p) or np.isneginf(parent_log_p):
            log_w = -np.inf
        
        else:
            log_w = prior + log_p - parent_log_p

        return Particle(log_p, log_w, parent, parent_path + [value])
    
    def _resample(self, swarm):
        log_W = swarm.log_weights
        
        def f(x):
            y = log_W - x
            y[y >= 0] = 0
            return log_sum_exp(y)
        
        log_l = scipy.optimize.bisect(lambda x:f(x) - np.log(self.max_particles), log_W.min(), 1000)
        
        new_swarm = ParticleSwarm()
        
        if log_W[0] >= log_l:
            new_swarm.add_particle(log_W[0], swarm[0])
        
        else:
            new_swarm.add_particle(log_l, swarm[0])
        
        for i in range(1, swarm.num_particles):
            if log_W[i] >= log_l:
                new_swarm.add_particle(log_W[i], swarm[i])
            
            else:
                if bernoulli_rvs(np.exp(log_W[i] - log_l)) == 1:
                    new_swarm.add_particle(log_l, swarm[i])
        
        return new_swarm
# 
# #     def _resample(self, swarm):
# #         log_W = swarm.log_weights      
# #  
# #         keep_idxs, resample_idxs, log_C = _split_particles(log_W, self.max_particles)
# #   
# #         num_resampled_particles = self.max_particles - len(keep_idxs)
# #           
# #         resample_log_w = np.array([swarm.unnormalized_log_weights[i] for i in resample_idxs])
# #         
# #         if num_resampled_particles > 0:
# #             if 0 in keep_idxs:
# #                 resample_idxs_sub = stratified_resampling(resample_log_w, num_resampled_particles)
# #       
# #             else:
# #                 assert resample_idxs[0] == 0
# #       
# #                 resample_idxs_sub = conditional_stratified_resampling(resample_log_w, num_resampled_particles)
# #   
# #             resample_idxs = [resample_idxs[i] for i in resample_idxs_sub]
# #         
# #         else:
# #             resample_idxs = []
# #   
# #         idxs = sorted(keep_idxs + resample_idxs)
# #           
# #         try:
# #             assert idxs[0] == 0
# #           
# #         except AssertionError:
# #             print(resample_idxs)
# #             print(keep_idxs)
# #             print(-log_C, log_W[0])
# #   
# #         new_swarm = ParticleSwarm()
# #   
# #         for i in idxs:
# #             if i in keep_idxs:
# #                 new_swarm.add_particle(log_W[i], swarm[i])
# #   
# #             else:
# #                 new_swarm.add_particle(-log_C, swarm[i])
# #  
# #         return new_swarm
# 
# 
# def _split_particles(log_W, N):    
# 
#     def f(x):
#         y = log_W + x
#         y[y > 0] = 0
#         return log_sum_exp(y)
#     
#     log_C = scipy.optimize.bisect(lambda x:f(x) - np.log(N), log_W.min(), 1000)
#     
#     kept, resamp = [], []
# 
#     for i in range(len(log_W)):
#         if log_W[i] > -log_C:
#             kept.append(i)
# 
#         else:
#             resamp.append(i)
#     
#     # TODO: Need this to hack around the case we won't resample, but also don't keep 0 (conditional path)
#     if (len(kept) == N) and (0 not in kept):
#         kept, resamp, log_C = _split_particles(log_W, N - 1)
# 
#     return kept, resamp, log_C    
# 
# # @numba.njit(cache=True)
# # def _split_particles(log_W, N):
# #     for idx in np.argsort(log_W):
# #         log_kappa = log_W[idx]
# # 
# #         log_ratio = log_W - log_kappa
# # 
# #         log_ratio[log_ratio > 0] = 0
# # 
# #         total = log_sum_exp(log_ratio)
# # 
# #         if total <= np.log(N):
# #             break
# # 
# #     A = np.sum(log_W > log_kappa)
# # 
# #     B = log_sum_exp(log_W[log_W <= log_kappa])
# # 
# #     log_C = np.log(N - A) - B
# #     
# #     W = np.exp(log_W)
# #     
# #     def f(x):
# #         y = W / x
# #         y[y >= 1] = 1
# #         return np.sum(y)
# #     
# #     print(scipy.optimize.bisect(lambda x:f(x) - N, 0, 1000), np.exp(-log_C), np.exp(log_kappa))
# #     
# # #     print(log_C, -log_kappa)
# #     
# #     kept, resamp = [], []
# # 
# #     for i in range(len(log_W)):
# #         if log_W[i] > log_kappa:
# #             kept.append(i)
# # 
# #         else:
# #             resamp.append(i)
# # 
# #     return kept, resamp, log_C

