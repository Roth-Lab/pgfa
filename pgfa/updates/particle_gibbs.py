import numpy as np

from pgfa.data_structures import Particle, ParticleSwarm
from pgfa.math_utils import conditional_multinomial_resampling, conditional_stratified_resampling, discrete_rvs, log_sum_exp
from pgfa.updates.base import FeatureAllocationMatrixUpdater


class ParticleGibbsUpdater(FeatureAllocationMatrixUpdater):

    def __init__(self, annealed=False, num_particles=10, resample_scheme='multinomial', resample_threshold=0.5, singletons_updater=None):
        self.annealed = annealed

        self.num_particles = num_particles
        
        self.resample_scheme = resample_scheme

        self.resample_threshold = resample_threshold

        self.singletons_updater = singletons_updater
    
    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        T = len(cols)

        conditional_path = params.Z[row_idx, cols].copy()
        
        log_feat_probs = np.row_stack([np.log1p(-feat_probs), np.log(feat_probs)])

        swarm = ParticleSwarm()
        
        for _ in range(self.num_particles):
            swarm.add_particle(0, None)

        for t in range(T):           
            if t > 0:
                particles = self._resample(swarm)

                assert np.all(particles[0].path == conditional_path[:t])

            new_swarm = ParticleSwarm()

            annealing_factor = self._get_annealing_factor(t, T)
            
            col = cols[t]

            for i, (parent_particle, log_W) in enumerate(zip(swarm.particles, swarm.log_weights)):
                if parent_particle is not None:
                    params.Z[row_idx, cols[:t]] = parent_particle.path
                
                if i == 0:
                    value = conditional_path[t]
                
                else:
                    value = None
                
                particle = self._get_new_particle(
                    annealing_factor, col, data, dist, log_feat_probs, params, parent_particle, row_idx, value=value
                )
                
                new_swarm.add_particle(log_W + particle.log_w, particle)

            swarm = new_swarm

        params.Z[row_idx, cols] = swarm.sample().path

        return params
    
    def _get_annealing_factor(self, t, T):
        if self.annealed:
            annealing_factor = (t + 1) / T
        
        else:
            annealing_factor = 1
            
        return annealing_factor

    def _get_new_particle(self, annealing_factor, col, data, dist, log_feat_probs, params, parent, row_idx, value=None):
        if parent is None:
            parent_log_p = 0
            
            parent_path = []

        else:
            parent_log_p = parent.log_p
            
            parent_path = parent.path
        
        log_q = np.zeros(2)
        
        log_p = np.zeros(2)
        
        for i in range(2):
            params.Z[row_idx, col] = i
    
            log_p[i] = annealing_factor * dist.log_p_row(data, params, row_idx)
            
            log_q[i] = log_feat_probs[i, col] + log_p[i]
        
        log_norm = log_sum_exp(log_q)
        
        if value is None:
            value = discrete_rvs(np.exp(log_q - log_norm)) 
        
        log_w = log_norm - parent_log_p
        
        return Particle(log_p[value], log_w, parent, parent_path + [value])

    def _resample(self, swarm):
        if swarm.relative_ess <= self.resample_threshold:
            new_swarm = ParticleSwarm()
            
            if self.resample_scheme == 'multinomial':
                idxs = conditional_multinomial_resampling(swarm.unnormalized_log_weights, self.num_particles)
            
            elif self.resample_scheme == 'stratified':
                idxs = conditional_stratified_resampling(swarm.unnormalized_log_weights, self.num_particles)
            
            else:
                raise Exception('Unknown resampling scheme: {}'.format(self.resample_scheme))
            
            idxs = sorted(idxs)
            
            assert idxs[0] == 0

            for idx in idxs:
                new_swarm.add_particle(0, swarm.particles[idx])

        else:
            new_swarm = swarm

        return new_swarm

