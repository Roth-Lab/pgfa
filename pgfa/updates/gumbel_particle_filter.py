import numpy as np

from pgfa.data_structures import Particle, ParticleSwarm
from pgfa.math_utils import conditional_gumbel_resampling, \
    conditional_multinomial_resampling
from pgfa.updates.base import FeatureAllocationMatrixUpdater


class GumbelParticleFilterUpdater(FeatureAllocationMatrixUpdater):

    def __init__(self, annealing_power=0.0, max_particles=10, replacement=False, singletons_updater=None):
        self.annealing_power = annealing_power
        
        self.max_particles = max_particles
        
        self.replacement = replacement

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
        
        log_w = prior + log_p - parent_log_p

        return Particle(log_p, log_w, parent, parent_path + [value])

    def _resample(self, swarm):
        log_W = swarm.log_weights      
        
        if self.replacement:
            idxs = conditional_multinomial_resampling(swarm.unnormalized_log_weights, self.max_particles)
        
        else:
            idxs = conditional_gumbel_resampling(swarm.unnormalized_log_weights, self.max_particles)
        
        idxs = np.sort(idxs)
     
        new_swarm = ParticleSwarm()
  
        for i in idxs:
            new_swarm.add_particle(log_W[i], swarm[i])
  
        return new_swarm

