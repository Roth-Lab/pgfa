import numba
import numpy as np

from pgfa.data_structures import Particle, ParticleSwarm
from pgfa.math_utils import conditional_stratified_resampling, log_sum_exp, stratified_resampling
from pgfa.updates.base import FeatureAllocationMatrixUpdater


class DicreteParticleFilterUpdater(FeatureAllocationMatrixUpdater):

    def __init__(self, annealed=False, max_particles=10, singletons_updater=None):
        self.annealed = annealed
        
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
        if self.annealed:
            annealing_factor = (t + 1) / T
        
        else:
            annealing_factor = 1
            
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

        keep_idxs, resample_idxs, log_C = _split_particles(log_W, self.max_particles)

        num_resampled_particles = self.max_particles - len(keep_idxs)

        resample_particles = [swarm.particles[i] for i in resample_idxs]

        if 0 in keep_idxs:
            resample_idxs_sub = self._stratified_resample(num_resampled_particles, resample_particles)

        else:
            assert resample_idxs[0] == 0

            resample_idxs_sub = self._conditional_stratified_resample(num_resampled_particles, resample_particles)

        resample_idxs = [resample_idxs[i] for i in resample_idxs_sub]

        idxs = sorted(keep_idxs + resample_idxs)
        
        try:
            assert idxs[0] == 0
        
        except AssertionError:
            print(resample_idxs)
            print(keep_idxs)

        new_swarm = ParticleSwarm()

        for i in idxs:
            if i in keep_idxs:
                new_swarm.add_particle(log_W[i], swarm[i])

            else:
                new_swarm.add_particle(-log_C, swarm[i])

        return new_swarm

    def _conditional_stratified_resample(self, num_resampled, particles):
        log_w = np.array([p.log_w for p in particles])
        
        return conditional_stratified_resampling(log_w, num_resampled)

    def _stratified_resample(self, num_resampled, particles):
        log_w = np.array([p.log_w for p in particles])
        
        return stratified_resampling(log_w, num_resampled)


@numba.njit(cache=True)
def _split_particles(log_W, N):
    for idx in np.argsort(log_W):
        log_kappa = log_W[idx]

        log_ratio = log_W - log_kappa

        log_ratio[log_ratio > 0] = 0

        total = log_sum_exp(log_ratio)

        if total <= np.log(N):
            break

    A = np.sum(log_W > log_kappa)

    B = log_sum_exp(log_W[log_W <= log_kappa])

    log_C = np.log(N - A) - B

    kept, resamp = [], []

    for i in range(len(log_W)):
        if log_W[i] > -log_C:
            kept.append(i)

        else:
            resamp.append(i)

    return kept, resamp, log_C

