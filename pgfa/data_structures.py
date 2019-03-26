from collections import namedtuple

import numpy as np

from pgfa.math_utils import discrete_rvs_gumbel_trick, log_sum_exp


class ParticleSwarm(object):

    def __init__(self):
        self.particles = []

        self._log_norm_const = None

        self._unnormalized_log_weights = []

    def __getitem__(self, idx):
        return self.particles[idx]

    @property
    def ess(self):
        return 1 / np.sum(np.square(self.weights))

    @property
    def log_norm_const(self):
        if self._log_norm_const is None:
            self._log_norm_const = log_sum_exp(self.unnormalized_log_weights)

        return self._log_norm_const

    @property
    def log_weights(self):
        return self.unnormalized_log_weights - self.log_norm_const

    @property
    def num_particles(self):
        return len(self.particles)

    @property
    def relative_ess(self):
        return self.ess / self.num_particles

    @property
    def unnormalized_log_weights(self):
        return np.array(self._unnormalized_log_weights)

    @property
    def weights(self):
        weights = np.exp(self.log_weights)

        return weights

    def add_particle(self, log_w, particle):
        '''
        Args:
            log_weight: Unnormalized log weight of particle
            particle: Particle
        '''
        self.particles.append(particle)

        self._unnormalized_log_weights.append(log_w)

        self._log_norm_const = None

    def sample(self):
        idx = discrete_rvs_gumbel_trick(self.unnormalized_log_weights)

        return self.particles[idx]

    def to_dict(self):
        return dict(zip(self.particles, self.weights))

    def to_list(self):
        return list(zip(self.particles, self.weights))


Particle = namedtuple('Particle', ['log_p', 'log_w', 'parent', 'path'])

