import numba
import numpy as np
import scipy.optimize

from pgfa.data_structures import Particle, ParticleSwarm
from pgfa.math_utils import log_sum_exp
from pgfa.updates.base import FeatureAllocationMatrixUpdater


class DiscreteParticleFilterUpdater(FeatureAllocationMatrixUpdater):

    def __init__(
            self,
            annealing_power=0.0,
            conditional_update=True,
            num_particles=10,
            test_path='zeros',
            **kwargs):

        super().__init__(**kwargs)

        if conditional_update:
            self.row_updater = ConditionalDiscreteParticleFilterRowUpdater(
                annealing_power, num_particles, singletons_updater, test_path
            )

        else:
            self.row_updater = DiscreteParticleFilterRowUpdater(
                annealing_power, num_particles, singletons_updater, test_path
            )

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        return self.row_updater.update_row(cols, data, dist, feat_probs, params, row_idx)


class AbstractDiscreteParticleFilterRowUpdater(FeatureAllocationMatrixUpdater):

    def _resample(self, swarm):
        raise NotImplementedError

    def _update_row(self, cols, data, dist, feat_probs, params, row_idx, test_path):
        raise NotImplementedError

    def __init__(self, annealing_power=0.0, num_particles=10, singletons_updater=None, test_path='zeros'):
        self.annealing_power = annealing_power

        self.num_particles = num_particles

        self.singletons_updater = singletons_updater

        self.test_path = test_path

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        if self.test_path == 'conditional':
            test_path = params.Z[row_idx, cols]

        elif self.test_path == 'ones':
            test_path = np.ones(len(cols))

        elif self.test_path == 'zeros':
            test_path = np.zeros(len(cols))

        elif self.test_path == 'random':
            test_path = np.random.randint(0, 2, size=len(cols))

        elif self.test_path == 'unconditional':
            updater = DiscreteParticleFilterRowUpdater(
                annealing_power=self.annealing_power,
                num_particles=self.num_particles,
                singletons_updater=self.singletons_updater,
                test_path='random'
            )

            test_params = params.copy()

            test_params = updater.update_row(cols, data, dist, feat_probs, test_params, row_idx)

            test_path = test_params.Z[row_idx, cols]

        elif self.test_path == 'two-stage':
            updater = DiscreteParticleFilterRowUpdater(
                annealing_power=self.annealing_power,
                num_particles=self.num_particles,
                singletons_updater=self.singletons_updater,
                test_path='conditional'
            )

            test_params = params.copy()

            test_params = updater.update_row(cols, data, dist, feat_probs, test_params, row_idx)

            test_path = test_params.Z[row_idx, cols]

        return self._update_row(cols, data, dist, feat_probs, params, row_idx, test_path)

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

        log_w = _get_log_w(log_p, parent_log_p, prior)

        return Particle(log_p, log_w, parent, parent_path + [value])


class DiscreteParticleFilterRowUpdater(AbstractDiscreteParticleFilterRowUpdater):
    """ Unconditional discrete particle filter for sampling a row of a feature allocation matrix.

    NOTE: This cannot be used as a valid Gibbs update as it does not target the correct distribution.
    See PMMH paper for details.
    """

    def _update_row(self, cols, data, dist, feat_probs, params, row_idx, test_path):
        T = len(cols)

        log_feat_probs = np.row_stack([np.log1p(-feat_probs), np.log(feat_probs)])

        swarm = ParticleSwarm()

        swarm.add_particle(0, None)

        params.Z[row_idx, cols] = test_path

        for t in range(T):
            if swarm.num_particles > self.num_particles:
                swarm = self._resample(swarm)

            new_swarm = ParticleSwarm()

            annealing_factor = self._get_annealing_factor(t, T)

            col = cols[t]

            for log_W, parent_particle in zip(swarm.log_weights, swarm.particles):
                if parent_particle is not None:
                    params.Z[row_idx, cols[:t]] = parent_particle.path

                for s in [0, 1]:
                    particle = self._get_new_particle(
                        annealing_factor, col, data, dist, log_feat_probs, params, parent_particle, row_idx, s
                    )

                    new_swarm.add_particle(log_W + particle.log_w, particle)

            swarm = new_swarm

        params.Z[row_idx, cols] = swarm.sample().path

        return params

    def _resample(self, swarm):
        log_W = swarm.log_weights

        log_W[np.isneginf(log_W)] = -1e10

        log_l = scipy.optimize.bisect(_resample_opt_func, log_W.min(), 1000, args=(np.log(self.num_particles), log_W))

        new_swarm = ParticleSwarm()

        for i in range(swarm.num_particles):
            if log_W[i] >= log_l:
                new_swarm.add_particle(log_W[i], swarm[i])

            else:
                if bernoulli_rvs_log(log_W[i] - log_l):
                    new_swarm.add_particle(log_l, swarm[i])

        return new_swarm


class ConditionalDiscreteParticleFilterRowUpdater(AbstractDiscreteParticleFilterRowUpdater):

    def _update_row(self, cols, data, dist, feat_probs, params, row_idx, test_path):
        T = len(cols)

        conditional_path = params.Z[row_idx, cols].copy()

        params.Z[row_idx, cols] = test_path

        log_feat_probs = np.row_stack([np.log1p(-feat_probs), np.log(feat_probs)])

        swarm = ParticleSwarm()

        swarm.add_particle(0, None)

        for t in range(T):
            if swarm.num_particles > self.num_particles:
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

    def _resample(self, swarm):
        log_W = swarm.log_weights

        log_W[np.isneginf(log_W)] = -1e10

        log_l = scipy.optimize.bisect(_resample_opt_func, log_W.min(), 1000, args=(np.log(self.num_particles), log_W))

        new_swarm = ParticleSwarm()

        if log_W[0] >= log_l:
            new_swarm.add_particle(log_W[0], swarm[0])

        else:
            new_swarm.add_particle(log_l, swarm[0])

        for i in range(1, swarm.num_particles):
            if log_W[i] >= log_l:
                new_swarm.add_particle(log_W[i], swarm[i])

            else:
                if bernoulli_rvs_log(log_W[i] - log_l):
                    new_swarm.add_particle(log_l, swarm[i])

        return new_swarm


@numba.njit(cache=True)
def bernoulli_rvs_log(log_p):
    if np.log(np.random.random()) < log_p:
        return True

    else:
        return False


@numba.jit(cache=True)
def _get_log_w(log_p, parent_log_p, prior):
    """ Workaround to slow np.isneginf.
    """
    if np.isinf(log_p) and log_p < 0:
        log_w = -np.inf

    elif np.isinf(parent_log_p) and parent_log_p < 0:
        log_w = -np.inf

    else:
        log_w = prior + log_p - parent_log_p

    return log_w


@numba.njit(cache=True)
def _resample_opt_func(x, log_N, log_W):
    y = log_W - x

    y[y >= 0] = 0

    return log_sum_exp(y) - log_N
