import numba
import numpy as np

from pgfa.data_structures import Particle, ParticleSwarm
from pgfa.math_utils import conditional_multinomial_resampling, conditional_stratified_resampling, log_sum_exp, \
    multinomial_resampling, stratified_resampling
from pgfa.updates.base import FeatureAllocationMatrixUpdater


class ParticleGibbsUpdater(FeatureAllocationMatrixUpdater):

    def __init__(
            self,
            annealing_power=0.0,
            conditional_update=True,
            num_particles=10,
            resample_scheme='stratified',
            resample_threshold=0.5,
            singletons_updater=None,
            test_path='zeros'):

        super().__init__(singletons_updater=singletons_updater)

        if conditional_update:
            self.row_updater = ConditionalSequentialMonteCarloRowUpdater(
                annealing_power, num_particles, resample_scheme, resample_threshold, test_path
            )

        else:
            self.row_updater = SequentialMonteCarloRowUpdater(
                annealing_power, num_particles, resample_scheme, resample_threshold, test_path
            )

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        return self.row_updater.update_row(cols, data, dist, feat_probs, params, row_idx)


class AbstractSequentialMonteCarloRowUpdater(object):

    def _resample(self, swarm):
        raise NotImplementedError

    def _update_row(self, cols, data, dist, feat_probs, params, row_idx, test_path):
        raise NotImplementedError

    def __init__(
            self,
            annealing_power=0.0,
            num_particles=10,
            resample_scheme='stratified',
            resample_threshold=0.5,
            test_path='zeros'):

        self.annealing_power = annealing_power

        self.num_particles = num_particles

        self.resample_scheme = resample_scheme

        self.resample_threshold = resample_threshold

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
            updater = SequentialMonteCarloRowUpdater(
                annealing_power=self.annealing_power,
                num_particles=self.num_particles,
                resample_scheme=self.resample_scheme,
                resample_threshold=self.resample_threshold,
                test_path='random'
            )

            test_params = params.copy()

            test_params = updater.update_row(cols, data, dist, feat_probs, test_params, row_idx)

            test_path = test_params.Z[row_idx, cols]

        elif self.test_path == 'two-stage':
            updater = SequentialMonteCarloRowUpdater(
                annealing_power=self.annealing_power,
                num_particles=self.num_particles,
                resample_scheme=self.resample_scheme,
                resample_threshold=self.resample_threshold,
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
            value = bernoulli_rvs_log(log_q[1] - log_norm)

        log_w = _get_log_w(log_norm, parent_log_p)

        return Particle(log_p[value], log_w, parent, parent_path + [value])


class ConditionalSequentialMonteCarloRowUpdater(AbstractSequentialMonteCarloRowUpdater):

    def _update_row(self, cols, data, dist, feat_probs, params, row_idx, test_path):
        T = len(cols)

        conditional_path = params.Z[row_idx, cols].copy()

        params.Z[row_idx, cols] = test_path

        log_feat_probs = np.row_stack([np.log1p(-feat_probs), np.log(feat_probs)])

        swarm = ParticleSwarm()

        for _ in range(self.num_particles):
            swarm.add_particle(0, None)

        for t in range(T):
            if t > 0:
                try:
                    swarm = self._resample(swarm)

                except ValueError:
                    params.Z[row_idx, cols] = conditional_path

                    return params

                assert np.all(swarm.particles[0].path == conditional_path[:t])

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


class SequentialMonteCarloRowUpdater(AbstractSequentialMonteCarloRowUpdater):

    def _update_row(self, cols, data, dist, feat_probs, params, row_idx, test_path):
        T = len(cols)

        old_path = params.Z[row_idx, cols].copy()

        params.Z[row_idx, cols] = test_path

        log_feat_probs = np.row_stack([np.log1p(-feat_probs), np.log(feat_probs)])

        swarm = ParticleSwarm()

        for _ in range(self.num_particles):
            swarm.add_particle(0, None)

        for t in range(T):
            if t > 0:
                try:
                    swarm = self._resample(swarm)

                except ValueError:
                    params.Z[row_idx, cols] = old_path

                    return params

            new_swarm = ParticleSwarm()

            annealing_factor = self._get_annealing_factor(t, T)

            col = cols[t]

            for parent_particle, log_W in zip(swarm.particles, swarm.log_weights):
                if parent_particle is not None:
                    params.Z[row_idx, cols[:t]] = parent_particle.path

                particle = self._get_new_particle(
                    annealing_factor, col, data, dist, log_feat_probs, params, parent_particle, row_idx, value=None
                )

                new_swarm.add_particle(log_W + particle.log_w, particle)

            swarm = new_swarm

        params.Z[row_idx, cols] = swarm.sample().path

        return params

    def _resample(self, swarm):
        if swarm.relative_ess <= self.resample_threshold:
            new_swarm = ParticleSwarm()

            if self.resample_scheme == 'multinomial':
                idxs = multinomial_resampling(swarm.unnormalized_log_weights, self.num_particles)

            elif self.resample_scheme == 'stratified':
                idxs = stratified_resampling(swarm.unnormalized_log_weights, self.num_particles)

            else:
                raise Exception('Unknown resampling scheme: {}'.format(self.resample_scheme))

            for idx in idxs:
                new_swarm.add_particle(0, swarm.particles[idx])

        else:
            new_swarm = swarm

        return new_swarm


@numba.njit(cache=True)
def bernoulli_rvs_log(log_p):
    if np.log(np.random.random()) < log_p:
        return 1

    else:
        return 0


@numba.jit(cache=True)
def _get_log_w(log_norm, parent_log_p):
    """ Workaround to slow np.isneginf.
    """
    if np.isinf(log_norm) and log_norm < 0:
        log_w = -np.inf

    elif np.isinf(parent_log_p) and parent_log_p < 0:
        log_w = -np.inf

    else:
        log_w = log_norm - parent_log_p

    return log_w
