import numba
import numpy as np
import scipy.optimize

from pgfa.data_structures import Particle, ParticleSwarm
from pgfa.math_utils import log_sum_exp
from pgfa.updates.base import FeatureAllocationMatrixUpdater


class DiscreteParticleFilterUpdater(FeatureAllocationMatrixUpdater):

    def __init__(self,
                 annealing_power=0.0,
                 conditional_update=True,
                 max_particles=10,
                 singletons_updater=None,
                 test_path='conditional'):

        self.singletons_updater = singletons_updater

        if conditional_update:
            self.row_updater = ConditionalDiscreteParticleFilterRowUpdater(
                annealing_power, max_particles, singletons_updater, test_path
            )

        else:
            self.row_updater = DiscreteParticleFilterRowUpdater(
                annealing_power, max_particles, singletons_updater, test_path
            )

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        return self.row_updater.update_row(cols, data, dist, feat_probs, params, row_idx)


class AbstractDiscreteParticleFilterRowUpdater(FeatureAllocationMatrixUpdater):

    def _resample(self, swarm):
        raise NotImplementedError

    def _update_row(self, cols, data, dist, feat_probs, params, row_idx, test_path):
        raise NotImplementedError

    def __init__(self, annealing_power=0.0, max_particles=10, singletons_updater=None, test_path='conditional'):
        self.annealing_power = annealing_power

        self.max_particles = max_particles

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
                max_particles=self.max_particles,
                singletons_updater=self.singletons_updater,
                test_path='random'
            )

            test_params = params.copy()

            test_params = updater.update_row(cols, data, dist, feat_probs, test_params, row_idx)

            test_path = test_params.Z[row_idx, cols]

        elif self.test_path == 'two-stage':
            updater = DiscreteParticleFilterRowUpdater(
                annealing_power=self.annealing_power,
                max_particles=self.max_particles,
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
            if swarm.num_particles > self.max_particles:
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

        log_l = scipy.optimize.bisect(_resample_opt_func, log_W.min(), 1000, args=(np.log(self.max_particles), log_W))

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

    def _resample(self, swarm):
        log_W = swarm.log_weights

        log_W[np.isneginf(log_W)] = -1e10

        log_l = scipy.optimize.bisect(_resample_opt_func, log_W.min(), 1000, args=(np.log(self.max_particles), log_W))

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
