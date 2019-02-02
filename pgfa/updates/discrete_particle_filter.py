import numpy as np

from pgfa.math_utils import discrete_rvs, log_normalize, log_sum_exp

from pgfa.updates.base import FeatureAllocationMatrixUpdater


class DicreteParticleFilterUpdater(FeatureAllocationMatrixUpdater):
    def __init__(self, max_particles=10, singletons_updater=None):
        self.max_particles = max_particles

        self.singletons_updater = singletons_updater

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        T = len(cols)

        conditional_path = params.Z[row_idx, cols]

        particles = [None]

        for t in range(T):
            if t > 0:
                assert np.all(particles[0].genealogy == conditional_path[:t])

            if len(particles) > self.max_particles:
                particles = self._resample(particles)

            new_particles = []

            for i, p in enumerate(particles):
                if i == 0:
                    new_particles.append(
                        self._get_new_particle(
                            cols[:t + 1], data, dist, feat_probs, params, p, row_idx, conditional_path[t]
                        )
                    )

                for s in [0, 1]:
                    if (i == 0) and (s == conditional_path[t]):
                        continue

                    new_particles.append(
                        self._get_new_particle(
                            cols[:t + 1], data, dist, feat_probs, params, p, row_idx, s
                        )
                    )

            particles = new_particles

        W = np.exp(log_normalize(np.array([p.log_w for p in particles])))

        idx = discrete_rvs(W)

        params.Z[row_idx, cols] = particles[idx].genealogy

        return params

    def _get_new_particle(self, cols, data, dist, feat_probs, params, parent, row_idx, value):
        if parent is None:
            genealogy = []

            log_w = 0

            parent_log_p = 0

        else:
            genealogy = parent.genealogy

            log_w = parent.log_w

            parent_log_p = parent.log_p

        genealogy.append(value)

        genealogy = np.array(genealogy, dtype=np.int)

        params.Z[row_idx, cols] = genealogy

        log_p = np.sum(genealogy * np.log(feat_probs[cols]) + (1 - genealogy) * np.log(1 - feat_probs[cols]))

        log_p += dist.log_p_row(data, params, row_idx)

        return Particle(log_p, log_w + log_p - parent_log_p, parent, value)

    def _resample(self, particles):
        log_W = log_normalize(np.array([p.log_w for p in particles]))

        keep_idxs, resample_idxs, log_C = self._split_particles(particles)

        num_resampled_particles = self.max_particles - len(keep_idxs)

        resample_particles = [particles[i] for i in resample_idxs]

        if 0 in keep_idxs:
            resample_idxs_sub = self._stratified_resample(num_resampled_particles, resample_particles)

        else:
            assert resample_idxs[0] == 0

            resample_idxs_sub = self._conditional_stratified_resample(num_resampled_particles, resample_particles)

        resample_idxs = [resample_idxs[i] for i in resample_idxs_sub]

        idxs = sorted(keep_idxs + resample_idxs)

        assert idxs[0] == 0

        new_particles = []

        for i in idxs:
            p = particles[i]

            if i in keep_idxs:
                new_particles.append(Particle(p.log_p, log_W[i], p.parent, p.value))

            else:
                new_particles.append(Particle(p.log_p, -log_C, p.parent, p.value))

        return new_particles

    def _split_particles(self, particles):
        N = self.max_particles

        log_W = log_normalize(np.array([p.log_w for p in particles]))

        for idx in np.argsort(log_W):
            log_kappa = log_W[idx]

            total = log_W - log_kappa

            total[total > 0] = 0

            total = log_sum_exp(total)

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

    def _conditional_stratified_resample(self, num_resampled, particles):
        W = np.exp(log_normalize(np.array([p.log_w for p in particles])))

        U = np.random.uniform(0, W[0])

        positions = (U - np.floor(num_resampled * U) / num_resampled) + np.arange(num_resampled) / num_resampled

        indexes = []

        cumulative_sum = np.cumsum(W)

        i, j = 0, 0

        while i < num_resampled:
            if positions[i] < cumulative_sum[j]:
                indexes.append(j)

                i += 1

            else:
                j += 1

        return indexes

    def _stratified_resample(self, num_resampled, particles):
        W = np.exp(log_normalize(np.array([p.log_w for p in particles])))

        positions = (np.random.random() + np.arange(num_resampled)) / num_resampled

        indexes = []

        cumulative_sum = np.cumsum(W)

        i, j = 0, 0

        while i < num_resampled:
            if positions[i] < cumulative_sum[j]:
                indexes.append(j)

                i += 1

            else:
                j += 1

        return indexes


class Particle(object):
    def __init__(self, log_p, log_w, parent, value):
        self.log_p = log_p
        self.log_w = log_w
        self.parent = parent
        self.value = value

    @property
    def genealogy(self):
        genealogy = [self.value]
        parent = self.parent
        while parent is not None:
            genealogy.append(parent.value)
            parent = parent.parent
        return list(reversed(genealogy))
