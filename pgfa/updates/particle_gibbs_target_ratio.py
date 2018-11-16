from collections import namedtuple

import numba
import numpy as np

from pgfa.math_utils import discrete_rvs, log_normalize, log_sum_exp

from pgfa.updates.base import FeatureAllocationMatrixUpdater


class ParticleGibbsUpdater(FeatureAllocationMatrixUpdater):
    def __init__(self, annealed=False, num_particles=10, resample_threshold=0.5, singletons_updater=None):
        self.annealed = annealed

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

        self.singletons_updater = singletons_updater

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        return do_particle_gibbs_update(
            cols,
            data,
            dist,
            feat_probs,
            params,
            row_idx,
            annealed=self.annealed,
            num_particles=self.num_particles,
            resample_threshold=self.resample_threshold
        )


Particle = namedtuple('Particle', ['log_p', 'log_w', 'parent', 'value'])


def do_particle_gibbs_update(
        cols,
        data,
        dist,
        feat_probs,
        params,
        row_idx,
        annealed=True,
        num_particles=10,
        resample_threshold=0.5):

    T = len(cols)

    log_W = np.zeros(num_particles)

    particles = [None for _ in range(num_particles)]

    z = params.Z[row_idx].copy()

    Zs = np.zeros((num_particles, T))

    for t in range(T):
        log_W = log_normalize(log_W)

        if t > 0:
            log_W, particles = _resample(log_W, particles, conditional=True, threshold=resample_threshold)

        for i in range(num_particles):
            if i == 0:
                idx = z[cols[t]]

            else:
                idx = -1

            params.Z[row_idx, cols] = Zs[i]

            particles[i] = _propose(
                cols[t], data, dist, feat_probs, params, particles[i], row_idx, t + 1, T, annealed=annealed, idx=idx
            )

            Zs[i, t] = particles[i].value

            log_W[i] = log_W[i] + particles[i].log_w

    log_W = log_normalize(log_W)

    W = np.exp(log_W)

    idx = discrete_rvs(W)

    params.Z[row_idx, cols] = Zs[idx]

    return params


@numba.njit(cache=True)
def _propose_idx(col, feat_probs, log_p):
    # Target
    log_a = np.log(feat_probs[col])

    log_b = np.log(1 - feat_probs[col])

    log_target = np.zeros(2)

    log_target[0] = log_b

    log_target[1] = log_p[1] - log_p[0] + log_a

    # Sample
    log_norm = log_sum_exp(log_target)

    p = np.exp(log_target - log_norm)

    idx = discrete_rvs(p)

    return idx, log_norm


@numba.njit(cache=True)
def _propose_idx_annealed(col, feat_probs, log_p, t, T):
    # Target
    log_a = np.log(feat_probs[col])

    log_b = np.log(1 - feat_probs[col])

    log_target = np.zeros(2)

    log_target[0] = (1 / T) * log_p[0] + log_b

    log_target[1] = (t / T) * log_p[1] - ((t - 1) / T) * log_p[0] + log_a

    # Sample
    log_norm = log_sum_exp(log_target)

    p = np.exp(log_target - log_norm)

    idx = discrete_rvs(p)

    return idx, log_norm


def _propose(col, data, dist, feat_probs, params, parent_particle, row_idx, t, T, annealed=False, idx=-1):
    log_p = np.zeros(2)

    # Feature off
    if parent_particle is None:
        params.Z[row_idx, col] = 0

        log_p[0] = dist.log_p_row(data, params, row_idx)

    else:
        log_p[0] = parent_particle.log_p

    # Feature on
    params.Z[row_idx, col] = 1

    log_p[1] = dist.log_p_row(data, params, row_idx)

    if annealed:
        prop_idx, log_w = _propose_idx_annealed(col, feat_probs, log_p, t, T)

    else:
        prop_idx, log_w = _propose_idx(col, feat_probs, log_p)

    if idx == -1:
        idx = prop_idx

    return Particle(log_p[idx], log_w, parent_particle, idx)


def _get_ess(log_W):
    W = np.exp(log_W)

    return 1 / np.sum(np.square(W))


def _resample(log_W, particles, conditional=True, threshold=0.5):
    num_features = len(log_W)

    num_particles = len(particles)

    if (_get_ess(log_W) / num_particles) <= threshold:
        new_particles = []

        W = np.exp(log_W)

        W = W + 1e-10

        W = W / np.sum(W)

        if conditional:
            new_particles.append(particles[0])

            multiplicity = np.random.multinomial(num_particles - 1, W)

        else:
            multiplicity = np.random.multinomial(num_particles, W)

        for k in range(num_features):
            for _ in range(multiplicity[k]):
                new_particles.append(particles[k])

        log_W = -np.log(num_particles) * np.ones(num_particles)

        particles = new_particles

    return log_W, particles


def iter_particles(particle):
    while particle is not None:
        yield particle

        particle = particle.parent


def get_z(particle):
    z = []

    for p in iter_particles(particle):
        z.append(p.value)

    return z[::-1]
