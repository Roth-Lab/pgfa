import numpy as np
import numba


@numba.jit
def get_ess(log_W):
    W = np.exp(log_W)

    return 1 / np.sum(np.square(W))


@numba.jit
def resample(log_W, particles, conditional=True, threshold=0.5):
    num_features = len(log_W)

    num_particles = particles.shape[0]

    if (get_ess(log_W) / num_particles) <= threshold:
        new_particles = np.zeros(particles.shape, dtype=np.int64)

        W = np.exp(log_W)

        W = W + 1e-10

        W = W / np.sum(W)

        if conditional:
            new_particles[0] = particles[0]

            multiplicity = np.random.multinomial(num_particles - 1, W)

            idx = 1

        else:
            multiplicity = np.random.multinomial(num_particles, W)

            idx = 0

        for k in range(num_features):
            for _ in range(multiplicity[k]):
                new_particles[idx] = particles[k]

                idx += 1

        log_W = -np.log(num_particles) * np.ones(num_particles)

        particles = new_particles

    return log_W, particles
