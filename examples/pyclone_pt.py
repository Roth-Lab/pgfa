import numba
import numpy as np

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.pyclone
import pgfa.updates


def main():
    seed = 0
    np.random.seed(seed)
    set_numba_seed(seed)

    D = 4
    K = 10
    N = 100

    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K)

    data, params = simulate_data(feat_alloc_dist, N, D, alpha=4, kappa=0.1)

    singletons_updater = pgfa.models.pyclone.PriorSingletonsUpdater()

    singletons_updater = None

    feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(
        pgfa.updates.ParticleGibbsUpdater(
            annealed=True, num_particles=10, singletons_updater=singletons_updater
        )
    )

    feat_alloc_updater = pgfa.updates.GibbsUpdater(singletons_updater=singletons_updater)

    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K)

    model_updater = pgfa.models.pyclone.ParallelTemperingUpdater(data, feat_alloc_dist, feat_alloc_updater)
    
    model_updater = pgfa.models.pyclone.ModelUpdater(feat_alloc_updater)
    
    model = pgfa.models.pyclone.Model(data, feat_alloc_dist)

#     model.params.Z[:] = 0
#
#     model.params.Z[:, 0] = 1

    old_params = model.params.copy()

    model.params = params.copy()

    true_log_p = model.log_p

    model.params = old_params.copy()

    print(np.sum(params.Z, axis=0))

    for i in range(100000):
        if i % 10 == 0:
            print(
                i,
                model.params.K,
                model.params.alpha,
                model.log_p,
                (model.log_p - true_log_p) / abs(true_log_p),
                reconstruction_error(model.data, model.params)
            )

            print(np.sum(model.params.Z, axis=0))

            print(get_b_cubed_score(params.Z, model.params.Z))

            print('#' * 100)

        model_updater.update(model, param_updates=1)


def simulate_data(feat_alloc_prior, num_data_points, num_samples, alpha=1, kappa=1):
    D = num_samples
    N = num_data_points

    Z = feat_alloc_prior.rvs(alpha, N)

    K = Z.shape[1]

    eta = np.random.gamma(kappa, scale=1, size=(K, D))

    phi = eta / np.sum(eta, axis=0)

    F = Z @ phi

    data = np.zeros((N, D, 2))

    for d in range(D):
        for n in range(N):
            data[n, d, 0] = np.random.poisson(100000)

            data[n, d, 1] = np.random.binomial(data[n, d, 0], F[n, d])

    params = pgfa.models.pyclone.Parameters(alpha, np.ones(2), eta, Z)

    return data, params


def reconstruction_error(data, params):
    V = params.eta
    Z = params.Z
    X = data

    D = params.D
    N = params.N

    W = V / np.sum(V, axis=0)

    M = Z @ W

    error = 0

    for n in range(N):
        for d in range(D):
            e_x = M[n, d] * X[n, d, 0]

            error += abs(e_x - X[n, d, 1])

    error /= N * D

    return error


@numba.jit
def set_numba_seed(seed):
    np.random.seed(seed)


if __name__ == '__main__':
    main()
