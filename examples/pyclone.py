import numpy as np

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_priors
import pgfa.models.pyclone
import pgfa.updates


def main():
    np.random.seed(0)

    D = 4
    K = 5
    N = 100

    feat_alloc_prior = pgfa.feature_allocation_priors.BetaBernoulliFeatureAllocationDistribution(1.0, 1.0, K)

    data, params = simulate_data(feat_alloc_prior, N, D, alpha=1)

    singletons_updater = pgfa.models.pyclone.PriorSingletonsUpdater()

    feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(
        pgfa.updates.ParticleGibbsUpdater(
            annealed=True, num_particles=5, singletons_updater=singletons_updater
        )
    )

#     feat_alloc_updater = pgfa.updates.GibbsUpdater(singletons_updater=singletons_updater)

    model_updater = pgfa.models.pyclone.PyCloneFeatureAllocationModelUpdater(feat_alloc_updater)

    feat_alloc_prior = pgfa.feature_allocation_priors.IndianBuffetProcessDistribution()

    model = pgfa.models.pyclone.PyCloneFeatureAllocationModel(data, feat_alloc_prior)

    model.params.eta = np.random.gamma(1, 1, size=(D, 1))

    model.params.Z = np.random.randint(0, 2, size=(N, 1))

    print(np.sum(params.Z, axis=0))

    for i in range(100000):
        if i % 10 == 0:
            print(i, model.params.K, model.log_p)

            print(np.sum(model.params.Z, axis=0))

            print(get_b_cubed_score(params.Z, model.params.Z))

            print('#' * 100)

        model_updater.update(model)


def simulate_data(feat_alloc_prior, num_data_points, num_samples, alpha=1, kappa=10):
    D = num_samples
    N = num_data_points

    Z = feat_alloc_prior.rvs(N)

    K = Z.shape[1]

    eta = np.random.gamma(kappa, 1, size=(D, K))

    phi = eta / np.sum(eta, axis=1)[:, np.newaxis]

    data = np.zeros((N, D, 2))

    for d in range(D):
        for n in range(N):
            data[n, d, 0] = np.random.poisson(10000)

            f = np.sum(Z[n] * phi[d]) / np.sum(phi[d])

            data[n, d, 1] = np.random.binomial(data[n, d, 0], f)

    params = pgfa.models.pyclone.Parameters(kappa, eta, Z)

    return data, params


if __name__ == '__main__':
    main()
