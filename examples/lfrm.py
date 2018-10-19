import numpy as np

from pgfa.math_utils import bernoulli_rvs
from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_priors
import pgfa.models.lfrm
import pgfa.updates.feature_matrix


def main():
    np.random.seed(0)

    K = 4
    N = 100

    feat_alloc_prior = pgfa.feature_allocation_priors.BetaBernoulliFeatureAllocationDistribution(10, 1, K)

    data, data_true, params = simulate_data(feat_alloc_prior, N)

#     singletons_updater = None

    singletons_updater = pgfa.models.lfrm.PriorSingletonsUpdater()

    feat_alloc_updater = pgfa.updates.feature_matrix.ParticleGibbsUpdater(
        annealed=True, singletons_updater=singletons_updater
    )

    feat_alloc_updater = pgfa.updates.feature_matrix.GibbsMixtureUpdater(feat_alloc_updater)

#     feat_alloc_updater = pgfa.updates.feature_matrix.GibbsUpdater(singletons_updater=singletons_updater)

    model_updater = pgfa.models.lfrm.LatentFactorRelationalModelUpdater(feat_alloc_updater)

    feat_alloc_prior = pgfa.feature_allocation_priors.IndianBuffetProcessDistribution()

    model = pgfa.models.lfrm.LatentFactorRelationalModel(data, feat_alloc_prior, symmetric=False)

    print(np.sum(params.Z, axis=0))

    print('@' * 100)

    for i in range(10000):
        if i % 10 == 0:
            print(i, model.params.K, model.log_p, np.sum(np.abs(model.predict() - data_true)), model.params.tau)

            if model.params.K > 0:
                print(get_b_cubed_score(params.Z, model.params.Z))

            print(np.sum(model.params.Z, axis=0))

            print('#' * 100)

        model_updater.update(model)


def simulate_data(feat_alloc_prior, N, tau=0.1):
    Z = feat_alloc_prior.rvs(N)

    K = Z.shape[1]

    V = np.random.normal(0, 1 / np.sqrt(tau), size=(K, K))
    V = np.triu(V)
    V = V + V.T - np.diag(np.diag(V))

    params = pgfa.models.lfrm.Parameters(tau, V, Z)

    data_true = np.zeros((N, N))

#     for i in range(N):
#         for j in range(i, N):
#             if i == j:
#                 data_true[i, j] = 1
#
#             else:
#                 m = Z[i].T @ V @ Z[j]
#
#                 f = np.exp(-m)
#
#                 p = 1 / (1 + f)
#
#                 data_true[i, j] = bernoulli_rvs(p)
#
#                 data_true[j, i] = data_true[i, j]

    for i in range(N):
        for j in range(N):
            m = Z[i].T @ V @ Z[j]

            f = np.exp(-m)

            p = 1 / (1 + f)

            data_true[i, j] = bernoulli_rvs(p)

    data = data_true.copy()

    for i in range(N):
        for j in range(N):
            if np.random.random() < 0.0:
                data[i, j] = np.nan

    return data, data_true, params


if __name__ == '__main__':
    main()
