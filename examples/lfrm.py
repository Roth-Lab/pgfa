import numpy as np

from pgfa.feature_allocation_priors import BetaBernoulliFeatureAllocationDistribution
from pgfa.math_utils import bernoulli_rvs
from pgfa.models.lfrm import LatentFactorRelationalModel, Parameters
from pgfa.updates.feature_matrix import GibbsMixtureUpdater, ParticleGibbsUpdater
from pgfa.utils import get_b_cubed_score


def main():
    np.random.seed(0)

    K = 4
    N = 20

    feat_alloc_prior = BetaBernoulliFeatureAllocationDistribution(1, 1, K)

    data, data_true, params = simulate_data(feat_alloc_prior, N)

    feat_alloc_updater = ParticleGibbsUpdater(annealed=True)

    feat_alloc_updater = GibbsMixtureUpdater(feat_alloc_updater)

    model = LatentFactorRelationalModel(data, feat_alloc_prior, feat_alloc_updater)

    for i in range(10000):
        if i % 100 == 0:
            print(i, model.params.K, model.log_p)

            if model.params.K > 0:
                print(get_b_cubed_score(params.Z, model.params.Z))

            print(np.sum(model.params.Z, axis=0))

            print('#' * 100)

        model.update()


def simulate_data(feat_alloc_prior, N, tau=1):
    Z = feat_alloc_prior.rvs(N)

    K = Z.shape[1]

    V = np.random.normal(0, 1 / np.sqrt(tau), size=(K, K))
    V = np.triu(V)
    V = V + V.T - np.diag(np.diag(V))

    params = Parameters(1, tau, V, Z)

    data_true = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                data_true[i, j] = 1

            else:
                m = Z[i].T @ V @ Z[j]

                f = np.exp(-m)

                p = 1 / (1 + f)

                data_true[i, j] = bernoulli_rvs(p)

    data = data_true.copy()

    for i in range(N):
        for j in range(N):
            if np.random.random() < 0.1:
                data[i, j] = np.nan

    return data, data_true, params


if __name__ == '__main__':
    main()
