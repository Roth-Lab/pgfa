import numpy as np
import os
import scipy.stats

from pgfa.feature_allocation_priors import BetaBernoulliFeatureAllocationDistribution
from pgfa.models.linear_gaussian import LinearGaussianModel, Parameters
from pgfa.updates.feature_matrix import GibbsMixtureUpdater, ParticleGibbsUpdater
from pgfa.utils import get_b_cubed_score

os.environ['NUMBA_WARNINGS'] = '1'


def main():
    np.random.seed(0)

    num_iters = 10001
    D = 10
    K = 4
    N = 100

    feat_alloc_prior = BetaBernoulliFeatureAllocationDistribution(1, 1, K)

    data, params = simulate_data(feat_alloc_prior, D, N, tau_a=0.01, tau_x=100)

    feat_alloc_updater = GibbsMixtureUpdater(ParticleGibbsUpdater(annealed=True))

    model = LinearGaussianModel(data, feat_alloc_prior, feat_alloc_updater)

    print(np.sum(params.Z, axis=0))

    print('@' * 100)

    for i in range(num_iters):
        if i % 100 == 0:
            print(i, model.params.K, model.log_p)

            if model.params.K > 0:
                print(get_b_cubed_score(params.Z, model.params.Z))

            print(np.sum(model.params.Z, axis=0))

            print('#' * 100)

        model.update()


def simulate_data(feat_alloc_prior, D, N, tau_a=1, tau_x=1):
    Z = feat_alloc_prior.rvs(N)

    K = Z.shape[1]

    V = scipy.stats.matrix_normal.rvs(
        mean=np.zeros((K, D)),
        rowcov=(1 / tau_a) * np.eye(K),
        colcov=np.eye(D)
    )

    data = scipy.stats.matrix_normal.rvs(
        mean=np.dot(Z, V),
        rowcov=(1 / tau_x) * np.eye(N),
        colcov=np.eye(D)
    )

    return data, Parameters(tau_a, tau_x, V, Z)


if __name__ == '__main__':
    main()
