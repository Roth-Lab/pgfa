import numpy as np
import os
import scipy.stats

from pgfa.feature_allocation_priors import BetaBernoulliFeatureAllocationDistribution, IndianBuffetProcessDistribution
from pgfa.models.linear_gaussian import LinearGaussianModel, LinearGaussianModelUpdater, Parameters, SingletonsProposal
from pgfa.updates.feature_matrix import GibbsUpdater, GibbsMixtureUpdater, MetropolisHastingsSingletonUpdater, ParticleGibbsUpdater
from pgfa.utils import get_b_cubed_score

os.environ['NUMBA_WARNINGS'] = '1'


def main():
    np.random.seed(0)

    num_iters = 10001
    D = 10
    N = 100

    feat_alloc_prior = IndianBuffetProcessDistribution()

    data, params = simulate_data(BetaBernoulliFeatureAllocationDistribution(1, 1, 4), D, N, tau_v=0.01, tau_x=10)

    singletons_proposal = SingletonsProposal()

    singletons_updater = MetropolisHastingsSingletonUpdater(singletons_proposal)

    feat_alloc_updater = ParticleGibbsUpdater(annealed=True, singletons_updater=singletons_updater)

    feat_alloc_updater = GibbsMixtureUpdater(feat_alloc_updater)

#     feat_alloc_updater = GibbsUpdater(singletons_updater=singletons_updater)

    model = LinearGaussianModel(data, feat_alloc_prior, collapsed=True)

    model_updater = LinearGaussianModelUpdater(feat_alloc_updater)

    print(model.log_p)

    print(np.sum(params.Z, axis=0))

    print('@' * 100)

    for i in range(num_iters):
        if i % 1 == 0:
            print(i, model.params.K, model.log_p)

            if model.params.K > 0:
                print(get_b_cubed_score(params.Z, model.params.Z))

            print(np.sum(model.params.Z, axis=0))

            print('#' * 100)

        model_updater.update(model)


def simulate_data(feat_alloc_prior, D, N, tau_v=1, tau_x=1):
    Z = feat_alloc_prior.rvs(N)

    K = Z.shape[1]

    V = scipy.stats.matrix_normal.rvs(
        mean=np.zeros((K, D)),
        rowcov=(1 / tau_v) * np.eye(K),
        colcov=np.eye(D)
    )

    data = scipy.stats.matrix_normal.rvs(
        mean=Z @ V,
        rowcov=(1 / tau_x) * np.eye(N),
        colcov=np.eye(D)
    )

    return data, Parameters(tau_v, tau_x, V, Z)


if __name__ == '__main__':
    main()
