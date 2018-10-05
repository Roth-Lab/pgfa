import numpy as np

from pgfa.feature_allocation_priors import BetaBernoulliFeatureAllocationDistribution
from pgfa.math_utils import ibp_rvs, ffa_rvs
from pgfa.models.pyclone import Parameters, PyCloneFeatureAllocationModel
from pgfa.updates.feature_matrix import GibbsMixtureUpdater, GibbsUpdater, MetropolisHastingsUpdater, ParticleGibbsUpdater, RowGibbsUpdater
from pgfa.utils import get_b_cubed_score


def main():
    np.random.seed(0)

    D = 4
    K = 5
    N = 100

    data, params = simulate_data(D, N, alpha=1, K=K)

    feat_alloc_prior = BetaBernoulliFeatureAllocationDistribution(1.0, 1.0, K)

    mh_updater = MetropolisHastingsUpdater(adaptation_rate=100, flip_prob=0.5)

    feat_alloc_updater = mh_updater

#     feat_alloc_updater = GibbsMixtureUpdater(mh_updater)

#     feat_alloc_updater = GibbsMixtureUpdater(RowGibbsUpdater(max_cols=5))

    feat_alloc_updater = GibbsMixtureUpdater(ParticleGibbsUpdater(annealed=True, num_particles=10))

#     feat_alloc_updater = GibbsUpdater()

#     feat_alloc_updater = ParticleGibbsUpdater(annealed=True, num_particles=10)

#     feat_alloc_updater = RowGibbsUpdater(max_cols=5)

    model = PyCloneFeatureAllocationModel(data, feat_alloc_prior, feat_alloc_updater)

    model.params.eta = params.eta.copy()

    model.params.Z = params.Z.copy()

#     model.params.Z = np.random.randint(0, 2, size=params.Z.shape)

    model.params.Z = np.zeros(params.Z.shape, dtype=np.int64)

    model.params.Z[:, 0] = 1

    print(feat_alloc_updater)

    print(np.sum(params.Z, axis=0))

    for i in range(100000):
        if i % 100 == 0:
            print(i, model.params.K, model.params.alpha, model.log_p)
            print(np.sum(model.params.Z, axis=0))
            print(get_b_cubed_score(params.Z, model.params.Z))
#             if i > 0:
#                 print(mh_updater.accept_rate, mh_updater.flip_prob)
            print('#' * 100)

        model.update()


def simulate_data(D, N, alpha=1, kappa=10, K=None):
    if K is None:
        Z = ibp_rvs(alpha, N)

    else:
        Z = ffa_rvs(alpha / K, 1, K, N)

    K = Z.shape[1]

    eta = np.random.gamma(kappa, 1, size=(D, K))

    phi = eta / np.sum(eta, axis=1)[:, np.newaxis]

    data = np.zeros((N, D, 2))

    for d in range(D):
        for n in range(N):
            data[n, d, 0] = np.random.poisson(10000)

            f = np.sum(Z[n] * phi[d]) / np.sum(phi[d])

            data[n, d, 1] = np.random.binomial(data[n, d, 0], f)

    params = Parameters(alpha, kappa, eta, Z)

    return data, params


if __name__ == '__main__':
    main()
