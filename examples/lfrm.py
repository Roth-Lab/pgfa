import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import bernoulli_rvs
from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.lfrm
import pgfa.updates


def main():
    seed = 1
    np.random.seed(seed)
    set_numba_seed(seed)

    ibp = True
    pg = False
    num_iters = int(1e5)

    K = 5
    N = 20

    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K)

    data, data_true, params = simulate_data(feat_alloc_dist, N, alpha=K, tau=0.1)

    if ibp:
        singletons_updater = pgfa.models.lfrm.PriorSingletonsUpdater()

        feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(None)

    else:
        singletons_updater = None

        feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K)

    if pg:
        feat_alloc_updater = pgfa.updates.ParticleGibbsUpdater(
            annealed=True, singletons_updater=singletons_updater
        )

        feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(feat_alloc_updater)

    else:
        feat_alloc_updater = pgfa.updates.GibbsUpdater(singletons_updater=singletons_updater)

    model_updater = pgfa.models.lfrm.ModelUpdater(feat_alloc_updater)

    model = pgfa.models.lfrm.Model(data, feat_alloc_dist, symmetric=False)

    old_params = model.params.copy()

    model.params = params.copy()

    true_log_p = model.log_p

    model.params = old_params.copy()

#     model.params = params.copy()

    print(np.sum(params.Z, axis=0))

    print('@' * 100)

    for i in range(num_iters):
        if i % 10 == 0:
            print(
                i,
                model.params.K,
                model.params.alpha,
                model.log_p,
                (model.log_p - true_log_p) / abs(true_log_p),
                np.sum(np.abs(model.predict(method='prob') - data_true)),
                model.params.tau
            )

            if model.params.K > 0:
                print(get_b_cubed_score(params.Z, model.params.Z))

            print(np.sum(model.params.Z, axis=0))

#             print('x' * 100)
#
#             print(
#                 model.data_dist.log_p(model.data, model.params),
#                 model.feat_alloc_dist.log_p(model.params),
#                 model.params_dist.log_p(model.params)
#             )

            print('#' * 100)

        model_updater.update(model, alpha_updates=1, param_updates=1)


def simulate_data(feat_alloc_dist, N, alpha=1, tau=0.1):
    Z = feat_alloc_dist.rvs(alpha, N)

    K = Z.shape[1]

    V = np.random.normal(0, 1 / np.sqrt(tau), size=(K, K))
#     V = np.triu(V)
#     V = V + V.T - np.diag(np.diag(V))

    params = pgfa.models.lfrm.Parameters(alpha, np.ones(2), tau, np.ones(2), V, Z)

    data_true = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            m = Z[i].T @ V @ Z[j]

            f = np.exp(-m)

            p = 1 / (1 + f)

            data_true[i, j] = bernoulli_rvs(p)

    data = data_true.copy()

    for i in range(N):
        for j in range(N):
            if np.random.random() < -0.1:
                data[i, j] = np.nan

    return data, data_true, params


@numba.jit
def set_numba_seed(seed):
    np.random.seed(seed)


if __name__ == '__main__':
    main()
