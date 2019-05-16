import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import bernoulli_rvs
from pgfa.utils import get_b_cubed_score, Timer

import pgfa.feature_allocation_distributions
import pgfa.models.lfrm
import pgfa.updates


def main():
    seed = 2

    set_seed(seed)

    ibp = False
    time = 1000
    updater = 'rg'
    updater_mixed = False
    N = 100
    K = 5

    data, data_true, params = simulate_data(N, K, alpha=2, prop_missing=0, tau=0.1)

    model_updater = get_model_updater(
        feat_alloc_updater_type=updater, ibp=ibp, mixed_updates=updater_mixed
    )

    model = get_model(data, ibp=ibp, K=K)

    print(np.sum(params.Z, axis=0))

    print('@' * 100)

    old_params = model.params.copy()

    model.params = params.copy()

    log_p_true = model.log_p

    model.params = old_params.copy()

    print(log_p_true)

    timer = Timer()

    i = 0

    while timer.elapsed < time:
        if i % 10 == 0:
            print(
                i,
                model.params.K,
                model.log_p,
                (model.log_p - log_p_true) / abs(log_p_true),
                np.sum(np.abs(model.predict(method='max') - data_true)),
                model.params.tau
            )

            print(get_b_cubed_score(params.Z, model.params.Z))

            print(
                model.data_dist.log_p(model.data, model.params),
                model.feat_alloc_dist.log_p(model.params),
                model.params_dist.log_p(model.params)
            )

            print('#' * 100)

        timer.start()

        model_updater.update(model)

        timer.stop()

        i += 1


def get_model(data, ibp=False, K=None):
    if ibp:
        feat_alloc_dist = pgfa.feature_allocation_distributions.IndianBuffetProcessDistribution()

    else:
        feat_alloc_dist = pgfa.feature_allocation_distributions.BetaBernoulliFeatureAllocationDistribution(K)

    return pgfa.models.lfrm.Model(data, feat_alloc_dist, symmetric=False)


def get_model_updater(feat_alloc_updater_type='g', ibp=True, mixed_updates=False):
    if ibp:
        singletons_updater = pgfa.models.nsfa.CollapsedSingletonsUpdater()

    else:
        singletons_updater = None

    if feat_alloc_updater_type == 'g':
        feat_alloc_updater = pgfa.updates.GibbsUpdater(singletons_updater=singletons_updater)

    elif feat_alloc_updater_type == 'pg':
        feat_alloc_updater = pgfa.updates.ParticleGibbsUpdater(
            annealed=False, num_particles=20, singletons_updater=singletons_updater
        )

    elif feat_alloc_updater_type == 'pga':
        feat_alloc_updater = pgfa.updates.ParticleGibbsUpdater(
            annealed=True, num_particles=20, singletons_updater=singletons_updater
        )

    elif feat_alloc_updater_type == 'rg':
        feat_alloc_updater = pgfa.updates.RowGibbsUpdater(singletons_updater=singletons_updater)

    if mixed_updates:
        feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(feat_alloc_updater)

    return pgfa.models.lfrm.ModelUpdater(feat_alloc_updater)


def set_seed(seed):
    np.random.seed(seed)

    set_numba_seed(seed)


@numba.jit
def set_numba_seed(seed):
    np.random.seed(seed)


def simulate_data(N, K=None, alpha=1, prop_missing=0, tau=1):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K=K)

    Z = feat_alloc_dist.rvs(alpha, N)

    K = Z.shape[1]

    V = scipy.stats.norm.rvs(0, 1 / np.sqrt(tau), size=(K, K))

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
            if np.random.random() < prop_missing:
                data[i, j] = np.nan

    return data, data_true, params


if __name__ == '__main__':
    main()
