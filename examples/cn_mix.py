import numba
import numpy as np
import scipy.stats

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.cn_mix
import pgfa.updates

from pgfa.utils import Timer


def main():
    seed = 0

    set_seed(seed)

    ibp = False
    time = 10000
    D = 2
    K = 2
    N = 10

    data, params = simulate_data(D, N, K=K, alpha=2)

    for d in range(D):
        assert not np.all(np.isnan(data[:, d]))

    for n in range(N):
        assert not np.all(np.isnan(data[n]))

    model_updater = get_model_updater(
        feat_alloc_updater_type='g', ibp=ibp, mixed_updates=False
    )

    model = get_model(data, ibp=ibp, K=K)

    print(np.sum(params.Z, axis=0))

    print('@' * 100)

    old_params = model.params.copy()

    model.params = params.copy()

    log_p_true = model.log_p

    model.params = old_params.copy()

    model.params.alpha = 1

    print(log_p_true)

    timer = Timer()

    i = 0

    while timer.elapsed < time:
        if i % 1 == 0:
            print(
                i,
                model.params.K,
                model.log_p,
                (model.log_p - log_p_true) / abs(log_p_true)
            )

            print(
                get_b_cubed_score(params.Z, model.params.Z)
            )

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


def set_seed(seed):
    np.random.seed(seed)

    set_numba_seed(seed)


@numba.jit
def set_numba_seed(seed):
    np.random.seed(seed)


def get_model(data, ibp=False, K=None):
    if ibp:
        feat_alloc_dist = pgfa.feature_allocation_distributions.IndianBuffetProcessDistribution()

    else:
        feat_alloc_dist = pgfa.feature_allocation_distributions.BetaBernoulliFeatureAllocationDistribution(K)

    return pgfa.models.cn_mix.Model(data, feat_alloc_dist)


def get_model_updater(feat_alloc_updater_type='g', ibp=True, mixed_updates=False):
    if ibp:
        singletons_updater = pgfa.models.linear_gaussian.PriorSingletonsUpdater()

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

    return pgfa.models.cn_mix.ModelUpdater(feat_alloc_updater)


def simulate_data(D, N, K=None, alpha=1):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K=K)

    Z = feat_alloc_dist.rvs(alpha, D)

    K = Z.shape[1]

    h = scipy.stats.gamma.rvs(10, 1, size=D)

    t = scipy.stats.uniform.rvs(0.5, 0.4, size=D)

    C = scipy.stats.randint.rvs(0, 8, size=(N, K))

    E = 2 * np.ones((N, D))

    V = scipy.stats.gamma.rvs(1, 1, size=(D, K))

    F = (Z * V) / np.sum(Z * V, axis=1)[:, np.newaxis]

    F[np.isnan(F)] = 0

    data = np.zeros((N, D))

    for n in range(N):
        for d in range(D):
            m = h[d] * (t[d] * np.sum(F[d] * C[n]) + (1 - t[d]) * E[n, d])

        data[n, d] = scipy.stats.poisson.rvs(m)

    params = pgfa.models.cn_mix.Parameters(
        alpha, np.ones(2), h, np.ones(2), t, np.ones(2), C, np.array([0, 8]), E, V, np.ones(2), Z
    )

    return data, params


if __name__ == '__main__':
    main()
