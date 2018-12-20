import numba
import numpy as np
import scipy.stats

from pgfa.utils import get_b_cubed_score, Timer

import pgfa.feature_allocation_distributions
import pgfa.models.nsfa
import pgfa.updates


def main():
    np.random.seed(0)

    ibp = False
    time = 1000
    updater = 'rg'
    updater_mixed = False
    D = 100
    K = 5
    N = 1000

    data, data_true, params = simulate_data(D, N, K=K, alpha=5, prop_missing=0, tau_v=0.25, tau_x=25)

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
        if i % 100 == 0:
            print(
                i,
                model.params.K,
                (model.log_p - log_p_true) / abs(log_p_true),
                pgfa.models.nsfa.log_held_out_pdf(data_true, np.isnan(data), model.params),
                pgfa.models.nsfa.rmse(data, model.params)
            )

            print(get_b_cubed_score(params.Z, model.params.Z))

            print(
                model.data_dist.log_p(model.data, model.params),
                model.feat_alloc_dist.log_p(model.params),
                model.params_dist.log_p(model.params)
            )

            print(np.sum(model.params.Z, axis=0))

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

    return pgfa.models.nsfa.Model(data, feat_alloc_dist)


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
            annealed=True, num_particles=5, singletons_updater=singletons_updater
        )

    elif feat_alloc_updater_type == 'rg':
        feat_alloc_updater = pgfa.updates.RowGibbsUpdater(singletons_updater=singletons_updater)

    if mixed_updates:
        feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(feat_alloc_updater)

    return pgfa.models.nsfa.ModelUpdater(feat_alloc_updater)


def set_seed(seed):
    np.random.seed(seed)

    set_numba_seed(seed)


@numba.jit
def set_numba_seed(seed):
    np.random.seed(seed)


def simulate_data(D, N, K=None, alpha=1, prop_missing=0, tau_v=1, tau_x=1):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K=K)

    Z = feat_alloc_dist.rvs(alpha, D)

    Z[:, 4:] = 0

    K = Z.shape[1]

    tau_v = tau_v * np.ones(K)

    tau_x = tau_x * np.ones(D)

    V = np.zeros((D, K))

    F = np.zeros((K, N))

    if K > 0:
        for k in range(K):
            V[:, k] = scipy.stats.multivariate_normal.rvs(np.zeros(D), (1 / tau_v[k]) * np.eye(D))

            F[k] = scipy.stats.norm.rvs(0, 1, size=N)

    params = pgfa.models.nsfa.Parameters(
        alpha, np.ones(2), tau_v, np.ones(2), tau_x, np.ones(2), F, V, Z
    )

    data = params.Y

    data += np.random.multivariate_normal(np.zeros(params.D), np.diag(1 / params.tau_x), size=params.N)

    data_true = data.copy()

    data[np.random.random(data.shape) <= prop_missing] = np.nan

    return data, data_true, params


if __name__ == '__main__':
    main()
