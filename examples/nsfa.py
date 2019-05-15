import numba
import numpy as np
import scipy.stats

from pgfa.math_utils import bernoulli_rvs
from pgfa.utils import get_b_cubed_score, Timer

import pgfa.feature_allocation_distributions
import pgfa.updates


def main():
    annealing_power = 1.0
    num_particles = 20
    
    data_seed = 0
    param_seed = 0
    run_seed = 1
    updater = 'dpf'

    ibp = False
    time = np.inf
    K = 5
    N = 200

    set_seed(data_seed)

    data, data_true, params = simulate_data(N, K, alpha=1, prop_missing=0.1, tau=0.1)

    model_updater = get_model_updater(
        annealing_power=annealing_power, feat_alloc_updater_type=updater, ibp=ibp, mixed_updates=ibp, num_particles=num_particles
    )

    set_seed(param_seed)

    model = get_model(data, ibp=ibp, K=K)

    old_params = model.params.copy()

    model.params = params.copy()

    log_p_true = model.log_p

    print(np.sum(params.Z, axis=0))
    print(log_p_true)
    print(compute_error(data_true, model))

    model.params = old_params.copy()

    print('@' * 100)
    
    set_seed(run_seed)

    timer = Timer()

    i = 0

    while timer.elapsed < time:
        if i % 10 == 0:
            print(
                i,
                model.params.K,
                model.log_p,
                (model.log_p - log_p_true) / abs(log_p_true),
                compute_error(data_true, model),
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


def get_model_updater(feat_alloc_updater_type='g', annealing_power=0, ibp=True, mixed_updates=False, num_particles=20):
    if ibp:
        singletons_updater = pgfa.models.lfrm.PriorSingletonsUpdater()

    else:
        singletons_updater = None

    if feat_alloc_updater_type == 'dpf':
        feat_alloc_updater = pgfa.updates.DicreteParticleFilterUpdater(
            annealing_power=annealing_power, max_particles=num_particles, singletons_updater=singletons_updater
        )

    elif feat_alloc_updater_type == 'g':
        feat_alloc_updater = pgfa.updates.GibbsUpdater(singletons_updater=singletons_updater)
        
    elif feat_alloc_updater_type == 'gpf':
        feat_alloc_updater = pgfa.updates.GumbelParticleFilterUpdater(
            annealing_power=annealing_power, max_particles=num_particles
            )

    elif feat_alloc_updater_type == 'pg':
        feat_alloc_updater = pgfa.updates.ParticleGibbsUpdater(
            annealing_power=annealing_power, num_particles=num_particles, singletons_updater=singletons_updater
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
    
    data, data_true = sim_data(N, V, Z.astype(np.float64), prop_missing)

    return data, data_true, params


@numba.njit(cache=True)
def sim_data(N, V, Z, prop_missing=0):
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
    
    return data, data_true


def compute_error(data_true, model):
    idxs = np.isnan(model.data)
    
    if not np.any(idxs):
        return 0
    
    return np.sum(np.abs(model.predict(method='max')[idxs] - data_true[idxs]))


if __name__ == '__main__':
    main()
