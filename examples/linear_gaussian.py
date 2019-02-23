import numba
import numpy as np
import scipy.stats

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.linear_gaussian
import pgfa.updates

from pgfa.utils import Timer


def main():
    data_seed = 2
    param_seed = 0
    updater = 'pga'

    set_seed(data_seed)

    ibp = False
    time = 1000
    D = 10
    K = 20
    N = 100

    data, data_true, params = simulate_data(D, N, K=K, alpha=2, prop_missing=0.01, tau_v=0.25, tau_x=25)

    for d in range(D):
        assert not np.all(np.isnan(data[:, d]))

    for n in range(N):
        assert not np.all(np.isnan(data[n]))

    model_updater = get_model_updater(
        feat_alloc_updater_type=updater, ibp=ibp, mixed_updates=False
    )

    set_seed(param_seed)

    model = get_model(data, ibp=ibp, K=K)

    print(np.sum(params.Z, axis=0))

    print('@' * 100)

    old_params = model.params.copy()

    model.params = params.copy()

    log_p_true = model.log_p

    model.params = old_params.copy()

    model.params.alpha = 1

    print(log_p_true)

    set_seed(0)

    timer = Timer()

    i = 0

    while timer.elapsed < time:
        if i % 10 == 0:
            print(
                i,
                model.params.K,
                model.log_p,
                (model.log_p - log_p_true) / abs(log_p_true),
                compute_l2_error(data, data_true, model.params)
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

    return pgfa.models.linear_gaussian.Model(data, feat_alloc_dist)


def get_model_updater(feat_alloc_updater_type='g', ibp=True, mixed_updates=False):
    if ibp:
        singletons_updater = pgfa.models.linear_gaussian.PriorSingletonsUpdater()

    else:
        singletons_updater = None

    if feat_alloc_updater_type == 'dpf':
        feat_alloc_updater = pgfa.updates.DicreteParticleFilterUpdater(
            annealed=False, max_particles=20, singletons_updater=singletons_updater
        )

    elif feat_alloc_updater_type == 'dpfa':
        feat_alloc_updater = pgfa.updates.DicreteParticleFilterUpdater(
            annealed=True, max_particles=20, singletons_updater=singletons_updater
        )

    elif feat_alloc_updater_type == 'g':
        feat_alloc_updater = pgfa.updates.GibbsUpdater(singletons_updater=singletons_updater)

    elif feat_alloc_updater_type == 'pg':
        feat_alloc_updater = pgfa.updates.ParticleGibbsUpdater(
            annealed=False, num_particles=20, singletons_updater=singletons_updater
        )

    elif feat_alloc_updater_type == 'pga':
        feat_alloc_updater = pgfa.updates.ParticleGibbsUpdater(
            annealed=True, num_particles=20, singletons_updater=singletons_updater
        )

    elif feat_alloc_updater_type == 'pga-a':
        feat_alloc_updater = pgfa.updates.ParticleGibbsUpdater(
            ancestor_resample_prob=1.0, annealed=True, num_particles=20, resample_threshold=0.5, singletons_updater=singletons_updater
        )

    elif feat_alloc_updater_type == 'rg':
        feat_alloc_updater = pgfa.updates.RowGibbsUpdater(singletons_updater=singletons_updater)

    if mixed_updates:
        feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(feat_alloc_updater)

    return pgfa.models.linear_gaussian.ModelUpdater(feat_alloc_updater)


def simulate_data(D, N, K=None, alpha=1, prop_missing=0, tau_v=1, tau_x=1):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K=K)

    Z = feat_alloc_dist.rvs(alpha, N)

    K = Z.shape[1]

    V = scipy.stats.matrix_normal.rvs(
        mean=np.zeros((K, D)),
        rowcov=(1 / tau_v) * np.eye(K),
        colcov=np.eye(D)
    )

    data_true = scipy.stats.matrix_normal.rvs(
        mean=Z @ V,
        rowcov=(1 / tau_x) * np.eye(N),
        colcov=np.eye(D)
    )

    mask = np.random.uniform(0, 1, size=data_true.shape) <= prop_missing

    data = data_true.copy()

    data[mask] = np.nan

    params = pgfa.models.linear_gaussian.Parameters(alpha, np.ones(2), tau_v, np.ones(2), tau_x, np.ones(2), V, Z)

    return data, data_true, params


def compute_l2_error(data, data_true, params):
    idxs = np.isnan(data)

    data_pred = params.Z @ params.V

    return np.sqrt(np.mean(np.square(data_pred[idxs] - data_true[idxs])))


if __name__ == '__main__':
#     import line_profiler
#  
#     profiler = line_profiler.LineProfiler()
#  
#     profiler.add_function(pgfa.updates.DicreteParticleFilterUpdater.update_row)
#      
#     profiler.add_function(pgfa.updates.DicreteParticleFilterUpdater._get_new_particle)
#     
#     profiler.add_function(pgfa.updates.discrete_particle_filter.DicreteParticleFilterUpdater._resample)
#  
#     profiler.run('main()')
#      
#     profiler.print_stats()

    main()
