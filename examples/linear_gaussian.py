import numba
import numpy as np

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.linear_gaussian
import pgfa.updates

from pgfa.utils import Timer

np.seterr(all='warn')


def main():
    print_freq = 10

    annealing_power = 1.0
    num_particles = 20

    data_seed = 2
    param_seed = 2
    run_seed = 1
    updater = 'dpf'
    test_path = 'two-stage'

    set_seed(data_seed)

    ibp = False
    time = np.inf
    D = 10
    K = 20
    N = 200

    params = pgfa.models.linear_gaussian.simulate_params(alpha=2, tau_v=0.25, tau_x=25, D=D, K=K, N=N)

    data, data_true = pgfa.models.linear_gaussian.simulate_data(params, prop_missing=0.1)

    for d in range(D):
        assert not np.all(np.isnan(data[:, d]))

    for n in range(N):
        assert not np.all(np.isnan(data[n]))

    model_updater = get_model_updater(
        annealing_power=annealing_power,
        feat_alloc_updater_type=updater,
        ibp=ibp,
        mixture_prob=0.5,
        num_particles=num_particles,
        test_path=test_path
    )

    set_seed(param_seed)

    model = get_model(data, ibp=ibp, K=K)

    print(sorted(np.sum(params.Z, axis=0)))

    old_params = model.params.copy()

    model.params = params.copy()

    log_p_true = model.log_p

    model.params = old_params.copy()

    print(log_p_true)

    set_seed(run_seed)

    print('@' * 100)

    timer = Timer()

    i = 0

    last_print_time = -np.float('inf')

    while timer.elapsed < time:
        if (timer.elapsed - last_print_time) > print_freq:
            last_print_time = timer.elapsed

            print(
                i,
                model.log_p,
                model.params.K,
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

            print(sorted(np.sum(model.params.Z, axis=0)))

            print(model.params.alpha, model.params.tau_v, model.params.tau_x)

            Vs = np.abs(model.params.V).min(axis=1)

            Zs = np.sum(model.params.Z, axis=0)

            idxs = np.argsort(np.sum(model.params.Z, axis=0))

            print([(Zs[i], Vs[i]) for i in idxs])

            print(Vs.min())

            print('#' * 100)

        timer.start()

        model_updater.update(model)  # , alpha_updates=0)#, param_updates=0)

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


def get_model_updater(
        feat_alloc_updater_type='g',
        annealing_power=0,
        ibp=True,
        mixture_prob=0.0,
        num_particles=20,
        test_path='conditional'):

    if ibp:
        singletons_updater = pgfa.models.linear_gaussian.PriorSingletonsUpdater()

    else:
        singletons_updater = None

    if feat_alloc_updater_type == 'dpf':
        feat_alloc_updater = pgfa.updates.DiscreteParticleFilterUpdater(
            annealing_power=annealing_power,
            conditional_update=True,
            max_particles=num_particles,
            singletons_updater=singletons_updater,
            test_path=test_path
        )

    elif feat_alloc_updater_type == 'g':
        feat_alloc_updater = pgfa.updates.GibbsUpdater(singletons_updater=singletons_updater)

    elif feat_alloc_updater_type == 'gpf':
        feat_alloc_updater = pgfa.updates.GumbelParticleFilterUpdater(
            annealing_power=annealing_power, max_particles=num_particles
            )

    elif feat_alloc_updater_type == 'pg':
        feat_alloc_updater = pgfa.updates.ParticleGibbsUpdater(
            annealing_power,
            conditional_update=True,
            num_particles=num_particles,
            resample_scheme='stratified',
            resample_threshold=0.5,
            singletons_updater=singletons_updater,
            test_path=test_path
        )

    elif feat_alloc_updater_type == 'rg':
        feat_alloc_updater = pgfa.updates.RowGibbsUpdater(singletons_updater=singletons_updater)

    if mixture_prob > 0:
        feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(feat_alloc_updater, gibbs_prob=mixture_prob)

    return pgfa.models.linear_gaussian.ModelUpdater(feat_alloc_updater)


def compute_l2_error(data, data_true, params):
    idxs = np.isnan(data)

    if not np.all(idxs):
        return 0

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
