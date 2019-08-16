import numba
import numpy as np

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.linear_gaussian
import pgfa.updates

from pgfa.utils import Timer

np.seterr(all='warn')


def main():
    annealing_power = 1.0
    num_particles = 20
    
    data_seed = 1
    param_seed = 1
    run_seed = 0
    updater = 'dpf'

    set_seed(data_seed)

    ibp = False
    time = 10
    D = 10
    K = 20
    N = 100
    
    params = pgfa.models.linear_gaussian.simulate_params(alpha=1, tau_v=0.25, tau_x=25, D=D, K=K, N=N)
    
    data, data_true = pgfa.models.linear_gaussian.simulate_data(params, prop_missing=0)

    for d in range(D):
        assert not np.all(np.isnan(data[:, d]))

    for n in range(N):
        assert not np.all(np.isnan(data[n]))

    model_updater = get_model_updater(
        annealing_power=annealing_power, feat_alloc_updater_type=updater, ibp=ibp, mixed_updates=False, num_particles=num_particles
    )

    set_seed(param_seed)

    model = get_model(data, ibp=ibp, K=K)

    print(sorted(np.sum(params.Z, axis=0)))

    print('@' * 100)

    old_params = model.params.copy()

    model.params = params.copy()

    log_p_true = model.log_p
    
#     model.params.Z = old_params.Z

    model.params = old_params.copy()
    
#     model.params.Z = params.Z.copy()
# 
#     model.params.alpha = params.alpha

    print(log_p_true)

    set_seed(run_seed)

    timer = Timer()

    i = 0
    
    if updater == 'g':
        thin = 10
    
    else:
        thin = 1

    while timer.elapsed < time:
        if i % thin == 0:
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


def get_model_updater(feat_alloc_updater_type='g', annealing_power=0, ibp=True, mixed_updates=False, num_particles=20):
    if ibp:
        singletons_updater = pgfa.models.linear_gaussian.PriorSingletonsUpdater()

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
