import numba
import numpy as np
import scipy.stats

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.mutation_signatures
import pgfa.updates

from pgfa.utils import Timer


def main():
    annealing_power = 1.0
    num_particles = 20
    
    data_seed = 0
    param_seed = 1
    run_seed = 0
    updater = 'g'

    ibp = False
    time = np.inf
    K = 5
    N = 20
    D = 100

    set_seed(data_seed)

    data, params = simulate_data(N, D=D, K=K, alpha=2)

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

    model.params = old_params.copy()
    
    model.params.S = params.S.copy()
    
    model.params.V = params.V.copy()

    print('@' * 100)
    
    set_seed(run_seed)

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

            print(get_b_cubed_score(params.Z, model.params.Z))

            print(
                model.data_dist.log_p(model.data, model.params),
                model.feat_alloc_dist.log_p(model.params),
                model.params_dist.log_p(model.params)
            )
            
            print(np.sum(model.params.Z, axis=0))

            print('#' * 100)

        timer.start()

        model_updater.update(model, param_updates=0)

        timer.stop()

        i += 1


def get_model(data, ibp=False, K=None):
    if ibp:
        feat_alloc_dist = pgfa.feature_allocation_distributions.IndianBuffetProcessDistribution()
    
    else:
        feat_alloc_dist = pgfa.feature_allocation_distributions.BetaBernoulliFeatureAllocationDistribution(K)

    return pgfa.models.mutation_signatures.Model(data, feat_alloc_dist)


def get_model_updater(feat_alloc_updater_type='g', annealing_power=0, ibp=True, mixed_updates=False, num_particles=20):
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

    return pgfa.models.mutation_signatures.ModelUpdater(feat_alloc_updater)


def set_seed(seed):
    np.random.seed(seed)

    set_numba_seed(seed)


@numba.jit
def set_numba_seed(seed):
    np.random.seed(seed)


def simulate_data(N, D=1, K=None, alpha=1):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K=K)

    Z = feat_alloc_dist.rvs(alpha, N)
    
    Z[:, 0] = 1

    K = Z.shape[1]
        
    Z = np.random.randint(0, 2, size=(N, K))

    S = scipy.stats.dirichlet.rvs(np.ones(D), size=K)

    V = scipy.stats.gamma.rvs(1, 1, size=(N, K))
    
    params = pgfa.models.mutation_signatures.Parameters(alpha, np.ones(2), S, np.ones(D), V, np.ones(2), Z)
    
    pi = params.pi

    data = np.zeros((N, D))
    
    for n in range(N):
        num_mutations = np.random.poisson(1000)
        
        data[n] = scipy.stats.multinomial.rvs(num_mutations, pi[n])

    return data, params


if __name__ == '__main__':
    main()
