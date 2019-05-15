import numba
import numpy as np
import scipy.stats

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.pyclone
import pgfa.updates

from pgfa.utils import Timer


def main():
    seed = 0

    set_seed(seed)

    ibp = True
    time = 10000
    D = 6
    K = 4
    N = 100

    data, params = simulate_data(D, N, K=K, alpha=2)

    model_updater = get_model_updater(
        feat_alloc_updater_type='g', ibp=ibp, mixed_updates=False
    )

    sm_updater = pgfa.models.pyclone.SplitMergeUpdater()

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
        
        if ibp:
            for _ in range(20):
                sm_updater.update(model)
        
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

    return pgfa.models.pyclone.Model(data, feat_alloc_dist)


def get_model_updater(feat_alloc_updater_type='g', annealing_power=0, ibp=True, mixed_updates=False, num_particles=20):
    if ibp:
        singletons_updater = pgfa.models.pyclone.PriorSingletonsUpdater()

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

    return pgfa.models.pyclone.ModelUpdater(feat_alloc_updater)


def simulate_data(D, N, K=None, alpha=1, eps=1e-3):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K=K)

    Z = feat_alloc_dist.rvs(alpha, N)

    K = Z.shape[1]
    
    V = scipy.stats.gamma.rvs(1, 1, size=(K, D))
    
    params = pgfa.models.pyclone.Parameters(alpha, np.ones(2), 100, np.array([1e-2, 1e-2]), V, np.ones(2), Z)
    
    F = params.F

    data = []
    
    cn_n = 2
    
    cn_r = 2
    
    mu_n = eps
    
    mu_r = eps
    
    t = np.ones(D)
    
    for n in range(N):
        phi = Z[n] @ F
        
        cn_total = 2  # scipy.stats.poisson.rvs(1) + 1
        
        cn_major = scipy.stats.randint.rvs(1, cn_total + 1)
        
        cn_minor = cn_total - cn_major
        
        cn_var = scipy.stats.randint.rvs(1, cn_major + 1)
        
        a = []
        
        b = []
         
        for s in range(D):
            mu_v = min(cn_var / cn_total, 1 - eps)
            
            xi = (1 - t[s]) * phi[s] * cn_n * mu_n + t[s] * (1 - phi[s]) * cn_r * mu_r + t[s] * phi[s] * cn_total * mu_v
            
            xi /= (1 - t[s]) * phi[s] * cn_n + t[s] * (1 - phi[s]) * cn_r + t[s] * phi[s] * cn_total
            
            d = scipy.stats.poisson.rvs(1000)
            
            b.append(scipy.stats.binom.rvs(d, xi))
            
            a.append(d - b[-1])
      
        data.append(
            pgfa.models.pyclone.get_data_point(a, b, cn_major, cn_minor)
        )

    return data, params


if __name__ == '__main__':
    main()
