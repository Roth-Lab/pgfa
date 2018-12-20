import numba
import numpy as np
import scipy.stats

from pgfa.utils import get_b_cubed_score, summarize_feature_allocation_matrix

import pgfa.feature_allocation_distributions
import pgfa.models.cn_poisson
import pgfa.updates

from pgfa.utils import Timer


def main():
    seed = 1

    np.random.seed(seed)

    set_seed(seed)
    
    ibp = False
    time = 100
    K = 4
    N = 100

    data, params = simulate_data(N, D=10, K=K, alpha=2 * K)

    model_updater = get_model_updater(
        feat_alloc_updater_type='rg', ibp=ibp, mixed_updates=False
    )

    model = get_model(data, ibp=ibp, K=K)

    print(np.sum(params.Z, axis=0))

    print('@' * 100)

    timer = Timer()

    i = 0

    old_params = model.params.copy()

    model.params = params.copy()

    true_log_p = model.log_p

    model.params = old_params.copy()

    print(true_log_p)
    
    Zs = []

    while timer.elapsed < time:
        if i % 10 == 0:
            print(
                i,
                model.params.K,
                model.params.alpha,
                model.log_p,
                (model.log_p - true_log_p) / abs(true_log_p)
            )
            
            print(
                get_b_cubed_score(params.Z, model.params.Z)
            )

            print(
                model.data_dist.log_p(model.data, model.params),
                model.feat_alloc_dist.log_p(model.params),
                model.params_dist.log_p(model.params)
            )
            
#             print(model.params.nu)
#             
#             print(model.params.mu)

            print('#' * 100)

        timer.start()

        model_updater.update(model)

        timer.stop()

        i += 1
        
        Zs.append(model.params.Z)
    
    Zs = np.array(Zs)
    
    Z_max = summarize_feature_allocation_matrix(Zs, burnin=100, thin=10)
    
    print(get_b_cubed_score(params.Z, Z_max))


@numba.jit
def set_seed(seed):
    np.random.seed(seed)


def get_model(data, ibp=False, K=None):
    if ibp:
        feat_alloc_dist = pgfa.feature_allocation_distributions.IndianBuffetProcessDistribution()
    
    else:
        feat_alloc_dist = pgfa.feature_allocation_distributions.BetaBernoulliFeatureAllocationDistribution(K)

    return pgfa.models.cn_poisson.Model(data, feat_alloc_dist)


def get_model_updater(feat_alloc_updater_type='g', ibp=True, mixed_updates=False):
    if ibp:
        singletons_updater = pgfa.models.cn_poisson.PriorSingletonsUpdater()

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

    return pgfa.models.cn_poisson.ModelUpdater(feat_alloc_updater)


def simulate_data(N, D=1, K=None, alpha=1):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K=K)

    Z = feat_alloc_dist.rvs(alpha, N)

    K = Z.shape[1]
    
    mu = scipy.stats.gamma.rvs(10, 10, size=(K, D))
    
    nu = scipy.stats.gamma.rvs(10, 10, size=(D,))

    data = np.zeros((N, D))
    
    for n in range(N):
        for d in range(D):
            mean = np.sum(mu[:, d] * Z[n]) + nu[d]
            
            data[n, d] = scipy.stats.poisson.rvs(mean)
    
    params = pgfa.models.cn_poisson.Parameters(alpha, np.ones(2), mu, np.ones(2), nu, np.ones(2), Z)
    
    return data, params


if __name__ == '__main__':
    main()
