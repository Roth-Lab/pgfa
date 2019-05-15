import numba
import numpy as np
import pandas as pd
import scipy.stats
import scipy.spatial.distance

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.pyclone
import pgfa.updates

from pgfa.utils import Timer


def main(num_iters=100000):
    seed = 0

    set_seed(seed)

    ibp = False
    time = 100000
    K = 10
    
    data, Z_true, mutations = load_data()
    
#     idxs = np.random.choice(len(data), 50, replace=False)
#      
#     data = [data[i] for i in idxs]
#      
#     Z_true = Z_true[idxs]
     
    model_updater = get_model_updater(
        feat_alloc_updater_type='dpf', ibp=ibp, mixed_updates=False
    )

    model = get_model(data, ibp=ibp, K=K)
    
    model.params.alpha = 1e-3
    
#     model.params.V_prior = np.array([1e4, 1])

    annealing_ladder = np.array([1.0])
    
#     annealing_ladder = np.linspace(0, 1, 20)
    
    annealing_idx = 0
    
    sm_updater = pgfa.models.pyclone.SplitMergeUpdater()

    print('@' * 100)

    timer = Timer()

    i = 0

    while timer.elapsed < time:
        if i % 20 == 0:
            if annealing_idx < len(annealing_ladder):
                model.data_dist.annealing_power = annealing_ladder[annealing_idx]
                
#                 sm_updater.annealing_factor = annealing_ladder[annealing_idx]
                
                print('Setting annealing power to: {}'.format(annealing_ladder[annealing_idx]))
                
                annealing_idx += 1
            
        if i % 1 == 0:
            print(
                i,
                model.params.K,
                model.params.alpha,
                model.params.precision,
                model.log_p,
            )

            print(
                get_b_cubed_score(Z_true, model.params.Z)
            )

            print(
                model.data_dist.log_p(model.data, model.params),
                model.feat_alloc_dist.log_p(model.params),
                model.params_dist.log_p(model.params)
            )
            
            print(model.params.Z.sum(axis=0))
            
            print(np.around(model.params.F, 3)[np.sum(model.params.Z, axis=0) > 0])

            print('#' * 100)

        timer.start()

        model_updater.update(model, alpha_updates=0)
        
#         if ibp:
#             for _ in range(1):
#                 sm_updater.update(model)

        timer.stop()

        i += 1
        
        if i >= num_iters:
            break


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


def load_data():
    file_name = 'data/pyclone/mixing_full.tsv'
    
    df = pd.read_csv(file_name, sep='\t')
    
    Z = []
    
    cases = set()
    
    for x in df['variant_cases'].unique():
        cases.update(set(x.split(',')))
    
    cases = sorted(cases)
    
    data = []
    
    mutations = []
    
    for x, mut_df in df.groupby('mutation_id'):
        mutations.append(x)
        
        mut_df = mut_df.set_index('sample_id')
        
        data.append(
            pgfa.models.pyclone.get_data_point(
                mut_df['a'], mut_df['b'], mut_df['cn_major'].iloc[0], mut_df['cn_minor'].iloc[0])
        )
        
        Z_mut = np.zeros(4)
        
        for case in mut_df['variant_cases'].iloc[0].split(','):
            Z_mut[cases.index(case)] = 1
        
        Z.append(Z_mut)
    
    Z = np.array(Z, dtype=np.int8)
    
    return data, Z, mutations


if __name__ == '__main__':
    main(num_iters=np.inf)
    
#     import line_profiler
#   
#     profiler = line_profiler.LineProfiler()
#   
#     profiler.add_function(pgfa.models.pyclone.ModelUpdater.update)
#     
#     profiler.add_function(pgfa.models.pyclone.DataDistribution._log_p)
#   
#     profiler.run('main(num_iters=100)')
#       
#     profiler.print_stats()
