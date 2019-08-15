import numba
import numpy as np
import pandas as pd

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.pyclone
import pgfa.updates

from pgfa.utils import Timer


def main():
    seed = 2

    set_seed(seed)

    ibp = True
    time = 100000
    K = 6
    updater = 'g'
    num_split_merge_updates = 0
    
    if not ibp:
        num_split_merge_updates = 0
    
    data, Z_true, mutations, samples = load_data()

    model_updater = get_model_updater(
        feat_alloc_updater_type=updater, ibp=ibp, mixed_updates=False
    )
    
    sm_updater = pgfa.models.pyclone.SplitMergeUpdater()

    model = get_model(data, ibp=ibp, K=K)

    print('@' * 100)

    timer = Timer()

    i = 0

    while timer.elapsed < time:
        if i % 1 == 0:
            print(
                i,
                model.params.K,
                model.params.alpha,
                model.params.precision,
                model.log_p,
            )
            
            Z = np.column_stack([model.params.Z[d] for d, m in enumerate(mutations) if m in Z_true.columns])

            print(
                get_b_cubed_score(Z_true.values, Z)
            )

            print(
                model.data_dist.log_p(model.data, model.params),
                model.feat_alloc_dist.log_p(model.params),
                model.params_dist.log_p(model.params)
            )
            
            print(model.params.Z.sum(axis=0))
            
            print(samples)
            
            print(np.around(model.params.F, 3)[np.sum(model.params.Z, axis=0) > 0])

            print('#' * 100)

        timer.start()

        model_updater.update(model)
        
        for _ in range(num_split_merge_updates):
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


def load_data():
    
    file_name = 'data/pyclone/ith_sc_genotypes.tsv'
    
    geno_df = pd.read_csv(file_name, sep='\t')
    
    file_name = 'data/pyclone/ith_sc_prev.tsv'
    
    prev_df = pd.read_csv(file_name, sep='\t')
    
    prev_df = prev_df.groupby('cluster').filter(lambda x: x['mean_posterior_prevalence'].max() >= 0.1)

    clusters = prev_df['cluster'].unique()
    
    geno_df = geno_df.set_index('cluster').loc[clusters].reset_index()
    
    geno_df = geno_df.drop('cluster', axis=1)
    
    geno_df = (geno_df > 0).astype(int)
    
    file_name = 'data/pyclone/ith_tumour_content.tsv' 
    
    tc = pd.read_csv(file_name, sep='\t')
    
    tc = tc.set_index('sample_id')['tumour_content']
    
    file_name = 'data/pyclone/ith.tsv'
    
    df = pd.read_csv(file_name, sep='\t')
    
    data = []
    
    mutations = []
    
    samples = sorted(df['sample_id'].unique())
    
    for x, mut_df in df.groupby('mutation_id'):
        mutations.append(x)
        
        sample_data_points = []
        
        mut_df = mut_df.set_index('sample_id')
        
        for sample in samples:
            row = mut_df.loc[sample]
         
            sample_data_points.append(
                pgfa.models.pyclone.get_sample_data_point(
                    row['a'], row['b'], row['cn_major'], row['cn_minor'], tumour_content=tc.loc[sample]
                )
            )
        
        data.append(pgfa.models.pyclone.DataPoint(sample_data_points))
    
    geno_df = geno_df[[x for x in mutations if x in geno_df.columns]]
    
    geno_df = geno_df[geno_df.sum(axis=1) > 0]
    
    print('Genotypes dim: {}'.format(geno_df.shape))
    
    return data, geno_df, mutations, samples


if __name__ == '__main__':
    main()

