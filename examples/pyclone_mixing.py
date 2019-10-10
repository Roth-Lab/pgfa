import numba
import numpy as np
import pandas as pd

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.pyclone.beta_binomial
import pgfa.models.pyclone.singletons_updates
import pgfa.models.pyclone.utils
import pgfa.updates

from pgfa.utils import Timer


def main():
    seed = 9

    set_seed(seed)

    ibp = True
    time = 100000
    K = 4
    updater = 'dpf'
    test_path = 'random'
    num_split_merge_updates = 0

    if not ibp:
        num_split_merge_updates = 0

    data, Z_true, _ = load_data()

    model_updater = get_model_updater(
        annealing_power=1.0, feat_alloc_updater_type=updater, ibp=ibp, mixture_prob=0.0, test_path=test_path
    )

    sm_updater = pgfa.models.pyclone.singletons_updates.SplitMergeUpdater()

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

    return pgfa.models.pyclone.beta_binomial.Model(data, feat_alloc_dist)


def get_model_updater(
        feat_alloc_updater_type='g',
        annealing_power=0,
        ibp=True,
        mixture_prob=0.0,
        num_particles=20,
        test_path='conditional'
    ):

    if ibp:
        singletons_updater = pgfa.models.pyclone.singletons_updates.PriorSingletonsUpdater()

    else:
        singletons_updater = None

    if feat_alloc_updater_type == 'dpf':
        feat_alloc_updater = pgfa.updates.DiscreteParticleFilterUpdater(
            annealing_power=annealing_power,
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
            annealing_power=annealing_power,
            num_particles=num_particles,
            singletons_updater=singletons_updater,
            test_path=test_path
        )

    elif feat_alloc_updater_type == 'rg':
        feat_alloc_updater = pgfa.updates.RowGibbsUpdater(singletons_updater=singletons_updater)

    if mixture_prob > 0:
        feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(feat_alloc_updater, gibbs_prob=mixture_prob)

    return pgfa.models.pyclone.beta_binomial.ModelUpdater(feat_alloc_updater)


def load_data():
    file_name = 'data/pyclone/mixing.tsv'

    df = pd.read_csv(file_name, sep='\t')

    Z = []

    cases = set()

    for x in df['variant_cases'].unique():
        cases.update(set(x.split(',')))

    cases = sorted(cases)

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
                pgfa.models.pyclone.utils.get_sample_data_point(
                    row['a'], row['b'], row['cn_major'], row['cn_minor']
                )
            )

        data.append(pgfa.models.pyclone.utils.DataPoint(sample_data_points))

        Z_mut = np.zeros(4)

        for case in mut_df['variant_cases'].iloc[0].split(','):
            Z_mut[cases.index(case)] = 1

        Z.append(Z_mut)

    Z = np.array(Z, dtype=np.int8)

    return data, Z, mutations


if __name__ == '__main__':
    main()

